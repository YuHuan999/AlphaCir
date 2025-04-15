import math
import time

import numpy as np
import ray
import torch

import models
import copy

from collections import namedtuple
from typing import Optional, Sequence, Dict, NamedTuple, Any, Tuple, List

import random

from models import AlphaCirNetwork, NetworkOutput

from games.CirsysGame import Game
from games.CirsysGame import AlphaCirConfig

from shared_storage import SharedStorage
from replay_buffer import ReplayBuffer
#################################################################
####################### Self_play-Start- #######################

MAXIMUM_FLOAT_VALUE = float('inf')

KnownBounds = namedtuple('KnownBounds', ['min', 'max'])

class Action(object):
  """Action representation."""

  def __init__(self, index: int):
    self.index = index

  def __hash__(self):
    return self.index

  def __eq__(self, other):
    return self.index == other.index

  def __gt__(self, other):
    return self.index > other.index

class MinMaxStats(object):
  """A class that holds the min-max values of the tree."""

  def __init__(self, known_bounds: Optional[KnownBounds]):
    self.maximum = known_bounds.max if known_bounds else -MAXIMUM_FLOAT_VALUE
    self.minimum = known_bounds.min if known_bounds else MAXIMUM_FLOAT_VALUE

  def update(self, value: float):
    self.maximum = max(self.maximum, value)
    self.minimum = min(self.minimum, value)

  def normalize(self, value: float) -> float:
    if self.maximum > self.minimum:
      # We normalize only when we have set the maximum and minimum values.
      return (value - self.minimum) / (self.maximum - self.minimum)
    return value

class Player:
    def __init__(self, player_id, player_type="self_play"):
        self.player_id = player_id      # 例如 0 或 1
        self.player_type = player_type  # "human" 或 "self_play"

    def __repr__(self):
        return f"Player({self.player_id}, type={self.player_type})"

def softmax_sample(visit_counts: List[Tuple[int, Any]], temperature: float):
    visits = np.array([v for v, _ in visit_counts], dtype=np.float32)
    
    if temperature == 0.0:
        # 贪婪：选择访问次数最多的动作
        max_visit = np.max(visits)
        indices = np.where(visits == max_visit)[0]
        selected = random.choice(indices)
        return visit_counts[selected]
    
    # 否则按 softmax 分布采样
    visits = visits ** (1 / temperature)  # 温度调整
    probs = visits / np.sum(visits)      # softmax 概率
    index = np.random.choice(len(visit_counts), p=probs)
    return visit_counts[index]


class ActionHistory(object):
  """Simple history container used inside the search.

  Only used to keep track of the actions executed.
  """

  def __init__(self, history: Sequence[Action], action_space_size: int):
    self.history = list(history)
    self.action_space_size = action_space_size

  def clone(self):
    return ActionHistory(self.history, self.action_space_size)

  def add_action(self, action: Action):
    self.history.append(action)

  def last_action(self) -> Action:
    return self.history[-1]

  def action_space(self) -> Sequence[Action]:
    return [Action(i) for i in range(self.action_space_size)]

  def to_play(self) -> Player:
    return Player()

class Node(object):
  """MCTS node."""

  def __init__(self, prior: float):
    self.visit_count = 0         # 当前节点被访问了多少次
    self.to_play = -1            # 当前轮到哪个玩家（-1为默认值）
    self.prior = prior           # 从策略网络中获得的先验概率
    self.value_sum = 0           # 当前节点累积的 value 值
    self.children = {}           # 子节点，结构为 action -> Node
    self.hidden_state = None     # 用于存储模型隐状态（在 MuZero 中很关键）
    self.reward = 0              # 当前节点上的即时奖励

  def expanded(self) -> bool:
    return bool(self.children)

  def value(self) -> float:
    if self.visit_count == 0:
      return 0
    return self.value_sum / self.visit_count
  
  def __repr__(self):
    return (
        f"Node("
        f"prior={self.prior:.4f}, "
        f"visits={self.visit_count}, "
        f"value={self.value():.4f}, "
        f"value_sum={self.value_sum:.4f}, "
        f"reward={self.reward:.4f}, "
        f"num_children={len(self.children)}"
        f")"
    )

# Each self-play job is independent of all others; it takes the latest network
# snapshot, produces a game and makes it available to the training job by
# writing it to a shared replay buffer.
def run_selfplay(
    config: AlphaCirConfig, storage: SharedStorage, replay_buffer: ReplayBuffer
):
  while True:
    network = storage.latest_network()
    game = play_game(config, network)
    replay_buffer.save_game(game)


def play_game(config: AlphaCirConfig, network: AlphaCirNetwork) -> Game:
  """Plays an AlphaCir game.

  Each game is produced by starting at the initial empty program, then
  repeatedly executing a Monte Carlo Tree Search to generate moves until the end
  of the game is reached.

  Args:
    config: An instance of the AlphaDev configuration.
    network: Networks used for inference.

  Returns:
    The played game.
  """

  game = Game(config.task_spec) 

  while not game.terminal() and len(game.history) < config.max_moves:
    min_max_stats = MinMaxStats(config.known_bounds)

    # Initialisation of the root node and addition of exploration noise
    root = Node(0)
    current_observation = game.make_observation(-1)
    network_output = network.inference(current_observation)
    _expand_node(
        root, game.to_play(), game.legal_actions(), network_output, reward=0
    )
    _backpropagate(
        [root],
        network_output.value,
        game.to_play(),
        config.discount,
        min_max_stats,
    )
    _add_exploration_noise(config, root)

    # We then run a Monte Carlo Tree Search using the environment.
    run_mcts(
        config,
        root,
        game.action_history(),
        network,
        min_max_stats,
        game,
    )
    action = _select_action(config, len(game.history), root, network)
    game.apply(action)
    game.store_search_statistics(root)
  return game


def run_mcts(
    config: AlphaCirConfig,
    root: Node,
    action_history: ActionHistory,
    network: AlphaCirNetwork,
    min_max_stats: MinMaxStats,
    env: Game,
):
  """Runs the Monte Carlo Tree Search algorithm.

  To decide on an action, we run N simulations, always starting at the root of
  the search tree and traversing the tree according to the UCB formula until we
  reach a leaf node.

  Args:
    config: AlphaDev configuration
    root: The root node of the MCTS tree from which we start the algorithm
    action_history: history of the actions taken so far.
    network: instances of the networks that will be used.
    min_max_stats: min-max statistics for the tree.
    env: an instance of the AssemblyGame.
  """

  for _ in range(config.num_simulations):
    history = action_history.clone()
    node = root
    search_path = [node]
    sim_env = env.clone()

    while node.expanded():
      action, node = _select_child(config, node, min_max_stats)
      sim_env.step(action)
      history.add_action(action)
      search_path.append(node)

    # Inside the search tree we use the environment to obtain the next
    # observation and reward given an action.
    observation, reward = sim_env.step(action)
    network_output = network.inference(observation)
    _expand_node(
        node, history.to_play(), history.action_space(), network_output, reward
    )

    _backpropagate(
        search_path,
        network_output.value,
        history.to_play(),
        config.discount,
        min_max_stats,
    )


def _select_action(
    # pylint: disable-next=unused-argument
    config: AlphaCirConfig, num_moves: int, node: Node, network: AlphaCirNetwork
):
  visit_counts = [
      (child.visit_count, action) for action, child in node.children.items()
  ]
  t = config.visit_softmax_temperature_fn(
      training_steps=network.training_steps()
  )
  _, action = softmax_sample(visit_counts, t)
  return action


def _select_child(
    config: AlphaCirConfig, node: Node, min_max_stats: MinMaxStats
):
  """Selects the child with the highest UCB score."""
  _, action, child = max(
      (_ucb_score(config, node, child, min_max_stats), action, child)
      for action, child in node.children.items()
  )
  return action, child


def _ucb_score(
    config: AlphaCirConfig,
    parent: Node,
    child: Node,
    min_max_stats: MinMaxStats,
) -> float:
  """Computes the UCB score based on its value + exploration based on prior."""
  pb_c = (
      math.log((parent.visit_count + config.pb_c_base + 1) / config.pb_c_base)
      + config.pb_c_init
  )
  pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

  prior_score = pb_c * child.prior
  if child.visit_count > 0:
    value_score = min_max_stats.normalize(
        child.reward + config.discount * child.value()
    )
  else:
    value_score = 0
  return prior_score + value_score


def _expand_node(
    node: Node,
    to_play: Player,
    actions: Sequence[Action],
    network_output: NetworkOutput,
    reward: float,
):
  """Expands the node using value, reward and policy predictions from the NN."""
  node.to_play = to_play
  node.hidden_state = network_output.hidden_state
  node.reward = reward
  policy = {a: math.exp(network_output.policy_logits[a]) for a in actions}
  policy_sum = sum(policy.values())
  for action, p in policy.items():
    node.children[action] = Node(p / policy_sum)


def _backpropagate(
    search_path: Sequence[Node],
    value: float,
    to_play: Player,
    discount: float,
    min_max_stats: MinMaxStats,
):
  """Propagates the evaluation all the way up the tree to the root."""
  for node in reversed(search_path):
    node.value_sum += value if node.to_play == to_play else -value
    node.visit_count += 1
    min_max_stats.update(node.value())

    value = node.reward + discount * value


def _add_exploration_noise(config: AlphaCirConfig, node: Node):
  """Adds dirichlet noise to the prior of the root to encourage exploration."""
  actions = list(node.children.keys())
  noise = np.random.dirichlet([config.root_dirichlet_alpha] * len(actions))
  frac = config.root_exploration_fraction
  for a, n in zip(actions, noise):
    node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac




#################################################################
####################### Self_play- End - #######################











@ray.remote
class SelfPlay:
    """
    Class which run in a dedicated thread to play games and save them to the replay-buffer.
    """

    def __init__(self, initial_checkpoint, Game, config, seed):
        self.config = config
        self.game = Game(seed = seed)

        # Fix random generator seed
        numpy.random.seed(seed)
        torch.manual_seed(seed)

        # Initialize the network
        self.model = models.MuZeroNetwork(self.config)
        self.model.set_weights(initial_checkpoint["weights"])
        self.model.to(torch.device("cuda" if self.config.selfplay_on_gpu else "cpu"))
        self.model.eval()

    def continuous_self_play(self, shared_storage, replay_buffer, test_mode=False):
        while ray.get(
            shared_storage.get_info.remote("training_step")
        ) < self.config.training_steps and not ray.get(
            shared_storage.get_info.remote("terminate")
        ):
            self.model.set_weights(ray.get(shared_storage.get_info.remote("weights")))

            if not test_mode:
                game_history = self.play_game(
                    self.config.visit_softmax_temperature_fn(
                        trained_steps=ray.get(
                            shared_storage.get_info.remote("training_step")
                        )
                    ),
                    self.config.temperature_threshold,
                    False,
                    "self",
                    0,
                )

                replay_buffer.save_game.remote(game_history, shared_storage)

            ## test mode
            else:
                # Take the best action (no exploration) in test mode
                game_history = self.play_game(
                    0,
                    self.config.temperature_threshold,
                    False,
                    "self" if len(self.config.players) == 1 else self.config.opponent,
                    self.config.muzero_player,
                )

                # Save to the shared storage
                shared_storage.set_info.remote(
                    {
                        "episode_length": len(game_history.action_history) - 1,
                        "total_reward": sum(game_history.reward_history),
                        "mean_value": numpy.mean(
                            [value for value in game_history.root_values if value]
                        ),
                    }
                )
                if 1 < len(self.config.players):
                    shared_storage.set_info.remote(
                        {
                            "muzero_reward": sum(
                                reward
                                for i, reward in enumerate(game_history.reward_history)
                                if game_history.to_play_history[i - 1]
                                == self.config.muzero_player
                            ),
                            "opponent_reward": sum(
                                reward
                                for i, reward in enumerate(game_history.reward_history)
                                if game_history.to_play_history[i - 1]
                                != self.config.muzero_player
                            ),
                        }
                    )

            # Managing the self-play / training ratio
            if not test_mode and self.config.self_play_delay:
                time.sleep(self.config.self_play_delay)
            if not test_mode and self.config.ratio:
                while (
                    ray.get(shared_storage.get_info.remote("training_step"))
                    / max(
                        1, ray.get(shared_storage.get_info.remote("num_played_steps"))
                    )
                    < self.config.ratio
                    and ray.get(shared_storage.get_info.remote("training_step"))
                    < self.config.training_steps
                    and not ray.get(shared_storage.get_info.remote("terminate"))
                ):
                    time.sleep(0.5)

        self.close_game()

    def play_game(
        self, temperature, temperature_threshold, render, muzero_player, opponent = "self"
    ):
        """
        Play one game with actions based on the Monte Carlo tree search at each moves.
        """
        game_history = GameHistory()
        state = self.game.reset()
        game_history.action_history = self.game.actions
        game_history.state_history.append(state)
        game_history.correstness_reward_history = self.game.correstness_rewards ## correstness_rewards is list
        game_history.length_reward_history = self.game.length_rewards ## length_rewards is list
        game_history.to_play_history.append(self.game.to_play())

        done = False

        if render:
            self.game.render()

        with torch.no_grad():
            while (
                not done and len(game_history.action_history) <= self.config.max_moves
            ):
                # assert (
                #     numpy.array(state).shape == self.config.observation_shape
                # ), f"Observation should match the observation_shape defined in MuZeroConfig. Expected {self.config.observation_shape} but got {numpy.array(observation).shape}."
                
                stacked_observations = game_history.state_history[-1]

                # Choose the action
                if opponent == "self" or muzero_player == self.game.to_play():
                    ## MCTS in real env
                    root, mcts_info = MCTS(self.config).run(
                        self.game,
                        self.model,
                        stacked_observations,
                        self.game.legal_actions(),
                        self.game.to_play(),
                        True, #add exploration noise
                    )

                    action = self.select_action(
                        root,
                        temperature
                        if not temperature_threshold
                        or len(game_history.action_history) < temperature_threshold
                        else 0,
                    )

                    if render:
                        print(f'Tree depth: {mcts_info["max_tree_depth"]}')
                        print(
                            f"Root value for player {self.game.to_play()}: {root.value():.2f}"
                        )
                
                else: ## ignore
                    action, root = self.select_opponent_action(
                        opponent, stacked_observations )
                
                is_target, state, correstness_reward, length_reward, done = self.game.step(action)

                if render:
                    print(f"Played action: {self.game.action_to_string(action)}")
                    self.game.render()

                game_history.store_search_statistics(root, self.config.action_space)

                # Next batch
                game_history.state_history.append(state)
                game_history.correstness_reward_history.append(correstness_reward)
                game_history.length_reward_history.append(length_reward)
                game_history.to_play_history.append(self.game.to_play())

        return game_history

    def close_game(self):
        self.game.close()

    def select_opponent_action(self, opponent, stacked_observations):
        """
        Select opponent action for evaluating MuZero level.
        """
        if opponent == "human":
            root, mcts_info = MCTS(self.config).run(
                self.model,
                stacked_observations,
                self.game.legal_actions(),
                self.game.to_play(),
                True, 
            )
            print(f'Tree depth: {mcts_info["max_tree_depth"]}')
            print(f"Root value for player {self.game.to_play()}: {root.value():.2f}")
            print(
                f"Player {self.game.to_play()} turn. MuZero suggests {self.game.action_to_string(self.select_action(root, 0))}"
            )
            return self.game.human_to_action(), root
        elif opponent == "expert":
            return self.game.expert_agent(), None
        elif opponent == "random":
            assert (
                self.game.legal_actions()
            ), f"Legal actions should not be an empty array. Got {self.game.legal_actions()}."
            assert set(self.game.legal_actions()).issubset(
                set(self.config.action_space)
            ), "Legal actions should be a subset of the action space."

            return numpy.random.choice(self.game.legal_actions()), None
        else:
            raise NotImplementedError(
                'Wrong argument: "opponent" argument should be "self", "human", "expert" or "random"'
            )

    @staticmethod
    def select_action(node, temperature):
        """
        Select action according to the visit count distribution and the temperature.
        The temperature is changed dynamically with the visit_softmax_temperature function
        in the config.
        """
        visit_counts = numpy.array(
            [child.visit_count for child in node.children.values()], dtype="int32"
        )
        actions = [action for action in node.children.keys()]
        if temperature == 0:
            action = actions[numpy.argmax(visit_counts)]
        elif temperature == float("inf"):
            action = numpy.random.choice(actions)
        else:
            # See paper appendix Data Generation
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(
                visit_count_distribution
            )
            action = numpy.random.choice(actions, p=visit_count_distribution)

        return action


# Game independent
class MCTS:
    """
    Core Monte Carlo Tree Search algorithm.
    To decide on an action, we run N simulations, always starting at the root of
    the search tree and traversing the tree according to the UCB formula until we
    reach a leaf node.
    """

    def __init__(self, config):
        self.config = config
        

    def run(
        self,
        game,   
        model,
        observation,
        legal_actions,
        to_play,
        add_exploration_noise,
        override_root_with=None,
    ):
        """Runs the Monte Carlo Tree Search algorithm.

        To decide on an action, we run N simulations, always starting at the root of
        the search tree and traversing the tree according to the UCB formula until we
        reach a leaf node.
        """

        if override_root_with:
            root = override_root_with
            root_predicted_value = None
        else:
            ## initialize the root node
            root = Node(0)
            observation = (
                torch.tensor(observation)
                .float()
                .unsqueeze(0)
                .to(next(model.parameters()).device)
            )
            (   
                root_predicted_value, ## value = correctness_value["mean"] + cirlength_value["mean"]
                correctness_value_logits,
                length_value_logits, 
                policy_logits,  
            ) = model.initial_inference(observation)

            assert (
                legal_actions
            ), f"Legal actions should not be an empty array. Got {legal_actions}."
            assert set(legal_actions).issubset(
                set(self.config.action_space)
            ), "Legal actions should be a subset of the action space."
            
            root.expand(
                legal_actions,
                to_play,
                policy_logits,
            )

        if add_exploration_noise:
            root.add_exploration_noise(
                dirichlet_alpha=self.config.root_dirichlet_alpha,
                exploration_fraction=self.config.root_exploration_fraction,
            )

        ## run in mcts 
        min_max_stats = MinMaxStats()
        max_tree_depth = 0
        for _ in range(self.config.num_simulations):
            virtual_to_play = to_play
            node = root
            search_path = [node]
            current_tree_depth = 0
            game_inside = copy.deepcopy(game)

            while node.expanded():
                current_tree_depth += 1
                action, node = self.select_child(node, min_max_stats)
                is_target, state, corrcectness_reward, length_reward, done = game_inside.step(action)
                node.correctness_reward = corrcectness_reward
                node.length_reward = length_reward  
                search_path.append(node)

                # Players play turn by turn
                if virtual_to_play + 1 < len(self.config.players):
                    virtual_to_play = self.config.players[virtual_to_play + 1]
                else:
                    virtual_to_play = self.config.players[0]

            # Inside the search tree we use the environment to obtain the next
            # observation and reward given an action.
            
            state = game_inside.states[-1]
            legal_actions = game_inside.legal_actions()
            value, correctness_value_logits, length_value_logits, policy_logits = model.initial_inference(state)
            correctness_value = model.support_to_scalar_simply(correctness_value_logits, self.config.support_size)
            length_value = model.support_to_scalar_simply(length_value_logits, self.config.support_size)
            
            node.expand(
                legal_actions,
                virtual_to_play,
                policy_logits
            )

            self.backpropagate(search_path, value, correctness_value, length_value, virtual_to_play, min_max_stats)

            max_tree_depth = max(max_tree_depth, current_tree_depth)

        extra_info = {
            "max_tree_depth": max_tree_depth,
            "root_predicted_value": root_predicted_value,
        }
        return root, extra_info

    def select_child(self, node, min_max_stats):
        """
        Select the child with the highest UCB score.
        """
        max_ucb = max(
            self.ucb_score(node, child, min_max_stats)
            for action, child in node.children.items()
        )
        action = numpy.random.choice(
            [
                action
                for action, child in node.children.items()
                if self.ucb_score(node, child, min_max_stats) == max_ucb
            ]
        )
        return action, node.children[action]

    def ucb_score(self, parent, child, min_max_stats):
        """
        The score for a node is based on its value, plus an exploration bonus based on the prior.
        """
        pb_c = (
            math.log(
                (parent.visit_count + self.config.pb_c_base + 1) / self.config.pb_c_base
            )
            + self.config.pb_c_init
        )
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

        prior_score = pb_c * child.prior

        if child.visit_count > 0:
            # Mean value Q
            value_score = min_max_stats.normalize(
                child.reward
                + self.config.discount
                * (child.value() if len(self.config.players) == 1 else -child.value())
            )
        else:
            value_score = 0

        return prior_score + value_score

    def backpropagate(self, search_path, value, correctness_value, length_value,  to_play, min_max_stats):
        """
        At the end of a simulation, we propagate the evaluation all the way up the tree
        to the root.
        """
        if len(self.config.players) == 1:
            for node in reversed(search_path):
                node.correctness_value_sum += correctness_value
                node.length_value_sum += length_value
                node.visit_count += 1
                min_max_stats.update(node.reward + self.config.discount * node.value())  
                correctness_value = node.correctness_reward + self.config.discount * correctness_value
                length_value = node.length_reward + self.config.discount * length_value


        elif len(self.config.players) == 2:
            for node in reversed(search_path):
                node.value_sum += value if node.to_play == to_play else -value
                node.visit_count += 1
                min_max_stats.update(node.reward + self.config.discount * -node.value())

                value = (
                    -node.reward if node.to_play == to_play else node.reward
                ) + self.config.discount * value

        else:
            raise NotImplementedError("More than two player mode not implemented.")


class Node:
    def __init__(self, prior):
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.value = 0
        self.correctness_value_sum = 0
        self.correctness_value = 0
        self.length_value_sum = 0
        self.length_value = 0
        self.children = {}
        self.hidden_state = None
        self.correctness_reward = 0
        self.length_reward = 0
        
        

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        self.correctness_value = self.correctness_value_sum / self.visit_count
        self.length_value = self.length_value_sum / self.visit_count    
        self.value = self.correctness_value + self.length_value
        return self.value

    def expand(self, actions, to_play, policy_logits):
        """
        We expand nodes next layer based the legal actions and policy distribution by NN.
        """
        self.to_play = to_play

        policy_values = torch.softmax(
            torch.tensor([policy_logits[0][a] for a in actions]), dim=0
        ).tolist()
        policy = {a: policy_values[i] for i, a in enumerate(actions)}
        for action, p in policy.items():
            self.children[action] = Node(p)

    def add_exploration_noise(self, dirichlet_alpha, exploration_fraction):
        """
        At the start of each search, we add dirichlet noise to the prior of the root to
        encourage the search to explore new actions.
        """
        actions = list(self.children.keys())
        noise = numpy.random.dirichlet([dirichlet_alpha] * len(actions))
        frac = exploration_fraction
        for a, n in zip(actions, noise):
            self.children[a].prior = self.children[a].prior * (1 - frac) + n * frac


class GameHistory:
    """
    Store only usefull information of a self-play game.
    """

    def __init__(self):
        self.observation_history = []
        self.action_history = []
        self.correstness_reward_history = []
        self.length_reward_history = []
        self.to_play_history = []
        self.child_visits = []
        self.root_correctness_values = []
        self.root_length_values = []
        self.reanalysed_predicted_root_values = None
        # For PER
        self.priorities = None
        self.game_priority = None

    def store_search_statistics(self, root, action_space):
        # Turn visit count from root into a policy
        if root is not None:
            sum_visits = sum(child.visit_count for child in root.children.values())
            self.child_visits.append(
                [
                    root.children[a].visit_count / sum_visits
                    if a in root.children
                    else 0
                    for a in action_space
                ]
            )
            root.value() ## update the value of root
            self.root_correctness_values.append(root.correctness_value)
            self.root_length_values.append(root.length_value)
        else:
            # self.root_values.append(None)
            root.value() ## update the value of root
            self.root_correctness_values.append(None)
            self.root_length_values.append(None)

    def get_stacked_observations(
        self, index, num_stacked_observations, action_space_size
    ):
        """
        Generate a new observation with the observation at the index position
        and num_stacked_observations past observations and actions stacked.
        
        observation is  circuit
        
        """
        # Convert to positive index
        index = index % len(self.observation_history)

        stacked_observations = self.observation_history[index].copy()
        for past_observation_index in reversed(
            range(index - num_stacked_observations, index)
        ):
            if 0 <= past_observation_index:
                previous_observation = numpy.concatenate(
                    (
                        self.observation_history[past_observation_index],
                        [
                            numpy.ones_like(stacked_observations[0])
                            * self.action_history[past_observation_index + 1]
                            / action_space_size
                        ],
                    )
                )
            else:
                previous_observation = numpy.concatenate(
                    (
                        numpy.zeros_like(self.observation_history[index]),
                        [numpy.zeros_like(stacked_observations[0])],
                    )
                )

            stacked_observations = numpy.concatenate(
                (stacked_observations, previous_observation)
            )

        return stacked_observations


class MinMaxStats:
    """
    A class that holds the min-max values of the tree.
    """

    def __init__(self):
        self.maximum = -float("inf")
        self.minimum = float("inf")

    def update(self, value):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value):
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value
