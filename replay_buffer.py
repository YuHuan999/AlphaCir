import copy
import time

import numpy
import ray
import torch

import models
from typing import Optional, Sequence, Dict, NamedTuple

from games.CirsysGame import Game
from games.CirsysGame import AlphaCirConfig
from self_play import Sample


#################################################################
####################### Replay_Buffer-Start- #######################
class ReplayBuffer(object):
  """Replay buffer object storing games for training."""

  def __init__(self, config: AlphaCirConfig):
    self.window_size = config.window_size #最多存多少局游戏（FIFO）
    self.batch_size = config.batch_size #每次训练采样多少个训练样本
    self.buffer = [] #保存所有游戏的列表，每一项是一个完整 Game 实例

  def save_game(self, game):
    if len(self.buffer) > self.window_size:
      self.buffer.pop(0)
    self.buffer.append(game)

  def sample_batch(self, td_steps: int) -> Sequence[Sample]:
    games = [self.sample_game() for _ in range(self.batch_size)]
    game_pos = [(g, self.sample_position(g)) for g in games]
    # pylint: disable=g-complex-comprehension
    return [
        Sample(
            observation=g.make_observation(i), ## observation该包括什么呢？
            bootstrap_observation=g.make_observation(i + td_steps), ## observation该包括什么呢？
            target=g.make_target(i, td_steps, g.to_play()),
        )
        for (g, i) in game_pos
    ]
    # pylint: enable=g-complex-comprehension

  def sample_game(self) -> Game:
    # Sample game from buffer either uniformly or according to some priority.
    return self.buffer[0]

  # pylint: disable-next=unused-argument
  def sample_position(self, game) -> int:
    # Sample position from game either uniformly or according to some priority.
    return -1






#################################################################
####################### Replay_Buffer- End - #######################


@ray.remote
class ReplayBuffer:
    """
    Class which run in a dedicated thread to store played games and generate batch.
    """

    def __init__(self, initial_checkpoint, initial_buffer, config):
        self.config = config
        self.buffer = copy.deepcopy(initial_buffer)
        self.num_played_games = initial_checkpoint["num_played_games"]
        self.num_played_steps = initial_checkpoint["num_played_steps"]
        self.total_samples = sum(
            [len(game_history.root_values) for game_history in self.buffer.values()]
        )
        if self.total_samples != 0:
            print(
                f"Replay buffer initialized with {self.total_samples} samples ({self.num_played_games} games).\n"
            )

        # Fix random generator seed
        numpy.random.seed(self.config.seed)
    
    ## finish a game save the game
    def save_game(self, game_history, shared_storage=None):
        if self.config.PER:
            if game_history.priorities is not None:
                # Avoid read only array when loading replay buffer from disk
                game_history.priorities = numpy.copy(game_history.priorities)
            else:
                # Initial priorities for the prioritized replay (See paper appendix Training)
                priorities = []
                for i, root_value in enumerate(game_history.root_values):
                    priority = (
                        numpy.abs(
                            root_value - self.compute_target_value(game_history, i)
                        )
                        ** self.config.PER_alpha
                    )
                    priorities.append(priority)

                game_history.priorities = numpy.array(priorities, dtype="float32")
                game_history.game_priority = numpy.max(game_history.priorities)

        self.buffer[self.num_played_games] = game_history
        self.num_played_games += 1
        self.num_played_steps += len(game_history.root_values)
        self.total_samples += len(game_history.root_values)

        if self.config.replay_buffer_size < len(self.buffer):
            del_id = self.num_played_games - len(self.buffer)
            self.total_samples -= len(self.buffer[del_id].root_values)
            del self.buffer[del_id]

        if shared_storage:
            shared_storage.set_info.remote("num_played_games", self.num_played_games)
            shared_storage.set_info.remote("num_played_steps", self.num_played_steps)

    def get_buffer(self):
        return self.buffer

    #### 这里, 了解如何 get batch
    def get_batch(self):
        (
            index_batch,
            observation_batch,
            action_batch,
            correctness_value_batch,
            length_value_batch,
            policy_batch,
            gradient_scale_batch,
        ) = ([], [], [], [], [], [], [])
        weight_batch = [] if self.config.PER else None

        for game_id, game_history, game_prob in self.sample_n_games(
            self.config.batch_size ## batch_size = n_games
        ):
            game_pos, pos_prob = self.sample_position(game_history)

            ## 这里准备 correctness_value length_value
            correctness_values, length_values, policies, actions = self.make_target(
                game_history, game_pos
            )

            index_batch.append([game_id, game_pos])
            observation_batch.append(
                game_history.get_stacked_observations(
                    game_pos,
                    self.config.stacked_observations,
                    len(self.config.action_space),
                )
            )
            action_batch.append(actions)
            correctness_value_batch.append(correctness_values)
            length_value_batch.append(length_values)
            policy_batch.append(policies)
            gradient_scale_batch.append(
                [
                    min(
                        self.config.num_unroll_steps,
                        len(game_history.action_history) - game_pos,
                    )
                ]
                * len(actions)
            )
            if self.config.PER:
                weight_batch.append(1 / (self.total_samples * game_prob * pos_prob))

        if self.config.PER:
            weight_batch = numpy.array(weight_batch, dtype="float32") / max(
                weight_batch
            )

        # observation_batch: batch, channels, height, width
        # action_batch: batch, num_unroll_steps+1
        # value_batch: batch, num_unroll_steps+1
        # reward_batch: batch, num_unroll_steps+1
        # policy_batch: batch, num_unroll_steps+1, len(action_space)
        # weight_batch: batch
        # gradient_scale_batch: batch, num_unroll_steps+1
        return (
            index_batch,
            (
                observation_batch,
                action_batch,
                correctness_value_batch,
                length_value_batch,
                policy_batch,
                weight_batch,
                gradient_scale_batch,
            ),
        )

    def sample_game(self, force_uniform=False):
        """
        Sample game from buffer either uniformly or according to some priority.
        See paper appendix Training.
        """
        game_prob = None
        if self.config.PER and not force_uniform:
            game_probs = numpy.array(
                [game_history.game_priority for game_history in self.buffer.values()],
                dtype="float32",
            )
            game_probs /= numpy.sum(game_probs)
            game_index = numpy.random.choice(len(self.buffer), p=game_probs)
            game_prob = game_probs[game_index]
        else:
            game_index = numpy.random.choice(len(self.buffer))
        game_id = self.num_played_games - len(self.buffer) + game_index

        return game_id, self.buffer[game_id], game_prob

    def sample_n_games(self, n_games, force_uniform=False):
        if self.config.PER and not force_uniform:
            game_id_list = []
            game_probs = []
            for game_id, game_history in self.buffer.items():
                game_id_list.append(game_id)
                game_probs.append(game_history.game_priority)
            game_probs = numpy.array(game_probs, dtype="float32")
            game_probs /= numpy.sum(game_probs)
            game_prob_dict = dict(
                [(game_id, prob) for game_id, prob in zip(game_id_list, game_probs)]
            )
            selected_games = numpy.random.choice(game_id_list, n_games, p=game_probs)
        else:
            selected_games = numpy.random.choice(list(self.buffer.keys()), n_games)
            game_prob_dict = {}
        ret = [
            (game_id, self.buffer[game_id], game_prob_dict.get(game_id))
            for game_id in selected_games
        ]
        return ret

    def sample_position(self, game_history, force_uniform=False):
        """
        Sample position from game either uniformly or according to some priority.
        See paper appendix Training.
        """
        position_prob = None
        if self.config.PER and not force_uniform:
            position_probs = game_history.priorities / sum(game_history.priorities)
            position_index = numpy.random.choice(len(position_probs), p=position_probs)
            position_prob = position_probs[position_index]
        else:
            position_index = numpy.random.choice(len(game_history.root_values))

        return position_index, position_prob

    def update_game_history(self, game_id, game_history):
        # The element could have been removed since its selection and update
        if next(iter(self.buffer)) <= game_id:
            if self.config.PER:
                # Avoid read only array when loading replay buffer from disk
                game_history.priorities = numpy.copy(game_history.priorities)
            self.buffer[game_id] = game_history

    def update_priorities(self, priorities, index_info):
        """
        Update game and position priorities with priorities calculated during the training.
        See Distributed Prioritized Experience Replay https://arxiv.org/abs/1803.00933
        """
        for i in range(len(index_info)):
            game_id, game_pos = index_info[i]

            # The element could have been removed since its selection and training
            if next(iter(self.buffer)) <= game_id:
                # Update position priorities
                priority = priorities[i, :]
                start_index = game_pos
                end_index = min(
                    game_pos + len(priority), len(self.buffer[game_id].priorities)
                )
                self.buffer[game_id].priorities[start_index:end_index] = priority[
                    : end_index - start_index
                ]

                # Update game priorities
                self.buffer[game_id].game_priority = numpy.max(
                    self.buffer[game_id].priorities
                )
    ## target value = discounted future value + discounted sum of rewards
    def compute_target_value(self, game_history, index):
        # The value target is the discounted root value of the search tree td_steps into the
        # future, plus the discounted sum of all rewards until then.
        bootstrap_index = index + self.config.td_steps
        if bootstrap_index < len(game_history.correstness_values):
            correstness_values = game_history.correstness_values
            length_values = game_history.length_values

            ## future consider
            ## set reanalysed_predicted_root_values None
            #  use the last model to provide the value after n-step
            #  if game_history.reanalysed_predicted_root_values is None 
            #  else game_history.reanalysed_predicted_root_values
            

            
            last_step_correstness = correstness_values[bootstrap_index] # only one player in this game 
            last_step_length = length_values[bootstrap_index]

            value_correstness = last_step_correstness * self.config.discount**self.config.td_steps
            value_length = last_step_length * self.config.discount**self.config.td_steps
        else:
            value_correstness, value_length  = 0

        for i, correctness_reward in enumerate(
            game_history.correstness_reward_history[index + 1 : bootstrap_index + 1]
        ):  
            length_reward = game_history.length_reward_history[i]

            # The value is oriented from the perspective of the current player
            value_correstness += (
                # reward
                # if game_history.to_play_history[index]
                # == game_history.to_play_history[index + i]
                # else -reward
                correctness_reward # only one player in this game
            ) * self.config.discount**i
            value_length += (
                length_reward 
            ) * self.config.discount**i
        return value_correstness, value_length
    def make_target(self, game_history, state_index):
        """
        Generate targets for every unroll steps.
        """
        target_correctness_values, target_length_values, target_policies, actions = [], [], [], []
        for current_index in range(
            state_index, state_index + self.config.num_unroll_steps + 1
        ):
            value_correctness, value_length = self.compute_target_value(game_history, current_index)

            if current_index < len(game_history.root_values):
                target_correctness_values.append(value_correctness)
                target_length_values.append(value_length)
                target_policies.append(game_history.child_visits[current_index])
                actions.append(game_history.action_history[current_index])
            elif current_index == len(game_history.root_values):
                target_correctness_values.append(0)
                target_length_values.append(0)
                # Uniform policy
                target_policies.append(
                    [
                        1 / len(game_history.child_visits[0])
                        for _ in range(len(game_history.child_visits[0]))
                    ]
                )
                actions.append(game_history.action_history[current_index])
            else:
                # States past the end of games are treated as absorbing states
                target_correctness_values.append(0)
                target_length_values.append(0)
                # Uniform policy
                target_policies.append(
                    [
                        1 / len(game_history.child_visits[0])
                        for _ in range(len(game_history.child_visits[0]))
                    ]
                )
                actions.append(numpy.random.choice(self.config.action_space))

        return target_correctness_values, target_length_values, target_policies, actions


@ray.remote
class Reanalyse:
    """
    Class which run in a dedicated thread to update the replay buffer with fresh information.
    See paper appendix Reanalyse.
    """

    def __init__(self, initial_checkpoint, config):
        self.config = config

        # Fix random generator seed
        numpy.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        # Initialize the network
        self.model = models.MuZeroNetwork(self.config)
        self.model.set_weights(initial_checkpoint["weights"])
        self.model.to(torch.device("cuda" if self.config.reanalyse_on_gpu else "cpu"))
        self.model.eval()

        self.num_reanalysed_games = initial_checkpoint["num_reanalysed_games"]

    def reanalyse(self, replay_buffer, shared_storage):
        while ray.get(shared_storage.get_info.remote("num_played_games")) < 1:
            time.sleep(0.1)

        while ray.get(
            shared_storage.get_info.remote("training_step")
        ) < self.config.training_steps and not ray.get(
            shared_storage.get_info.remote("terminate")
        ):
            self.model.set_weights(ray.get(shared_storage.get_info.remote("weights")))

            game_id, game_history, _ = ray.get(
                replay_buffer.sample_game.remote(force_uniform=True)
            )

            # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
            if self.config.use_last_model_value:
                observations = numpy.array(
                    [
                        game_history.get_stacked_observations(
                            i,
                            self.config.stacked_observations,
                            len(self.config.action_space),
                        )
                        for i in range(len(game_history.root_values))
                    ]
                )

                observations = (
                    torch.tensor(observations)
                    .float()
                    .to(next(self.model.parameters()).device)
                )
                values = models.support_to_scalar(
                    self.model.initial_inference(observations)[0],
                    self.config.support_size,
                )
                game_history.reanalysed_predicted_root_values = (
                    torch.squeeze(values).detach().cpu().numpy()
                )

            replay_buffer.update_game_history.remote(game_id, game_history)
            self.num_reanalysed_games += 1
            shared_storage.set_info.remote(
                "num_reanalysed_games", self.num_reanalysed_games
            )
