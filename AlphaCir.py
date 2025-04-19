import datetime
import pathlib

from tqdm import tqdm

import math
import copy
import numpy as np
import random
import ml_collections

import torch
import torch.nn as nn
import torch.nn.functional as F

import cirq
from qiskit import QuantumCircuit
import qiskit.circuit.library as qlib
from qiskit.quantum_info import Operator

from collections import namedtuple
from typing import Optional, Sequence, Dict, NamedTuple, Any, Tuple, List

from games.abstract_game import AbstractGame # for local test
# from abstract_game import AbstractGame  

import pdb
# Test model
from global_config import test_model, test_model_run
#################################################################
######################### Tools -Start- #######################

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
    # if test_model:
    #   print("value_sum init", self.value_sum, type(self.value_sum))
  def expanded(self) -> bool:
    return bool(self.children)

  def value(self) -> float:
    if self.visit_count == 0:
      return 0
    # if test_model:
    #   print("visit_count", self.visit_count, type(self.visit_count))
    #   print("value_sum", self.value_sum, type(self.value_sum))
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

class Player:
    def __init__(self, player_id, player_type="self_play"):
        self.player_id = player_id      # 例如 0 或 1
        self.player_type = player_type  # "human" 或 "self_play"

    def __repr__(self):
        return f"Player({self.player_id}, type={self.player_type})"

GATES = {"H": 0, "S": 1, "T": 2, "S†": 3, "T†": 4}

def index2action(index: int, num_qubits: int, num_actions: int):
    """
    将 index 转换为动作，返回 [gate, location, control]。
    """
    location, ope = divmod(index, num_actions // num_qubits)
    
    if ope < 5:
        gates = ("H", "S", "T", "S†", "T†")
        return [gates[ope], location, location]
    else:
        op = ope - 5
        control = op if op < location else op + 1
        return ["CX", location, control]
    
## action:["CX", location, control]
def action2index(action, num_qubits: int, num_actions: int) -> int: 
    gate, location, control = action
    num_actions_per_qubit = num_actions // num_qubits
    if gate != "CX":
        ope = GATES[gate]
    else:
        op = control if control < location else control - 1
        ope = 5 + op
    return location * num_actions_per_qubit + ope

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

class ActionHistory(object):
  """Simple history container used inside the search.

  Only used to keep track of the actions executed.
  """

  def __init__(self, history: Sequence[Action], action_space_size: int): ## action_space_size 所有可能的门，包括作用在不同量子比特上的相同门
    self.history = list(history) ## list of gate(Action) index
    self.action_space_size = action_space_size

  def clone(self):
    return copy.deepcopy(ActionHistory(self.history, self.size))

  def add_action(self, action: Action):
    self.history.append(action.index)

  def last_action(self) -> Action:
    return self.history[-1]
  ## action space 给出的是 Action list
  def action_space(self) -> Sequence[Action]:
    return [Action(i) for i in range(self.action_space_size)]

  def to_play(self) -> Player:
    return Player(0) ## only self-play


def soft_cross_entropy(logits, target_probs):
    log_probs = F.log_softmax(logits, dim=-1)
    return -(target_probs * log_probs).sum(dim=-1).mean()


def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    if test_model:
        print("test model collate_batch", batch)
        for item in batch:
            print("test model circuit", item["circuit"])
        


    return {
        "circuits": torch.stack([torch.tensor(obs["circuit"]) for obs in batch], dim=0),
        "circuits_length": torch.tensor([obs["circuit_length"] for obs in batch], dtype=torch.long),
        "matrices": torch.stack([torch.tensor(obs["matrix"]) for obs in batch])
    }


#################################################################
######################### Tools - End - #######################



#################################################################
######################## Configs -Start- ######################

class TaskSpec(NamedTuple):
    max_circuit_length: int 
    num_qubits: int
    fidelity_reward: float 
    fidelity_reward_weight: float 
    length_reward: float 
    length_reward_weight: float 
    num_ops: int  # type of gates H S T S† T† CX
    num_actions: int # number of actions, gates in different qubits 
    fidelity_threshold: float # 保真度阈值 less than 1.0
    actions_index: List[int]
    discount: float = 0.99 
    
    @classmethod
    def create(cls, max_circuit_length: int, num_qubits: int, num_ops: int,
               fidelity_reward: float, fidelity_reward_weight: float,
               length_reward: float, length_reward_weight: float, fidelity_threshold: float, discount: float = 0.99):   
        """
        类方法，先做计算，再实例化 NamedTuple
        """
        num_actions = num_qubits * ((num_ops - 1) + (num_qubits - 1))
        actions_index = list(range(num_actions))
        return cls(
            max_circuit_length=max_circuit_length,
            num_qubits=num_qubits,
            num_ops=num_ops,
            fidelity_reward=fidelity_reward,
            fidelity_reward_weight=fidelity_reward_weight,
            length_reward = length_reward,
            length_reward_weight=length_reward_weight,
            fidelity_threshold=fidelity_threshold,
            num_actions=num_actions,
            actions_index=actions_index,
            discount=discount,
        )
    
class AlphaCirConfig(object):
  """AlphaCir configuration."""  
  # need to be changed

  def __init__(
      self,
  ):
    ### Self-Play
    self.num_actors = 128  # TPU actors
    # pylint: disable-next=g-long-lambda
    self.visit_softmax_temperature_fn = lambda steps: (
        1.0 if steps < 500e3 else 0.5 if steps < 750e3 else 0.25
    )
    self.max_moves = 10   ## 10 for test
    self.num_simulations = 80 ## 80 for test
    self.discount = 1.0

    # Root prior exploration noise.
    self.root_dirichlet_alpha = 0.03
    self.root_exploration_fraction = 0.25

    # UCB formula
    self.pb_c_base = 19652
    self.pb_c_init = 1.25

    self.known_bounds = KnownBounds(-6.0, 6.0)

    # Environment: spec of the Variable Sort 3 task
    self.task_spec = TaskSpec.create(
        max_circuit_length=50,  # 50
        num_qubits=2,
        num_ops=6,  # type of all gates H S T S† T† CX
        fidelity_reward=1.0,
        fidelity_reward_weight=1.0,
        length_reward=-0.1,
        length_reward_weight=0.01,
        fidelity_threshold=0.99,  # 保真度阈值 less than 1.0
        discount=0.99
    )

    ### Network architecture
    self.hparams = ml_collections.ConfigDict()
    self.hparams.embedding_dim = 512
    self.hparams.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.hparams.representation = ml_collections.ConfigDict()
    self.hparams.representation.input_dim = 10 # 输入维度，通常是门的 one-hot 编码长度
    self.hparams.representation.head_depth = 128
    self.hparams.representation.num_heads = 4
    self.hparams.representation.dropout = 0.1
    self.hparams.representation.num_mqa_blocks = 6
    
    # self.hparams.representation.attention.position_encoding = 'absolute'
    # self.hparams.representation.repr_net_res_blocks = 8
    # self.hparams.representation.attention = ml_collections.ConfigDict()
    # self.hparams.representation.use_program = True
    # self.hparams.representation.use_locations = True
    # self.hparams.representation.use_locations_binary = False
    # self.hparams.representation.use_permutation_embedding = False

    self.hparams.prediction = ml_collections.ConfigDict()
    self.hparams.prediction.value_max = 3.0  # These two parameters are task / reward-
    self.hparams.prediction.num_bins = 51  # dependent and need to be adjusted.
    self.hparams.prediction.num_resblocks = 1  # dependent and need to be adjusted.
    self.hparams.prediction.hidden_dim = 512  # hidden_dim of the prediction network
    
    ### Training
    self.training_steps = int(1000e3)
    self.checkpoint_interval = 500
    self.target_network_interval = 100
    self.window_size = int(1e6)
    self.batch_size = 8 ## 8 for test 
    self.td_steps = 5
    self.lr_init = 2e-4
    self.momentum = 0.9

    # Build action maps
    self._build_action_maps()


  def new_game(self):
    return Game(self.task_spec)
  
  def _build_action_maps(self):
    """构建 index <-> action:["CX", location, control] 的映射表."""
    self.index2action_map = {}
    self.action2index_map = {}
    num_qubits = self.task_spec.num_qubits
    num_actions = self.task_spec.num_actions
    num_per_qubit = num_actions // num_qubits

    gates = ["H", "S", "T", "S†", "T†"]
    idx = 0
    for location in range(num_qubits):
        for op in range(num_per_qubit):
            if op < 5:
                action = [gates[op], location, location]
            else:
                ctrl = op - 5
                control = ctrl if ctrl < location else ctrl + 1
                action = ["CX", location, control]
            self.index2action_map[idx] = action
            self.action2index_map[tuple(action)] = idx
            idx += 1

    def get_action_from_index(self, index: int):
        return self.index2action_map[index]

    def get_index_from_action(self, action: list):
        return self.action2index_map[tuple(action)]

#################################################################
######################## Configs - End - ######################


    

#################################################################
###################### CirSysGame -Start- #####################




class Target(NamedTuple):
    # 该状态的“正确性”评估值，比如量子电路的 fidelity（越高越好）
    # 来自 MCTS 回传的 value 或直接从环境评估
    fidelity_rewards: float

    # 该状态的“延迟”或“成本”评估值，比如电路长度、运行时间等（越低越好）
    # 可以定义为负数形式（-length），便于统一作为 value 使用
    length_rewards: float

    # 策略分布（动作的概率分布），由 MCTS 的 visit count 归一化得到
    # 例如：MCTS 搜索中 [action0: 1次, action1: 3次] → [0.25, 0.75]
    policy: Sequence[int]

    # 对 future value 的折扣因子，比如 γ^k，表示从当前状态往后 k 步的影响衰减程度
    # 也可以用于 lambda-return 等 bootstrapped value 的加权
    bootstrap_discount: float


class Sample(NamedTuple):  ##一次训练样本
  observation: Dict[str, np.ndarray]
  bootstrap_observation: Dict[str, np.ndarray]
  target: Target





## function not enough, need to be added, make...

class Game(AbstractGame):
    """Game wrapper."""
    class CirSysSimulator(object):

        def __init__(self, task_spec: TaskSpec):
            self.num_qubits = task_spec.num_qubits
            self.qc = QuantumCircuit(self.num_qubits)
            self.length_qc = self.qc.size()

        def apply_action(self, gate, location, control):
            if gate == "H": self.qc.h(location)
            elif gate == "S": self.qc.s(location)
            elif gate == "T": self.qc.t(location)
            elif gate == "S†": self.qc.sdg(location)  
            elif gate == "T†": self.qc.tdg(location) 
            elif gate == "CX": self.qc.cx(control, location)
            # print("电路矩阵", Operator(self.qc).data)

    def __init__(self, task_spec: TaskSpec):
        self.task_spec = task_spec
        self.simulator = self.CirSysSimulator(task_spec)
        self.action_space = self.task_spec.actions_index
        self.action_space_size = self.task_spec.num_actions
        self.circuit = {i: [] for i in range(self.task_spec.num_qubits)}
        self.circuit_depth = 0
        self.QFT = qlib.QFT(self.task_spec.num_qubits)
        self.QFT_matrix = Operator(self.QFT).data
        self.pre_fidelity = 0.0
        self.player = None
        self.qc = QuantumCircuit(self.task_spec.num_qubits)
        self.is_terminal = False
        self.discount = task_spec.discount
        self.fidelity_reward_weight = task_spec.fidelity_reward_weight
        self.length_reward_weight = task_spec.length_reward_weight

        self.fidelities = []
        self.rewards = []
        self.fidelity_rewards = []
        self.length_rewards = []
        self.history = [] ##list of gate index  
        self.child_visits = []
        self.root_values = []


    def step(self, action:list):
        """
        Apply action to the game.
        Args:
            action : action of the action_space to take, i.e.["CX", target, control]].
        Returns:
            The new observation
        """
        # is_terminal = False
        action_index = action2index(action, self.task_spec.num_qubits, self.task_spec.num_actions)
        print(" self.history 前",  self.history)
        self.history.append(action_index)
        print(" self.history 后",  self.history)
        gate, location, control = action
        if gate == "CX":
            self.circuit[location].append(action)
            self.circuit[control].append(None)
            
            self.qc.cx(control, location)
        else:
            self.circuit[location].append(action)
            if gate == "H": self.qc.h(location)
            elif gate == "S": self.qc.s(location)
            elif gate == "T": self.qc.t(location)
            elif gate == "S†": self.qc.sdg(location)  
            elif gate == "T†": self.qc.tdg(location) 
    
        self.simulator.apply_action(gate, location, control)
        matrix =Operator(self.simulator.qc).data
        self.simulator.length_qc = self.simulator.qc.size()
        
        length_reward = self.task_spec.length_reward
        self.length_rewards.append(length_reward)
        
        matrix_in = Operator(self.qc).data
        assert (matrix_in == matrix).all()

        d = matrix.shape[0]

        hs_inner = cirq.hilbert_schmidt_inner_product(matrix, self.QFT_matrix)
        fidelity = np.abs(hs_inner) / d

        fidelity_reward = self.fidelity_reward(fidelity)
        self.fidelity_rewards.append(fidelity_reward)

        self.pre_fidelity = fidelity
        self.fidelities.append(self.pre_fidelity)

        if fidelity >= self.task_spec.fidelity_threshold:
            self.is_terminal = True
            fidelity_reward += 1.0 # 具体后面再说
        if self.simulator.length_qc >= self.task_spec.max_circuit_length:
            self.is_terminal = True
        reward = length_reward * self.length_reward_weight + fidelity_reward * self.fidelity_reward_weight
        self.rewards.append(reward)

        return self.render(), reward, fidelity_reward, length_reward, self.is_terminal
    
    def store_search_statistics(self, root: Node):
        sum_visits = sum(child.visit_count for child in root.children.values())
        ## list of all Actions
        action_space = self.action_history().action_space() 
        self.child_visits.append(
            [
                root.children[a].visit_count / sum_visits
                if a in root.children
                else 0
                for a in action_space
            ]
        )
        self.root_values.append(root.value())



    def terminal(self) -> bool:
        return self.is_terminal
    
    def current_fidelity(self) -> float:
        return self.pre_fidelity

    
    def fidelity_reward(self, fidelity_new: float):
        """
        Calculate the reward based on fidelity.
        Args:
            fidelity_new: The new fidelity value.
        Returns:
            The reward based on the change in fidelity.
        """
        reward = fidelity_new - self.pre_fidelity
        return reward

    def depth_reward(self):
       pass

        

    def legal_actions(self): 
        """
        遍历每一个比特上的最后一个门，然后将动作空间中与这些门冲突的门移除
        return: actions_legal 是合法动作的参数集
        """
        num_qubits = self.task_spec.num_qubits
        actions_legal = copy.deepcopy(self.action_space)
        
        for i in range(num_qubits):
            # 检查当前比特的电路是否为空，或者最后一个门是否为 None
            if not self.circuit[i] or self.circuit[i][-1] is None:
                continue

            pre_gate = self.circuit[i][-1]
            index_pre = action2index(pre_gate, self.task_spec.num_qubits, self.task_spec.num_actions)

            if pre_gate[0] in ("H", "T", "T†", "CX"):
                # 对于 H, T, T†, CX 门，不允许重复使用同一个门
                if index_pre in actions_legal:
                    actions_legal.remove(index_pre)
            if pre_gate[0] == "S":
                # 对于 S 门，不允许使用 S† 和 T†
                ilegal_S_gd = index_pre + 2 
                ilegal_T_gd = index_pre + 3
                if ilegal_S_gd in actions_legal:
                    actions_legal.remove(ilegal_S_gd)
                if ilegal_T_gd in actions_legal:
                    actions_legal.remove(ilegal_T_gd)
            if pre_gate[0] == "T":
                # 对于 T 门，不允许使用 S† 和 T†
                ilegal_S_gd = index_pre + 1 
                ilegal_T_gd = index_pre + 2
                if ilegal_S_gd in actions_legal:
                    actions_legal.remove(ilegal_S_gd)
                if ilegal_T_gd in actions_legal:
                    actions_legal.remove(ilegal_T_gd)
            if pre_gate[0] == "S†":
                # 对于 S† 门，不允许使用 S 和 T
                ilegal_S = index_pre - 2 
                ilegal_T = index_pre - 1
                if ilegal_S in actions_legal:
                    actions_legal.remove(ilegal_S)
                if ilegal_T in actions_legal:
                    actions_legal.remove(ilegal_T)
            if pre_gate[0] == "T†":
                # 对于 T† 门，不允许使用 S 和 T
                ilegal_S = index_pre - 3 
                ilegal_T = index_pre - 2
                if ilegal_S in actions_legal:
                    actions_legal.remove(ilegal_S)
                if ilegal_T in actions_legal:
                    actions_legal.remove(ilegal_T)
                    
        return actions_legal
    
    def to_play(self):
        """
        Return the current player.
        Player is always MCTS player in self-play.
        Returns:
            The current player, it should be an element of the players list in the config.
        """
        return self.player

    def reset(self):
        """
        Reset the game for a new game.

        Returns:
            Initial observation of the game.
        """
        return Game(self.task_spec)

    def human_play(self):
        self.player = "human"
        self.reset()
        
        while True:
            # 显示当前电路、保真度或其他信息，方便玩家决策
            obs = self.render()
            print(f"\n当前电路长度: {obs['circuit_length']}")
            print("当前电路: ", obs['circuit'])
            
            # 计算可用动作（索引），如果想限制玩家输入非法门
            # 可以先将这些动作翻译成 [gate, location, control] 的可读形式
            legal_indices = self.legal_actions()
            legal_actions_list = [index2action(idx, self.task_spec.num_qubits, self.task_spec.num_actions) 
                                for idx in legal_indices]
            
            print("可用动作列表(部分示例)：", legal_actions_list)


            # 让玩家选择：可以让玩家直接输入门名称、location、control，也可以让玩家输入一个动作索引
            user_input = input("请输入动作(格式: gate location [control])，或 Q 退出: ").strip()
            if user_input.lower() == 'q':
                print("退出手动模式。")
                break
            
            # 解析玩家输入
            # 例如玩家输入: "H 0" 或 "CX 0 1" 等
            parts = user_input.split()
            if not parts:
                print("输入为空，请重新输入。")
                continue
            
            gate = parts[0]
            if gate not in ("H", "S", "T", "S†", "T†", "CX"):
                print("非法门类型，请重新输入。")
                continue
            
            # 检查 location、control 是否存在
            try:
                location = int(parts[1])
            except (IndexError, ValueError):
                print("location 输入有误，请重新输入。")
                continue
            
            if gate == "CX":
                # 需要第三个参数 control
                try:
                    control = int(parts[2])
                except (IndexError, ValueError):
                    print("控制比特 control 输入有误，请重新输入。")
                    continue
            else:
                # 对于单量子比特门，control = location
                control = location
            
            # 组装成 action
            action = [gate, location, control]

            # 检查这个 action 是否在 legal_actions 中
            action_index = action2index(action, self.task_spec.num_qubits, self.task_spec.num_actions)
            if action_index not in legal_indices:
                print("该动作不合法或已被限制，请重新输入。")
                continue
            
            # 执行一步
            obs_next, reward, f_reward, l_reward, is_terminal = self.step(action)
            
            print(f"执行动作: {action}, 得到奖励: {reward:.4f}, 保真度奖励: {f_reward:.4f}, 长度奖励: {l_reward:.4f}")
            
            # 判断是否结束
            if is_terminal:
                print("到达终止状态！")
                print("最终电路: ", obs_next['circuit'])
                print("电路长度: ", obs_next['circuit_length'])
                break
    def action_history(self) -> ActionHistory:
        return ActionHistory(self.history, self.action_space_size)

    ## render 获取的数据可能不够？？？
    def render(self):
        return {
            'matrix': Operator(self.simulator.qc).data,
            'circuit_length': self.simulator.length_qc,
            'circuit': self.history
        }
    def make_observation(self, state_index: int):
        if state_index == -1:
            return self.render()
        env = self.reset()
        for action in self.history[:state_index]:
            env.step(action)
            observation = env.render()
        return observation
    ## 可能有问题，取出来的数据可能不足
    def make_target(
      self, state_index: int, td_steps: int, to_play: Player
  ) -> Target:
        """Creates the value target for training."""
    # The value target is the discounted sum of all rewards until N steps
    # into the future, to which we will add the discounted boostrapped future
    # value.
    
        bootstrap_index = state_index + td_steps

        for i, (f_reward, l_reward) in enumerate(zip(
                self.fidelity_rewards[state_index:bootstrap_index],
                self.length_rewards[state_index:bootstrap_index])):
            discount = self.discount ** i
            f_rewards += f_reward * discount
            l_rewards += l_reward * discount
        if bootstrap_index < len(self.root_values):
            bootstrap_discount = self.discount**td_steps
        else:
            bootstrap_discount = 0

        return Target(
            f_rewards,
            l_rewards,
            self.child_visits[state_index],
            bootstrap_discount,
        )

    # def clone(self):
    #     """Create a deep copy of the current game state for MCTS simulation."""
    #     new_game = copy.deepcopy(Game(self.task_spec))
        
    #     # 复制电路
    #     new_game.qc = copy.deepcopy(self.qc)
    #     new_game.simulator = self.CirSysSimulator(self.task_spec)
    #     new_game.simulator.qc = copy.deepcopy(self.simulator.qc)
    #     new_game.simulator.length_qc = self.simulator.length_qc

    #     # 复制电路结构
    #     new_game.circuit = copy.deepcopy(self.circuit)

    #     # 状态变量
    #     new_game.pre_fidelity = self.pre_fidelity
    #     new_game.is_terminal = self.is_terminal
    #     new_game.player = self.player  

    #     # 历史轨迹和统计信息
    #     new_game.history = copy.deepcopy(self.history)
    #     new_game.fidelities = copy.deepcopy(self.fidelities)
    #     new_game.rewards = copy.deepcopy(self.rewards)
    #     new_game.fidelity_rewards = copy.deepcopy(self.fidelity_rewards)
    #     new_game.length_rewards = copy.deepcopy(self.length_rewards)
    #     new_game.child_visits = copy.deepcopy(self.child_visits)
    #     new_game.root_values = copy.deepcopy(self.root_values)

    #     return new_game

#################################################################
###################### CirSysGame - End - #####################



#################################################################
#################### AlphaCirNetwork -Start- #####################


class NetworkOutput(NamedTuple):
  value: np.ndarray
  fidelity_value_logits: np.ndarray
  length_value_logits: np.ndarray  ## 代表执行动作后余下的电路长短
  policy_logits: Dict[Action, float] 
  hidden_state: Optional[np.ndarray] = None


class AlphaCirNetwork(nn.Module):
    def __init__(self, hparams, task_spec):
        super().__init__()
        self.representation = AlphaCirRepNet(
            hparams.representation, task_spec, hparams.embedding_dim, hparams.device
        ).to(hparams.device)
        self.prediction = AlphaCirPreNet(
            task_spec=task_spec,
            value_max=hparams.prediction.value_max,
            num_bins=hparams.prediction.num_bins,
            embedding_dim= hparams.embedding_dim,
            num_resblocks=hparams.prediction.num_resblocks,
            hidden_dim=hparams.prediction.hidden_dim,
        ).to(hparams.device)
    def forward(self, observation):
        embedding = self.representation(observation)
        return self.prediction(embedding)
    
    def inference(self, observation):
        ## 参考uniform network infernence function
        self.eval()
        with torch.no_grad():
            embedding = self.representation.inference(observation)
            return self.prediction.inference(embedding)

    def training_forward(self, observation):
        embedding = self.representation(observation)
        return self.prediction(embedding)


    def get_params(self):
        return self.state_dict()

    def update_params(self, updates_dict):
        self.load_state_dict(updates_dict)
        return copy.deepcopy(self.state_dict())


    def training_steps(self):
        pass  


class UniformNetwork(nn.Module):    
    
    """
    UniformNetwork: 测试pipeline, 始终返回固定输出。
    """
    def __init__(self, num_actions: int):
        super().__init__()  # 正确初始化 nn.Module
        self.num_actions = num_actions
        self.params = {}

    def forward(self, observation: torch.Tensor) -> NetworkOutput:
        batch_size = observation.shape[0] if observation.ndim > 1 else 1

        outputs = [NetworkOutput(
            value=np.zeros(self.num_actions), 
            fiedlity_value_logits=np.zeros(self.num_actions),
            length_value_logits=np.zeros(self.num_actions),
            policy_logits={Action(a): 1.0 / self.num_actions for a in range(self.num_actions)}
        ) for _ in range(batch_size)]

        return outputs
    
    def inference(self, observation: torch.Tensor) -> NetworkOutput:
        self.eval()

        return {
            "value": float(0), 
            "fiedlity_value_logits": float(0),
            "length_value_logits" : float(0),
            "policy_logits":{Action(a): 1.0 / self.num_actions for a in range(self.num_actions)},
            "hidden_state": None    
        }
    
    def get_params(self):
        return self.params

    def update_params(self, updates) -> None:
        self.params = updates

    def training_steps(self) -> int:
        return 0  # 不训练


###Representation Network
class MultiQueryAttentionBlock_old(torch.nn.Module):
    """
    - Multi-Query Attention Layer:
    - Multi-head Q
    - Shared K and V
    """
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = self.hid_dim // self.n_heads

        self.fc_q = torch.nn.Linear( self.hid_dim, self.hid_dim)
        self.fc_k = torch.nn.Linear( self.hid_dim, self.head_dim)
        self.fc_v = torch.nn.Linear(self.hid_dim, self.head_dim)  
        self.fc_o = torch.nn.Linear(self.hid_dim, self.hid_dim)
        
        self.dropout = torch.nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
    def forward(self, query, key, value, mask = None):
        
        batch_size = query.shape[0]
        
        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]
               
        Qbank = self.fc_q(query).view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        Kbank = self.fc_k(key).view(batch_size, -1, 1, self.head_dim).permute(0, 2, 3, 1)
        Vbank = self.fc_v(value).view(batch_size, -1, 1, self.head_dim).permute(0, 2, 1, 3)   
        
        #Qbank = [batch size, n heads, query len, head dim]
        #Kbank = [batch size, 1, head dim, key len]
        #Vbank = [batch size, 1, value len, head dim]

        energy = torch.matmul(Qbank, Kbank) / self.scale

        #energy = [batch size, n heads, query len, key len]
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        attention = F.softmax(energy, dim = -1)
                
        #attention = [batch size, n heads, query len, key len]

        x = torch.matmul(self.dropout(attention), Vbank)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.hid_dim)     
        #x = [batch size, seq len, hid dim]
        
        x = self.fc_o(x)
        return x, attention




class MultiQueryAttentionBlock(nn.Module):
    """
    Multi-Query Attention Layer:
    - Multi-head Q
    - Shared K and V (复制给每个 head)
    """

    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        assert hid_dim % n_heads == 0, "hid_dim 必须能被 n_heads 整除"

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(hid_dim, hid_dim)           # 每个 head 一个 Q
        self.fc_k = nn.Linear(hid_dim, self.head_dim)     # 共享的 K
        self.fc_v = nn.Linear(hid_dim, self.head_dim)     # 共享的 V
        self.fc_o = nn.Linear(hid_dim, hid_dim)           # 输出层

        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)).to(device)

    def forward(self, query, key, value, mask=None):
        B, QL, _ = query.shape   # batch size, query len
        _, KL, _ = key.shape     # key len
        _, VL, _ = value.shape   # value len（通常与 key 相同）

        # ----- Qbank: 多头 -----
        Q = self.fc_q(query)  # [B, QL, hid_dim]
        Qbank = Q.view(B, QL, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        # → [B, n_heads, QL, head_dim]

        # ----- Kbank: 共享，但复制 n_heads 个 -----
        K = self.fc_k(key).view(B, KL, self.head_dim).unsqueeze(1)
        # → [B, 1, KL, head_dim]
        Kbank = K.expand(-1, self.n_heads, -1, -1).permute(0, 1, 3, 2)
        # → [B, n_heads, head_dim, KL]

        # ----- Vbank: 同样共享复制 -----
        V = self.fc_v(value).view(B, VL, self.head_dim).unsqueeze(1)
        # → [B, 1, VL, head_dim]
        Vbank = V.expand(-1, self.n_heads, -1, -1)
        # → [B, n_heads, VL, head_dim]

        # ----- Scaled Dot-Product Attention -----
        energy = torch.matmul(Qbank, Kbank) / self.scale
        # → [B, n_heads, QL, KL]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = F.softmax(energy, dim=-1)
        x = torch.matmul(self.dropout(attention), Vbank)
        # → [B, n_heads, QL, head_dim]

        # ----- 拼接多头 -----
        x = x.permute(0, 2, 1, 3).contiguous()  # [B, QL, n_heads, head_dim]
        x = x.view(B, QL, self.hid_dim)         # [B, QL, hid_dim]

        return self.fc_o(x), attention
    
    
    
class AlphaCirRepNet(nn.Module):
    def __init__(self, hparams_r, task_spec, embedding_dim, device):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_gate_types = task_spec.num_ops  # 比如 ['H', 'CX', 'T', 'Tdg', ...]
        self.num_qubits = task_spec.num_qubits  # 量子比特数量
        self.max_circuit_length = task_spec.max_circuit_length
        self.pooling = "last"  # 'mean', 'last', 'cls' ## 获取最后一个门
        
        # 嵌入层        
        self.input_dim = hparams_r.input_dim  # 输入维度，通常是门的 one-hot 编码长度
        self.num_mqa_blocks = hparams_r.num_mqa_blocks  # MQA 的数量
        self.num_heads = hparams_r.num_heads  # MQA 的头数
        self.dropout = hparams_r.dropout  # dropout 概率
        self.device = device  # 设备类型（CPU 或 GPU）

        ## circuit embedding
        ## 先经过mlp -> a
        self.mlp_embedder = nn.Sequential(
            nn.Linear(self.input_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )

        ## 在经过MQAttention -> b
        self.mqas_embedder = nn.ModuleList([
        MultiQueryAttentionBlock(
        hid_dim=self.embedding_dim,
        n_heads=self.num_heads,
        dropout=self.dropout,
        device=self.device
    )
    for _ in range(self.num_mqa_blocks)
    ])

    def forward(self, inputs):   
        ## inputs: {
        # "circuits": torch.stack([obs["circuit"] for obs in batch], dim=0),
        # "circuits_length": torch.tensor([obs["circuit_length"] for obs in batch], dtype=torch.long),
        # "matrices": torch.stack([obs["matrix"] for obs in batch]) }

        if not isinstance(inputs["circuits"], torch.Tensor):
            circuits = torch.as_tensor(inputs["circuits"], device=self.device)
            circuits_length = torch.as_tensor(inputs["circuits_length"], device=self.device)  # 程序长度

        else:
            circuits = inputs["circuits"].to(self.device)
            circuits_length = inputs["circuits_length"].to(self.device)  # 程序长度

        # circuits = torch.as_tensor(inputs["circuits"], device=self.device) # 电路数据
        # circuits_length = torch.as_tensor(inputs["circuits_length"], device=self.device)  # 程序长度
        print("print circuits", circuits)  # [batch_size, max_length_circuit, 3]
        
        batch_size = circuits.shape[0]  # 批大小
        max_length_circuit = self.max_circuit_length  # 最大电路长度
        if circuits.shape[1] == 0:
            # 空电路处理：返回全零嵌入或 learnable 的默认嵌入
            return torch.zeros((batch_size, self.embedding_dim), device=self.device)
        else:
            batch_size, seq_len = circuits.shape  # 添加 input_dim 解包

        circuits_onehot = self.circuits2onehot(circuits, batch_size, max_length_circuit).to(self.device)  # 将电路转换为 one-hot 编码
        # print(f"circuits_onehot: {circuits_onehot.shape}")  # [batch_size, max_length_circuit, input_dim]
        # print("device",circuits_onehot.device)
        circuits_onehot_view = circuits_onehot.view(-1, self.input_dim)
        # print("device",circuits_onehot_view.device)
    # === 建议2：reshape 以适配 mlp_embedder ===
        embedded_mlp = self.mlp_embedder(circuits_onehot.view(-1, self.input_dim)).view(batch_size, seq_len, -1)
        _, seq_size, feat_size = embedded_mlp.shape

        pos_enc = self.get_position_encoding(seq_size, feat_size, embedded_mlp.device)  ## take care of device
        embedded = embedded_mlp + pos_enc.unsqueeze(0) 

        for i in range(self.num_mqa_blocks):
            embedded, _ = self.mqas_embedder[i](embedded, embedded, embedded)

        batch_size = embedded.size(0) ## 只留最后一个门
        idx = (circuits_length - 1).clamp(min=0, max=self.max_circuit_length - 1)
        batch_idx = torch.arange(batch_size, device=embedded.device)
        output = embedded[batch_idx, idx] 

        return output  # [batch_size, embedding_dim]    

    def inference(self, input):
        ## input : game.render()  
        ## {'matrix': Operator(self.simulator.qc).data,
        ##     'circuit_length': self.simulator.length_qc,
        ##     'circuit': self.history }   
        self.eval()
        with torch.no_grad():

            input = [input] 
            observation = move_to_device(collate_batch(input), self.device)
            return self.forward(observation)  


    
    ## 真正有一个batch时可能会不同
    def circuits2onehot(self, circuits, batch_size, max_length_circuit):
        
        pad_value = -1
        batch_size, seq_len = circuits.shape
        pad_right = max_length_circuit - seq_len
        circuits_padded = F.pad(circuits, (0, pad_right), value=pad_value)

        gate_indexs = circuits[:, :, 0].to(self.device)
        gates= circuits[:, :, 0].to(self.device)
        locations = circuits[:, :, 1]
        control = circuits[:, :, 2]

        gates_one_hot = F.one_hot(gates, num_classes=self.num_gate_types).float()
        locations_one_hot = F.one_hot(locations, num_classes=self.num_qubits).float()
        control_one_hot = F.one_hot(control, num_classes=self.num_qubits).float()  ##考虑单量子比特门控制比特onehot为零

        circuits_onehot = torch.cat([gates_one_hot, locations_one_hot, control_one_hot], dim=-1)


        assert circuits_onehot.shape[:2] == (batch_size, max_length_circuit), \
                    f"Shape mismatch: got {circuits_onehot.shape}, expected ({batch_size}, {max_length_circuit}, ?)"
        return circuits_onehot
    
    
    def get_position_encoding(self, seq_len, dim, device):
        """
        获取 sinusoidal 位置编码，shape 为 [seq_len, dim]
        """
        position = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)  # [seq_len, 1]
        #dim 要为偶数？？？
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float, device=device) * (-math.log(10000.0) / dim))
    
        pe = torch.zeros(seq_len, dim, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维
    
        return pe  # [seq_len, dim]



###Prediction Network
class ResBlockV2(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = None):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim or dim

        self.ln1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, self.hidden_dim)

        self.ln2 = nn.LayerNorm(self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, dim)

    def forward(self, x):
        residual = x

        out = self.ln1(x)
        out = F.relu(out)
        out = self.fc1(out)

        out = self.ln2(out)
        out = F.relu(out)
        out = self.fc2(out)

        return out + residual

def make_policy_head(embedding_dim: int, num_actions: int, num_ResBlock: int, hidden_dim: int):
    layers = []
    
    for _ in range(num_ResBlock):
        layers.append(ResBlockV2(embedding_dim, hidden_dim))
    
    layers.append(nn.LayerNorm(embedding_dim))   # 最后加一个 LayerNorm，防止输出失控
    layers.append(nn.ReLU())                     # 非线性激活（可选）
    layers.append(nn.Linear(embedding_dim, num_actions))  # 输出动作 logits
    
    return nn.Sequential(*layers)

class DistributionSupport(nn.Module):
    def __init__(self, value_max: float, num_bins: int):
        super().__init__()
        self.value_max = value_max
        self.num_bins = num_bins
        self.value_min = -value_max
        self.support = torch.linspace(self.value_min, self.value_max, num_bins)  # shape: (num_bins,)
        self.delta = self.support[1] - self.support[0]  # bin 宽度
        # self.register_buffer("support", torch.linspace(self.value_min, self.value_max, num_bins))

    def scalar_to_two_hot(self, scalar: torch.Tensor) -> torch.Tensor:
        """
        将标量值映射为 two-hot 分布形式
        支持批量输入，scalar shape: (batch,)
        返回 shape: (batch, num_bins)
        """
    #     if scalar.ndim != 1:
    #         raise IndexError(
    #     f"[scalar_to_two_hot] 输入维度错误：期望输入为一维张量 (batch_size,)，"
    #     f"但收到的是形状 {scalar.shape}，维度数为 {scalar.ndim}。"
    # )
        
        scalar = scalar.clamp(self.value_min, self.value_max)
        print(f"scalar: {scalar}")
        batch_size = scalar.shape[0]
        
        # 缩放到 bin 的 float 下标（不取整）
        # eps = 1e-6
        # pos = ((scalar - self.value_min) / self.delta).clamp(0, self.num_bins - 1 - eps)
        pos = (scalar - self.value_min) / self.delta
        print(f"pos: {pos}")
        lower_idx = pos.floor().long()
        print(f"lower_idx: {lower_idx}")
        # upper_idx = lower_idx + 1
        upper_idx = (lower_idx + 1).clamp(0, self.num_bins - 1)
        print(f"upper_idx: {upper_idx}")

        # 权重分配
        upper_w = pos - lower_idx.float()
        lower_w = 1.0 - upper_w

        # 创建 zero 初始化的分布
        dist = torch.zeros(batch_size, self.num_bins, device=scalar.device)

        # 边界处理
        
        upper_idx = upper_idx.clamp(0, self.num_bins - 1)
        lower_idx = lower_idx.clamp(0, self.num_bins - 1)

        # 填入概率（two-hot）
        
        dist.scatter_(1, upper_idx.unsqueeze(1), upper_w.unsqueeze(1))
        dist.scatter_(1, lower_idx.unsqueeze(1), lower_w.unsqueeze(1))
        return dist

    def mean(self, logits: torch.Tensor) -> torch.Tensor:
        """
        从 logits 中解码出期望值（支持批量）
        logits shape: (batch, num_bins)
        返回 shape: (batch,)
        """
        probs = F.softmax(logits, dim=-1)  # 转成概率分布
        expected = torch.sum(probs * self.support.to(logits.device), dim=-1)
        return expected

class CategoricalHead(nn.Module):
    def __init__(self, embedding_dim: int, support: DistributionSupport, num_ResBlock: int, hidden_dim: int):
        super().__init__()
        self.support = support
        self.embedding_dim = embedding_dim
        self.num_bins = support.num_bins

        self.res_blocks = nn.ModuleList([ResBlockV2(embedding_dim, hidden_dim) for _ in range(num_ResBlock)])
        self.fc_out = nn.Linear(embedding_dim, self.num_bins)

    def forward(self, x):
        for block in self.res_blocks:
            x = block(x)

        logits = self.fc_out(x)
        mean = self.support.mean(logits)
        return dict(logits=logits, mean=mean)
    
class AlphaCirPreNet(nn.Module):

    def __init__(
        self,
        task_spec,
        value_max: float,
        num_bins: int,
        embedding_dim: int,
        num_resblocks: int,
        hidden_dim: int
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.support = DistributionSupport(value_max, num_bins)
        self.num_bins = num_bins

        # Heads
        self.policy_head = make_policy_head(embedding_dim, task_spec.num_actions, num_resblocks, hidden_dim)
        self.fidelity_value_head = CategoricalHead(embedding_dim, self.support, num_resblocks, hidden_dim)
        self.length_value_head = CategoricalHead(embedding_dim, self.support, num_resblocks, hidden_dim)

    def forward(self, embedding: torch.Tensor):
        """
        embedding: (batch_size, embedding_dim)
        returns:
            dict with logits and mean value
        """
        fidelity_value = self.fidelity_value_head(embedding)  # logits + mean
        length_value = self.length_value_head(embedding)

        total_value = fidelity_value["mean"] + length_value["mean"]
        policy_logits = self.policy_head(embedding)

        return NetworkOutput(
            value = total_value,  # shape: (batch,)
            fidelity_value_logits = fidelity_value["logits"],  # shape: (batch, num_bins)
            length_value_logits = length_value["logits"],          # shape: (batch, num_bins)
            policy_logits = policy_logits                                    # shape: (batch, num_actions)
        )

    def inference(self, embedding: torch.Tensor):
        ## 参考uniform network inference function
        self.eval()
        with torch.no_grad():
           return self.forward(embedding)
       
    

    

#################################################################
#####################  AlphaCirNetwork-End- #########################



#################################################################
####################### Replay_Buffer-Start- #######################
class ReplayBuffer(object):
  """Replay buffer object storing games for training."""

  def __init__(self, config: AlphaCirConfig):
    self.window_size = config.window_size #最多存多少局游戏（FIFO）
    self.batch_size = config.batch_size #每次训练采样多少个训练样本
    self.buffer = [] #保存所有游戏的列表，每一项是一个完整 Game 实例

  def save_game(self, game):
    if len(self.buffer) >= self.window_size:
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

#################################################################
####################### Shared_Storage-Start- #######################
def make_uniform_network(num_actions: int) -> UniformNetwork:
    return UniformNetwork(num_actions)

class SharedStorage(object):
  """Controls which network is used at inference."""

  def __init__(self, num_actions: int):
    self._num_actions = num_actions
    self._networks = {}

  def latest_network(self) -> AlphaCirNetwork:
    if self._networks:
      return self._networks[max(self._networks.keys())]
    else:
      # policy -> uniform, value -> 0, reward -> 0
      return make_uniform_network(self._num_actions)

  def save_network(self, step: int, network: AlphaCirNetwork):
    self._networks[step] = network




#################################################################
####################### Shared_Storage- End - #######################

#################################################################
####################### Self_play-Start- #######################

MAXIMUM_FLOAT_VALUE = float('inf')

KnownBounds = namedtuple('KnownBounds', ['min', 'max'])



class MinMaxStats(object):
  """A class that holds the min-max values of the tree."""

  def __init__(self, known_bounds: Optional[KnownBounds]):
    self.maximum = known_bounds.max if known_bounds else -MAXIMUM_FLOAT_VALUE
    self.minimum = known_bounds.min if known_bounds else MAXIMUM_FLOAT_VALUE

  def update(self, value):
    if isinstance(value, np.ndarray):
        self.maximum = max(self.maximum, float(value.max()))
        self.minimum = min(self.minimum, float(value.min()))
    else:
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



# Each self-play job is independent of all others; it takes the latest network
# snapshot, produces a game and makes it available to the training job by
# writing it to a shared replay buffer.
def run_selfplay(
    config: AlphaCirConfig, storage: SharedStorage, replay_buffer: ReplayBuffer
):
  if test_model:
    for _ in tqdm(range(test_model_run)):
        network = storage.latest_network()
        game = play_game(config, network)
        replay_buffer.save_game(game)
     
  else:
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
    # if test_model:
    #     print("test model INI: root", root)
    current_observation = game.make_observation(-1)
    if test_model:
        print("test model: current_observation", current_observation)
    network_output = network.inference(current_observation)
    ## game.legal_actions(): the list of indices of legal actions
    # legal_actions: the list of legal Actions
    legal_actions_index =  game.legal_actions() ## 未生效


    _expand_node(
        root, game.to_play(), game.action_history().action_space(), network_output, reward=0
    )
    # if test_model:
    #     print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
    #     print("test model: root EXPAND", root)

    _backpropagate(
        [root],
        network_output.value,
        game.to_play(),
        config.discount,
        min_max_stats,
    )

    # if test_model:
    #    print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
    #    print("test model BACK: root", root)

    _add_exploration_noise(config, root)

    # if test_model:
    #     print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
    #     print("test model NOISE: root", root)
    # We then run a Monte Carlo Tree Search using the environment.
    run_mcts(
        config,
        root,
        game.action_history(),
        network,
        min_max_stats,
        game,
    )
    # action:Action  选择动作完全根据概率分布 
    action = _select_action(config, len(game.history), root, network, legal_actions_index)
    index = action.index
    gate = index2action(index, config.task_spec.num_qubits, config.task_spec.num_actions)  # ["H", 0, 0]
    game.step(gate)
    if test_model:
        print("test model: gate", gate)
        print("test model: game history", game.history)
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
    # if test_model:
    #    print("test model: ", _)
    history = copy.deepcopy(action_history)
    node = root
    search_path = [node]
    sim_env = copy.deepcopy(env)

    while node.expanded():
      action, node = _select_child(config, node, min_max_stats)
      index = action.index
      gate = index2action(index, config.task_spec.num_qubits, config.task_spec.num_actions)  # ["H", 0, 0]
      observation, reward, _, _, _ = sim_env.step(gate)
      history.add_action(action)
      search_path.append(node)

    # Inside the search tree we use the environment to obtain the next
    # observation and reward given an action.   self.render(), reward, fidelity_reward, length_reward, self.is_terminal
    # observation, reward, f_reward, l_reward, terminal = sim_env.step(gate)
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
    config: AlphaCirConfig, num_moves: int, node: Node, network: AlphaCirNetwork, legal_actions_index: List[int]
):
  ## 让未在合法动作列表中的动作的访问次数为0

    visit_counts = []
    for action, child in node.children.items():
        if action.index not in legal_actions_index:
            child.visit_count = 0  # 非法动作访问次数设为 0
        visit_counts.append((child.visit_count, action))  

    t = config.visit_softmax_temperature_fn(
      steps=network.training_steps()
  )
    _, action = softmax_sample(visit_counts, t)
    return action ## Action


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
#   if test_model:
#      print("test model: pb_c",type(pb_c))
  pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

  prior_score = pb_c * child.prior
  if child.visit_count > 0:
    # x = child.reward + config.discount * child.value()
    # if test_model:
    #     print("test model: child.reward",type(child.reward), child.reward)
    #     print("test model: config.discount",type(config.discount), config.discount)
    #     print("test model: child.value()",type(child.value()), child.value())
    value_score = min_max_stats.normalize(
        child.reward + config.discount * child.value()
    )
  else:
    value_score = 0
#   if test_model:
#      print("test model: value_score",type(value_score))
#      print("test model: value_score",value_score)
    
  return float(prior_score + value_score)


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
#   if test_model:    
#     print("test model:network_output.policy_logits", network_output.policy_logits )
#     print("test model:actions", actions)
#   for a in actions:
#     print("test model: a", a.index)
#     item = network_output.policy_logits[0, a.index]
#     print("test model: network_output.policy_logits[a.index]", item)
#     prob = torch.exp(item)
#     print("test model: prob", prob)
  policy = { a: torch.exp(network_output.policy_logits[0, a.index]) for a in actions}

#   if test_model:
#     print("test model:policy", policy)
  policy_sum = sum(policy.values())
#   sum_p = 0
  for action, p in policy.items():
    node.children[action] = Node(p / policy_sum)
    # print(p/policy_sum)
    # assert (p/policy_sum) >= 0, f"negative policy {p/policy_sum}"
    # sum_p += p/policy_sum
#   assert round(sum_p) == 1, f"sum of policy is not 1.0, but {sum_p}"


def _backpropagate(
    search_path: Sequence[Node],
    value: float,
    to_play: Player,
    discount: float,
    min_max_stats: MinMaxStats,
):
  """Propagates the evaluation all the way up the tree to the root."""
#   if test_model:     
#     print("test model in back : value")
#     print(type(value), value)
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



#################################################################
######################## Trainer -Start- #######################

def train_network(config: AlphaCirConfig, storage, replay_buffer):
    """Trains the network on data stored in the replay buffer."""
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = config.hparams.device
    network = AlphaCirNetwork(config.hparams, config.task_spec).to(device)
    target_network = AlphaCirNetwork(config.hparams, config.task_spec).to(device)
    target_network.load_state_dict(network.state_dict())  # 初始化目标网络

    optimizer = torch.optim.SGD(network.parameters(), lr=config.lr_init, momentum=config.momentum)

    for i in range(config.training_steps):
        if i % config.checkpoint_interval == 0:
            storage.save_network(i, network)

        if i % config.target_network_interval == 0:
            target_network.load_state_dict(network.state_dict())

        batch = replay_buffer.sample_batch(config.td_steps)
        optimizer.zero_grad()
        loss = compute_loss(network, target_network, batch, device)
        loss.backward()
        optimizer.step()

    storage.save_network(config.training_steps, network)


def compute_loss(network, target_network, batch, device):
    observations, bootstrap_obs, targets = zip(*batch)

    obs_batch = move_to_device(collate_batch(list(observations)), device)
    boot_batch = move_to_device(collate_batch(list(bootstrap_obs)), device)

    # 解包 targets
    fidelity_value, length_value, policy, discount = zip(*targets)
    fidelity_value = torch.stack(fidelity_value).to(device)
    length_value = torch.stack(length_value).to(device)
    policy = torch.stack(policy).to(device)
    discount = torch.tensor(discount).to(device)

    # 推理
    pred = network(obs_batch)
    pred_boot = target_network(boot_batch)

    # TD Bootstrapped target
    fidelity_value += discount.unsqueeze(1) * pred_boot["fidelity_value_logits"].detach()
    length_value += discount.unsqueeze(1) * pred_boot["length_value_logits"].detach()

    # Loss
    policy_loss = soft_cross_entropy(pred["policy"], policy)
    fidelity_loss = scalar_loss(pred["fidelity_value_logits"], fidelity_value, network)
    length_loss = scalar_loss(pred["length_value_logits"], length_value, network)

    return (policy_loss + fidelity_loss + length_loss)/ len(batch)


    # total_loss = 0.0

    # for observation, bootstrap_obs, target in batch:
    #     # Move tensors to device
    #     observation = move_to_device(observation, device)
    #     bootstrap_obs = move_to_device(bootstrap_obs, device)
    #     ## target_fidelity_value, target_length_value: reward[1] + γ * reward[2] + γ² * reward[3]
    #     target_fidelity_value, target_length_value, target_policy, bootstrap_discount = target

    #     # Forward pass
    #     # observation: [{"circuit":..., "matrix":..., "circuit_length":...}, {}...]

    #     predictions = network(observation)
    #     bootstrap_predictions = target_network(bootstrap_obs)

    #     # Unpack predictions
    #     policy_logits = predictions["policy"]
    #     fidelity_logits = predictions["fidelity_value_logits"]
    #     length_logits = predictions["length_value_logits"]

    #     # Target for value（加上 TD bootstrapping）
    #     bootstrap_fidelity_value = bootstrap_predictions["fidelity_value_logits"].detach()
    #     bootstrap_length_value = bootstrap_predictions["length_value_logits"].detach()
    #     # target_fidelity = reward[1] + γ * reward[2] + γ² * reward[3] + γ³ * V(s_4)
    #     target_fidelity_value += bootstrap_discount* bootstrap_fidelity_value
    #     target_length_value += bootstrap_discount *  bootstrap_length_value
    #     ## 可能会存在问题
    #     policy_loss = soft_cross_entropy(policy_logits, target_policy)  
    #     fidelity_loss = scalar_loss(fidelity_logits, target_fidelity_value, network)
    #     length_loss = scalar_loss(length_logits, target_length_value, network)

    #     total_loss += policy_loss + fidelity_loss + length_loss

    # return total_loss / len(batch)

def scalar_loss(pred_logits, scalar_target, network):
    """
    pred_logits: shape (B, num_bins)
    scalar_target: shape (B,)
    """
    two_hot_target = network.prediction.support.scalar_to_two_hot(scalar_target)
    log_probs = F.log_softmax(pred_logits, dim=-1)
    loss = -(two_hot_target * log_probs).sum(dim=-1).mean()
    return loss

def move_to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [move_to_device(x, device) for x in data]
    else:
        return data


#################################################################
######################## Trainer - End - #######################

#################################################################
########################### Main -Start- #######################


def AlphaCir(config: AlphaCirConfig):
  storage = SharedStorage()
  replay_buffer = ReplayBuffer(config)

  for _ in range(config.num_actors):
    launch_job(run_selfplay, config, storage, replay_buffer)

  train_network(config, storage, replay_buffer)

  return storage.latest_network()


def launch_job(f, *args):
  f(*args)
#################################################################
########################### Main - End - #######################



if __name__ == "__main__":
    # print(test_model_run) 
    config = AlphaCirConfig()
    num_qubits = config.task_spec.num_qubits
    type_single_gates = config.task_spec.num_ops





