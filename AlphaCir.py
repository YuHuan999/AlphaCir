import glob
import re
import pickle
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYTORCH_JIT_LOG_LEVEL"] = ">>"

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
## 线程和多进程
import threading
from multiprocessing import Process, Lock, Queue, Manager, current_process
from multiprocessing.queues import Empty as mp_Empty
from typing import Sequence
import queue
import time

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
recorder_fid = SummaryWriter(log_dir="records/fidelities")  # 可自定义路径
recorder_loss = SummaryWriter(log_dir="records/losses")  # 可自定义路径
recorder_params = SummaryWriter(log_dir="records/params")  # 可自定义路径
GATES2Index = {"H": 0, "S": 1, "T": 2, "S†": 3, "T†": 4, "CX": 5}
Index2GATES = {0:"H", 1:"S", 2:"T", 3: "S†", 4: "T†", 5:"CX"}
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
    def safe_float(x):
        return x.item() if isinstance(x, torch.Tensor) else x

    return (
        f"Node("
        f"prior={safe_float(self.prior):.4f}, "
        f"visits={self.visit_count}, "
        f"value={safe_float(self.value()):.4f}, "
        f"value_sum={safe_float(self.value_sum):.4f}, "
        f"reward={safe_float(self.reward):.4f}, "
        f"num_children={len(self.children)}"
        f")"
    )

class Player:
    def __init__(self, player_id, player_type="self_play"):
        self.player_id = player_id      # 例如 0 或 1
        self.player_type = player_type  # "human" 或 "self_play"

    def __repr__(self):
        return f"Player({self.player_id}, type={self.player_type})"



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
        ope = GATES2Index[gate]
    else:
        op = control if control < location else control - 1
        ope = 5 + op
    return location * num_actions_per_qubit + ope

# class Action(object):
#   """Action representation."""

#   def __init__(self, index: int):
#     self.index = index

#   def __hash__(self):
#     return self.index

#   def __eq__(self, other):
#     return self.index == other.index

#   def __gt__(self, other):
#     return self.index > other.index

class Action:
    def __init__(self, index: int):
        self.index = index

    def __hash__(self):
        return hash(self.index)

    def __eq__(self, other):
        return isinstance(other, Action) and self.index == other.index

    def __repr__(self):
        return f"Action({self.index})"

    def __getstate__(self):
        return self.index

    def __setstate__(self, state):
        self.index = state



class ActionHistory(object):
  """Simple history container used inside the search.

  Only used to keep track of the actions executed.
  """

  def __init__(self, history: Sequence[Action], action_space_size: int): ## action_space_size 所有可能的门，包括作用在不同量子比特上的相同门
    self.history = list(history) ## list of gate(Action) index
    self.action_space_size = action_space_size

  def clone(self):
    return copy.deepcopy(ActionHistory(self.history, self.action_space_size))

  def add_action(self, action: Action):
    self.history.append(action.index)

  def last_action(self): 
    return self.history[-1]
  ## action space 给出的是 Action list
  def action_space(self) -> Sequence[Action]:
    return [Action(i) for i in range(self.action_space_size)]

  def to_play(self) -> Player:
    return Player(0) ## only self-play


def soft_cross_entropy(logits, target_probs):
    
    log_probs = F.log_softmax(logits, dim=-1)
    # print("log_probs", log_probs, type(log_probs))
    return -(target_probs * log_probs).sum(dim=-1).mean()

## 并行化处理
def collate_batch(batch: List[Dict[str, Any]], max_length) -> Dict[str, Any]:
    # if test_model:
    #     print("test model collate_batch", batch)
    #     for item in batch:
    #         print("test model circuit", item["circuit"])
    
    # pad_token = np.full((3,), -1)
    circuits = []
    circuits_length = []
    matrices = []
    for idx in range(len(batch)):
        circuit = batch[idx]["circuit"]
        circuit_length = batch[idx]["circuit_length"]
        d2 = max_length - len(circuit) 
        pad_token = np.full((d2, 3), -1)
        # print("circuit", circuit, type(circuit))
    #    print("gate", type(circuit[0]))

        if circuit.size == 0:
            padded = pad_token
        else:
            padded = np.concatenate([circuit, pad_token], axis=0)
           
        circuits.append(padded)
        circuits_length.append(circuit_length)
        matrices.append(torch.tensor(batch[idx]["matrix"], dtype=torch.complex64))  # ✅ 保留复数

    circuits_np = np.array(circuits)  # 自动转换成统一形状的 np.ndarray
    circuits = torch.from_numpy(circuits_np).to(dtype=torch.long)

    return {
        "circuits": circuits,
        "circuits_length": torch.tensor(circuits_length, dtype=torch.long),
        "matrices": torch.stack(matrices),
    }

def move_to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [move_to_device(x, device) for x in data]
    else:
        return data
    
def move_to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cpu()
    elif isinstance(obj, dict):
        return {k: move_to_cpu(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to_cpu(v) for v in obj]
    elif hasattr(obj, '__dict__'):
        for k, v in vars(obj).items():
            setattr(obj, k, move_to_cpu(v))
        return obj
    else:
        return obj
    
def save_game_to_file(game, filename="games.pkl", folder="saved_games"):
    # print("save_game_to_file", filename, folder)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    with open(path, "wb") as f:
        pickle.dump(game, f) 
    # print(f"游戏已保存至 {filename}")   

# def load_games(filename="games.pkl", folder="saved_games"):
#     path = os.path.join(folder, filename)
#     if os.path.exists(path):
#         with open(path, "rb") as f:
#             games = pickle.load(f)
#         return games
#     else:
#         print("No saved games now, initialization.")
#         return []
    
def save_network_to_file(network, filename="network_dump.pkl", folder="saved_networks"):
   pass

def networks_equal(net1: nn.Module, net2: nn.Module) -> bool:
    state_dict1 = net1.state_dict()
    state_dict2 = net2.state_dict()

    if state_dict1.keys() != state_dict2.keys():
        return False

    for key in state_dict1:
        if not torch.equal(state_dict1[key], state_dict2[key]):
            return False

    return True


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
    self.num_actors = 4  # num of actors
    # pylint: disable-next=g-long-lambda
    # self.visit_softmax_temperature_fn = lambda steps: (
    #     1.0 if steps < 500e3 else 0.5 if steps < 750e3 else 0.25
    # )
    self.max_moves = 20 ## 10 for test
    self.num_simulations = 100 ## 80 for test
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
        max_circuit_length = self.max_moves,  # 50
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
    self.hparams.embedding_dim = 128
    self.hparams.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.hparams.representation = ml_collections.ConfigDict()
    self.hparams.representation.input_dim = 10 # 输入维度，通常是门的 one-hot 编码长度
    self.hparams.representation.head_depth = 128
    self.hparams.representation.num_heads = 4
    self.hparams.representation.dropout = 0.2
    self.hparams.representation.num_mqa_blocks = 1 ## 1 for test
    
    # self.hparams.representation.attention.position_encoding = 'absolute'
    # self.hparams.representation.repr_net_res_blocks = 8
    # self.hparams.representation.attention = ml_collections.ConfigDict()
    # self.hparams.representation.use_program = True
    # self.hparams.representation.use_locations = True
    # self.hparams.representation.use_locations_binary = False
    # self.hparams.representation.use_permutation_embedding = False

    self.hparams.prediction = ml_collections.ConfigDict()
    self.hparams.prediction.value_max = 10.0  # These two parameters are task / reward-
    self.hparams.prediction.num_bins = 51  # dependent and need to be adjusted.
    self.hparams.prediction.num_resblocks = 2  # dependent and need to be adjusted.
    self.hparams.prediction.hidden_dim = 128  # hidden_dim of the prediction network

    ### Training
    self.max_training_steps = 1000# for test
    self.checkpoint_interval = 100 ## for test
    self.target_network_interval = 200
    self.window_size = int(1e6)
    self.batch_size = 16 ## 8 for test 
    self.td_steps = 5
    self.lr_init = 1e-4 ## 1e-3 for test
    self.momentum = 0.9

    # Build action maps
    self._build_action_maps()
  
  def visit_softmax_temperature_fn(self, steps):
    return 1.0 if steps < 500e3 else 0.5 if steps < 750e3 else 0.25

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
        self.player = 0
        self.qc = QuantumCircuit(self.task_spec.num_qubits)
        self.is_terminal = False
        self.discount = task_spec.discount
        self.fidelity_reward_weight = task_spec.fidelity_reward_weight
        self.length_reward_weight = task_spec.length_reward_weight

        self.fidelities = []
        self.rewards = []
        self.fidelity_rewards = []
        self.length_rewards = []
        self.history = [] ## list of gate index
        self.history_Instruction = [] ## list of instructions could be implemented ["CX", 1, 0]
        self.history_numed = [] ## [gate_type_index, control bit, target bit] [2, 0, 0]
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
        # print(" self.history 前",  self.history)
        self.history.append(action_index)   ##list of gate index
        # print(" self.history 后",  self.history)
        action_numed = copy.deepcopy(action)
        action_numed[0] = GATES2Index[action_numed[0]]
        self.history_numed.append(action_numed)  ## for collate batch [gate_type_index, control bit, target bit] [2, 0, 0]

        self.history_Instruction.append(action)  
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
        # print("root in store_search_statistics", root.children)

        # print("root.children", root.children)
        # for child in root.children:
        #     print("child", child)

        
        
        sum_visits = 0
        for child in root.children.values():
            # print("child", child)
            sum_visits += child.visit_count
            # print("child visit count", child.visit_count)

        # print("sum_visits", sum_visits)

        ## list of all Actions
        action_space = self.action_history().action_space() 

        # for a in action_space:
        #     visit_count = root.children[a].visit_count 
        #     print("visit count", visit_count)
        #     self.child_visits.append


        #     print("node children", isinstance(root.children[a], Node))
                
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
        try:
            mat = Operator(self.simulator.qc).data
        except Exception as e:
            print("Operator 生成失败！当前电路指令：")
            print(self.simulator.qc)
            raise e  # 继续抛出错误，帮助你调试
        mat_cpu = np.array(mat)  # 确保是 numpy.ndarray 且在 CPU
        circuit_length_cpu = int(self.simulator.length_qc)  # 转为 python int
        circuit_cpu = np.array(self.history_numed, dtype=np.int32)
        # print("self.fidelities", self.fidelities)
        fidelity = float(self.fidelities[-1])
        f_reward = float(self.fidelity_rewards[-1])
        l_reward = float(self.length_rewards[-1])
        player = int(self.player)
        discount = float(self.discount)
        child_visits = np.array(self.child_visits[-1], dtype=np.float32) if self.child_visits else None

        return {
            'matrix': mat_cpu,
            'circuit_length': circuit_length_cpu,
            'circuit': circuit_cpu,
            'fidelity': fidelity,
            "f_reward":f_reward,
            "l_reward":l_reward,
            "player": player,
            "discount": discount,
            "child_visits": child_visits,
        }
    def render_ob(self):
        try:
            mat = Operator(self.simulator.qc).data
        except Exception as e:
            print("Operator 生成失败！当前电路指令：")
            print(self.simulator.qc)
            raise e  # 继续抛出错误，帮助你调试
        mat_cpu = np.array(mat)  # 确保是 numpy.ndarray 且在 CPU
        circuit_length_cpu = int(self.simulator.length_qc)  # 转为 python int
        circuit_cpu = np.array(self.history_numed, dtype=np.int32)
        if self.fidelities:
            fidelity = self.fidelities[-1]
        else:
            fidelity = 0.0
        return {
            'matrix': mat_cpu,
            'circuit_length': circuit_length_cpu,
            'circuit': circuit_cpu,
            'fidelity': fidelity,
        }
    def make_observation(self, state_index: int):
        if state_index == -1:
            return self.render_ob()
        env = self.reset()
        for action in self.history_Instruction[:state_index]:
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
        f_rewards = 0.0
        l_rewards = 0.0

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
            torch.tensor(self.child_visits[state_index], dtype=torch.float32),
            # self.child_visits[state_index],
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
  policy_logits: torch.Tensor 
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

        self.training_steps = 0  # 训练步数



        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.LayerNorm):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)



    def forward(self, observation):
        embedding = self.representation(observation)
        return self.prediction(embedding)
    
    def inference(self, observation):
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
        return self.training_steps  # 训练步数


class UniformNetwork(nn.Module):       ## 输出有问题！！！ 
    
    """
    UniformNetwork: 测试pipeline, 始终返回固定输出。
    """
    def __init__(self, num_actions: int):
        super().__init__()  # 正确初始化 nn.Module
        self.num_actions = num_actions
        self.params = {}
        self.training_steps = 0  # 训练步数

    def forward(self, observation: torch.Tensor) -> NetworkOutput:
        batch_size = observation.shape[0] if observation.ndim > 1 else 1

        outputs = [NetworkOutput(
            value=np.zeros(self.num_actions), 
            fiedlity_value_logits=np.zeros(self.num_actions),
            length_value_logits=np.zeros(self.num_actions),
            policy_logits = torch.full((self.num_actions,), fill_value=1.0 / self.num_actions),  # shape: (num_actions,)
        ) for _ in range(batch_size)]

        return outputs
    
    def inference(self, observation: torch.Tensor): 
        self.eval()

        return NetworkOutput(
            value = float(0),  # shape: (batch,)
            fidelity_value_logits = float(0),  # shape: (batch, num_bins)
            length_value_logits = float(0),          # shape: (batch, num_bins)
            policy_logits = torch.full((self.num_actions,), fill_value=1.0 / self.num_actions),  # shape: (num_actions,)
        )

    
    def get_params(self):
        return self.params

    def update_params(self, updates) -> None:
        self.params = updates

    def training_steps(self) -> int:
        ## 训练增加 
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
        # print("print circuits", circuits)  # [batch_size, max_length_circuit, 3]
        
        batch_size = circuits.shape[0]  # 批大小
        max_length_circuit = self.max_circuit_length  # 最大电路长度
        # if test_model:
        #    print("test model circuits", circuits.shape)  
        if circuits.shape[1] == 0:
            # 空电路处理：返回全零嵌入或 learnable 的默认嵌入
            return torch.zeros((batch_size, self.embedding_dim), device=self.device)
        else:
            batch_size, seq_len, _ = circuits.shape  # 添加 input_dim 解包

        circuits_onehot = self.circuits2onehot(circuits, batch_size, max_length_circuit).to(self.device)  # 将电路转换为 one-hot 编码
        # print(f"circuits_onehot: {circuits_onehot.shape}")  # [batch_size, max_length_circuit, input_dim]
        # print("device",circuits_onehot.device)
        circuits_onehot_view = circuits_onehot.view(-1, self.input_dim)
        # print("device",circuits_onehot_view.device)
    # === 建议2：reshape 以适配 mlp_embedder ===
        embedded_mlp = self.mlp_embedder(circuits_onehot.view(-1, self.input_dim)).view(batch_size, seq_len, -1)
        _, seq_size, feat_size = embedded_mlp.shape

        # pos_enc = self.get_position_encoding(seq_size, feat_size, embedded_mlp.device)  ## take care of device
        # embedded = embedded_mlp + pos_enc.unsqueeze(0)
        # 获取 sinusoidal 位置编码，支持 mask 掉 padding 部分。
        pos_enc = self.get_position_encoding_with_mask(seq_size, feat_size, embedded_mlp.device, circuits_length) 
        embedded = embedded_mlp + pos_enc  # 加入位置编码

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
            # print("input", input)

            input = [input] 
            observation = move_to_device(collate_batch(input, self.max_circuit_length), self.device)
            return self.forward(observation)  


    
    ## 真正有一个batch时可能会不同
    def circuits2onehot(self, circuits, batch_size, max_length_circuit):
        """
        将 circuits 三列 (gate, location, control) 编码为 one-hot 格式，
        并正确处理 padding（值为 -1 的项将被编码为全 0 向量）。
        """
        gates = circuits[:, :, 0].clone().to(self.device)
        locations = circuits[:, :, 1].clone().to(self.device)
        control = circuits[:, :, 2].clone().to(self.device)

        # === Create masks to detect valid entries ===
        gates_mask = gates != -1
        loc_mask = locations != -1
        ctrl_mask = control != -1

        # === Replace -1 with dummy value for one_hot ===
        gates[gates == -1] = 0
        locations[locations == -1] = 0
        control[control == -1] = 0

        # === One-hot encode ===
        gates_one_hot = F.one_hot(gates, num_classes=self.num_gate_types).float()
        locations_one_hot = F.one_hot(locations, num_classes=self.num_qubits).float()
        control_one_hot = F.one_hot(control, num_classes=self.num_qubits).float()

        # === Zero-out padding locations ===
        gates_one_hot[~gates_mask] = 0.0
        locations_one_hot[~loc_mask] = 0.0
        control_one_hot[~ctrl_mask] = 0.0

        # === Concatenate all one-hot encodings ===
        circuits_onehot = torch.cat([gates_one_hot, locations_one_hot, control_one_hot], dim=-1)

        # === Sanity check ===
        assert circuits_onehot.shape[:2] == (batch_size, max_length_circuit), \
            f"Shape mismatch: got {circuits_onehot.shape}, expected ({batch_size}, {max_length_circuit}, ?)"

        return circuits_onehot

    
    
    def get_position_encoding(self, seq_len, dim, device):
        """
        获取 sinusoidal 位置编码，shape 为 [seq_len, dim]
        """
        position = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)  # [seq_len, 1]
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float, device=device) * (-math.log(10000.0) / dim))
    
        pe = torch.zeros(seq_len, dim, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维
    
        return pe  # [seq_len, dim]

    def get_position_encoding_with_mask(self, seq_len, dim, device, circuits_length=None):
        """
        获取 sinusoidal 位置编码，支持 mask 掉 padding 部分。
        
        Args:
            seq_len (int): 电路的最大长度（padding 后的长度）
            dim (int): 编码维度，等于嵌入维度
            device: 当前计算设备
            circuits_length (Tensor or None): shape (batch_size,)
                每个样本真实的电路长度。如果为 None，则不做 mask。
        
        Returns:
            Tensor: shape [batch_size, seq_len, dim]，padding 位置编码为 0
        """

        # [seq_len, 1]
        position = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)

        # 频率因子 [dim/2]
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float, device=device) * (-math.log(10000.0) / dim))

        # [seq_len, dim]
        pe = torch.zeros(seq_len, dim, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        if circuits_length is None:
            # 没有 mask：返回 [1, seq_len, dim]
            return pe.unsqueeze(0)

        # circuits_length: shape [batch_size]
        batch_size = circuits_length.shape[0]

        # 构造 mask：[batch_size, seq_len]
        mask = torch.arange(seq_len, device=device).unsqueeze(0) < circuits_length.unsqueeze(1)

        # mask shape: [batch_size, seq_len, 1]
        mask = mask.unsqueeze(-1).float()

        # 将 pe 扩展为 [batch_size, seq_len, dim] 后乘以 mask
        return pe.unsqueeze(0) * mask


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
        # print(f"scalar: {scalar}")
        batch_size = scalar.shape[0]
        
        # 缩放到 bin 的 float 下标（不取整）
        # eps = 1e-6
        # pos = ((scalar - self.value_min) / self.delta).clamp(0, self.num_bins - 1 - eps)
        pos = (scalar - self.value_min) / self.delta
        # print(f"pos: {pos}")
        lower_idx = pos.floor().long()
        # print(f"lower_idx: {lower_idx}")
        # upper_idx = lower_idx + 1
        upper_idx = (lower_idx + 1).clamp(0, self.num_bins - 1)
        # print(f"upper_idx: {upper_idx}")

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

def to_cpu_output(output: NetworkOutput) -> NetworkOutput:
    return NetworkOutput(
        value=output.value.cpu() if isinstance(output.value, torch.Tensor) else output.value,
        fidelity_value_logits=output.fidelity_value_logits.cpu(),
        length_value_logits=output.length_value_logits.cpu(),
        policy_logits=output.policy_logits.cpu(),
        hidden_state=output.hidden_state.cpu() if isinstance(output.hidden_state, torch.Tensor) else output.hidden_state
    )
    

    

#################################################################
#####################  AlphaCirNetwork-End- #########################



#################################################################
####################### Replay_Buffer-Start- #######################

## Single-prcessing verison
# class ReplayBuffer(object):
#   """Replay buffer object storing games for training."""
  

#   def __init__(self, config: AlphaCirConfig):
#     self.window_size = config.window_size  # 最多存多少局游戏（FIFO）
#     self.batch_size = config.batch_size    # 每次训练采样多少个训练样本
#     self.buffer = load_games()             # 保存所有游戏的列表，每一项是一个完整 Game 实例

#   def save_game(self, game):
#     if len(self.buffer) >= self.window_size:
#         self.buffer.pop(0)
#     self.buffer.append(game)

#   def sample_batch(self, td_steps: int) -> Sequence[Sample]:
    
#     games = [self.sample_game() for _ in range(self.batch_size)]
#     game_pos = [(g, self.sample_position(g)) for g in games]
#     return [
#         Sample(
#             observation=g.make_observation(i),
#             bootstrap_observation=g.make_observation(i + td_steps),
#             target=g.make_target(i, td_steps, g.to_play()),
#         )
#         for (g, i) in game_pos
#     ]

#   def sample_game(self) -> Game:
    
#     return random.choice(self.buffer) if self.buffer else None

#   def sample_position(self, game) -> int:
#     return random.randint(0, len(game.history) - 1) if game and len(game.history) > 0 else -1


def move_game_to_cpu(game):
    """
    将 Game 对象中所有在 GPU 上的 torch.Tensor 成员移到 CPU。
    """
    for name, value in game.__dict__.items():
        # 如果是 Tensor，则移动到 CPU
        if isinstance(value, torch.Tensor):
            game.__dict__[name] = value.detach().cpu()

        # 如果是 list，则尝试处理其中每个元素
        elif isinstance(value, list):
            new_list = []
            for v in value:
                if isinstance(v, torch.Tensor):
                    new_list.append(v.detach().cpu())
                else:
                    new_list.append(v)
            game.__dict__[name] = new_list

        # 如果是 dict，则尝试处理其中每个值
        elif isinstance(value, dict):
            new_dict = {}
            for k, v in value.items():
                if isinstance(v, torch.Tensor):
                    new_dict[k] = v.detach().cpu()
                else:
                    new_dict[k] = v
            game.__dict__[name] = new_dict

    return game


def load_games(path = None, filename="games.pkl", folder="saved_games" ):
    # print("path:", path)
    if path is None:
        path = os.path.join(folder, filename)
    # print("os.path.exists(path)", os.path.exists(path))
    if os.path.exists(path):
        with open(path, "rb") as f:
            games = pickle.load(f)
        return games
    else:
        print("No saved games now, initialization.")
        return []


class ReplayBuffer:
    """基于多进程 Manager 的安全 ReplayBuffer，供多个 self-play actor 写入、trainer 读取训练数据。"""

    def __init__(self, config: AlphaCirConfig):
        self.window_size = config.window_size
        self.batch_size = config.batch_size
        self._buffer = load_games(path="saved_games/games.pkl")  # ⬅️ 初始化加载 


    @property
    def buffer(self):
        self._buffer = load_games(path="saved_games/games.pkl")  # ⬅️ 读取最新的游戏数据
        return self._buffer  # 读取游戏数据
        # print(" in buffer")
        # buffer_cpu = []
        # print(" in buffer 1")
        # buffer = load_games()
        # print(" in buffer 2")
        # for game in buffer:
        #     print("game", game)
        # for game in buffer:
        #     # print("game", game)
        #     # game = move_game_to_cpu(game)
        #     # print("game after cpu", game)
        #     buffer_cpu.append(game)
        #     # print("game append", game)
        # return buffer
    
    def save_buffer_to_file(self, buffer_new, path):
        path_saved = "saved_games/games.pkl"
        buffer_temp = copy.deepcopy(self.buffer)
        buffer_temp.extend(buffer_new)
        if len(self._buffer) > self.window_size:
            buffer_temp = buffer_temp[-self.window_size:]
        with open(path, "wb") as f:
            pickle.dump(buffer_temp, f)
        os.replace(path, path_saved)  # 原子替换



    def sample_batch(self, td_steps: int) -> Sequence[Sample]:
        
        game_histories = [self.sample_game() for _ in range(self.batch_size)]
        game_pos = [(gh, self.sample_position(gh)) for gh in game_histories]

    

        return [
            Sample(
                observation=copy.deepcopy(gh[i]),
                bootstrap_observation=copy.deepcopy(gh[i + td_steps]) if i + td_steps < len(gh) else copy.deepcopy(gh[-1]),
                target=make_target(gh, i, td_steps),
            )

            for (gh, i) in game_pos
        ]

    def sample_game(self) -> Game:    
        return random.choice(self.buffer) if self.buffer else None

    def sample_position(self, game_history) -> int:
        return random.randint(0, len(game_history) - 1) if game_history and len(game_history) > 0 else -1

# def make_target(game_history:list, state_index, td_steps: int) -> np.ndarray: #In CPU
#         """Creates the value target for training."""
#     # The value target is the discounted sum of all rewards until N steps
#     # into the future, to which we will add the discounted boostrapped future
#     # value.
#         f_rewards = 0.0
#         l_rewards = 0.0
#         gamme = game_history[state_index]["discount"]
#         bootstrap_index = state_index + td_steps
#         for i in range(state_index, min(bootstrap_index, len(game_history) )):
          
#             discount = gamme**( i - state_index )
#             f_rewards += game_history[i]["f_reward"] * discount
#             l_rewards += game_history[i]["l_reward"] * discount



#         if bootstrap_index < len(game_history):
#             bootstrap_discount = gamme**td_steps
#         else:
#             bootstrap_discount = 0

#         return Target(
#             f_rewards,
#             l_rewards,
#             game_history[state_index]["child_visits"], ## np.array
#             bootstrap_discount,
#         )
    
def make_target(game_history: list, state_index: int, td_steps: int) -> Target:
    f_rewards = 0.0
    l_rewards = 0.0
    gamma = game_history[state_index]["discount"]

    for i in range(state_index, min(state_index + td_steps, len(game_history))):
        factor = gamma ** (i - state_index)
        f_rewards += game_history[i]["f_reward"] * factor
        l_rewards += game_history[i]["l_reward"] * factor

    bootstrap_discount = gamma ** td_steps if state_index + td_steps < len(game_history) else 0.0

    return Target(
        f_rewards,
        l_rewards,
        game_history[state_index]["child_visits"],
        bootstrap_discount,
    )










# class ReplayBuffer(object):
#   """Replay buffer object storing games for training."""
#   ### Multi-prcessing verison

#   def __init__(self, config: AlphaCirConfig):
#     self.window_size = config.window_size  # 最多存多少局游戏（FIFO）
#     self.batch_size = config.batch_size    # 每次训练采样多少个训练样本
#     self.buffer = load_games()             # 保存所有游戏的列表，每一项是一个完整 Game 实例
#     self.lock = Lock()                     # 多进程锁，确保进程安全读写 buffer

#   def save_game(self, game):
#     with self.lock:
#         if len(self.buffer) >= self.window_size:
#             self.buffer.pop(0)
#         self.buffer.append(game)

#   def sample_batch(self, td_steps: int) -> Sequence[Sample]:
#     with self.lock:
#         games = [self.sample_game() for _ in range(self.batch_size)]
#         game_pos = [(g, self.sample_position(g)) for g in games]
#         return [
#             Sample(
#                 observation=g.make_observation(i),
#                 bootstrap_observation=g.make_observation(i + td_steps),
#                 target=g.make_target(i, td_steps, g.to_play()),
#             )
#             for (g, i) in game_pos
#         ]

#   def sample_game(self) -> Game:
#     with self.lock:
#         return random.choice(self.buffer) if self.buffer else None

#   def sample_position(self, game) -> int:
#     return random.randint(0, len(game.history) - 1) if game and len(game.history) > 0 else -1


#################################################################
####################### Replay_Buffer- End - #######################

#################################################################
####################### Shared_Storage-Start- #######################
def make_uniform_network(num_actions: int) -> UniformNetwork:
    return UniformNetwork(num_actions)

class SharedStorage(object):
  """Controls which network is used at inference."""

  def __init__(self, num_actions: int, config: AlphaCirConfig, model_dir="saved_models"):
    self._num_actions = num_actions
    self.config = config
    self._networks = {}
    self._optimizers = {}
    self.model_dir = model_dir
    self._optimizer_state_dict = None

    # 尝试加载已有模型
    os.makedirs(model_dir, exist_ok=True)
    model_files = glob.glob(os.path.join(model_dir, "network_step_*.pt"))

    if model_files:
        extract_step = lambda f: int(re.findall(r"network_step_(\d+).pt", f)[0])
        latest_file = max(model_files, key=extract_step)
        latest_step = extract_step(latest_file)

        print(f"已存储的最新模型: {latest_file}")
        checkpoint = torch.load(latest_file)
        # print("checkpoint", checkpoint.keys())

        model = AlphaCirNetwork(self.config.hparams, self.config.task_spec)
        model.load_state_dict(checkpoint["network_state"])
        self._networks[latest_step] = model

        # ✅ 存储优化器状态
        self._optimizer_state_dict = checkpoint["optimizer_state"]
    # else: 
    #    print("初始化失败")


  def latest_network(self) -> AlphaCirNetwork:
    if self._networks:
    #   print("get latest network")
      # Return the latest network (highest step number).
      print("stored network exists, start with latest network")

      return self._networks[max(self._networks.keys())]
    else:
      # policy -> uniform, value -> 0, reward -> 0
      print("None stored network, start with uniform network")
      return make_uniform_network(self._num_actions)

  def save_network(self, step: int, network: AlphaCirNetwork, optimizer: torch.optim.Optimizer):
    ## 存储在saved_networks文件中
    ## 只保存AlphaCirNetwork
    if isinstance(network, AlphaCirNetwork):
        # === 及时存储 ===
        # print("存储")
        self._networks[step] = copy.deepcopy(network)

        # === 硬盘存储 ===
        os.makedirs(self.model_dir, exist_ok=True)  # 如果不存在则创建
        path = os.path.join(self.model_dir, f"network_step_{step}.pt")  # 文件名中包含 step 方便排序
        
        save_dict = {
            "network_state": network.state_dict(),  
        }
        if optimizer is not None:
            save_dict["optimizer_state"] = optimizer.state_dict()  ## 保存优化器状态
        torch.save(save_dict, path)  # 保存权重到指定路径
        print(f"保存神经网络到文件: {path}")  # 提示用户保存成功





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
    # print("visit_counts", visit_counts)
    visits = np.array([v for v, _ in visit_counts], dtype=np.float32)
    # print("visits", visits)
    # print("temperature", temperature)
    if temperature == 0.0:
        # 贪婪：选择访问次数最多的动作
        max_visit = np.max(visits)
        indices = np.where(visits == max_visit)[0]
        selected = random.choice(indices)
        return visit_counts[selected]
    
    # 否则按 softmax 分布采样
    sum_visits = np.sum(visits)
    # if sum_visits == 0:
    #     sum_visits = num_simulations  # 避免除以零
    visits = visits ** (1 / temperature)  # 温度调整
    probs = visits / sum_visits      # softmax 概率
    index = np.random.choice(len(visit_counts), p=probs)
    return visit_counts[index]



# Each self-play job is independent of all others; it takes the latest network
# snapshot, produces a game and makes it available to the training job by
# writing it to a shared replay buffer.
def run_selfplay(
    config: AlphaCirConfig, storage: SharedStorage, replay_buffer: ReplayBuffer, games_Queue = None, net = None
):
#   print("run_selfplay start")
  if test_model:
    for _ in tqdm(range(test_model_run)):
        if net is not None:
            network = net
        else:
            network = storage.latest_network()
        
        game = play_game(config, network)
        replay_buffer.save_game(game)
        save_game_to_file(replay_buffer.buffer, filename=f"games.pkl")
     
  else:
    # print("run_selfplay start with multiprocessing")
    
    # print("num:", num)
    
    # print("标签0")
    num = len(replay_buffer.buffer)
    pbar = tqdm(desc=f"[{current_process().name}] Stored Games", unit=" games", initial=len(replay_buffer.buffer))

    while True:
        # print("in while:")

        buffer_size = len(replay_buffer.buffer)
        pbar.n = buffer_size
        pbar.refresh()
        
        network = storage.latest_network()
        # network = AlphaCirNetwork(config.hparams, config.task_spec).to("cpu")
        # network.load_state_dict(storage.latest_network().state_dict())
        network.eval()
        # print("get net:")
        game_history = play_game(config, network)
        # print(f"{current_process().name} generated game history:", game_history)
        recorder_fid.add_scalar("SelfPlay/Fidelity", game_history[-1]["fidelity"], num)
        # print("标签1")
        games_Queue.put(game_history)  # 将游戏放入队列供训练使用
        # print("标签2")
        ## 存储数据
        # replay_buffer.save_game(game)
        # save_game_to_file(replay_buffer.buffer, filename=f"games.pkl")
        
        ## tqdm 进度条更新
        # pbar.update(1) 
        # print("标签3")
        num += 1


def move_network_output_to_cpu(output: NetworkOutput) -> NetworkOutput:
    return NetworkOutput(
        value=output.value.cpu() if isinstance(output.value, torch.Tensor) else output.value,
        fidelity_value_logits=output.fidelity_value_logits.cpu(),
        length_value_logits=output.length_value_logits.cpu(),
        policy_logits=output.policy_logits.cpu(),
        hidden_state=None  # 或 output.hidden_state.cpu() 如果有的话
    )

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
  print("play_game start")
  game = Game(config.task_spec)
  game_history = [] 


  while not game.terminal() and len(game.history) < config.max_moves:
    min_max_stats = MinMaxStats(config.known_bounds)

    # Initialisation of the root node and addition of exploration noise
    root = Node(0)
    # if test_model:
    #     print("test model INI: root", root)
    current_observation = game.make_observation(-1)
    # if test_model:
    # print("test model: current_observation", current_observation)
    network_output = network.inference(current_observation)
    # for name, value in network_output.__dict__.items():
    #     if isinstance(value, torch.Tensor) and value.is_cuda:
    #         print(f"⚠️ GPU Tensor in play game 未释放：{name}")
    #     else:
    #         print(f"✅ {name} 正常 in play game")
    # if test_model:
    #     print("test model: current_network_output", network_output)
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
    # print("root", root.children)
    # action:Action  选择动作完全根据概率分布 
    action = _select_action(config, len(game.history), root, network, legal_actions_index)
    index = action.index
    gate = index2action(index, config.task_spec.num_qubits, config.task_spec.num_actions)  # ["H", 0, 0]
    game.step(gate)
    # print("visit situation", game.child_visits)
    
    # if test_model:
    #     # print("test model: gate", gate)
    #     print("test model: game history in play game, actual applied ", game.history)
    game.store_search_statistics(root)
    # print("visit situation after store", game.child_visits)
    # game = move_to_cpu(game)
    game_step = game.render()
    game_history.append(game_step)  # 将每一步的游戏状态添加到历史记录中

  return game_history


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
        ## MCTS中也不许超过最大长度
        if sim_env.simulator.length_qc >= config.task_spec.max_circuit_length:
            break

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
#   print(f"[MCTS Done] Root has {len(root.children)} children.")


def _select_action(
    # pylint: disable-next=unused-argument
    config: AlphaCirConfig, num_moves: int, node: Node, network: AlphaCirNetwork, legal_actions_index: List[int]
):
  
    # print("Expected indices:", legal_actions_index)
    # print("Node children keys:", [a.index for a in node.children.keys()])

    visit_counts = []
    # visit_counts_before =[]
    # print("legal_actions_index",legal_actions_index)
    for action, child in node.children.items():
        # visit_counts_before.append((child.visit_count, action))
        # print("action index", action.index)
        if action.index not in legal_actions_index:
            # print("illegal action", action.index)
            # print("visit count before ilegal ", child.visit_count)
            child.visit_count = 0  # 非法动作访问次数设为 0
        visit_counts.append((child.visit_count, action))  

    total_visits = sum([v for v, _ in visit_counts])
    visit_average = int(config.num_simulations/len(legal_actions_index))
    if total_visits == 0:
        visit_counts = []
        for action, child in node.children.items():
            if action.index not in legal_actions_index:
                child.visit_count = 0  # 非法动作访问次数设为 0
            else:
                child.visit_count = visit_average
            visit_counts.append((child.visit_count, action)) 

    
    # print("visit_counts_before", visit_counts_before)
    # print("visit_counts", visit_counts)
    steps = network.training_steps
    t = config.visit_softmax_temperature_fn(steps)
    _, action = softmax_sample(visit_counts, t)
    return action ## Action


def _select_child(
    config: AlphaCirConfig, node: Node, min_max_stats: MinMaxStats
):
  """Selects the child with the highest UCB score."""

  _, action, child = max(
        (( _ucb_score(config, node, child, min_max_stats), action, child )
        for action, child in node.children.items()),
        key=lambda x: x[0]  
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
#     print("test model: actions", actions)
#   if test_model:    
#     print("test model:network_output.policy_logits", network_output.policy_logits )
#     print("test model:actions", actions)
#   for a in actions:
#     print("test model: a", a.index)
#     item = network_output.policy_logits[0, a.index]
#     print("test model: network_output.policy_logits[a.index]", item)
#     prob = torch.exp(item)
#     print("test model: prob", prob)
#   if test_model:
#     print("test model: network_output.policy_logits", network_output.policy_logits)
  policy_logits = network_output.policy_logits.squeeze(0)  # shape: [num_actions]
  policy = { a: torch.exp(policy_logits[a.index]) for a in actions}

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
class Trainer:
    def __init__(self, config: AlphaCirConfig, storage: SharedStorage, replay_buffer: ReplayBuffer):    
       self.config = config
       self.storage = storage
       self.replay_buffer = replay_buffer
       self.device = self.config.hparams.device
       self.network = self.storage.latest_network().to(self.device)

       if isinstance(self.network, AlphaCirNetwork):
           print("Trainer: network is AlphaCirNetwork")

       if isinstance(self.network, UniformNetwork):
          self.network = AlphaCirNetwork(self.config.hparams, self.config.task_spec).to(self.device)
          self.train_step = 0
       else:
            # 从已有网络中取出最大的 step
          self.train_step = max(self.storage._networks.keys())
       self.optimizer = torch.optim.SGD(self.network.parameters(), lr=self.config.lr_init, momentum=self.config.momentum)
       if storage._optimizer_state_dict is not None:
            self.optimizer.load_state_dict(self.storage._optimizer_state_dict)
            print("已恢复 optimizer 状态") 
            for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(self.device)


    def train_network(self):
        """Trains the network on data stored in the replay buffer."""
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("train start")
        losses = []


        
        target_network = AlphaCirNetwork(self.config.hparams, self.config.task_spec).to(self.device)
        target_network.load_state_dict(self.network.state_dict())  # 初始化目标网络

        for i in tqdm(range(self.train_step, self.train_step + self.config.max_training_steps), desc="Training Network"):
            
            
            for name, param in self.network.named_parameters():
                if torch.isnan(param).any():
                    print(f"⚠️ 网络参数 {name} 含 NaN")
                # else:
                #     print(f"网络参数 {name} 正常")
                    
            
            for name, param in target_network.named_parameters():
                if torch.isnan(param).any():
                    print(f"⚠️ 目标网络参数 {name} 含 NaN")
                # else:
                #     print(f"目标网络参数 {name} 正常")

            batch = self.replay_buffer.sample_batch(self.config.td_steps)
            self.optimizer.zero_grad()
            loss = self.compute_loss(self.network, target_network, batch, self.device, i)
            recorder_loss.add_scalar("Loss/total", loss.item(), i)

            # print("loss", loss)
            assert not math.isnan(loss), "Loss 是 NaN"
            assert loss.item() >= 0, "Loss 是负数"

            
            losses.append(loss.item())
            model_before = copy.deepcopy(self.network)

            # print("👉 反向传播前，检查梯度是否为 NaN")
            for name, param in self.network.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    print(f"⚠️ 梯度 NaN: {name}")

            loss.backward()

            for name, param in self.network.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    print(f"⚠️ 梯度 NaN: {name}")
            # print("👉 反向传播后，检查梯度是否为 NaN")




            self.optimizer.step()
            model_after = copy.deepcopy(self.network)

            ## 记录参数变化量
            total_change = 0.0
            with torch.no_grad():
                for (name_before, param_before), (name_after, param_after) in zip(model_before.named_parameters(), model_after.named_parameters()):
                    assert name_before == name_after, "参数名称不一致，模型结构可能不同"
                    delta = (param_before - param_after).abs().sum().item()
                    total_change += delta
                    # print(f"{name_before} 参数变化量: {delta:.6f}")

            recorder_params.add_scalar("param_change/total", total_change, i)

            # print(f"💡 参数总变化量: {total_change:.6f}")

            ## 记录平均梯度范数
            total_norm = 0.0
            count = 0
            for _, param in self.network.named_parameters():
                if param.grad is not None:
                    total_norm += param.grad.norm().item()
                    count += 1
            avg_grad_norm = total_norm / count if count > 0 else 0
            recorder_params.add_scalar("grad_norm/avg", avg_grad_norm, i)


            if i % self.config.checkpoint_interval == 0:
                if i > self.train_step:
                    index = max(self.storage._networks.keys())
                    old_net =copy.deepcopy(self.storage._networks[index])
                    current_network = copy.deepcopy(self.network)                
                    assert not networks_equal(old_net, current_network), "旧的网络和当前网络相同"
                self.storage.save_network(i, self.network, self.optimizer)
            time.sleep(1)          

            if i % self.config.target_network_interval == 0:
                target_network.load_state_dict(self.network.state_dict())
        
        self.storage.save_network(self.config.max_training_steps, self.network, self.optimizer)
        return losses

    def compute_loss(self, network, target_network, batch, device, i):
        observations, bootstrap_obs, targets = zip(*batch)

        obs_batch = move_to_device(collate_batch(list(observations), network.representation.max_circuit_length), device)
        boot_batch = move_to_device(collate_batch(list(bootstrap_obs), network.representation.max_circuit_length), device)

        # 解包 targets
        fidelity_value, length_value, policy, discount = zip(*targets)
        fidelity_value = torch.tensor(fidelity_value, dtype=torch.float32, device=device)
        length_value   = torch.tensor(length_value, dtype=torch.float32, device=device)
        policy = np.stack(policy)  # policy 是 list/tuple of np.array
        policy = torch.from_numpy(policy).to(dtype=torch.float32, device=device)
        # print("看看 policy", policy)
        # policy_tensor = torch.tensor(policy, dtype=torch.float32)
        # policy = torch.stack(policy_tensor).to(device)
        discount = torch.tensor(discount, dtype=torch.float32, device = device)

        # target_network.support.mean()
        # 推理
        # print("obs_prenet", obs_batch)
        # print("boot_prenet", boot_batch)

        assert not torch.isnan(obs_batch['circuits']).any(), "obs_batch['circuits'] 含 NaN"
        assert not torch.isinf(obs_batch['circuits']).any(), "obs_batch['circuits'] 含 Inf"


        # 在这里调用：
        assert_tensor_on_cuda(
            fidelity_value=fidelity_value,
            length_value=length_value,
            policy=policy,
            discount=discount
        )

        for name, param in network.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"⚠️ 参数 {name} 包含 NaN 或 Inf")
            # else:
            #     print(f"参数 {name} 正常")
        for name, param in target_network.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"⚠️ 目标参数 {name} 包含 NaN 或 Inf")
            # else:
            #     print(f"目标参数 {name} 正常")

        # for name, param in network.named_parameters():
        #     print(f"{name}: mean={param.data.mean().item():.6f}, std={param.data.std().item():.6f}")


        for name, module in network.named_modules():
            if isinstance(module, (nn.Linear, nn.LayerNorm, nn.Conv1d, nn.ReLU)):
                register_nan_hook(module)

        # for name, module in network.named_modules():
        #     if isinstance(module, (nn.Linear, nn.LayerNorm, nn.ReLU, nn.Softmax, nn.Conv1d)):
        #         module.register_forward_hook(print_tensor_hook)

        pred = network(obs_batch)
        pred_boot = target_network(boot_batch)
        # print("pred", pred)
        # print("pred_boot", pred_boot)
        # assert not torch.all(policy == 0), "Policy target 全为 0，loss 计算无效"
        # print("policy target", policy) 
        # print("policy_logits", pred.policy_logits)

        # TD Bootstrapped target
        # print("fidelity_value", fidelity_value.shape)
        # print("discount", discount.shape)
        # print("pred_boot.fidelity_value_logits", pred_boot.fidelity_value_logits.shape)
        pred_f_value = target_network.prediction.support.mean(pred_boot.fidelity_value_logits)
        fidelity_value += discount * pred_f_value.detach()
        pre_l_value = target_network.prediction.support.mean(pred_boot.length_value_logits)
        length_value += discount * pre_l_value.detach()

        # Loss
        policy_loss = soft_cross_entropy(pred.policy_logits, policy)
        fidelity_loss = scalar_loss(pred.fidelity_value_logits, fidelity_value, network)
        length_loss = scalar_loss(pred.length_value_logits, length_value, network)


        recorder_loss.add_scalar("loss/policy", policy_loss.item(), i)
        recorder_loss.add_scalar("loss/fidelity", fidelity_loss.item(), i)
        recorder_loss.add_scalar("loss/length", length_loss.item(), i)
        
        # print("final policy_loss", policy_loss)
        # print("final fidelity_loss", fidelity_loss)
        # print("final length_loss", length_loss)   


        # return (policy_loss + fidelity_loss + length_loss)/ len(batch)
        return policy_loss + fidelity_loss + length_loss

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
def print_tensor_hook(module, input, output):
    if torch.isnan(output).any():
        print(f"❌ {module.__class__.__name__} 输出包含 NaN")
    elif torch.isinf(output).any():
        print(f"⚠️ {module.__class__.__name__} 输出包含 Inf")
    else:
        print(f"✅ {module.__class__.__name__} 正常: mean={output.mean():.4f}, std={output.std():.4f}")

def register_nan_hook(module):
    def hook(module, input, output):
        if isinstance(output, torch.Tensor) and torch.isnan(output).any():
            print(f"⚠️ 输出 NaN in {module}")
        elif isinstance(output, tuple):
            for i, o in enumerate(output):
                if torch.isnan(o).any():
                    print(f"⚠️ 输出 NaN in {module}, output[{i}]")
    return module.register_forward_hook(hook)

def assert_tensor_on_cuda(**tensors):
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            print(f"❌ {name} 不是 Tensor 类型，而是 {type(tensor)}")
        elif not tensor.is_cuda:
            print(f"⚠️ 警告：{name} 不在 CUDA 上，而是在 {tensor.device}")


def scalar_loss(pred_logits, scalar_target, network):
    """
    pred_logits: shape (B, num_bins)
    scalar_target: shape (B,)
    """
    two_hot_target = network.prediction.support.scalar_to_two_hot(scalar_target)
    log_probs = F.log_softmax(pred_logits, dim=-1)
    loss = -(two_hot_target * log_probs).sum(dim=-1).mean()
    return loss




#################################################################
######################## Trainer - End - #######################

#################################################################
########################### Main -Start- #######################




def monitor_buffer_tqdm(replay_buffer, maximum_games):
    from tqdm import tqdm
    import time

    with tqdm(total=maximum_games, desc="ReplayBuffer Size Monitor", position=0, leave=True) as pbar:
        last_size = 0
        while True:
            current_size = len(replay_buffer.buffer)
            delta = current_size - last_size
            if delta > 0:
                pbar.update(delta)
                last_size = current_size
            time.sleep(1)


## single actor version
def AlphaCir_single(config: AlphaCirConfig):
    # maximum_games = 1000000
    storage = SharedStorage(config.task_spec.num_actions, config)
    replay_buffer = ReplayBuffer(config)
    # print("buffer size:", len(replay_buffer.buffer))
    # print(replay_buffer.buffer)
    trainer = Trainer(config, storage, replay_buffer)
    # losses = None

    selfplay_thread = threading.Thread(
        target=run_selfplay,
        args=(config, storage, replay_buffer),
        daemon=True  # 设置为守护线程，主程序结束就自动退出
    )
    selfplay_thread.start()

    with tqdm(total=config.batch_size, desc="Filling ReplayBuffer") as pbar:
        last = 0
        while len(replay_buffer.buffer) <= config.batch_size:
            current = len(replay_buffer.buffer)
            pbar.update(current - last)
            last = current
            time.sleep(1)

    # buffer_monitor_thread = threading.Thread(
    #     target=monitor_buffer_tqdm,
    #     args=(replay_buffer, maximum_games),  # 你可以改成 config.window_size
    #     daemon=True
    # )
    # buffer_monitor_thread.start()
            


    print("ReplayBuffer已填满，开始训练！")
    losses = trainer.train_network()

    recorder_fid.close()
    recorder_loss.close()
    recorder_params.close()
    return storage.latest_network(), losses

def game_collector(buffer, games_queue):
    buffer_temp = []
    while True:
        try:
            game_history = games_queue.get(timeout = 1000)
           # for name, value in game.__dict__.items():
            #     if isinstance(value, torch.Tensor) and value.is_cuda:
            #         print(f"⚠️ GPU Tensor 未释放：{name}")
            #     else:
            #         print(f"✅ {name} 正常")
            buffer_temp.append(game_history)
        except (queue.Empty, mp_Empty):
            print("[Collector] No data for 1000 seconds, exit.")
            break
        if len(buffer_temp) >= 2:
            buffer.save_buffer_to_file(buffer_temp, path="temp_games/games.pkl")
            buffer_temp = []  # 清空临时缓冲区
        # print("Game saved to buffer. Current size:", len(buffer.buffer))

## multi actors version
def AlphaCir_multi(config: AlphaCirConfig):     
    de_daemon = True
    storage = SharedStorage(config.task_spec.num_actions, config)     
    replay_buffer = ReplayBuffer(config)   
    print("buffer size:", len(replay_buffer.buffer))  
    trainer = Trainer(config, storage, replay_buffer) 
     # 启动多个 self-play actor 进程     
    num_actors = config.num_actors if hasattr(config, 'num_actors') else 4  # 默认 4 个 actor  
    # print("num_actors", num_actors)
    games_Queue = Queue()   
    collector_thread = threading.Thread(target=game_collector, args=(replay_buffer, games_Queue))
    collector_thread.daemon = de_daemon  # 主程序结束它也结束
    collector_thread.start()

    # print("⚠️ here")
    selfplay_processes = []     
    for i in range(num_actors):  
        p = Process(             
            target=run_selfplay,  
            name=f"Actor-{i+1}",           
            args=(config, storage, replay_buffer, games_Queue),             
            daemon = de_daemon,   ## trainer没有问题了 改回True
                )         
        p.start()  
     
        selfplay_processes.append(p) 
     # 等待 ReplayBuffer 填满初始数据     
  

    with tqdm(total=config.batch_size, desc="Filling ReplayBuffer") as pbar:         
        last = 0         
        # print("ReplayBuffer size:", len(replay_buffer.buffer))
        while len(replay_buffer.buffer) <= config.batch_size:             
            current = len(replay_buffer.buffer)             
            pbar.update(current - last)             
            last = current             
            time.sleep(10) 
    print("ReplayBuffer已填满，开始训练！") 




    loss = trainer.train_network()
         
    # 你可以根据需要监控进度或在外部控制退出 
    recorder_fid.close()
    recorder_loss.close()
    recorder_params.close()
    return storage.latest_network()


def launch_job(f, *args):
  f(*args)
#################################################################
########################### Main - End - #######################



if __name__ == "__main__":
    # print(test_model_run) 
    config = AlphaCirConfig()
    output = AlphaCir_multi(config)
    
    # num_qubits = self.config.task_spec.num_qubits
    # type_single_gates = self.config.task_spec.num_ops
    




