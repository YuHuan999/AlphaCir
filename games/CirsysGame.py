import qiskit.circuit.library as qlib
import datetime
import pathlib
import torch
import cirq
from qiskit import QuantumCircuit
import numpy as np
import copy
from games.abstract_game import AbstractGame # for local test
# from abstract_game import AbstractGame  
from typing import NamedTuple, List
from qiskit.quantum_info import Operator

class MuZeroConfig:
    def __init__(self):
        # fmt: off
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = 0  # Seed for numpy, torch and the game
        self.max_num_gpus = 1  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available



        ### Game
        # self.observation_shape = (1, 1, 4)  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.n_qubits = 2 # number of qubits
        # Clifford + T gate: [gate, control_bit, target_bit] for 2 qubits
        self.real_actions = [["H", 0, 0], ["S", 0, 0], ["T", 0, 0], ["S†", 0, 0], ["T†", 0, 0], ["CX", 1, 0], 
                                  ["H", 1, 1], ["S", 1, 1], ["T", 1, 1], ["S†", 1, 1], ["T†", 1, 1], ["CX", 0, 1]], 
        self.action_space = list(range(self.real_actions))  # Fixed list of all possible actions index. 
        self.players = [0]  # List of players. You should only edit the length
        self.stacked_observations = 0  # Number of previous observations and previous actions to add to the current observation

        # Evaluate
        ## use a test set to evaluate the performance of the model


        ### Self-Play
        self.num_workers = 128  # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.selfplay_on_gpu = False
        self.max_moves = 50  # Maximum number of moves if game is not finished before
        self.num_simulations = 800  # Number of future moves self-simulated
        self.discount = 0.997  # Chronological discount of the reward
        self.temperature_threshold = None  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25



        ### Network
        self.network = "Transformer"  # "resnet" / "fullyconnected / "Transformer"
        self.support_size = 10  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))

        # Transformer Network
        self.value_max = 1
        self.value_min = -1
        self.input_dim = 8, ## not sure 
        self.embedding_dim = 128,
        self.nhead = 4 ,
        self.num_encoderLayer = 3,
        self.policy_layers = 2,
        self.correctness_value_layers = 2,
        self.length_value_layers = 2,

        # Residual Network
        self.downsample = False  # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
        self.blocks = 1  # Number of blocks in the ResNet
        self.channels = 2  # Number of channels in the ResNet
        self.reduced_channels_reward = 2  # Number of channels in reward head
        self.reduced_channels_value = 2  # Number of channels in value head
        self.reduced_channels_policy = 2  # Number of channels in policy head
        self.resnet_fc_reward_layers = []  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = []  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = []  # Define the hidden layers in the policy head of the prediction network

        # Fully Connected Network
        self.encoding_size = 8
        self.fc_representation_layers = []  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = [16]  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = [16]  # Define the hidden layers in the reward network
        self.fc_value_layers = [16]  # Define the hidden layers in the value network
        self.fc_policy_layers = [16]  # Define the hidden layers in the policy network



        ### Training
        self.results_path = pathlib.Path(__file__).resolve().parents[1] / "results" / pathlib.Path(__file__).stem / datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")  # Path to store the model weights and TensorBoard logs
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = 1000000  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 128  # Number of parts of games to train on at each training step
        self.checkpoint_interval = 10  # Number of training steps before using the model for self-playing
        self.target_interval = 100  # Number of training steps before updating the target networks
        self.value_loss_weight = 1  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.train_on_gpu = torch.cuda.is_available()  # Train on GPU if available

        self.optimizer = "SGD"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.02  # Initial learning rate
        self.lr_decay_rate = 0.8  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 1000



        ### Replay Buffer
        self.replay_buffer_size = 500  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 10  # Number of game moves to keep for every batch element
        self.td_steps = 50  # Number of steps in the future to take into account for calculating the target value
        self.PER = True  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = 0.5  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

        # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.use_last_model_value = False  ## this program is not necessary
        self.reanalyse_on_gpu = False



        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = 1.5  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it
        # fmt: on

    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        if trained_steps < 0.5 * self.training_steps:
            return 1.0
        elif trained_steps < 0.75 * self.training_steps:
            return 0.5
        else:
            return 0.25
        
class TaskSpec(NamedTuple):
    max_circuit_length: int 
    num_qubits: int
    correct_reward: float 
    correctness_reward_weight: float 
    length_reward: float 
    length_reward_weight: float 
    num_ops: int  # type of gates H S T S† T† CX
    num_actions: int # number of actions, gates in different qubits 
    fidelity_threshold: float # 保真度阈值 less than 1.0
    actions_index: List[int]
    
    @classmethod
    def create(cls, max_circuit_length: int, num_qubits: int, num_ops: int,
               correct_reward: float, correctness_reward_weight: float,
               length_reward: float, length_reward_weight: float, fidelity_threshold: float):
        """
        类方法，先做计算，再实例化 NamedTuple
        """
        num_actions = num_qubits * ((num_ops - 1) + (num_qubits - 1))
        actions_index = list(range(num_actions))
        return cls(
            max_circuit_length=max_circuit_length,
            num_qubits=num_qubits,
            num_ops=num_ops,
            correct_reward=correct_reward,
            correctness_reward_weight=correctness_reward_weight,
            length_reward = length_reward,
            length_reward_weight=length_reward_weight,
            fidelity_threshold=fidelity_threshold,
            num_actions=num_actions,
            actions_index=actions_index
        )
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
    
GATES = {"H": 0, "S": 1, "T": 2, "S†": 3, "T†": 4}

def action2index(action, num_qubits: int, num_actions: int) -> int:
    gate, location, control = action
    num_actions_per_qubit = num_actions // num_qubits
    if gate != "CX":
        ope = GATES[gate]
    else:
        op = control if control < location else control - 1
        ope = 5 + op
    return location * num_actions_per_qubit + ope



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
        self.circuit = {i: [] for i in range(self.task_spec.num_qubits)}
        self.QFT = qlib.QFT(self.task_spec.num_qubits)
        self.QFT_matrix = Operator(self.QFT).data
        self.pre_fidelity = 0.0
        self.player = None
        self.qc = QuantumCircuit(self.task_spec.num_qubits)

    def step(self, action:list):
        """
        Apply action to the game.
        Args:
            action : action of the action_space to take, i.e.["CX", target, control]].
        Returns:
            The new observation
        """
        is_terminal = False
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
        
        matrix_in = Operator(self.qc).data
        assert (matrix_in == matrix).all()

        d = matrix.shape[0]

        hs_inner = cirq.hilbert_schmidt_inner_product(matrix, self.QFT_matrix)
        fidelity = np.abs(hs_inner) / d
        # print("保真度：", fidelity)
        fidelity_reward = self.fidelity_reward(fidelity)
        self.pre_fidelity = fidelity

        if fidelity >= self.task_spec.fidelity_threshold:
            is_terminal = True
            fidelity_reward += 1.0 # 具体后面再说
        if self.simulator.length_qc >= self.task_spec.max_circuit_length:
            is_terminal = True
        reward = length_reward + fidelity_reward

        return self.render(), reward, fidelity_reward, length_reward, is_terminal
    
    def render(self):
        return {
            'matrix': Operator(self.simulator.qc).data,
            'circuit_length': self.simulator.length_qc,
            'circuit': self.circuit,
        }
    
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

    def length_reward(self):
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
        self.simulator = self.CirSysSimulator(self.task_spec)
        self.circuit = {i: [] for i in range(self.task_spec.num_qubits)}
        self.pre_fidelity = 0.0
        self.qc = QuantumCircuit(self.task_spec.num_qubits)
        return self.render()

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






  
# class CircuitSys:
#     def __init__(self, 
#                        n_qubit: int,
#                        fidelity_threshold: float = 0.99,
#                        max_steps: int = 20,
#                        num_testSet: int = 20, # equal to max_steps
#                        seed: int = 0,

#                        correct_reward = 10.0, # additional reward for all correct
#                        correctness_reward_weight = 1.0, # float = 1.0,
#                        alpha = 1.0,# length reward coefficient
#                        length_threshold = -1, # synthesized circuit length threshold
#                        gateSet = [["H", 0, 0], ["S", 0, 0], ["T", 0, 0], ["S†", 0, 0], ["T†", 0, 0], ["CX", 0, 1], 
#                                   ["H", 1, 1], ["S", 1, 1], ["T", 1, 1], ["S†", 1, 1], ["T†", 1, 1], ["CX", 1, 0]], #Clifford + T gate: [gate, control_bit, target_bit]
#                        render_model = None, # Visual mode
#                        target = qlib.QFT #QFT
#                  ):
        
#         super().__init__()
#         self.n_qubit = n_qubit
#         self.fidelity_threshold = fidelity_threshold
#         self.max_steps = max_steps
#         self.num_testSet = num_testSet ## number of test set
#         self.seed = seed

#         self.correct_reward = correct_reward # additional reward for all correct
#         self.correctness_reward_weight = correctness_reward_weight
#         self.alpha = alpha # length reward coefficient
#         self.length_threshold = length_threshold
#         self.gateSet = gateSet
#         self.render_model = render_model
#         self.target = target

#         self.testSet = [] ## consist of random states as initial states 
#         self.qc_inis = [] ## quantum circuit initialize in random initial state
#         self.ideal_OutputState = [] ## ideal output state of QFT acting on the initial states
#         self.qc = QuantumCircuit(self.n_qubit)
#         self.actions = [] # index of gateSet
#         self.operations = [] # specific gate i.e. ["H", 0]
#         self.states = [] # consist of operations i.e. [["H", 0, 0], ["CX", 1, 0],...]
#         self.correstness_rewards = []
#         self.length_rewards = []
#         self.previous_correct_items = 0

#         ## preparing the testSet
#         if self.num_testSet:
#             self.testSet_prepare()
        
        

#     def testSet_prepare(self):
#         ## according target prepare testset and label
#         ## 1. random initial states as testset
#         ## 2. using ideal QFT in qiskit acting on the initial states as label 
#         for i in range(self.num_testSet):
#             ## random initial state preparation
#             #  
#             random_sv = random_statevector(2**self.n_qubit, self.seed)
#             self.testSet.append(random_sv.data)
#             ## quantum circuit initialize in random initial state
#             #
#             qc = QuantumCircuit(self.n_qubit)
#             list_qubits = list(range(self.n_qubit))
#             qc.initialize(random_sv.data, list_qubits)
#             qc_ini = copy.deepcopy(qc)
#             self.qc_inis.append(qc_ini)
#             ## ideal output state preparation as label
#             #
#             qft = qlib.QFT(self.n_qubit).to_gate()
#             qc.append(qft, list_qubits)
#             state = Statevector.from_instruction(qc)
#             self.ideal_OutputState.append(state)

#             ## noisy model in furture
#             #
#             # qc = transpile(qc, Aer)
#             # state = Aer.get_backend('statevector_simulator').run(qc, shots=1).result().get_statevector()

#     @property    
#     def reward(self) -> float:
#         # correctness_reward + alpha(length_reward)
#         is_target, correctness_reward = self.correctness_reward
#         reward = correctness_reward + self.alpha * self.length_reward
#         return reward, is_target

#     @property
#     def correctness_reward(self) -> float:
#         ## according to the test set performance to calculate the reward
#         #
#         labels =  self.ideal_OutputState
#         correct_items = 0
#         for i in range(len(self.qc_inis)):
#             qc_current = self.qc_inis[i].compose(self.qc)
#             state = Statevector.from_instruction(qc_current)
#             fidelity = state_fidelity(state, labels[i]) 
#             if fidelity > self.fidelity_threshold:
#                 correct_items += 1
#         reward = self.correctness_reward_weight * (
#         correct_items - self.previous_correct_items)
#         self.previous_correct_items = correct_items
#         ## additional reward for all correct
#         #
#         all_correct = True if correct_items == self.num_testSet else False
#         reward += self.correct_reward * all_correct
#         return reward, all_correct
    
#     @property
#     def length_reward(self) -> float:
#         ## if the length of the circuit exceed length thresold, give negative reward
#         #
#         if len(self.actions) > self.length_threshold:
#             return -1.0
#         return 0.0
        
#     def legal_actions(self):

#         previous_action = self.actions[-1]
#         actions_leagal =  self.gateSet.copy()

#         if previous_action == 0 or previous_action == 2 or previous_action == 4 or previous_action == 5 or previous_action == 6 or previous_action == 8 or previous_action == 10 or previous_action == 11:
#         ## repeat gate is not legal
#         # H, T, T†, CX
#             actions_leagal.remove(previous_action)   

#         if previous_action == 1 or previous_action == 7:
#         ## some ilegal gate for S gate
#         #  S†, T†
#             ilegal_S_gd = previous_action + 2 
#             ilegal_T_gd = previous_action + 3
#             actions_leagal.remove(ilegal_S_gd)
#             actions_leagal.remove(ilegal_T_gd)

#         if previous_action == 2 or previous_action == 8:
#         ## some ilegal gate for T gate
#         #  S†, T†
#             ilegal_S_gd = previous_action + 1 
#             ilegal_T_gd = previous_action + 2
#             actions_leagal.remove(ilegal_S_gd)
#             actions_leagal.remove(ilegal_T_gd)
#         if previous_action == 3 or previous_action == 9:
#         ## some ilegal gate for S† gate
#         #  S, T
#             ilegal_S = previous_action - 2 
#             ilegal_T = previous_action -1
#             actions_leagal.remove(ilegal_S)
#             actions_leagal.remove(ilegal_T)
        
#         if previous_action == 4 or previous_action == 10:
#         ## some ilegal gate for T† gate
#         #  S, T
#             ilegal_S = previous_action - 3 
#             ilegal_T = previous_action - 2
#             actions_leagal.remove(ilegal_S)
#             actions_leagal.remove(ilegal_T)
#         return actions_leagal

#     def step(self, action):
#         ## store actions
#         #
#         self.actions.append(action)
#         ## execute action
#         #
#         operation = self.gateSet[action] 
#         gate, qubits = operation[0], operation[1] 
#         if gate == "H": self.qc.h(qubits)
#         if gate == "S": self.qc.s(qubits)
#         if gate == "T": self.qc.t(qubits)
#         if gate == "S†": self.qc.sdg(qubits)  
#         if gate == "T†": self.qc.tdg(qubits) 
#         if gate == "CX": self.qc.cx(qubits[0], qubits[1])
#         ## update state, state consist of one-hot of actions
#         # 

#         self.state.append(operation) ## state i.e.: [["H",1, 0], ["CX", 1, 0],...] 
#         ## calculate reward
#         # 
#         corrcectness_reward, is_target = self.correctness_reward
#         length_reward = self.length_reward
#         self.correstness_rewards.append(corrcectness_reward)
#         self.length_rewards.append(length_reward)
        
#         ## check if the game is over
#         #
#         if len(self.actions) >= self.max_steps or is_target:
#             done = self.close()


#         return is_target, self.state, corrcectness_reward, length_reward, done

#     def to_play(self):
#         ## always 1 player
#         return 0
    
#     def reset(self):
#         ## reset the game
#         self.testSet = [] ## consist of random states as initial states 
#         self.qc = QuantumCircuit(self.n_qubit)
#         self.actions = [] # index of gateSet
#         self.operations = [] # specific gate i.e. ["H", 0]
#         self.states = [] # consist of operations i.e. [["H", 0, 0], ["CX", 1, 0],...]
#         self.correstness_rewards = []
#         self.length_rewards = []
#         self.previous_correct_items = 0

#         return self.states[0]
        

#     def render(self, render = False):## render the boolen value
#         ## show circuit or not
#         #
#         if self.render_mode is False and render is False : return None
#         return self.qc.draw()

#     def close(self):
#         ## game over
#         # collect the data to replay buffer ???


#         return True
