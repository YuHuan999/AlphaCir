import gymnasium as gym
import qiskit.circuit.library as qlib
from qiskit.quantum_info import Statevector, random_statevector, state_fidelity
import datetime
import pathlib
import torch

from qiskit import QuantumCircuit, transpile
import numpy as np
from qiskit_aer import Aer
import copy
from abstract_game import AbstractGame 
class MuZeroConfig:
    def __init__(self):
        # fmt: off
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = 0  # Seed for numpy, torch and the game
        self.max_num_gpus = None  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available



        ### Game
        self.observation_shape = (1, 1, 4)  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = list(range(2))  # Fixed list of all possible actions. You should only edit the length
        self.players = list(range(1))  # List of players. You should only edit the length
        self.stacked_observations = 0  # Number of previous observations and previous actions to add to the current observation

        # Evaluate
        self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = None  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class



        ### Self-Play
        self.num_workers = 1  # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.selfplay_on_gpu = False
        self.max_moves = 500  # Maximum number of moves if game is not finished before
        self.num_simulations = 50  # Number of future moves self-simulated
        self.discount = 0.997  # Chronological discount of the reward
        self.temperature_threshold = None  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25



        ### Network
        self.network = "fullyconnected"  # "resnet" / "fullyconnected"
        self.support_size = 10  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))
        
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
        self.training_steps = 10000  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 128  # Number of parts of games to train on at each training step
        self.checkpoint_interval = 10  # Number of training steps before using the model for self-playing
        self.value_loss_weight = 1  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.train_on_gpu = torch.cuda.is_available()  # Train on GPU if available

        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
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

        # Reanalyze (See paper appendix Reanalyse)
        self.use_last_model_value = True  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
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
        
class CircuitSys(AbstractGame):
    def __init__(self, 
                       n_qubit: int,
                       fidelity_threshold: float = 0.99,
                       max_steps: int = 10,
                       num_testSet: int = 20,
                       seed: int = 0,

                       correct_reward = 1.0, # additional reward for all correct
                       correctness_reward_weight = 1.0, # float = 1.0,
                       alpha: float = 0.1, # length reward coefficient
                       length_threshold: int = 10, # synthesized circuit length threshold
                       gateSet = [["H", 0], ["S", 0], ["T", 0], ["S†", 0], ["T†", 0], ["CX", [0, 1]], 
                                  ["H", 1], ["S", 1], ["T", 1], ["S†", 1], ["T†", 1], ["CX", [1, 0] ]], #Clifford + T gate
                       render_model = None, # Visual mode
                       target = qlib.QFT #QFT
                 ):
        
        super().__init__()
        self.n_qubit = n_qubit
        self.fidelity_threshold = fidelity_threshold
        self.max_steps = max_steps
        self.num_testSet = num_testSet ## number of test set
        self.seed = seed

        self.correct_reward = correct_reward
        self.correctness_reward_weight = correctness_reward_weight
        self.alpha = alpha # length reward coefficient
        self.length_threshold = length_threshold
        self.gateSet = gateSet
        self.render_model = render_model
        self.target = target

        self.testSet = [] ## consist of random states as initial states 
        self.qc_inis = [] ## quantum circuit initialize in random initial state
        self.ideal_OutputState = [] ## ideal output state of QFT acting on the initial states
        self.qc = QuantumCircuit(self.n_qubit)
        self.actions = [] # index of gateSet
        self.state = [] # consist of one-hot of actions
        self.previous_correct_items = 0

        ## preparing the testSet
        if self.num_testSet:
            self.testSet_prepare()
        
        

    def testSet_prepare(self):
        ## according target prepare testset and label
        ## 1. random initial states as testset
        ## 2. using ideal QFT in qiskit acting on the initial states as label 
        for i in range(self.num_testSet):
            ## random initial state preparation
            #  
            random_sv = random_statevector(2**self.n_qubit, self.seed)
            self.testSet.append(random_sv.data)
            ## quantum circuit initialize in random initial state
            #
            qc = QuantumCircuit(self.n_qubit)
            list_qubits = list(range(self.n_qubit))
            qc.initialize(random_sv.data, list_qubits)
            qc_ini = copy.deepcopy(qc)
            self.qc_inis.append(qc_ini)
            ## ideal output state preparation as label
            #
            qft = qlib.QFT(self.n_qubit).to_gate()
            qc.append(qft, list_qubits)
            state = Statevector.from_instruction(qc)
            self.ideal_OutputState.append(state)

            ## noisy model in furture
            #
            # qc = transpile(qc, Aer)
            # state = Aer.get_backend('statevector_simulator').run(qc, shots=1).result().get_statevector()

    @property    
    def reward(self) -> float:
        # correctness_reward + alpha(length_reward)
        is_target, correctness_reward = self.correctness_reward
        reward = correctness_reward + self.alpha * self.length_reward
        return reward, is_target

    @property
    def correctness_reward(self) -> float:
        ## according to the test set performance to calculate the reward
        #
        labels =  self.ideal_OutputState
        correct_items = 0
        for i in range(len(self.qc_inis)):
            qc_current = self.qc_inis[i].compose(self.qc)
            state = Statevector.from_instruction(qc_current)
            fidelity = state_fidelity(state, labels[i]) 
            if fidelity > self.fidelity_threshold:
                correct_items += 1
        reward = self.correctness_reward_weight * (
        correct_items - self.previous_correct_items)
        self.previous_correct_items = correct_items
        ## additional reward for all correct
        #
        all_correct = True if correct_items == self.num_testSet else False
        reward += self.correct_reward * all_correct
        return reward, all_correct
    
    @property
    def length_reward(self) -> float:
        ## if the length of the circuit exceed length thresold, give negative reward
        #
        if len(self.actions) > self.length_threshold:
            return -1.0
        return 0.0
        

    def action_to_one_hot(self, action):
        one_hot = np.zeros(len(self.gateSet))
        if 0 <= action < len(self.gateSet):
            one_hot[action] = 1
        return one_hot

    def legal_actions(self):

        previous_action = self.actions[-1]
        actions_leagal =  self.gateSet.copy()

        if previous_action == 0 or previous_action == 2 or previous_action == 4 or previous_action == 5 or previous_action == 6 or previous_action == 8 or previous_action == 10 or previous_action == 11:
        ## repeat gate is not legal
        # H, T, T†, CX
            actions_leagal.remove(previous_action)   

        if previous_action == 1 or previous_action == 7:
        ## some ilegal gate for S gate
        #  S†, T†
            ilegal_S_gd = previous_action + 2 
            ilegal_T_gd = previous_action + 3
            actions_leagal.remove(ilegal_S_gd)
            actions_leagal.remove(ilegal_T_gd)

        if previous_action == 2 or previous_action == 8:
        ## some ilegal gate for T gate
        #  S†, T†
            ilegal_S_gd = previous_action + 1 
            ilegal_T_gd = previous_action + 2
            actions_leagal.remove(ilegal_S_gd)
            actions_leagal.remove(ilegal_T_gd)
        if previous_action == 3 or previous_action == 9:
        ## some ilegal gate for S† gate
        #  S, T
            ilegal_S = previous_action - 2 
            ilegal_T = previous_action -1
            actions_leagal.remove(ilegal_S)
            actions_leagal.remove(ilegal_T)
        
        if previous_action == 4 or previous_action == 10:
        ## some ilegal gate for T† gate
        #  S, T
            ilegal_S = previous_action - 3 
            ilegal_T = previous_action - 2
            actions_leagal.remove(ilegal_S)
            actions_leagal.remove(ilegal_T)
        return actions_leagal

    def step(self, action):
        ## store actions
        #
        self.actions.append(action)
        ## execute action
        #
        operation = self.gateSet[action] 
        gate, qubits = operation[0], operation[1] 
        if gate == "H": self.qc.h(qubits)
        if gate == "S": self.qc.s(qubits)
        if gate == "T": self.qc.t(qubits)
        if gate == "S†": self.qc.sdg(qubits)  
        if gate == "T†": self.qc.tdg(qubits) 
        if gate == "CX": self.qc.cx(qubits[0], qubits[1])
        ## update state, state consist of one-hot of actions
        # 
        self.state.append(self.action_to_one_hot(action))
        ## calculate reward
        # 
        reward, is_target = self.reward
        ## check if the game is over
        #
        if len(self.actions) >= self.max_steps or is_target:
            done = self.close()


        return is_target, self.state, reward, done

    def to_play(self):
        ## always 1 player
        return 0
    
    def reset(self):
        ## Call __init__ to reset all arguments and status
        #
        self.__init__(self.n_qubit, 
                      self.fidelity_threshold,         
                      self.max_steps,
                      self.num_testSet,
                      self.seed,
                      self.correct_reward,
                      self.correctness_reward_weight,
                      self.alpha,
                      self.length_threshold,
                      self.gateSet,
                      self.render_model,
                      self.target)
        ##Preparing the testSet, seed should not be the same？
        #
        # self.testSet_prepare()
        

    def render(self, render = False):## render the boolen value
        ## show circuit or not
        #
        if self.render_mode is False and render is False : return None
        return self.qc.draw()

    def close(self):
        ## game over
        # collect the data to replay buffer ???


        return True
