import gymnasium as gym
import qiskit.circuit.library as qlib
from qiskit.quantum_info import Statevector, random_statevector, state_fidelity

from qiskit import QuantumCircuit, transpile
import numpy as np
from qiskit_aer import Aer
import copy
class CircuitSys(gym.Env):
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

    def filter_legal_actions(self):
        pass

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
        done = True if len(self.actions) >= self.max_steps else False
        return is_target, self.state, reward, done


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
        self.testSet_prepare()
        

    def render(self):
        if self.render_mode is None: return None
        return self.qc.draw(self.render_mode)

    def close(self):
        pass
