import sys

sys.path.append("C:\projects\AlphaCir\AlphaCir_dos")


import pytest 
from AlphaCir.games.CirsysGame import Game, TaskSpec, action2index
from qiskit.quantum_info import Operator
import qiskit.circuit.library as qlib
from qiskit import QuantumCircuit
from AlphaCir.games.palette import valid_data


@pytest.fixture
def game_instance():
    # 构造一个 TaskSpec 实例（参数值可根据需要调整）
    task_spec = TaskSpec.create(
        max_circuit_length=50,
        num_qubits=4,
        num_ops=6,  # 假设有6种门：H, S, T, S†, T†, CX
        correct_reward=1.0,
        correctness_reward_weight=1.0,
        length_reward= -0.1,
        length_reward_weight=0.1,
        fidelity_threshold=0.9
    )
    game = Game(task_spec)
    return game



test_data = valid_data(num_qubits=4, num_examples=10)

@pytest.mark.parametrize(["circuit", "fidelity"], test_data)
def test_implement_game(game_instance, circuit, fidelity, num_qubits=4, num_gates=32):
    for gate in circuit:
        gate_index = action2index(gate, num_qubits, num_gates)  # 将门操作转换为索引
        if gate_index not in game_instance.legal_actions():
            assert False, f"Gate {gate} {gate_index} is not a legal action"
        else:
            game_instance.step(gate)
    
    assert game_instance.pre_fidelity == fidelity, f"Expected fidelity {fidelity}, but got {game_instance.pre_fidelity}"