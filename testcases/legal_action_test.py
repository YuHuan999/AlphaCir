import pytest
import sys

sys.path.append("C:\projects\AlphaCir\AlphaCir_dos")
from games.CirsysGame import Game, TaskSpec, action2index, index2action


# @pytest.fixture
# def game_instance():
#     # 构造一个 TaskSpec 实例（参数值可根据需要调整）
#     task_spec = TaskSpec.create(
#         max_circuit_length=5,
#         num_qubits=2,
#         num_ops=6,  # 假设有6种门：H, S, T, S†, T†, CX
#         correct_reward=1.0,
#         correctness_reward_weight=1.0,
#         length_reward_weight=0.1,
#         fidelity_threshold=0.9
#     )
#     game = Game(task_spec)
#     return game



# @pytest.mark.ship(reason="测试过了，暂时不需要")
# def test_legal_actions_no_gate(game_instance):
#     """
#     当电路为空时，legal_actions 应返回初始动作空间，即所有动作。
#     """
#     expected_initial = list(range(game_instance.task_spec.num_actions))
#     legal = game_instance.legal_actions()
#     assert sorted(legal) == expected_initial


# @pytest.mark.parametrize(["q1", "q2", "expects"], 
#                          [(["H", 0, 0], ["S", 1, 1], [ 1, 2, 3, 4, 5, 6, 7, 8, 11]),
#                          (["H", 0, 0], ["T†", 1, 1], [ 1, 2, 3, 4, 5, 6, 9, 11]),
#                          (["T", 0, 0], ["CX", 1, 0], [ 0, 1, 5, 6, 7, 8, 9, 10]),
#                          (["T†", 0, 0], ["S", 1, 1], [ 0, 3, 5, 6, 7, 8, 11]),
#                         (["CX", 0, 1], ["H", 1, 1], [ 0, 1, 2, 3, 4, 7, 8, 9, 10, 11]),
#                         (["S", 0, 0], ["T", 1, 1], [ 0, 1, 2, 5, 6, 7, 11]),
#                         (["S†", 0, 0], None, [ 0, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
                                                   
                                                   
                                                   
                                                   
                                                   
                                                   
#                                                    ])
# @pytest.mark.ship(reason="测试过了，暂时不需要")
# def test_legal_actions_with_gates(game_instance, q1, q2, expects):
    
    
#     game_instance.circuit[0].append(q1)
   
    
#     game_instance.circuit[1].append(q2)
    
#     # 计算预期合法动作
#     legal = game_instance.legal_actions()
#     assert legal == expects, f"Expected {expects}, but got {legal}"


@pytest.fixture
def game_instance():
    # 构造一个 TaskSpec 实例（参数值可根据需要调整）
    task_spec = TaskSpec.create(
        max_circuit_length=21,
        num_qubits=3,
        num_ops=6,  # 假设有6种门：H, S, T, S†, T†, CX
        correct_reward=1.0,
        correctness_reward_weight=1.0,
        length_reward_weight=0.1,
        fidelity_threshold=0.9
    )
    game = Game(task_spec)
    return game

@pytest.mark.parametrize(["q1", "q2", "q3", "expects"], 
                         [(["H", 0, 0], ["S", 1, 1], ["T", 2, 2], [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 19, 20]),
                         (["CX", 0, 2], ["T†", 1, 1], ["CX", 2, 1], [0, 1, 2, 3, 4, 5, 7, 10, 12, 13, 14, 15, 16, 17, 18, 19]),
                         (["T", 0, 0], ["CX", 1, 0], ["S", 2, 2], [0, 1, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 19, 20]),
                         (["T†", 0, 0], ["S", 1, 1], ["T", 2, 2], [0, 3, 5, 6, 7, 8, 9, 12, 13, 14, 15, 19, 20]),
                        (["CX", 0, 1], ["H", 1, 1], ["S†", 2, 2], [0, 1, 2, 3, 4, 6, 8, 9, 10, 11, 12, 13, 14, 17, 18, 19, 20]),
                        (["S", 0, 0], ["T", 1, 1], ["H", 2, 2],   [0, 1, 2, 5, 6, 7, 8, 12, 13, 15, 16, 17, 18, 19, 20]),
                        (["S†", 0, 0], ["CX", 1, 2], ["H", 2, 2], [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19, 20]),
                                                                          
                                                   
                                                   ])
# @pytest.mark.ship(reason="测试过了，暂时不需要")
def test_legal_actions_with_gates(game_instance, q1, q2, q3,expects):
    
    
    game_instance.circuit[0].append(q1)
    game_instance.circuit[1].append(q2)
    game_instance.circuit[2].append(q3)
    
    # 计算预期合法动作
    legal = game_instance.legal_actions()
    assert legal == expects, f"Expected {expects}, but got {legal}"


@pytest.fixture
def game_instance():
    # 构造一个 TaskSpec 实例（参数值可根据需要调整）
    task_spec = TaskSpec.create(
        max_circuit_length=32,
        num_qubits=4,
        num_ops=6,  # 假设有6种门：H, S, T, S†, T†, CX
        correct_reward=1.0,
        correctness_reward_weight=1.0,
        length_reward_weight=0.1,
        fidelity_threshold=0.9
    )
    game = Game(task_spec)
    return game

@pytest.mark.parametrize(["q1", "q2", "q3", "q4", "expects"], 
                         [(["H", 0, 0], ["S", 1, 1], ["T", 2, 2], ["T†", 3, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 17, 
                                                                                 21, 22, 23, 24, 27, 29, 30, 31]),
                         (["CX", 0, 3], ["T†", 1, 1], ["CX", 2, 3], ["CX", 3, 0], [0, 1, 2, 3, 4, 5, 6, 8, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                                                                                   24, 25, 26, 27, 28, 30, 31]),
                         (["T", 0, 0], ["CX", 1, 3], ["S", 2, 2], ["CX", 3, 2], [0, 1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 21, 22, 23, 
                                                                                 24, 25, 26, 27, 28, 29, 30]),
                         (["T†", 0, 0], ["S", 1, 1], ["T", 2, 2], ["H", 3, 3], [0, 3,  5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 17,21, 22, 
                                                                                23, 25, 26, 27, 28, 29, 30, 31]),
                        (["CX", 0, 2], ["H", 1, 1], ["S†", 2, 2], ["S†", 3, 3], [0, 1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 13, 14, 15, 16, 19, 20, 21, 22, 23, 
                                                                                 24, 27, 28, 29, 30, 31]),
                        (["S", 0, 0], ["T", 1, 1], ["H", 2, 2],   ["S", 3, 3], [0, 1, 2,  5, 6, 7, 8, 9, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 
                                                                                24, 25, 26, 29, 30, 31]),
                        (["S†", 0, 0], ["CX", 1, 0], ["H", 2, 2], ["T", 3, 3], [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,14, 15, 17, 18, 19, 20, 21, 22, 23, 
                                                                                24, 25, 29, 30, 31]),
                                                                          
                                                   
                                                   ])
# @pytest.mark.ship(reason="测试过了，暂时不需要")
def test_legal_actions_with_gates(game_instance, q1, q2, q3, q4, expects):
    
    
    game_instance.circuit[0].append(q1)
    game_instance.circuit[1].append(q2)
    game_instance.circuit[2].append(q3)
    game_instance.circuit[3].append(q4)
    
    # 计算预期合法动作
    legal = game_instance.legal_actions()
    assert legal == expects, f"Expected {expects}, but got {legal}"