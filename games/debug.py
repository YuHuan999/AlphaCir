from CirsysGame import Game, TaskSpec, action2index, index2action 
    # 构造一个 TaskSpec 实例（参数值可根据需要调整）
task_spec = TaskSpec.create(
        max_circuit_length=50,
        num_qubits=3,
        num_ops=6,  # 假设有6种门：H, S, T, S†, T†, CX
        correct_reward=1.0,
        correctness_reward_weight=1.0,
        length_reward=-0.1,
        length_reward_weight=0.1,
        fidelity_threshold=0.9
    )

game = Game(task_spec)
game.human_play()
# game.step(['S', 0, 0])
# game.step(['H', 1, 1])
# game.step(['S†', 1, 1])
# game.step(['T†', 1, 1])
# game.step(['CX', 1, 0])
# game.step(['T†', 1, 1])
# game.step(['S', 0, 0])
# game.step(['CX', 0, 1])