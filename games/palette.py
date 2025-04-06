from qiskit.quantum_info import Operator
import qiskit.circuit.library as qlib
from qiskit import QuantumCircuit
import cirq
import numpy as np
from games.CirsysGame import index2action # for local test
# from CirsysGame  import index2action
import random
import time
# QFT = qlib.QFT(2)
# matrix = Operator(QFT).data
# for _ in range(5):   
#     random_number1 = random.randrange(10) 
#     gate_list = [] # 从 0 到 11
#     for a in range(random_number1):
        
#         random_number = random.randrange(12)  # 从 0 到 11
#         gate =index2action(random_number, 2, 12)  # 将数字转换为门操作
#         gate_list.append(gate)
    
#     time.sleep(1)
#     print("门操作：", gate_list)
#     qc = QuantumCircuit(2)
#     for gate in gate_list:
#         if gate[0] == "H":
#             qc.h(gate[1])
#         elif gate[0] == "S":
#             qc.s(gate[1])
#         elif gate[0] == "T":
#             qc.t(gate[1])
#         elif gate[0] == "S†":
#             qc.sdg(gate[1])
#         elif gate[0] == "T†":
#             qc.tdg(gate[1])
#         elif gate[0] == "CX":
#             qc.cx(gate[2], gate[1])

#     matrix_q = Operator(qc).data
#     d = matrix.shape[0]
#     print("矩阵的维度：", d)
#     hs_inner = cirq.hilbert_schmidt_inner_product(matrix, matrix_q)
#     fidelity = np.abs(hs_inner) / d
#     print("保真度：", fidelity)

def valid_data(num_qubits, num_examples):
    QFT = qlib.QFT(num_qubits)
    matrix = Operator(QFT).data
    num_gates = num_qubits * (num_qubits - 1 + 5 )  
    data = []  # 用于存储生成的门操作和保真度
    for _ in range(num_examples):   
        random_number1 = random.randrange(1, 20) 
        gate_list = [] # 从 0 到 11
        for a in range(random_number1):
            

            random_number = random.randrange(num_gates)  
            gate =index2action(random_number, num_qubits, num_gates)  # 将数字转换为门操作
            gate_list.append(gate)
    
        qc = QuantumCircuit(num_qubits)
        for gate in gate_list:
            if gate[0] == "H":
                qc.h(gate[1])
            elif gate[0] == "S":
                qc.s(gate[1])
            elif gate[0] == "T":
                qc.t(gate[1])
            elif gate[0] == "S†":
                qc.sdg(gate[1])
            elif gate[0] == "T†":
                qc.tdg(gate[1])
            elif gate[0] == "CX":
                qc.cx(gate[2], gate[1])

        matrix_q = Operator(qc).data
        d = matrix.shape[0]

        hs_inner = cirq.hilbert_schmidt_inner_product(matrix, matrix_q)
        fidelity = np.abs(hs_inner) / d

        item = (gate_list, fidelity)
        print(item[0])
        print(item[1])
        data.append(item)
    return data

if __name__ == "__main__":
    # num_qubits = 2
    # num_examples = 10
    # data = valid_data(num_qubits, num_examples)
    # for item in data:
    #     print(item[0])
    #     print(item[1])
    
    # [['S', 0, 0], ['H', 1, 1], ['S†', 1, 1], ['T†', 1, 1], 
    # ['CX', 1, 0], ['T†', 1, 1], ['S', 0, 0], ['CX', 0, 1]]
# 0.3535533905932736 0.24999999999999992
        QFT = qlib.QFT(2)
        matrix = Operator(QFT).data
        qc = QuantumCircuit(2)
        qc.s(0)
        qc.h(1)     
        qc.sdg(1)
        qc.tdg(1)   
        qc.cx(0, 1)
        # qc.tdg(1)
        # qc.s(0)
        # qc.cx(1, 0)
        
        matrix_q = Operator(qc).data

        a=[[ 0.70710678+0.j,          0.        +0.j,          0.70710678+0.j,
   0.        +0.j        ],
 [ 0.        +0.j,          0.        +0.70710678j,  0.        +0.j,
   0.        +0.70710678j],
 [ 0.        +0.j,          0.        -0.70710678j,  0.        +0.j,
   0.        +0.70710678j],
 [-0.70710678+0.j,          0.        +0.j,          0.70710678+0.j,
   0.        +0.j        ]]
    
        print("q矩阵：", matrix_q)
        if a.all() == matrix_q.all():
            print("相等")
        d = matrix.shape[0]

        hs_inner = cirq.hilbert_schmidt_inner_product(matrix, matrix_q)
        fidelity = np.abs(hs_inner) / d
        print("保真度：", fidelity)
    # [['CX', 0, 1], ['T†', 0, 0], ['S†', 0, 0]]
# 0.17677669529663684  0.4619397662556433

# [[ 0.70710678+0.j          0.        +0.j          0.70710678+0.j
#    0.        +0.j        ]
#  [ 0.        +0.j          0.        +0.70710678j  0.        +0.j
#    0.        +0.70710678j]
#  [ 0.        +0.j          0.5       -0.5j         0.        +0.j
#   -0.5       +0.5j       ]
#  [-0.5       -0.5j         0.        +0.j          0.5       +0.5j
#    0.        +0.j        ]]

[['S†', 1, 1], ['H', 1, 1], ['T†', 0, 0], ['T', 1, 1], ['S†', 1, 1], ['S†', 1, 1], ['H', 0, 0]]