# from utils import read_transform
# from backtrack import BacktrackSolver
from mcts import MCTS
# from deep_iterative_solver import DeepIterativeSolver
from alpha_nurikabe import AlphaNurikabe
from tensorflow.keras.models import load_model
import time
import numpy as np

NROWS = 2
DATASET_PATH = './assets/nurikabe.npz'
# data_X, data_Y = read_transform(NROWS=NROWS)

all_solutions = np.load(DATASET_PATH)

single_solution = all_solutions[
    all_solutions.files[0]
]
single_board = np.where(
    (single_solution == 0) | (single_solution == -1),
    -2, single_solution
)

print(single_solution)
print(single_board)

model = load_model('./alpha_nurikabe/policy_network.keras')

for i in range(1, NROWS):
    print('---------', i)
    # print('--------- Backtrack ')
    # start_time = time.time()
    # # back_solver = BacktrackSolver(data_X[i])
    # back_solver = BacktrackSolver(single_board)
    # print(back_solver.solve())
    # print(round(time.time() - start_time, 2))
    # print(back_solver.iterations)

    print('--------- MCTS ')
    start_time = time.time()
    mcts_solver = MCTS(single_board, single_solution, max_iterations=20000)
    # mcts_solver = MCTS(data_X[i], max_iterations=100000)
    print(mcts_solver.solve())
    print(round(time.time() - start_time, 2))
    print(mcts_solver.iterations)

    # print('--------- DeepIterativeSolver ')
    # start_time = time.time()
    # deep_solver = DeepIterativeSolver(data_X[i], model=model)
    # deep_solver.solve()
    # print(round(time.time() - start_time, 2))
    # print(deep_solver.iterations)

    print('--------- AlphaNurikabe ')
    start_time = time.time()
    alpha_solver = AlphaNurikabe(single_board, single_solution, model=model)
    print(alpha_solver.solve())
    print(round(time.time() - start_time, 2))
    print(alpha_solver.iterations)
