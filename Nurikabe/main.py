from mcts import MCTS
from alpha_nurikabe import AlphaNurikabe
from tensorflow.keras.models import load_model
import time
import numpy as np

NROWS = 2
DATASET_PATH = './assets/nurikabe.npz'

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
    print('--------- MCTS ')
    start_time = time.time()
    mcts_solver = MCTS(single_board, single_solution, max_iterations=20000)
    print(mcts_solver.solve())
    print(round(time.time() - start_time, 2))
    print(mcts_solver.iterations)

    print('--------- AlphaNurikabe ')
    start_time = time.time()
    alpha_solver = AlphaNurikabe(single_board, single_solution, model=model)
    print(alpha_solver.solve())
    print(round(time.time() - start_time, 2))
    print(alpha_solver.iterations)
