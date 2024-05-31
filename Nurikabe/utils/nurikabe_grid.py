from .grid import SmartGrid
import random
import numpy as np


class NurikabeGrid:

    def __init__(self, grid, solution):
        if isinstance(grid, np.ndarray):
            grid = SmartGrid.from_grid(grid.copy(), solution.copy())
        self.grid = grid
        self.solution = solution
    
    def __str__(self):
        print('-' * 80)
        print('  grid:')
        print(self.grid)
        print('-' * 80)

    def find_children(self):
        if self.is_terminal():
            return set()
        else:
            possible_moves = []
            pos = self.grid.possibilities
            min_nb_pos_ind = min([len(v) for v in pos.values()])
            # print('min_nb_pos_ind:', min_nb_pos_ind)
            # breakpoint()
            for index in pos:
                # just look at indeces with min pos
                if len(pos[index]) == min_nb_pos_ind:
                    for value in pos[index]:
                        possible_moves.append((index, value))
                    # we just return children on 1 cell --> sufficient
                    return {self.take_action(a[0], a[1])
                            for a in possible_moves}

    def find_random_child(self):
        pos = self.grid.possibilities
        # breakpoint()
        min_nb_pos_ind = min([len(v) for v in pos.values()])
        if len(pos) == 0 or min_nb_pos_ind == 0:
            return None
        pos_considered = []
        for k, v in pos.items():
            # just look at indeces with min pos
            if len(v) == min_nb_pos_ind:
                pos_considered.append(k)
        index = random.choice(pos_considered)
        action = random.choice(self.grid.possibilities[index])
        child = self.take_action(index, action)
        return child

    def take_action(self, index, action):
        new_grid = NurikabeGrid(self.grid.grid.copy(), self.solution.copy())
        new_grid.grid.fill_cell(*index, action)
        return new_grid

    def is_terminal(self):
        if len(self.grid.possibilities) == 0:
            return True
        elif self.grid.is_complete() or not self.grid.is_correct(self.solution) or \
                min([len(v) for v in self.grid.possibilities.values()]) == 0:
            return True
        return False

    def reward(self):
        if (self.grid.grid == self.solution).all():
            return 100
        return np.sum(self.grid.grid == self.solution)
        # return np.count_nonzero(self.grid.grid) / 81

    def __hash__(self):
        return hash(str(self.grid.grid))

    def __eq__(self, grid2):
        if np.array_equal(self.grid.grid, grid2.grid.grid):
            return True
        return False

    def __str__(self):
        return str(self.grid.grid)
