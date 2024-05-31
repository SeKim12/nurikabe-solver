
from utils import NurikabeGrid
import numpy as np
from math import log, sqrt

# This code is an adaptation of the code here:
# https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1
# in the case of nurikabe


class MCTS:

    def __init__(self, nurikabe_grid, solution, exploration_weight=1,
                 max_depth_tree=1000, max_iterations=20000):
        if isinstance(nurikabe_grid, np.ndarray):
            nurikabe_grid = NurikabeGrid(nurikabe_grid, solution)
        self.nurikabe_grid = nurikabe_grid
        self.solution = solution
        self.exploration_weight = exploration_weight
        self.Q = {}
        self.N = {}
        self.children = {}
        self.max_depth_tree = max_depth_tree
        self.iterations = 0
        self.max_iterations = max_iterations

    def solve(self):
        # # breakpoint()
        while not self.nurikabe_grid.is_terminal():
            for i in range(self.max_depth_tree):
                # breakpoint()
                self.do_rollout()
            self.nurikabe_grid = self.choose_best_action()
            # breakpoint()
            if self.iterations > self.max_iterations:
                print("Solver failed, you might want to increase "
                      "the number of iterations.")
                break
        return self.nurikabe_grid

    def choose_best_action(self):
        self.iterations += 1
        if self.nurikabe_grid.is_terminal():
            return self.nurikabe_grid

        if self.nurikabe_grid not in self.children:
            return self.nurikabe_grid.find_random_child()

        def score(n):
            if self.N.get(n, 0) == 0:
                return -1
            return self.Q.get(n, 0) / self.N[n]

        if len(self.children[self.nurikabe_grid]) == 0:
            if self.nurikabe_grid.find_random_child() is not None:
                return self.nurikabe_grid.find_random_child()
            else:
                return RuntimeError("Solver failed")

        return max(self.children[self.nurikabe_grid],
                   key=score)

    def do_rollout(self):
        path = self._select(self.nurikabe_grid)
        leaf = path[-1]
        self._expand(leaf)
        reward = self._simulate(leaf)
        self._backpropagate(path, reward)

    def _select(self, nurikabe_grid):
        path = []
        while True:
            path.append(nurikabe_grid)
            # breakpoint()
            if nurikabe_grid not in self.children \
                    or not self.children[nurikabe_grid]:
                self.iterations += 1
                # breakpoint()
                return path
            unexplored = self.children[nurikabe_grid] - self.children.keys()
            # breakpoint()
            if unexplored:
                child = unexplored.pop()
                path.append(child)
                self.iterations += 1
                # breakpoint()
                return path
            nurikabe_grid = self._action_selection(nurikabe_grid)
            # breakpoint()

    def _simulate(self, nurikabe_grid):
        while not nurikabe_grid.is_terminal():
            self.iterations += 1
            nurikabe_grid = nurikabe_grid.find_random_child()
        if nurikabe_grid.grid.is_complete() and nurikabe_grid.grid.is_correct(self.solution):
            self.nurikabe_grid = nurikabe_grid
        return nurikabe_grid.reward()

    def _expand(self, nurikabe_grid):
        if nurikabe_grid in self.children:
            return None
        self.iterations += 1
        self.children[nurikabe_grid] = nurikabe_grid.find_children()

    def _backpropagate(self, path, reward):
        for nurikabe_grid in reversed(path):
            self.N[nurikabe_grid] = self.N.get(nurikabe_grid, 0) + 1
            self.Q[nurikabe_grid] = self.Q.get(nurikabe_grid, 0) + reward

    def _action_selection(self, nurikabe_grid):
        # All children of node should already be expanded:
        assert all(child in self.children
                   for child in self.children[nurikabe_grid])

        log_N_parent = log(self.N[nurikabe_grid])

        def uct(child):
            "Upper confidence bound for trees"
            return self.Q[child] / self.N[child] + self.exploration_weight * \
                sqrt(log_N_parent / self.N[child])

        # breakpoint()
        return max(self.children[nurikabe_grid], key=uct)
