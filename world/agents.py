import random
import numpy as np
import pickle
from world import grids

class NurikabeAgent:
    def __init__(self, max_horizon):
        self.max_horizon = max_horizon
        self.combined_trajectories = []
    
    def step(grid: grids.NurikabeGrid):
        raise NotImplementedError()

    def play(self, grid: grids.NurikabeGrid):
        taus = [grid.get_current_state()]  # [s, a, r, s, ...]
        for _ in range(self.max_horizon):
            taus.extend(list(self.step(grid)))
            if grid.is_solved():
                break
        self.combined_trajectories.append(taus)
        return taus

    def save_trajectory(self, fp):
        with open(fp, 'wb') as f:
            pickle.dump(self.combined_trajectories) 


# TODO: probably need a smarter random walk agent
class RandomWalkNurikabeAgent(NurikabeAgent):
    def __init__(self, max_horizon) -> None:
        super().__init__(max_horizon)
    
    def step(self, grid: grids.NurikabeGrid):
        row = random.randint(0, grid.get_num_rows() - 1)
        col = random.randint(0, grid.get_num_cols() - 1)
        action = random.choice([-1, 0])
        reward = grid.update(row, col, action)
        return action, reward, grid.get_current_state()


# TODO: translate nurikabe.cpp
class NonMLNurikabeAgent(NurikabeAgent):
    def __init__(self, max_horizon) -> None:
        super().__init__(max_horizon)

    def step(self, grid: grids.NurikabeGrid):
        action = -1
        reward = grid.update(0, 0, -1)
        return action, reward, grid.get_current_state()