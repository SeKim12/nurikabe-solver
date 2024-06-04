import numpy as np
import random
from typing import List
import numpy as np

import gym
from gym import spaces

from dt import constants

class NurikabeEnv:
    def __init__(self, h, w, solution_files):
        self.h, self.w = h, w
        self.solution_files = solution_files
        self.board_index = 0

    def reset(self):    
        self.solution = np.load(self.solution_files[self.board_index])

        self.board = np.where(
            (self.solution == 0) | (self.solution == -1),
            -2, self.solution
        )

        self.board_index += 1

        return np.copy(self.board), self.get_optimal_rtg()
    
    def step(self, action, position):
        i, j = position // self.h, position % self.w
        current_val = self.board[i, j]

        if current_val >= 1:  # Already-numbered cell
            reward = constants.NUM_CELL_RWD  # Heavy penalty
            done = False
            return np.copy(self.board), reward, done
        
        prev_score = np.count_nonzero(self.solution == self.board)
        self.board[i, j] = action 
        cur_score = np.count_nonzero(self.solution == self.board) 

        return np.copy(self.board), cur_score - prev_score, cur_score == self.h * self.w
    
    def get_optimal_rtg(self):
        return np.count_nonzero(self.board <= 0)