import gym
from gym import spaces
import numpy as np
from enum import Enum

class Cell(Enum):
    EMPTY = -2
    SEA = -1
    ISLAND = 0

class NurikabeEnv(gym.Env):
    # do we need more metadata fields? I guess so only if we are using a CNN
    metadata = {'render_modes': ['human']}

    def __init__(self, all_solution_path, full_width=25, full_height=25):
        self.full_width = full_width
        self.full_height = full_height

        self.observation_space = spaces.Box(
            low=-3, high=np.inf, shape=(self.full_height, self.full_width),
            dtype=np.int32
        )

        self.action_space = spaces.Tuple(
            (spaces.Discrete(2, start=Cell.SEA),  # 0 or -1
             spaces.Discrete(self.full_width, start=0),
             spaces.Discrete(self.full_height, start=0))
        )

        self.board = None
        self.all_solution_path = all_solution_path  # can take a more lightweight approach by not saving in memory
        self._load_all_solutions()
        self.solution_board_index = 0
        self.reset(self.solution_board_index)


    def _load_all_solutions(self):
        self.all_solutions = np.load(self.all_solution_path)
        
    
    def step(self, action):
        
        _action, x, y = action

        if not self.action_space.contains(action):
            print(f'{action} is not a valid action!')
            observation = np.copy(self.board)
            reward = -np.inf
            terminated = False
            out_of_bound = True
            done = False
            info = {}
            
            return observation, reward, terminated, out_of_bound, done, info

        done = False

        if self.solution_board[x, y] == _action:
            self.board_copy = np.copy(self.board)
            self.board_copy[x, y] = _action
            if (self.board_copy == self.solution_board).all():
                done = True
                observation = np.copy(self.board_copy)
                reward = 100
                terminated = True
                out_of_bound = False
                info = {}
                return observation, reward, terminated, out_of_bound, done, info
            else:
                done = False
                observation = np.copy(self.board_copy)
                reward = 1
                terminated = False
                out_of_bound = False
                info = {}
                return observation, reward, terminated, out_of_bound, done, info
        else:
            done = False
            observation = np.copy(self.board)
            reward = -np.inf if self.solution_board[x, y] >= 1 else -1
            terminated = False
            out_of_bound = False
            info = {}
            return observation, reward, terminated, out_of_bound, done, info
            

    def reset(self, solution_board_index):
        self.original_solution_board = self.all_solutions[self.all_solutions.files[solution_board_index]]
        self.solution_board_index += 1
        (self.width, self.height) = self.original_solution_board.shape
        
        self.solution_board = np.full((self.full_width, self.full_height), -3)
        self.solution_board[:self.width, :self.height] = self.original_solution_board
      
        observation = np.copy(self.solution_board)
        info = {}
        return observation, info
    

    def _compute_reward(self):
        pass

    def _record_info(self):
        pass
          
    def close():
        pass

    def render():
        pass
