import numpy as np
import random

import gym
from gym import spaces

class NurikabeEnv:
    def __init__(self, n, m, lazy=False, dataset_path=None):
        self.n, self.m = n, m

        self.state_space_size = self.n * self.m
        self.action_space_size = 2  # 0 for island, 1 for sea
        self.dataset_path = dataset_path

        self.board_index = 0

        if not lazy:
            assert dataset_path is not None
            self._load_all_solutions()
            self.reset()
    
    def _load_all_solutions(self):
        self.all_solutions = np.load(self.dataset_path)

    def reset(self):    
        # self.solution = self.all_solutions[
        #     self.all_solutions.files[self.board_index]
        # ]
        self.solution = np.load(self.all_solutions[self.board_index])
        
        self.board_index += 1
        self.board = np.where(
            (self.solution == 0) | (self.solution == -1),
            -2, self.solution
        )
        return self.board.flatten(), self.get_optimal_rtg(), self.get_state_index()
    
    def get_state_index(self):
        flat_board = self.board.flatten()
        unoccupied_indices = np.where(flat_board == -2)[0]
        if len(unoccupied_indices) == 0:
            return None
        return unoccupied_indices[0]
        # return np.ravel_multi_index(np.argwhere(self.board.flatten() == -2)[0], (self.n, self.m))

    def step(self, action, position):
        i, j = position // self.m, position % self.m
        current_val = self.board[i, j]

        if current_val >= 1:  # Already-numbered cell
            reward = -10000  # Heavy penalty
            done = False
            return np.copy(self.board.flatten()), reward, done
        
        prev_score = np.count_nonzero(self.solution == self.board)
        self.board[i, j] = action 
        cur_score = np.count_nonzero(self.solution == self.board) 

        return np.copy(self.board.flatten()), cur_score - prev_score, cur_score == self.m*self.n

        # if not self.check_correctness(i, j, action):
        #     reward = -1
        #     done = False
        #     return np.copy(self.board.flatten()), reward, done

        # if current_val == -2:  # Unoccupied cell
        #     self.board[i, j] = 0 if action == 0 else -1
        #     done = self.is_done()
        #     reward = 100 if done else 1
        # else:  # Attempt to overwrite sea or island
        #     reward = -0.5  # Penalty for incorrect attempt, do not write
        #     done = False
        # return np.copy(self.board.flatten()), reward, done
    
    def get_optimal_rtg(self):
        return np.count_nonzero(self.board <= 0)

    def is_done(self):
        return np.all(self.board != -2) and (self.solution == self.board).all()

    # checking logic for completeness
    def is_valid(self):
        return (self.check_contiguous_islands() and self.check_single_sea() and 
                self.check_no_2x2_sea() and self.check_island_numbers())

    def check_contiguous_islands(self):
        visited = np.zeros_like(self.board, dtype=bool)
        for i in range(self.n):
            for j in range(self.m):
                if self.board[i, j] == 0 and not visited[i, j]:
                    if not self.dfs(i, j, visited, 0):
                        return False
        return True
    
    def check_single_sea(self):
        visited = np.zeros_like(self.board, dtype=bool)
        sea_count = 0
        for i in range(self.n):
            for j in range(self.m):
                if self.board[i, j] == 1 and not visited[i, j]:
                    if sea_count > 0:
                        return False
                    self.dfs(i, j, visited, 1)
                    sea_count += 1
        return sea_count == 1
    
    def check_no_2x2_sea(self):
        for i in range(self.n - 1):
            for j in range(self.m - 1):
                if (self.board[i, j] == 1 and self.board[i+1, j] == 1 and
                    self.board[i, j+1] == 1 and self.board[i+1, j+1] == 1):
                    return False
        return True
    
    def check_correctness(self, i, j, action):
        return self.solution[i, j] == action
    
    def check_island_numbers(self):
        pass

    def dfs(self, i, j, visited, cell_type):
        stack = [(i, j)]
        visited[i, j] = True
        while stack:
            x, y = stack.pop()
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.n and 0 <= ny < self.m and not visited[nx, ny] and self.board[nx, ny] == cell_type:
                    visited[nx, ny] = True
                    stack.append((nx, ny))
        return True

def q_learning(env, num_episodes=1000, alpha=0.1, gamma=0.99, epsilon=0.1):
    q_table = np.zeros((env.state_space_size, env.action_space_size))
    for episode in range(num_episodes):
        print(f'Processing {episode}th episode')
        state, state_index = env.reset()
        done = False
        while not done:
            if random.uniform(0, 1) < epsilon:
                action = random.choice([0, 1])  # Explore action space
            else:
                action = np.argmax(q_table[state_index])  # Exploit learned values
            position = np.random.choice(env.state_space_size)
            while env.board[position // env.m, position % env.m] == -2:
                position = np.random.choice(env.state_space_size)
            next_state, reward, done = env.step(action, position)
            next_state_index = env.get_state_index()
            if next_state_index is None:
                break
            q_table[state_index, action] = q_table[state_index, action] + alpha * (
                reward + gamma * np.max(q_table[next_state_index]) - q_table[state_index, action])
            state_index = next_state_index
    return q_table


# env = NurikabeEnv(9, 9, '../data/nurikabe.npz')
# q_table = q_learning(env)

# print("Trained Q-table:")
# print(q_table)

