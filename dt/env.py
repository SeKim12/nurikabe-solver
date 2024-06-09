import numpy as np

from data import nurikabe_dataset


class NurikabeEnv:
    def __init__(
        self, nds: nurikabe_dataset.NurikabeDataset, num_worlds=-1, shuffle=False
    ):

        self.nds = nds
        self.board_index = 0
        self.num_worlds = num_worlds
        self.envs = nds.get_envs(num_worlds, shuffle)

    def reset(self):
        self.solution = np.load(self.envs[self.board_index])
        self.board = np.where(
            (self.solution == 0) | (self.solution == -1), -2, self.solution
        )
        self.board_index += 1
        return np.copy(self.board), self.get_optimal_rtg(self.solution)

    def step(self, action_y, action_x, action):
        sp = np.copy(self.board)
        sp[action_y, action_x] = action

        # noop for same action
        if np.all(sp == self.board):
            return self.board, 0, False

        # get reward
        rewards, terminated = self.nds.get_trajectory(
            np.stack((self.board, sp)), self.solution
        )[-2:]

        assert rewards.size == 1

        self.board = sp
        return np.copy(self.board), rewards[0], terminated

    def get_optimal_rtg(self, board):
        return self.nds.get_optimal_rtg(board)
