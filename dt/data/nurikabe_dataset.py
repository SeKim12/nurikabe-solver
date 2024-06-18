import os
import torch

from torch.utils import data
import numpy as np

import constants
from data import data_utils


class NurikabeDataset(data.Dataset):
    def __init__(
        self,
        data_dir,
        max_seq_len,
        delay_rewards=False,
        fix_k=False,
        fixed_h=None,
        fixed_w=None,
        variable_world_size=False,
    ):

        assert variable_world_size != (
            fixed_h is not None and fixed_w is not None
        ), "choose variable or fixed world size"

        self.data_dir = data_dir
        self.max_seq_len = max_seq_len

        self.fixed_h = fixed_h
        self.fixed_w = fixed_w
        self.variable_world_size = variable_world_size

        self.delay_rewards = delay_rewards
        self.fixed_k = fix_k

        self.action_space = constants.Actions.size()

        grid_dirs = os.listdir(self.data_dir)

        self.traj_files = []
        self.soln_files = []

        for grid_dir in grid_dirs:
            grid_dir = os.path.join(self.data_dir, grid_dir)
            self.traj_files.extend(
                [
                    os.path.join(grid_dir, file)
                    for file in os.listdir(grid_dir)
                    if file.endswith(".csv")
                ]
            )
            self.soln_files.append(os.path.join(grid_dir, "soln.npy"))

        self.traj_files = sorted(self.traj_files)
        self.soln_files = sorted(self.soln_files)

        print(
            f"Found {len(self.traj_files)} Trajectories ({len(self.soln_files)} Unique Grids)"
        )

    def get_envs(self, n_envs=-1, shuffle=False):
        if n_envs == -1:
            return self.soln_files
        if not shuffle:
            return self.soln_files[:n_envs]
        return (
            np.array(self.soln_files)[
                np.random.choice(np.arange(len(self.soln_files)), n_envs, replace=False)
            ]
        ).tolist()

    def get_trajectory(self, states, soln):
        T, H, W = states.shape
        if not self.variable_world_size:
            assert (
                H == self.fixed_h and W == self.fixed_w
            ), f"Expected fixed world size (H,W) of {self.fixed_h}x{self.fixed_w}, received {H}x{W}"

        # states - 1 ensures that all actions will be < 0
        # soln > 0 is a boolean mask for numbered cells
        # therefore if we apply the mask and get a negative, we've made an action on a number cell
        hit_numbered_cell = (
            np.count_nonzero(((states - 1) * (soln > 0)) < 0, axis=(1, 2))[-1] == 1
        )
        solved_grid = np.all(states[-1] == soln)
        terminated = hit_numbered_cell or solved_grid

        if not self.delay_rewards:
            # compute return-to-go
            # The return at each timestep is the improvement in number of positions that are equal to solution from last timestep
            cur_correct = (states == soln).sum((1, 2))
            prev_correct = np.roll(cur_correct, shift=1, axis=0)
            rewards = (cur_correct - prev_correct)[1:]

            if hit_numbered_cell:
                rewards[-1] = constants.NUM_CELL_RWD
        else:
            rewards = np.zeros((T - 1,))
            rewards[-1] = int(np.all(states[-1] == soln))

        # size T - 1
        rtgs = np.cumsum(rewards[::-1])[::-1]

        # compute actions
        # action can be recovered by doing diff(states[i], states[i-1])
        states_r = np.roll(states, shift=1, axis=0)
        states_r[0] = states[0]

        # one hot vector of places that are different
        states_diff = np.where((states - states_r) == 0, 0, 1)
        assert (
            np.count_nonzero(states_diff.sum((1, 2)) > 1) == 0
        ), f"States t and states t - 1 differ by more than one position!"

        # [[h1, w1], [h2, w2]...[ht, wt]] for t timesteps where each action occurs
        action_locs = np.stack(np.nonzero(states_diff)).T[:, 1:]
        action_y, action_x = action_locs[:, 0], action_locs[:, 1]
        # recover what the actual action was by looking at that location in next state
        actions = states[1:][np.arange(action_locs.shape[0]), action_y, action_x]

        # truncate last state and RTG
        states = states[:-1]
        # rtgs = rtgs[:-1]

        # map actions into unique index
        actions = data_utils.action_to_index_fixed(
            H, W, constants.Actions.size(), action_y, action_x, abs(actions)
        )
        # actions = action_y * W * self.action_space + action_x * self.action_space + abs(actions)
        assert (
            states.shape[0] == actions.shape[0] == rtgs.shape[0]
        ), f"Dimension mismatch: S ({states.shape[0]}) != A ({actions.shape[0]}) != R ({rtgs.shape})"

        return states, actions, rtgs, rewards, terminated

    def get_optimal_rtg(self, board):
        if self.delay_rewards:
            return 1
        else:
            return np.count_nonzero(board <= 0)

    def __len__(self):
        return len(self.traj_files)

    def __getitem__(self, index):
        traj_file = self.traj_files[index]
        soln_file = "/".join(self.traj_files[index].split("/")[:-1] + ["soln.npy"])

        states = data_utils.load_csv_trajectory(traj_file)
        soln = np.load(soln_file).astype(int)

        states, actions, rtgs, _, _ = self.get_trajectory(states, soln)

        T, H, W = states.shape

        # from trajectory, sample max_seq_len trajectory
        # if trajectory < max_seq_len, take entire trajectory and pad
        to_pad = self.max_seq_len - T
        idx = -1
        if to_pad >= 0:
            states = np.pad(states, ((0, to_pad), (0, 0), (0, 0)), mode="edge")
            actions = np.pad(actions, ((0, to_pad)), mode="edge")
            rtgs = np.pad(rtgs, ((0, to_pad)), mode="edge")
            idx = 0
        else:
            idx = np.random.choice(np.arange(actions.shape[0] - self.max_seq_len + 1))
            if self.fixed_k:
                idx = actions.shape[0] - self.max_seq_len

        states = states[idx : idx + self.max_seq_len]
        actions = actions[idx : idx + self.max_seq_len]
        rtgs = rtgs[idx : idx + self.max_seq_len]
        timesteps = np.array([idx])

        return (
            torch.tensor(states.copy(), dtype=torch.float32),
            torch.tensor(actions.copy()).unsqueeze(-1),
            torch.tensor(rtgs.copy(), dtype=torch.float32).unsqueeze(-1),
            torch.tensor(timesteps.copy()).unsqueeze(-1),
        )


def get_loader(
    data_dir,
    batch_size,
    reward_type,
    fix_k,
    max_seq_len,
    fixed_h=9,
    fixed_w=9,
    variable_world_size=False,
):

    delay_rewards = reward_type == "delayed"
    dataset = NurikabeDataset(
        data_dir, max_seq_len, delay_rewards, fix_k, fixed_h, fixed_w, False
    )

    return data.DataLoader(
        dataset, batch_size, shuffle=True, num_workers=8, pin_memory=True
    )


if __name__ == "__main__":
    loader = get_loader(
        data_dir="../data/trajectories_val",
        batch_size=1,
        reward_type="",
        fix_k=True,
        max_seq_len=50,
    )

    for _ in loader:
        pass
