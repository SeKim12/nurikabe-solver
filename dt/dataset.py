import tempfile
import zipfile
import os
import csv
import torch

from torch.utils import data
import torch.nn.functional as F
import numpy as np

from dt import constants
# import constants

def load_csv_trajectory(path_to_csv):
    arr = []
    with open(path_to_csv) as file:
        reader = csv.reader(file)
        w, h = 0, 0
        for i, row in enumerate(reader):
            # this indicates the number of cells solved
            if 'seconds' in row[-1]:
                continue
            row = [int(x) for x in row]
            if i == 0:
                w, h = row
            else:
                arr.append(np.array(row).reshape((h, w)))
    return np.array(arr)

class NurikabeDataset(data.Dataset):
    def __init__(self, 
                 path_to_data_dir, 
                 max_seq_len,
                 fixed_h=None,
                 fixed_w=None,
                 variable_world_size=False):

        # assert path_to_zipfile.endswith('.zip'), 'Please pass in a zipfile'
        assert variable_world_size != (fixed_h is not None and fixed_w is not None), 'choose variable or fixed world size'

        # self.tmpdir = tempfile.TemporaryDirectory()
        # self.tmpdir_name = self.tmpdir.name
        self.datadir = path_to_data_dir

        self.max_seq_len = max_seq_len 

        self.fixed_h = fixed_h
        self.fixed_w = fixed_w 
        self.variable_world_size = variable_world_size 

        self.action_space = constants.Actions.size()

        # zf = zipfile.ZipFile(path_to_zipfile)
        # zf.extractall(self.tmpdir_name)

        self.traj_files = []
        self.soln_files = []
        for file in os.listdir(self.datadir):
            if file.endswith('.csv'):
                self.traj_files.append(os.path.join(self.datadir, file))
            elif file.endswith('.npy'):
                self.soln_files.append(os.path.join(self.datadir, file))
        
        self.traj_files = sorted(self.traj_files)
        self.soln_files = sorted(self.soln_files)
        
        print(f'Found {len(self.traj_files)} Trajectories')
        
        # seq_lens = self.get_all_seq_lens()

        # self.max_trajectory_len = max(seq_lens) 
        # self.min_trajectory_len = min(seq_lens) 

        # if self.max_seq_len == -1:
        #     self.max_seq_len = self.min_trajectory_len

    
    def get_all_seq_lens(self):
        return [self.__getitem__(i, get_seq_len=True) for i in range(len(self))]

    def load_trajectory(self, index):
        states = load_csv_trajectory(self.traj_files[index])
        T, H, W = states.shape
        if not self.variable_world_size:
            assert H == self.fixed_h and W == self.fixed_w, \
                f'Expected fixed world size (H,W) of {self.fixed_h}x{self.fixed_w}, received {H}x{W}'

        try: 
            soln = np.load(self.soln_files[index]).astype(int)  # H, W
        except Exception as e:
            print(e)
            print(self.soln_files[index])
            os.remove(self.soln_files[index])
            os.remove(self.traj_files[index])
            assert False 

        # compute return-to-go
        # The return at each timestep is the improvement in number of positions that are equal to solution from last timestep
        cur_correct = (states == soln).sum((1,2))
        prev_correct = np.roll(cur_correct, shift=1, axis=0)
        rewards = cur_correct - prev_correct

        # states - 1 ensures that all actions will be < 0
        # soln > 0 is a boolean mask for numbered cells 
        # therefore if we apply the mask and get a negative, we've made an action on a number cell
        hit_numbered_cell = np.count_nonzero(((states - 1) * (soln > 0)) < 0, axis=(1,2))   # numbered
        # if np.count_nonzero(hit_numbered_cell) > 0:
        #     breakpoint()
        rewards[0] = 0
        rewards = np.where(hit_numbered_cell == 0, rewards, constants.NUM_CELL_RWD)
        rtgs = np.cumsum(rewards[::-1])[::-1]

        # compute actions
        # action can be recovered by doing diff(states[i], states[i-1])
        states_r = np.roll(states, shift=1, axis=0)
        states_r[0] = states[0] 

        # one hot vector of places that are different 
        states_diff = np.where((states - states_r) == 0, 0, 1) 
        assert np.count_nonzero(states_diff.sum((1,2)) > 1) == 0, f'States t and states t - 1 differ by more than one position!'
        
        # [[h1, w1], [h2, w2]...[ht, wt]] for t timesteps where each action occurs
        action_locs = np.stack(np.nonzero(states_diff)).T[:, 1:]  
        action_y, action_x = action_locs[:, 0], action_locs[:, 1]
        # recover what the actual action was by looking at that location in next state
        actions = states[1:][np.arange(action_locs.shape[0]), action_y, action_x]

        # truncate last state and RTG
        states = states[:-1] 
        rtgs = rtgs[:-1]

        # map actions into unique index
        # z = action_y, y = action_x, x = action
        actions = action_y * W * self.action_space + action_x * self.action_space + abs(actions)
        assert states.shape[0] == actions.shape[0] == rtgs.shape[0], f'Dimension mismatch at {index} S ({states.shape[0]}) != A ({actions.shape[0]}) != R ({rtgs.shape})'
        
        return states, actions, rtgs

    def __len__(self):
        return len(self.traj_files)
    
    def __getitem__(self, index, get_seq_len=False):
        states, actions, rtgs = self.load_trajectory(index)

        T, H, W = states.shape
        if get_seq_len:
            return T

        # from trajectory, sample max_seq_len trajectory
        # if trajectory < max_seq_len, take entire trajectory and pad 
        to_pad = self.max_seq_len - T
        idx = -1
        if to_pad >= 0:
            states = np.pad(states, ((0, to_pad), (0, 0), (0, 0)), mode='edge')
            actions = np.pad(actions, ((0, to_pad)), mode='edge')
            rtgs = np.pad(rtgs, ((0, to_pad)), mode='edge')
            idx = 0
        else:
            idx = np.random.choice(np.arange(actions.shape[0] - self.max_seq_len + 1))
        
        states = states[idx:idx+self.max_seq_len]
        actions = actions[idx:idx+self.max_seq_len]
        rtgs = rtgs[idx:idx+self.max_seq_len]
        timesteps = np.array([idx])

        return torch.tensor(states.copy(), dtype=torch.float32), torch.tensor(actions.copy()).unsqueeze(-1), torch.tensor(rtgs.copy(), dtype=torch.float32).unsqueeze(-1), torch.tensor(timesteps.copy()).unsqueeze(-1)

def get_loader(zf, bs, max_seq_len=-1, fixed_h=9, fixed_w=9, variable_world_size=False): 
    ds = NurikabeDataset(zf, max_seq_len, fixed_h, fixed_w, variable_world_size)
    return data.DataLoader(
        ds,
        bs, 
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )

if __name__ == '__main__':
    get_loader('data/logicgamesonline_trajectories_train_augmented.zip', 1)