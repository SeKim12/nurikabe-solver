from typing import Any

import torch
from torch.utils import data
import torch.nn.functional as F

class NurikabeDataset(data.Dataset):
    def __init__(self, seq_len, max_world_size) -> None:
        self.seq_len = seq_len
        self.max_world_size = max_world_size

    def load_trajectory(self, index):
        raise NotImplementedError()

    def __getitem__(self, index):
        states, actions, rewards = self.load_trajectory(index)

        T, H, W = states.size() 
        assert len(actions) == len(rewards) == T

        flat_states = states.resize(T, -1)

        # T, max_world_size**2
        flat_states_padded = F.pad(flat_states, (0, self.max_world_size**2 - H*W))
        # T
        flat_actions = [r*c + bit+1 for t in range(T) for (r, c), bit in actions[t]]

        return flat_states_padded[-self.seq_len:], flat_actions[-self.seq_len:], rewards[-self.seq_len:]