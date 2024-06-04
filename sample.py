"""
The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from model import decision_transformer
from data import nurikabe_dataset, constants


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

@torch.no_grad()
def sample(model, x, steps, temperature=1.0, sample=False, top_k=None, actions=None, rtgs=None, timesteps=None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    block_size = model.get_block_size()
    model.eval()
    for k in range(steps):
        # x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
        x_cond = x if x.size(1) <= block_size//3 else x[:, -block_size//3:] # crop context if needed
        if actions is not None:
            actions = actions if actions.size(1) <= block_size//3 else actions[:, -block_size//3:] # crop context if needed
        rtgs = rtgs if rtgs.size(1) <= block_size//3 else rtgs[:, -block_size//3:] # crop context if needed
        logits, _ = model(x_cond, actions=actions, targets=None, rtgs=rtgs, timesteps=timesteps)
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue
        # x = torch.cat((x, ix), dim=1)
        x = ix

    return x


if __name__ == '__main__':
    dataset = nurikabe_dataset.NurikabeDataset('data/microsoft_logicgamesonline_trajectories.zip', 1, 9, 9)

    mconf = decision_transformer.GPTConfig(9**2 * 3, 70 * 3,
                n_layer=6, n_head=8, n_embd=128, model_type='reward_conditioned', max_timestep=300)
    
    model = decision_transformer.GPT(mconf)
    model.load_state_dict(torch.load('ckpt.pt'))

    state, _, _, _ = (x.unsqueeze(0) for x in dataset[0])

    actions = []
    rtgs = [0] 

    all_states = state  # b, t, 9, 9

    sampled_action = sample(model, state, 1, temperature=1.0, sample=True, actions=None, 
        rtgs=torch.tensor(rtgs, dtype=torch.long).unsqueeze(0).unsqueeze(-1),  # add batch dimension, and add n_rwd 
        timesteps=torch.zeros((1, 1, 1), dtype=torch.int64))  # should be size (b, t) where b is 1
    
    # TODO: take sampled action to get new state
    # state = , reward = 
    reward = 0

    all_states = torch.cat([all_states, state], dim=1)  # b, t, 9, 9
    # breakpoint()
    actions += [sampled_action] 

    # breakpoint()
    rtgs += [rtgs[-1] - reward]

    sampled_action = sample(model, all_states, 1, temperature=1.0, sample=True,
                            actions=torch.tensor(actions, dtype=torch.long).unsqueeze(0).unsqueeze(1),  # add batch dim, add n_act
                            rtgs=torch.tensor(rtgs, dtype=torch.float).unsqueeze(0).unsqueeze(-1), 
                            timesteps=torch.ones((1, 1, 1), dtype=torch.int64))
    
    breakpoint()

