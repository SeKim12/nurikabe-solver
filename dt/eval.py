import torch 
import torch.nn.functional as F
from torch.utils.data import DataLoader
import random
import numpy as np
import matplotlib.pyplot as plt 

import env
import dataset
import dt_model

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
    block_size = 30 * 3
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

def convert_action(action):
    a_y = action // 27
    a_x = (action - a_y * 27) // 3
    a = action - a_y * 27 - a_x * 3
    return a_y * 9 + a_x, a

def run_model(model, loader, eval_iters, max_steps_in_env):
    world = env.NurikabeEnv(9, 9, lazy=True)
    world.all_solutions = loader.dataset.soln_files

    rtgs_per_timestep = []
    for _ in range(eval_iters):
        ts = 0
        is_done = False 

        state, init_rtg, _ = world.reset()       
        state = np.reshape(state, (9, 9)) 
        all_states = [state] 

        all_actions = []
        all_rtgs = [init_rtg]

        print(f'\n\nSTARTING NEW ENV\n\n')

        while not is_done and ts < max_steps_in_env:
            timestep = torch.tensor([ts], dtype=torch.int64).view((1,1,1))
            sampled_action = sample(model, torch.tensor(np.array(all_states), dtype=torch.float32).unsqueeze(0), 1, temperature=1.0, sample=True, actions=None if len(all_actions) == 0 else torch.tensor(all_actions, dtype=torch.long).unsqueeze(0).unsqueeze(-1), 
                                    rtgs=torch.tensor(all_rtgs, dtype=torch.float).unsqueeze(0).unsqueeze(-1),  # add batch dimension, and add n_rwd 
                                    timesteps=timestep)  # should be size (b, t) where b is 1
            
            position, action = convert_action(sampled_action.item())

            state, reward, is_done = world.step(-action, position)
            print(f'action: ({position}, {action}) reward: {reward}, equal to solution: {np.count_nonzero(world.board == world.solution)} / {np.size(world.solution)}')
            state = np.reshape(state, (9, 9)) 
            all_states.append(state) 
            all_rtgs.append(all_rtgs[-1] - reward)
            all_actions.append(sampled_action.item())
            ts += 1
        
        for i, rtg in enumerate(all_rtgs):
            if i < len(rtgs_per_timestep):
                rtgs_per_timestep[i].append(rtg)
            else:
                rtgs_per_timestep.append([rtg])
    
    mean_rtgs = []
    for i in range(len(rtgs_per_timestep)):
        mean_rtgs.append(np.mean(rtgs_per_timestep[i]))

    print(mean_rtgs)
    plt.plot(np.arange(len(mean_rtgs)), mean_rtgs)
    plt.savefig('avg_rtg.png') 
    plt.clf()


if __name__ == '__main__': 
    max_seq_len = 30
    max_timesteps = 140
    val_dataset = dataset.NurikabeDataset('data/logicgamesonline_trajectories_train_augmented.zip', max_seq_len, 9, 9)
    mconf = dt_model.GPTConfig(9**2 * 3, max_seq_len * 3,
                n_layer=6, n_head=8, n_embd=128, model_type='reward_conditioned', max_timestep=max_timesteps)
    model = dt_model.GPT(mconf)
    model.load_state_dict(torch.load('2024_06_01_01_41_23/ckpt.pt'))
    val_loader = DataLoader(val_dataset, shuffle=False,
                              batch_size=1,
                              num_workers=1)
    
    run_model(model, val_loader, 10, 100)

        









