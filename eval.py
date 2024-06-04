import torch 
import torch.nn.functional as F
from torch.utils.data import DataLoader
import random
import numpy as np
import matplotlib.pyplot as plt 
import os 
from scipy import stats


from dt import env, dataset, dt_model

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
    return a_y * 9 + a_x, -a

def run_model(model, loader, eval_iters, max_steps_in_env, results_dir=None):
    world = env.NurikabeEnv(9, 9, lazy=True)
    world.all_solutions = loader.dataset.soln_files[:120*eval_iters:120]
    rtgs_per_timestep = []

    returns_per_timestep = []
    max_all = -1
    for _ in range(eval_iters):
        ts = 0
        is_done = False 

        state, init_rtg, _ = world.reset()       
        state = np.reshape(state, (9, 9)) 
        all_states = [state] 

        all_actions = []
        all_rtgs = [init_rtg]

        print(f'\n\nSTARTING NEW ENV\n\n')

        returns = []
        while not is_done and ts < max_steps_in_env:
            timesteps = torch.tensor([min(ts, max_timesteps)], dtype=torch.int64).view((1,1,1))
            sampled_action = sample(model, torch.tensor(np.array(all_states), dtype=torch.float32).unsqueeze(0), 1, temperature=1.0, sample=True, actions=None if len(all_actions) == 0 else torch.tensor(all_actions, dtype=torch.long).unsqueeze(0).unsqueeze(-1), 
                                    rtgs=torch.tensor(all_rtgs, dtype=torch.float).unsqueeze(0).unsqueeze(-1),  # add batch dimension, and add n_rwd 
                                    timesteps=timesteps)  # should be size (b, t) where b is 1
            
            position, action = convert_action(sampled_action.item())

            state, reward, is_done = world.step(action, position)
            # print(f'action: ({position}, {action}) reward: {reward}, equal to solution: {np.count_nonzero(world.board == world.solution)} / {np.size(world.solution)}')
            state = np.reshape(state, (9, 9)) 
            all_states.append(state) 
            all_rtgs.append(all_rtgs[-1] - reward)
            all_actions.append(sampled_action.item())
            ts += 1

            returns.append(np.count_nonzero(world.board == world.solution)) 

        returns_per_timestep.append(returns)
        max_all = max(max_all, max(returns))
        # for i, rtg in enumerate(all_rtgs):
        #     if i < len(rtgs_per_timestep):
        #         rtgs_per_timestep[i].append(rtg)
        #     else:
        #         rtgs_per_timestep.append([rtg])
        
        # traj_file_name = f'{world.all_solutions[world.board_index - 1].split("/")[-1].split("_soln")[0]}'
        # with open(os.path.join(results_dir, f'{traj_file_name}.csv'), 'w') as f:
        #     f.write('9,9\n') 
        #     for t in range(len(all_states)):
        #         f.write(','.join([f'{int(x)}' for x in all_states[t].flatten()]) + '\n')
        #     f.write('seconds,seconds') 

    results = np.array(returns_per_timestep) 
    xs = np.arange(results.shape[1])
    results = np.array(results)
    xs = np.arange(results.shape[1])
    ys = np.mean(results, axis=0)
    yerrs = stats.sem(results, axis=0)
    plt.figure(figsize=(6,4))
    plt.fill_between(xs, ys - yerrs, ys + yerrs, alpha=0.25)
    plt.plot(xs, ys, label='DT')
    plt.axhline(81, color='g', linestyle='--', label='# cells in env')
    plt.xlabel('timesteps') 
    plt.ylabel('# cells correct')
    plt.legend()
    plt.title('DecisionTransformer Online Performance')

    print(max_all)
    # mean_rtgs = []
    # for i in range(len(rtgs_per_timestep)):
    #     mean_rtgs.append(np.mean(rtgs_per_timestep[i]))

    # print(mean_rtgs)
    # plt.plot(np.arange(len(mean_rtgs)), mean_rtgs)
    plt.savefig('cumsum_rewards_sem.png') 
    plt.clf()


if __name__ == '__main__': 
    eval_results_dir = 'dt_online_results'
    os.makedirs(eval_results_dir, exist_ok=True)

    max_seq_len = 30
    max_timesteps = 200
    val_dataset = dataset.NurikabeDataset('data/logicgamesonline_trajectories_val_expert720K', max_seq_len, 9, 9)
    mconf = dt_model.GPTConfig(9**2 * 3, max_seq_len * 3,
                n_layer=6, n_head=8, n_embd=128, model_type='reward_conditioned', max_timestep=max_timesteps)
    model = dt_model.GPT(mconf)
    model.load_state_dict(torch.load('2024_06_03_18_13_14/ckpt.pt')['model_state_dict'])
    val_loader = DataLoader(val_dataset, shuffle=False,
                              batch_size=1,
                              num_workers=1)
    
    run_model(model, val_loader, 30, 500, eval_results_dir)

        









