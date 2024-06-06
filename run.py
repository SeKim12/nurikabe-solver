"""
Very quick and dirty way to test the training flow. Need to set up proper training / eval pipeline
"""
import torch 
import numpy as np
from tqdm import tqdm
from datetime import datetime
import os 
import sys 
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from scipy import stats
from model import decision_transformer
# from data import nurikabe_dataset, constants

from dt import dataset, env2
import eval

eval.set_seed(1112)

@torch.no_grad()
def run_model_eval(model, dataset, n_worlds, max_timesteps, device='mps:0', split='train', max_steps_in_env=5000): 
    """
    Run online evaluation on fixed number of val set worlds and save 
    model evaluation results per world. 
    """
    model.eval()

    world = env2.NurikabeEnv(dataset, n_worlds)

    metrics = {
        'returns': [],  # rewards per world
        'timesteps': [],
        'solved': [], 
        'worlds': world.envs
    }

    pbar = tqdm(range(n_worlds), total=n_worlds, desc=f'online_{split}')
    returns_per_timestep = []
    max_all = -1

    for i in pbar:
        ts = 0
        is_done = False 

        state, init_rtg = world.reset()

        all_states = [state] 
        all_actions = []
        all_rtgs = [init_rtg] 

        reward_sum = 0
        returns = []
        while not is_done and ts < max_steps_in_env:
            states = torch.tensor(np.array(all_states), dtype=torch.float32).unsqueeze(0).to(device) # B, T, H, W 
            actions = None if len(all_actions) == 0 else torch.tensor(all_actions, dtype=torch.long).unsqueeze(0).unsqueeze(-1).to(device) # B, T, N_act
            rtgs = torch.tensor(all_rtgs, dtype=torch.float).unsqueeze(0).unsqueeze(-1).to(device)  # B, T, N_rtg
            timesteps = torch.tensor([min(ts, max_timesteps)], dtype=torch.int64).view((1,1,1)).to(device)

            sampled_action = eval.sample(model, 
                                         states, 
                                         steps=1, 
                                         temperature=1.0, 
                                         sample=False, 
                                         actions=actions, 
                                         rtgs=rtgs,
                                         timesteps=timesteps)
            
            action_y, action_x, action = eval.convert_action(sampled_action.item())

            state, reward, is_done = world.step(action_y, action_x, action)

            all_states.append(state)
            all_rtgs.append(all_rtgs[-1] - reward)
            all_actions.append(sampled_action.item())

            reward_sum += reward
            ts += 1

            returns.append(np.count_nonzero(world.board == world.solution)) 

        metrics['returns'].append(reward_sum)
        metrics['timesteps'].append(ts + 1) 
        metrics['solved'].append(is_done)

        # pbar.postfix = f'online_{split}_return={reward_sum:.3f}'
        pbar.postfix = f'online_{split}_return={np.mean(returns[-1]):.3f}'

        returns_per_timestep.append(returns)
        max_all = max(max_all, max(returns))

    
    model.train()

    ret = []
    print(f'timesteps: {[len(x) for x in returns_per_timestep]}')
    for x in returns_per_timestep:
        ret.append(x + [x[-1] for _ in range(max_steps_in_env - len(x))])

    results = np.array(ret) 
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

    plt.savefig('cumsum_rewards_sem_5000ts.png') 
    plt.clf()
    print(max_all)
    return metrics


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--path_to_ckpt', type=str) 
    args = parser.parse_args()

    # cfgs
    bs = 128
    max_seq_len = 50
    max_timesteps = 200

    eval_every = 1
    max_epochs = 100
    lr = 1e-4

    mps_device = torch.device("cuda:2")

    train_loader = dataset.get_loader('data/logicgamesonline_trajectories_train_expert720K', bs, max_seq_len)
    val_loader = dataset.get_loader('data/logicgamesonline_trajectories_val_expert720K', bs, max_seq_len)

    class TrainerConfig:
        # optimization parameters
        max_epochs = 500
        batch_size = 128
        learning_rate = 3e-4
        betas = (0.9, 0.95)
        grad_norm_clip = 1.0
        weight_decay = 0.1 # only applied on matmul weights
        # learning rate decay params: linear warmup followed by cosine decay to 10% of original
        lr_decay = False
        warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
        final_tokens = 260e9 # (at what point we reach 10% of original LR)
        # checkpoint settings
        ckpt_path = None
        num_workers = 0 # for DataLoader

        def __init__(self, **kwargs):
            for k,v in kwargs.items():
                setattr(self, k, v)
    
    # probably just need to fix this
    model_type = 'reward_conditioned'

    # fix these 
    # online_worlds_train = train_loader.dataset.soln_files[:2400:120] 
    # online_worlds_eval = val_loader.dataset.soln_files[:2400:120]
    
    tconf = TrainerConfig(max_epochs=max_epochs, 
                          batch_size=bs, 
                          learning_rate=lr,
                          lr_decay=False, 
                          warmup_tokens=512*20,
                          num_workers=4, 
                          model_type='reward_conditioned', 
                          max_timestep=max_timesteps)

    mconf = decision_transformer.GPTConfig(9**2 * 3, 
                                           max_seq_len * 3,
                                           n_layer=3, 
                                           n_head=8, 
                                           n_embd=128, 
                                           model_type=model_type, 
                                           max_timestep=max_timesteps)

    print(f'Train # trajectories: {len(train_loader.dataset)}\n'
          f'Eval # trajectories: {len(val_loader.dataset)}\n'
          f'max_seq_len: {max_seq_len}\n'
          f'max_timesteps: {max_timesteps}\n')
    

    model = decision_transformer.GPT(mconf)

    if args.eval:
        model.load_state_dict(torch.load(args.path_to_ckpt)['model_state_dict'])
        model.to(mps_device)
        eval_online = run_model_eval(model, val_loader.dataset, 100, max_timesteps, split='eval', device=mps_device)

        sys.exit()
        
    model.to(mps_device)
    
    logdir = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    print('delayed reward: ', logdir)
    os.makedirs(logdir, exist_ok=True)
    optimizer = model.configure_optimizers(tconf)

    best_val_loss = float('inf')
    train_losses_per_epoch = []

    # poor man's train/val loop...
    # for each epoch, even if we use the same trajectory
    # we would be getting a different K-length sequence from that trajectory
    for epoch in tqdm(range(tconf.max_epochs), desc='epoch'):

        pbar = tqdm(train_loader, total=len(train_loader), desc='train')

        # # this loss tracks how well the model is able to predict the next action
        train_seq_pred_losses = []

        for x, y, r, t in pbar:
            logits, loss = model(x.to(mps_device), y.to(mps_device), y.to(mps_device), r.to(mps_device), t.to(mps_device))
            loss = loss.mean()
            train_seq_pred_losses.append(loss.item())

            model.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.postfix = f'train_seq_pred_loss={train_seq_pred_losses[-1]:.3f}'

        train_losses_per_epoch.append(np.mean(train_seq_pred_losses))
        np.save(os.path.join(logdir, 'train_losses.npy'), np.array(train_losses_per_epoch))
        
        if epoch % eval_every == 0: 
            with torch.no_grad():
                model.eval()
                eval_dir = os.path.join(logdir, f'e{epoch:04d}')
                os.makedirs(eval_dir, exist_ok=True) 

                pbar = tqdm(val_loader, total=len(val_loader), desc='val')

                # this tracks how well the model is able to predict the next action
                eval_seq_pred_losses = []
                for x, y, r, t in pbar:
                    logits, loss = model(x.to(mps_device), y.to(mps_device), y.to(mps_device), r.to(mps_device), t.to(mps_device))
                    loss = loss.mean()
                    eval_seq_pred_losses.append(loss.item())
                    pbar.postfix = f'eval_loss={eval_seq_pred_losses[-1]:.3f}'

                # this tracks how well the model performs in an online setting
                eval_online = run_model_eval(model, val_loader.dataset, 20, max_timesteps, split='eval', device=mps_device)

                # we track the same thing for a small subset of the train set
                train_online = run_model_eval(model, train_loader.dataset, 20, max_timesteps, split='train', device=mps_device)

                # breakpoint()
                results = {
                    'online_results_eval': eval_online, 
                    'seq_pred_losses_eval': eval_seq_pred_losses,
                    'online_results_train': train_online,
                }

                torch.save(results, os.path.join(eval_dir, 'eval_results.pt'))

                mean_online_returns_eval = np.mean(results['online_results_eval']['returns']) 
                mean_online_returns_train = np.mean(results['online_results_train']['returns']) 
                mean_seq_pred_loss = np.mean(eval_seq_pred_losses)

                ckpt_updated = False 
                if mean_seq_pred_loss < best_val_loss: 
                    ckpt_updated = True
                    best_val_loss = mean_seq_pred_loss 
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(), 
                        'optimizer_state_dict': optimizer.state_dict()
                    }, os.path.join(logdir, 'ckpt.pt'))
            
                print(f'Epoch {epoch} evaluation results:\n'
                    f'\tEVAL mean online returns(N=10, ↑): {mean_online_returns_eval:.3f}{" (new best!)" if ckpt_updated else ""}\n'
                    f'\tTRAIN mean online returns(N=10, ↑): {mean_online_returns_train:.3f}'
                    f'\tmean seq pred loss(↓): {mean_seq_pred_loss:.3f}\n')
                
                model.train()