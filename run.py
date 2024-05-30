"""
Very quick and dirty way to test the training flow. Need to set up proper training / eval pipeline
"""
import torch 
import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader
from model import decision_transformer
from data import nurikabe_dataset, constants

if __name__ == '__main__':

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
    
    # this is K in paper
    max_seq_len = 70
    # this is how long an episode can be in general
    # the way I understood it was that an episode can be max_timesteps long 
    # and we sample a sequence of size K from an episode. I could be completely wrong on this though
    max_timesteps = 300

    # probably just need to fix this
    model_type = 'reward_conditioned'
    mps_device = torch.device("mps:0")

    # poor man's train/val split...
    train_dataset = nurikabe_dataset.NurikabeDataset('data/microsoft_logicgamesonline_trajectories.zip', max_seq_len, 9, 9)

    train_size = int(len(train_dataset) * 0.8)
    val_size = len(train_dataset) - train_size

    train_dataset.traj_files = train_dataset.traj_files[:train_size]
    train_dataset.soln_files = train_dataset.soln_files[:train_size]

    val_dataset = nurikabe_dataset.NurikabeDataset('data/microsoft_logicgamesonline_trajectories.zip', max_seq_len, 9, 9)
    val_dataset.traj_files = val_dataset.traj_files[train_size:]
    val_dataset.soln_files = val_dataset.soln_files[train_size:]
    
    print(f'Train: {len(train_dataset)}, Val: {len(val_dataset)}')

    # this is mostly from the code directly
    tconf = TrainerConfig(max_epochs=500, batch_size=128, learning_rate=1e-3,
                          lr_decay=True, warmup_tokens=512*20,
                          num_workers=1, model_type='reward_conditioned', max_timestep=max_timesteps)

    train_loader = DataLoader(train_dataset, shuffle=True,
                        batch_size=tconf.batch_size,
                        num_workers=tconf.num_workers)
    
    val_loader = DataLoader(val_dataset, shuffle=True,
                    batch_size=tconf.batch_size,
                    num_workers=tconf.num_workers)

    # same here
    mconf = decision_transformer.GPTConfig(9**2 * train_dataset.action_space, max_seq_len * 3,
                    n_layer=6, n_head=8, n_embd=128, model_type=model_type, max_timestep=max_timesteps)

    model = decision_transformer.GPT(mconf).to(mps_device)
    optimizer = model.configure_optimizers(tconf)

    eval_losses_per_epoch = []
    train_losses = []
    best_val_loss = float('inf')

    # poor man's train/val loop...
    # for each epoch, even if we use the same trajectory
    # we would be getting a different K-length sequence from that trajectory
    for epoch in tqdm(range(tconf.max_epochs), desc='epoch', position=0):
        pbar = tqdm(train_loader, total=len(train_loader), desc='train', position=1)
        for x, y, r, t in pbar:
            logits, loss = model(x.to(mps_device), y.to(mps_device), y.to(mps_device), r.to(mps_device), t.to(mps_device))
            loss = loss.mean()
            train_losses.append(loss.item())

            model.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.postfix = f'train_loss={train_losses[-1]:.3f}'

        eval_losses = []
        with torch.no_grad():
            pbar = tqdm(val_loader, total=len(val_loader), desc='val', position=1)
            for x, y, r, t in pbar:
                logits, loss = model(x.to(mps_device), y.to(mps_device), y.to(mps_device), r.to(mps_device), t.to(mps_device))
                loss = loss.mean()
                eval_losses.append(loss.item())

                pbar.postfix = f'eval_loss={eval_losses[-1]:.3f}'

        eval_losses_per_epoch.append(np.mean(eval_losses))

        np.save('eval_losses.npy', np.array(eval_losses_per_epoch))
        np.save('train_losses.npy', np.array(train_losses))

        if eval_losses_per_epoch[-1] < best_val_loss:
            torch.save(model, f'ckpt.pt')
            best_val_loss = eval_losses_per_epoch[-1]
