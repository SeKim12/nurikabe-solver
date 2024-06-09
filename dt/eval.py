import torch
import torch.nn.functional as F
import random
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy import stats
import io
from PIL import Image

from data import data_utils
import env


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def run_eval(model, loader, device):
    model.eval()
    pbar = tqdm(loader, total=len(loader), desc="val")
    losses = []

    for data in pbar:
        states, actions, rtgs, timesteps = (x.to(device) for x in data)
        target_actions = torch.clone(actions)

        _, loss = model(states, actions, target_actions, rtgs, timesteps)

        losses.append(loss.item())
        pbar.postfix = f"loss={losses[-1]:.3f}"

    model.train()
    return losses


@torch.no_grad()
def run_online(
    model, dataset, n_worlds, max_timestep, device, split="train", max_steps_in_env=5000
):
    """
    Run online evaluation on fixed number of val set worlds and save
    model evaluation results per world.
    """
    model.eval()

    world = env.NurikabeEnv(dataset, n_worlds)

    metrics = {
        "rewards": [],  # rewards per world
        "num_correct": [],  # num correct cells per world
        "timesteps": [],
        "solved": [],
        "worlds": world.envs,
    }

    pbar = tqdm(range(n_worlds), total=n_worlds, desc=f"online_{split}")
    for puzzle in pbar:
        ts = 0
        is_done = False

        state, init_rtg = world.reset()

        all_states = [state]
        all_actions = []
        all_rtgs = [init_rtg]

        rewards = []
        num_correct = []

        while not is_done and ts < max_steps_in_env:
            states = (
                torch.tensor(np.array(all_states), dtype=torch.float32)
                .unsqueeze(0)
                .to(device)
            )  # B, T, H, W
            actions = (
                None
                if len(all_actions) == 0
                else torch.tensor(all_actions, dtype=torch.long)
                .unsqueeze(0)
                .unsqueeze(-1)
                .to(device)
            )  # B, T, N_act
            rtgs = (
                torch.tensor(all_rtgs, dtype=torch.float)
                .unsqueeze(0)
                .unsqueeze(-1)
                .to(device)
            )  # B, T, N_rtg
            timesteps = (
                torch.tensor([min(ts, max_timestep)], dtype=torch.int64)
                .view((1, 1, 1))
                .to(device)
            )

            sampled_action = sample(
                model,
                states,
                steps=1,
                temperature=1.0,
                sample=False,
                actions=actions,
                rtgs=rtgs,
                timesteps=timesteps,
            )

            action_y, action_x, action = data_utils.index_to_action_fixed(
                dataset.fixed_h,
                dataset.fixed_w,
                dataset.action_space,
                sampled_action.item(),
            )
            # action_y, action_x, action = convert_action(sampled_action.item())

            state, reward, is_done = world.step(action_y, action_x, action)

            all_states.append(state)
            all_rtgs.append(all_rtgs[-1] - reward)
            all_actions.append(sampled_action.item())

            ts += 1
            rewards.append(reward)

            num_correct.append(np.count_nonzero(world.board == world.solution))

        # pad
        rewards = rewards + [rewards[-1]] * (max_steps_in_env - len(rewards))
        num_correct = num_correct + [num_correct[-1]] * (
            max_steps_in_env - len(num_correct)
        )

        metrics["rewards"].append(rewards)  #  (N_worlds, Timesteps)
        metrics["num_correct"].append(num_correct)  # (N_worlds, Timesteps)

        metrics["timesteps"].append(ts + 1)
        metrics["solved"].append(num_correct[-1] == world.solution.size)

        # pbar.postfix = f'online_{split}_return={reward_sum:.3f}'
        pbar.postfix = f"online_{split}=max {max(num_correct)} correct"

    return metrics
    #     returns_per_timestep.append(returns)
    #     max_all = max(max_all, max(returns))


def visualize_eval_metrics(metrics, return_image=False, save_path=None):
    rewards = metrics["rewards"]  # N_worlds, Timesteps
    num_correct = metrics["num_correct"]  #  N_worlds, Timesteps

    assert len(rewards) == len(num_correct)
    assert all([len(rewards[i]) == len(num_correct[i]) for i in range(len(rewards))])

    max_ts = max([len(x) for x in rewards])

    padded_rewards = []
    padded_correct = []

    for reward, correct in zip(rewards, num_correct):
        padded_rewards.append(reward + [reward[-1]] * (len(reward) - max_ts))
        padded_correct.append(correct + [correct[-1]] * (len(correct) - max_ts))

    rewards = np.array(padded_rewards)
    num_correct = np.array(padded_correct)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    titles = ["rewards", "num_correct"]
    arrs = [rewards, num_correct]

    for i in range(2):
        xs = np.arange(arrs[i].shape[-1])
        ys = np.mean(arrs[i], axis=0)
        yerrs = stats.sem(arrs[i], axis=0)

        axs[i].fill_between(xs, ys - yerrs, ys + yerrs, alpha=0.25)
        axs[i].plot(xs, ys)
        axs[i].set_xlabel("timesteps")
        axs[i].set_ylabel(titles[i])

        axs[i].set_title(f"DT online {titles[i]} ({rewards.shape[0]} worlds)")

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    if return_image:
        buf = io.BytesIO()
        fig.savefig(buf, bbox_inches="tight")
        buf.seek(0)
        plt.clf()
        return Image.open(buf)
    return None


def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float("Inf")
    return out


@torch.no_grad()
def sample(
    model,
    x,
    steps,
    temperature=1.0,
    sample=False,
    top_k=None,
    actions=None,
    rtgs=None,
    timesteps=None,
):
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
        x_cond = (
            x if x.size(1) <= block_size // 3 else x[:, -block_size // 3 :]
        )  # crop context if needed
        if actions is not None:
            actions = (
                actions
                if actions.size(1) <= block_size // 3
                else actions[:, -block_size // 3 :]
            )  # crop context if needed
        rtgs = (
            rtgs if rtgs.size(1) <= block_size // 3 else rtgs[:, -block_size // 3 :]
        )  # crop context if needed
        logits, _ = model(
            x_cond, actions=actions, targets=None, rtgs=rtgs, timesteps=timesteps
        )
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
    return a_y, a_x, -a
