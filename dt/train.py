import torch
import numpy as np
from tqdm import tqdm
import sys
import os
from datetime import datetime

from data import nurikabe_dataset
import dt_model, options, eval

wandb_logging = False

eval.set_seed(1112)

class TrainerConfig:
    # optimization parameters
    max_epochs = 500
    batch_size = 128
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1  # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6  # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9  # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 0  # for DataLoader

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def get_model_config(**model_args):
    return dt_model.GPTConfig(**model_args)


def get_train_config(**kwargs):
    return TrainerConfig(**kwargs)


def start_wandb(meta_args, cfg):
    run = wandb.init(
        entity="seungwoo-simon-kim",
        project="nubes",
        name=meta_args.exp_name,
        config=cfg,
    )

    run.define_metric("iter")
    run.define_metric("epoch")

    run.define_metric("iter/train_loss", step_metric="iter")

    run.define_metric("epoch/train_loss", step_metric="epoch")
    run.define_metric("epoch/val_loss", step_metric="epoch")

    run.define_metric("epoch/online_val_num_correct", step_metric="epoch")
    run.define_metric("epoch/online_train_num_correct", step_metric="epoch")

    run.define_metric("epoch/online_val", step_metric="epoch")
    run.define_metric("epoch/online_train", step_metric="epoch")


def train(model, optimizer, loader, device, itr):
    pbar = tqdm(loader, total=len(loader), desc="train")
    losses = []
    for idx, data in enumerate(pbar):
        states, actions, rtgs, timesteps = (x.to(device) for x in data)
        target_actions = torch.clone(actions)

        _, loss = model(states, actions, target_actions, rtgs, timesteps)

        loss = loss.mean()
        model.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        pbar.postfix = f"loss={losses[-1]:.3f}"

        if idx % 25 == 0 and wandb_logging:
            wandb.log({"iter/train_loss": losses[-1], "iter": itr})

        itr += 1

    if wandb_logging:
        wandb.log({"epoch/train_loss": np.mean(losses), "epoch": epoch})

    return itr


if __name__ == "__main__":
    opt_cmd = options.parse_arguments(sys.argv[1:])
    cfg = options.set(opt_cmd=opt_cmd, verbose=True)

    meta_args = cfg.meta
    train_args = cfg.train
    model_args = cfg.model
    data_args = cfg.data

    log_dir = os.path.join(
        "logs", meta_args.exp_name, datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    )
    os.makedirs(log_dir, exist_ok=True)

    if meta_args.get("wandb", False):
        import wandb

        start_wandb(meta_args, cfg)
        wandb_logging = True

    options.save_options_file(log_dir, cfg)

    mconf = get_model_config(**model_args)
    tconf = get_train_config(**train_args)
    model = dt_model.GPT(mconf)

    optimizer = model.configure_optimizers(tconf)
    if meta_args.get("ckpt"):
        sd = torch.load(meta_args.ckpt, map_location="cpu")
        model.load_state_dict(sd["model_state_dict"])
        optimizer.load_state_dict(sd["optimizer_state_dict"])

    device = torch.device(f'cuda:{meta_args.get("device", 0)}')

    model = model.to(device)

    train_loader = nurikabe_dataset.get_loader(
        data_dir=data_args.train_path,
        batch_size=data_args.batch_size,
        reward_type=model_args.reward_type,
        fix_k=data_args.fix_k,
        max_seq_len=model_args.max_seq_len,
    )

    val_loader = nurikabe_dataset.get_loader(
        data_dir=data_args.val_path,
        batch_size=data_args.batch_size,
        reward_type=model_args.reward_type,
        fix_k=data_args.fix_k,
        max_seq_len=model_args.max_seq_len,
    )

    itr = 0
    for epoch in tqdm(range(tconf.max_epochs), desc="epoch"):
        itr = train(model, optimizer, train_loader, device, itr)

        eval_losses = eval.run_eval(model, val_loader, device)

        if wandb_logging:
            wandb.log({"epoch/val_loss": np.mean(eval_losses)})

        epoch_dir = os.path.join(log_dir, f"e{epoch:05d}")
        os.makedirs(epoch_dir, exist_ok=True)

        val_online_metrics = eval.run_online(
            model,
            val_loader.dataset,
            meta_args.online_n_worlds,
            max_timestep=mconf.max_timestep,
            device=device,
            split="val",
            max_steps_in_env=meta_args.online_max_steps_in_world,
        )

        train_online_metrics = eval.run_online(
            model,
            train_loader.dataset,
            meta_args.online_n_worlds,
            max_timestep=mconf.max_timestep,
            device=device,
            split="train",
            max_steps_in_env=meta_args.online_max_steps_in_world,
        )

        torch.save(
            {
                "val_online_metrics": val_online_metrics,
                "train_online_metrics": train_online_metrics,
            },
            f"{epoch_dir}/metrics.pt",
        )

        if wandb_logging:
            wandb.log(
                {
                    "epoch/online_val_num_correct": np.max(
                        val_online_metrics["num_correct"]
                    ),
                    "epoch": epoch,
                }
            )

            wandb.log(
                {
                    "epoch/online_train_num_correct": np.max(
                        train_online_metrics["num_correct"]
                    ),
                    "epoch": epoch,
                }
            )

            wandb.log(
                {
                    "epoch/online_val": wandb.Image(
                        eval.visualize_eval_metrics(
                            val_online_metrics,
                            return_image=True,
                            save_path=f"{epoch_dir}/online_val.png",
                        )
                    ),
                    "epoch": epoch,
                }
            )

            wandb.log(
                {
                    "epoch/online_train": wandb.Image(
                        eval.visualize_eval_metrics(
                            train_online_metrics,
                            return_image=True,
                            save_path=f"{epoch_dir}/online_train.png",
                        )
                    ),
                    "epoch": epoch,
                }
            )
