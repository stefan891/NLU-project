import copy
import math
import os
import re
from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn
import wandb
from tqdm.auto import tqdm

from functions import eval_loop, init_weights, train_loop
from model import GPT2
from utils import ExperimentConfig, get_dataloaders


def _safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def run_experiment(config: ExperimentConfig, device: torch.device):
    train_loader, dev_loader, test_loader, tokenizer = get_dataloaders(config, device)

    model = GPT2.from_config(len(tokenizer), config).to(device)
    model.apply(init_weights)

    optimizer = config.optim(model.parameters(), lr=config.lr)
    criterion_train = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    criterion_eval = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    wandb_config = asdict(config)
    wandb_config["optim"] = config.optim.__name__
    wandb_config["n_params"] = sum(p.numel() for p in model.parameters())

    run = wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        mode=config.wandb_mode,
        name=config.name,
        config=wandb_config,
        reinit=True,
    )
    wandb.watch(model, log="gradients", log_freq=200)

    best_ppl = math.inf
    best_model = None
    patience = config.patience

    pbar = tqdm(range(config.n_epochs))
    for epoch in pbar:
        train_loss = train_loop(train_loader, optimizer, criterion_train, model)
        ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
        pbar.set_description(f"[{config.name}] PPL dev: {ppl_dev:.2f}")

        wandb.log({
            "epoch": epoch,
            "train/loss": float(train_loss),
            "dev/loss": float(loss_dev),
            "dev/ppl": float(ppl_dev),
            "patience": patience,
        })

        if ppl_dev < best_ppl:
            best_ppl = ppl_dev
            best_model = copy.deepcopy(model).to("cpu")
            patience = config.patience
        else:
            patience -= 1
            if patience <= 0:
                break

    best_model.to(device)
    final_ppl, final_loss = eval_loop(test_loader, criterion_eval, best_model)
    print(f"[{config.name}] Test PPL: {final_ppl:.2f}")

    os.makedirs(config.save_dir, exist_ok=True)
    ckpt_path = os.path.join(config.save_dir, f"{_safe_filename(config.name)}.pt")
    torch.save(
        {
            "state_dict": best_model.state_dict(),
            "config": asdict(config),
            "optim_name": config.optim.__name__,
            "vocab_size": len(tokenizer),
            "best_dev_ppl": float(best_ppl),
            "test_ppl": float(final_ppl),
        },
        ckpt_path,
    )
    print(f"[{config.name}] saved checkpoint → {ckpt_path}")

    wandb.log({
        "test/ppl": float(final_ppl),
        "test/loss": float(final_loss),
        "best_dev_ppl": float(best_ppl),
    })
    wandb.summary["test/ppl"] = float(final_ppl)
    wandb.summary["best_dev_ppl"] = float(best_ppl)
    wandb.summary["checkpoint_path"] = ckpt_path

    if config.log_artifact and config.wandb_mode != "disabled":
        artifact = wandb.Artifact(
            name=f"model-{_safe_filename(config.name)}",
            type="model",
            metadata={
                "best_dev_ppl": float(best_ppl),
                "test_ppl": float(final_ppl),
            },
        )
        artifact.add_file(ckpt_path)
        run.log_artifact(artifact)

    run.finish()

    return {
        "best_dev_ppl": float(best_ppl),
        "test_ppl": float(final_ppl),
        "checkpoint_path": ckpt_path,
    }


def load_checkpoint(path: str, device: torch.device) -> tuple[GPT2, dict]:
    """Reconstruct a GPT2 model from a checkpoint saved by run_experiment."""
    ckpt = torch.load(path, map_location=device)
    cfg_dict = ckpt["config"]
    model = GPT2(
        vocab_size=ckpt["vocab_size"],
        pos_emb_size=cfg_dict["pos_emb_size"],
        d_model=cfg_dict["d_model"],
        n_heads=cfg_dict["n_heads"],
        num_layers=cfg_dict["num_layers"],
        ff_dim=cfg_dict["ff_dim"],
        dropout_embedding=cfg_dict["dropout_embedding"],
        dropout_attn=cfg_dict["dropout_attn"],
        dropout_out_proj=cfg_dict["dropout_out_proj"],
        dropout_ff=cfg_dict["dropout_ff"],
        weight_tying=cfg_dict["weight_tying"],
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    baseline = ExperimentConfig(name="Baseline")
    run_experiment(baseline, device)
