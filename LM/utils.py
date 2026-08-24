from dataclasses import dataclass
from functools import partial
from typing import Type, Union

import torch
import torch.utils.data as data
from torch.optim import SGD, AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


@dataclass
class ExperimentConfig:
    """Configuration for a Part 1.A GPT2 experiment."""

    # bookkeeping
    name: str = "Baseline"

    # training
    lr: float = 1e-3
    n_epochs: int = 100
    patience: int = 3
    batch_size_train: int = 8
    batch_size_eval: int = 16
    optim: Union[Type[SGD], Type[AdamW]] = AdamW

    # model hyperparameters (step 1)
    pos_emb_size: int = 1024
    d_model: int = 20
    n_heads: int = 1
    num_layers: int = 1
    ff_dim: int = 20

    # dropout hyperparameters (step 2) — 0.0 disables the module
    dropout_embedding: float = 0.0    # after token + positional embeddings
    dropout_attn: float = 0.0         # after softmax(attn weights)
    dropout_out_proj: float = 0.0     # after MHA output projection
    dropout_ff: float = 0.0           # after last FF linear

    # step 3
    weight_tying: bool = False

    # wandb
    wandb_project: str = "NLU-LM"
    wandb_entity: str | None = None
    wandb_mode: str = "online"  # "online" | "offline" | "disabled"

    # checkpoints
    save_dir: str = "bin"
    log_artifact: bool = True  # upload best checkpoint as a wandb artifact


def read_file(path, eos_token="<eos>"):
    output = []
    with open(path, "r") as f:
        for line in f.readlines():
            output.append(line.strip() + " " + eos_token)
    return output


class PennTreeBank(data.Dataset):
    def __init__(self, corpus):
        self.sents = [sent for sent in corpus]

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx):
        return self.sents[idx]


def collate_fn(batch, tokenizer, device):
    tokenized = tokenizer(batch, padding=True, return_tensors="pt")

    input_ids = tokenized.input_ids[:, :-1].detach().clone().to(device)
    # labels are the input shifted left -> predict the next token
    labels = tokenized.input_ids[:, 1:].detach().clone().to(device)

    n_tokens = torch.sum(input_ids != tokenizer.pad_token_id)
    return input_ids, labels, n_tokens


def get_dataloaders(
    config: ExperimentConfig,
    device: torch.device,
    data_dir: str = "dataset/PennTreeBank",
):
    train_raw = read_file(f"{data_dir}/ptb.train.txt")
    dev_raw = read_file(f"{data_dir}/ptb.valid.txt")
    test_raw = read_file(f"{data_dir}/ptb.test.txt")

    train_dataset = PennTreeBank(train_raw)
    dev_dataset = PennTreeBank(dev_raw)
    test_dataset = PennTreeBank(test_raw)

    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    collate = partial(collate_fn, tokenizer=tokenizer, device=device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size_train,
        collate_fn=collate,
        shuffle=True,
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=config.batch_size_eval, collate_fn=collate
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size_eval, collate_fn=collate
    )

    return train_loader, dev_loader, test_loader, tokenizer
