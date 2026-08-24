"""Incremental sweep for Part 1.A.

The exercise says "add one modification at a time, keep it only if it helps".
We express that as a sequence of steps. Each step:
  1. takes the current best config,
  2. builds a list of variants (one field changed),
  3. runs them all, and
  4. either keeps the winning variant as the new base (if it beats the current
     best dev PPL) or discards the step.

Selection uses dev PPL; test PPL is only reported for the final run.
Every run is logged to wandb under `config.wandb_project`, so you can compare
them in the UI as well.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from typing import Callable

import numpy as np
import torch

from main import run_experiment
from utils import ExperimentConfig


Variant = Callable[[ExperimentConfig], list[ExperimentConfig]]


def lr_search(base: ExperimentConfig) -> list[ExperimentConfig]:
    return [
        replace(base, name=f"step0_lr={lr:g}", lr=lr)
        for lr in [5e-4, 1e-3, 3e-3, 1e-2]
    ]


def d_model_search(base: ExperimentConfig) -> list[ExperimentConfig]:
    # keep n_heads=1 so any d_model is divisible
    return [
        replace(base, name=f"step1_d_model={d}", d_model=d, ff_dim=4 * d)
        for d in [64, 128, 256]
    ]


def n_heads_search(base: ExperimentConfig) -> list[ExperimentConfig]:
    # only heads that divide the current d_model
    return [
        replace(base, name=f"step1_n_heads={h}", n_heads=h)
        for h in [2, 4, 8]
        if base.d_model % h == 0
    ]


def num_layers_search(base: ExperimentConfig) -> list[ExperimentConfig]:
    return [
        replace(base, name=f"step1_num_layers={n}", num_layers=n)
        for n in [2, 4, 6]
    ]


def ff_dim_search(base: ExperimentConfig) -> list[ExperimentConfig]:
    return [
        replace(base, name=f"step1_ff_dim={ff}", ff_dim=ff)
        for ff in [2 * base.d_model, 4 * base.d_model, 8 * base.d_model]
    ]


def dropout_embedding_search(base: ExperimentConfig) -> list[ExperimentConfig]:
    return [
        replace(base, name=f"step2_dropout_emb={p}", dropout_embedding=p)
        for p in [0.1, 0.2, 0.3]
    ]


def dropout_attn_search(base: ExperimentConfig) -> list[ExperimentConfig]:
    return [
        replace(base, name=f"step2_dropout_attn={p}", dropout_attn=p)
        for p in [0.1, 0.2]
    ]


def dropout_out_proj_search(base: ExperimentConfig) -> list[ExperimentConfig]:
    return [
        replace(base, name=f"step2_dropout_out_proj={p}", dropout_out_proj=p)
        for p in [0.1, 0.2]
    ]


def dropout_ff_search(base: ExperimentConfig) -> list[ExperimentConfig]:
    return [
        replace(base, name=f"step2_dropout_ff={p}", dropout_ff=p)
        for p in [0.1, 0.2]
    ]


def weight_tying_step(base: ExperimentConfig) -> list[ExperimentConfig]:
    return [replace(base, name="step3_weight_tying", weight_tying=True)]


STEPS: list[tuple[str, Variant]] = [
    ("Step 0 — learning rate", lr_search),
    ("Step 1a — d_model", d_model_search),
    ("Step 1b — n_heads", n_heads_search),
    ("Step 1c — num_layers", num_layers_search),
    ("Step 1d — ff_dim", ff_dim_search),
    ("Step 2a — dropout after embeddings", dropout_embedding_search),
    ("Step 2b — dropout on attention weights", dropout_attn_search),
    ("Step 2c — dropout after MHA out proj", dropout_out_proj_search),
    ("Step 2d — dropout after FF", dropout_ff_search),
    ("Step 3 — weight tying", weight_tying_step),
]


def run_step(
    label: str,
    variants: list[ExperimentConfig],
    base: ExperimentConfig,
    base_dev_ppl: float,
    device: torch.device,
) -> tuple[ExperimentConfig, float, list[dict]]:
    """Run every variant. Return (new_base, new_dev_ppl, results table)."""
    print("\n" + "=" * 70)
    print(f"{label}  |  base dev PPL to beat: {base_dev_ppl:.2f}")
    print("=" * 70)

    results = []
    for cfg in variants:
        metrics = run_experiment(cfg, device)
        results.append({"name": cfg.name, **metrics, "config": cfg})

    results.sort(key=lambda r: r["best_dev_ppl"])
    winner = results[0]

    print(f"\n{label} — variants (sorted by dev PPL):")
    for r in results:
        print(f"  {r['name']:<40}  dev={r['best_dev_ppl']:.2f}  test={r['test_ppl']:.2f}")

    if winner["best_dev_ppl"] < base_dev_ppl:
        print(f"  → keeping {winner['name']} (improved: {base_dev_ppl:.2f} → {winner['best_dev_ppl']:.2f})")
        return winner["config"], winner["best_dev_ppl"], results
    print(f"  → discarding step, no variant beat baseline dev PPL")
    return base, base_dev_ppl, results


def run_baseline(device: torch.device) -> tuple[ExperimentConfig, float]:
    base = ExperimentConfig(name="baseline")
    print("Running initial baseline before step 0...")
    metrics = run_experiment(base, device)
    return base, metrics["best_dev_ppl"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        nargs="+",
        type=int,
        help="Run only the given step indices (0-based). Default: all steps.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    base, base_ppl = run_baseline(device)
    print(f"Baseline dev PPL: {base_ppl:.2f}")

    step_indices = args.only if args.only is not None else range(len(STEPS))

    full_log = []
    for i in step_indices:
        label, variant_fn = STEPS[i]
        variants = variant_fn(base)
        if not variants:
            print(f"[{label}] no variants applicable, skipping.")
            continue
        base, base_ppl, results = run_step(label, variants, base, base_ppl, device)
        full_log.extend(results)

    print("\n" + "=" * 70)
    print("FINAL BEST CONFIG")
    print("=" * 70)
    for f, v in base.__dict__.items():
        print(f"  {f:<22} = {v}")
    print(f"\nBest dev PPL: {base_ppl:.2f}")


if __name__ == "__main__":
    main()
