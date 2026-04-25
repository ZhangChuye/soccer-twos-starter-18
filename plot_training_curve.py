"""Generate the training-curve figure shipped with the Luckets agent.

Reads the RLlib ``progress.csv`` produced by ``train_reward_shaped_curriculum.py``
and writes ``Luckets_agent/training_curve.png``.

Lives at the project root (NOT inside ``Luckets_agent/``) so the Gradescope
autograder doesn't pick it up as a second importable agent module.

Usage
-----
    python plot_training_curve.py \
        --progress ray_results/reward_shaped_curriculum/PPO_curriculum/\
PPO_Soccer_02fdb_00000_0_2026-04-22_00-09-09/progress.csv \
        --packaged-iter 600
"""
from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_PROGRESS = (
    "ray_results/reward_shaped_curriculum/PPO_curriculum/"
    "PPO_Soccer_02fdb_00000_0_2026-04-22_00-09-09/progress.csv"
)
DEFAULT_OUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "Luckets_agent",
    "training_curve.png",
)
WIN_RATE_THRESHOLD = 0.80
STAGE_NAMES = {0: "Random", 1: "Hybrid", 2: "Baseline", 3: "Self-play"}


def _stage_change_iters(df: pd.DataFrame) -> list[tuple[int, int]]:
    """Return [(iteration, new_stage), ...] when curriculum_stage changes."""
    stage = df["custom_metrics/curriculum_stage_max"].fillna(0).astype(int)
    changes: list[tuple[int, int]] = []
    prev = None
    for it, s in zip(df["training_iteration"], stage):
        if prev is None or s != prev:
            changes.append((int(it), int(s)))
            prev = s
    return changes


def _annotate_stages(ax, stage_changes, ymax_frac=0.95):
    ymin, ymax = ax.get_ylim()
    y_label = ymin + (ymax - ymin) * ymax_frac
    for it, stage in stage_changes:
        ax.axvline(it, color="grey", linestyle="--", linewidth=1.0, alpha=0.6)
        ax.text(
            it,
            y_label,
            f" S{stage} {STAGE_NAMES[stage]}",
            rotation=90,
            verticalalignment="top",
            fontsize=12,
            color="dimgray",
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--progress", default=DEFAULT_PROGRESS)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument(
        "--packaged-iter",
        type=int,
        default=600,
        help="Iteration of the checkpoint shipped in the submission.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.progress)
    df = df.sort_values("training_iteration").reset_index(drop=True)

    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "figure.titlesize": 22,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    fig.suptitle(
        "Luckets Agent — PPO + reward shaping + curriculum",
        fontweight="bold",
    )

    iters = df["training_iteration"]
    stage_changes = _stage_change_iters(df)
    packaged_iter = args.packaged_iter

    # 1. Win rate
    ax = axes[0, 0]
    ax.plot(iters, df["custom_metrics/win_rate_mean"], color="C0", linewidth=2,
            label="win_rate_mean")
    ax.fill_between(
        iters,
        df["custom_metrics/win_rate_min"],
        df["custom_metrics/win_rate_max"],
        color="C0", alpha=0.15, label="min/max",
    )
    ax.axhline(WIN_RATE_THRESHOLD, color="red", linestyle=":", linewidth=2,
               label=f"promotion gate ({WIN_RATE_THRESHOLD:.2f})")
    ax.axvline(packaged_iter, color="black", linestyle="-", linewidth=1.5,
               alpha=0.6, label=f"packaged ckpt (iter {packaged_iter})")
    ax.set_ylabel("Win rate (blue vs orange)")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    _annotate_stages(ax, stage_changes)

    # 2. Curriculum stage
    ax = axes[0, 1]
    ax.step(
        iters,
        df["custom_metrics/curriculum_stage_max"].fillna(0),
        where="post", color="C2", linewidth=2.5,
    )
    ax.axvline(packaged_iter, color="black", linestyle="-", linewidth=1.5, alpha=0.6)
    ax.set_ylabel("Curriculum stage")
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels([f"{i}\n{STAGE_NAMES[i]}" for i in range(4)])
    ax.grid(alpha=0.3)

    # 3. Episode reward (default policy)
    ax = axes[1, 0]
    ax.plot(iters, df["policy_reward_mean/default"], color="C1", linewidth=2,
            label="default (blue)")
    if "policy_reward_mean/opp_base" in df.columns:
        ax.plot(
            iters, df["policy_reward_mean/opp_base"],
            color="C3", linewidth=1.5, alpha=0.7, label="opp_base (orange)",
        )
    ax.axvline(packaged_iter, color="black", linestyle="-", linewidth=1.5, alpha=0.6)
    ax.set_ylabel("Mean episode reward (per policy)")
    ax.set_xlabel("Training iteration")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    _annotate_stages(ax, stage_changes)

    # 4. Entropy + episode length (twin axis)
    ax = axes[1, 1]
    ax.plot(
        iters,
        df["info/learner/default/learner_stats/entropy"],
        color="C4", linewidth=2, label="policy entropy",
    )
    ax.set_ylabel("Policy entropy", color="C4")
    ax.tick_params(axis="y", labelcolor="C4")
    ax.set_xlabel("Training iteration")
    ax.grid(alpha=0.3)

    ax_r = ax.twinx()
    ax_r.plot(iters, df["episode_len_mean"], color="C5", linewidth=2,
              alpha=0.8, label="episode length")
    ax_r.set_ylabel("Mean episode length", color="C5")
    ax_r.tick_params(axis="y", labelcolor="C5")

    ax.axvline(packaged_iter, color="black", linestyle="-", linewidth=1.5, alpha=0.6)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"Saved {args.out}")
    print(f"Stage transitions (iter, stage): {stage_changes}")


if __name__ == "__main__":
    main()
