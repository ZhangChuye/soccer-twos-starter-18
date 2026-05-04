"""Generate every training/comparison plot used in the report.

Outputs (under ``report/``):
  - ``training_reward_shaped.png`` -- Agent 1, single run (a6c6d)
  - ``training_selfplay.png``      -- Agent 2, single run (67194)
  - ``training_curriculum.png``    -- Agent 3, full STITCHED resume chain
                                       (c33b8 -> 6eb9a -> 02fdb)
  - ``comparison.png``             -- overlay of the three on env-step x-axis
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(ROOT)

P_REWARD_SHAPED = os.path.join(
    PROJECT, "ray_results/PPO_reward_shaped/"
    "PPO_Soccer_a6c6d_00000_0_2026-03-30_21-19-23/progress.csv")
P_SELFPLAY = os.path.join(
    PROJECT, "ray_results/PPO_SP/"
    "PPO_Soccer_67194_00000_0_2026-03-30_14-51-03/progress.csv")
P_CURR_FULL = os.path.join(ROOT, "curriculum_full.csv")

STAGE_NAMES = {0: "Random", 1: "Hybrid", 2: "Baseline", 3: "Self-play"}

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.titlesize": 14,
})


def smooth(y, w=5):
    y = np.asarray(y, dtype=float)
    if len(y) < w:
        return y
    k = np.ones(w) / w
    pad = w // 2
    yp = np.pad(y, (pad, pad), mode="edge")
    return np.convolve(yp, k, mode="valid")[: len(y)]


# ----------------------------------------------------------------------------
# Agent 1: reward-shaped
# ----------------------------------------------------------------------------
def plot_reward_shaped():
    df = pd.read_csv(P_REWARD_SHAPED).sort_values("training_iteration").reset_index(drop=True)
    ts = df["timesteps_total"] / 1e6

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    fig.suptitle("Agent 1 -- Reward-shaped PPO vs frozen random opponent",
                 fontweight="bold")

    ax = axes[0]
    ax.plot(ts, df["custom_metrics/win_rate_mean"], color="C0", linewidth=1.6)
    ax.fill_between(ts,
                    df["custom_metrics/win_rate_min"],
                    df["custom_metrics/win_rate_max"],
                    color="C0", alpha=0.15)
    ax.axhline(0.9, color="red", linestyle=":", linewidth=1.2, label="0.9")
    ax.set_xlabel("Environment steps (M)")
    ax.set_ylabel("Win rate (blue vs orange)")
    ax.set_title("Win rate")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")

    ax = axes[1]
    ax.plot(ts, df["policy_reward_mean/default"], color="C1", linewidth=1.6,
            label="default (blue)")
    if "policy_reward_mean/opponent" in df.columns:
        ax.plot(ts, df["policy_reward_mean/opponent"], color="C3", linewidth=1.6,
                alpha=0.7, label="opponent (random)")
    ax.set_xlabel("Environment steps (M)")
    ax.set_ylabel("Mean episode reward")
    ax.set_title("Per-policy episode reward")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")

    ax = axes[2]
    ent = df["info/learner/default/learner_stats/entropy"]
    ax.plot(ts, ent, color="C4", linewidth=1.6, label="entropy")
    ax.set_xlabel("Environment steps (M)")
    ax.set_ylabel("Policy entropy", color="C4")
    ax.tick_params(axis="y", labelcolor="C4")
    ax.grid(alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(ts, df["episode_len_mean"], color="C5", linewidth=1.6,
             alpha=0.8, label="ep_len")
    ax2.set_ylabel("Mean episode length", color="C5")
    ax2.tick_params(axis="y", labelcolor="C5")
    ax.set_title("Entropy / episode length")

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(ROOT, "training_reward_shaped.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# ----------------------------------------------------------------------------
# Agent 2: self-play (no shaping)
# ----------------------------------------------------------------------------
def plot_selfplay():
    df = pd.read_csv(P_SELFPLAY).sort_values("training_iteration").reset_index(drop=True)
    ts = df["timesteps_total"] / 1e6

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    fig.suptitle("Agent 2 -- Self-play PPO (no reward shaping)",
                 fontweight="bold")

    ax = axes[0]
    ax.plot(ts, df["episode_reward_mean"], color="C3", linewidth=1.6)
    ax.axhline(0.0, color="grey", linestyle=":", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("Environment steps (M)")
    ax.set_ylabel("Mean episode reward")
    ax.set_title("Episode reward (sparse only)")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ent_col = "info/learner/default_policy/learner_stats/entropy"
    if ent_col in df.columns:
        ax.plot(ts, df[ent_col], color="C4", linewidth=1.6, label="entropy")
        ax.set_ylabel("Policy entropy", color="C4")
        ax.tick_params(axis="y", labelcolor="C4")
    ax.set_xlabel("Environment steps (M)")
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(ts, df["episode_len_mean"], color="C5", linewidth=1.6, alpha=0.8)
    ax2.set_ylabel("Mean episode length", color="C5")
    ax2.tick_params(axis="y", labelcolor="C5")
    ax.set_title("Entropy / episode length")

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(ROOT, "training_selfplay.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# ----------------------------------------------------------------------------
# Agent 3: curriculum (stitched chain)
# ----------------------------------------------------------------------------
def plot_curriculum():
    df = pd.read_csv(P_CURR_FULL)
    ts = df["ts_global"] / 1e6

    seg_changes = []
    prev_seg = None
    for i, s in enumerate(df["segment"]):
        if s != prev_seg:
            seg_changes.append((float(ts.iloc[i]), s))
            prev_seg = s
    stage_changes = []
    prev_stage = None
    for i, s in enumerate(df["custom_metrics/curriculum_stage_max"].fillna(0).astype(int)):
        if s != prev_stage:
            stage_changes.append((float(ts.iloc[i]), int(s)))
            prev_stage = s

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.0))
    fig.suptitle(
        "Agent 3 (Luckets) -- Reward shaping + 4-stage opponent curriculum  "
        "[3-segment resume chain]",
        fontweight="bold")

    # 1. Win rate
    ax = axes[0, 0]
    ax.plot(ts, df["custom_metrics/win_rate_mean"],
            color="C0", linewidth=1.4, label="win_rate_mean")
    ax.fill_between(ts,
                    df["custom_metrics/win_rate_min"],
                    df["custom_metrics/win_rate_max"],
                    color="C0", alpha=0.10, label="min/max")
    ax.axhline(0.80, color="red", linestyle=":", linewidth=1.5,
               label="promotion gate 0.80")
    for x, _ in seg_changes[1:]:
        ax.axvline(x, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.set_ylabel("Win rate (blue vs orange)")
    ax.set_xlabel("Environment steps (M)")
    ax.set_title("Win rate (with stage transitions)")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    for x, stage in stage_changes:
        ax.text(x, 1.02, f"S{stage}", fontsize=9,
                color="dimgray", ha="left", va="bottom")

    # 2. Curriculum stage
    ax = axes[0, 1]
    stage_series = df["custom_metrics/curriculum_stage_max"].fillna(0)
    ax.step(ts, stage_series, where="post", color="C2", linewidth=2.0)
    for x, _ in seg_changes[1:]:
        ax.axvline(x, color="black", linestyle="--", linewidth=1.0, alpha=0.6,
                   label="resume" if "resume" not in [t.get_text() for t in ax.texts] else None)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels([f"{i}\n{STAGE_NAMES[i]}" for i in range(4)])
    ax.set_xlabel("Environment steps (M)")
    ax.set_title("Curriculum stage")
    ax.grid(alpha=0.3)

    # 3. Per-policy episode reward
    ax = axes[1, 0]
    cols = [
        ("policy_reward_mean/default",  "C1", "default (blue)"),
        ("policy_reward_mean/opp_rand", "C7", "opp_rand"),
        ("policy_reward_mean/opp_base", "C3", "opp_base (CEIA)"),
        ("policy_reward_mean/opp_self", "C9", "opp_self"),
    ]
    for col, color, label in cols:
        if col in df.columns and df[col].notna().any():
            ax.plot(ts, df[col], color=color, linewidth=1.3, alpha=0.85, label=label)
    for x, _ in seg_changes[1:]:
        ax.axvline(x, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.set_xlabel("Environment steps (M)")
    ax.set_ylabel("Mean episode reward")
    ax.set_title("Per-policy reward")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")

    # 4. Entropy + episode length
    ax = axes[1, 1]
    ent_col = "info/learner/default/learner_stats/entropy"
    if ent_col in df.columns:
        ax.plot(ts, df[ent_col], color="C4", linewidth=1.4)
        ax.set_ylabel("Policy entropy", color="C4")
        ax.tick_params(axis="y", labelcolor="C4")
    ax.set_xlabel("Environment steps (M)")
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(ts, df["episode_len_mean"], color="C5", linewidth=1.4, alpha=0.85)
    ax2.set_ylabel("Mean episode length", color="C5")
    ax2.tick_params(axis="y", labelcolor="C5")
    for x, _ in seg_changes[1:]:
        ax.axvline(x, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.set_title("Entropy / episode length")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(ROOT, "training_curriculum.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# ----------------------------------------------------------------------------
# Cross-agent comparison
# ----------------------------------------------------------------------------
def plot_comparison():
    df_rs = pd.read_csv(P_REWARD_SHAPED).sort_values("training_iteration").reset_index(drop=True)
    df_sp = pd.read_csv(P_SELFPLAY).sort_values("training_iteration").reset_index(drop=True)
    df_cu = pd.read_csv(P_CURR_FULL)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))

    # Left: episode reward (what RLlib optimises)
    ax = axes[0]
    ax.plot(df_rs["timesteps_total"]/1e6, smooth(df_rs["episode_reward_mean"]),
            color="C0", linewidth=1.7, label="Agent 1: reward-shaped (vs random)")
    ax.plot(df_sp["timesteps_total"]/1e6, smooth(df_sp["episode_reward_mean"]),
            color="C3", linewidth=1.7, label="Agent 2: self-play (no shaping)")
    ax.plot(df_cu["ts_global"]/1e6, smooth(df_cu["episode_reward_mean"]),
            color="C2", linewidth=1.7, label="Agent 3: shaping + curriculum")
    ax.set_xlabel("Environment steps (millions)")
    ax.set_ylabel("Mean episode reward (sparse + shaped)")
    ax.set_title("Training reward")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")

    # Right: win rate vs contemporaneous opponent
    ax = axes[1]
    if "custom_metrics/win_rate_mean" in df_rs.columns:
        ax.plot(df_rs["timesteps_total"]/1e6,
                smooth(df_rs["custom_metrics/win_rate_mean"]),
                color="C0", linewidth=1.7,
                label="Agent 1 (vs random)")
    sp_wr = (df_sp["episode_reward_mean"] > 0).astype(float).rolling(20, min_periods=1).mean()
    ax.plot(df_sp["timesteps_total"]/1e6, sp_wr, color="C3", linewidth=1.7,
            label="Agent 2 (proxy: \\#iters w/ +reward)")
    ax.plot(df_cu["ts_global"]/1e6,
            smooth(df_cu["custom_metrics/win_rate_mean"]),
            color="C2", linewidth=1.7,
            label="Agent 3 (vs curriculum opp.)")
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("Environment steps (millions)")
    ax.set_ylabel("Win rate")
    ax.set_title("Win rate vs.\\ contemporaneous opponent")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")

    fig.tight_layout()
    out = os.path.join(ROOT, "comparison.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def plot_per_agent_combined():
    """Compact 2x2 figure: Agent 1 (top row, win rate + per-policy reward),
    Agent 2 (bottom row, episode reward + entropy)."""
    df_rs = pd.read_csv(P_REWARD_SHAPED).sort_values("training_iteration").reset_index(drop=True)
    df_sp = pd.read_csv(P_SELFPLAY).sort_values("training_iteration").reset_index(drop=True)

    fig, axes = plt.subplots(2, 2, figsize=(11, 5.4))
    fig.suptitle(
        "Agents 1 & 2 -- isolated training traces",
        fontweight="bold")

    ts = df_rs["timesteps_total"]/1e6
    ax = axes[0, 0]
    ax.plot(ts, df_rs["custom_metrics/win_rate_mean"], color="C0", linewidth=1.5)
    ax.fill_between(ts, df_rs["custom_metrics/win_rate_min"],
                    df_rs["custom_metrics/win_rate_max"], color="C0", alpha=0.15)
    ax.axhline(0.9, color="red", linestyle=":", linewidth=1.0)
    ax.set_xlabel("Env steps (M)")
    ax.set_ylabel("Win rate")
    ax.set_title("Agent 1: win rate vs.\\ random")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(ts, df_rs["policy_reward_mean/default"],
            color="C1", linewidth=1.5, label="default (blue)")
    if "policy_reward_mean/opponent" in df_rs.columns:
        ax.plot(ts, df_rs["policy_reward_mean/opponent"],
                color="C3", linewidth=1.5, alpha=0.7, label="opponent (random)")
    ax.set_xlabel("Env steps (M)")
    ax.set_ylabel("Mean episode reward")
    ax.set_title("Agent 1: per-policy reward")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)

    ts = df_sp["timesteps_total"]/1e6
    ax = axes[1, 0]
    ax.plot(ts, df_sp["episode_reward_mean"], color="C3", linewidth=1.5)
    ax.axhline(0.0, color="grey", linestyle=":", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("Env steps (M)")
    ax.set_ylabel("Mean episode reward")
    ax.set_title("Agent 2: sparse-reward episode reward")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ent_col = "info/learner/default_policy/learner_stats/entropy"
    if ent_col in df_sp.columns:
        ax.plot(ts, df_sp[ent_col], color="C4", linewidth=1.5, label="entropy")
        ax.set_ylabel("Policy entropy", color="C4")
        ax.tick_params(axis="y", labelcolor="C4")
    ax.set_xlabel("Env steps (M)")
    ax.set_title("Agent 2: entropy / episode length")
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(ts, df_sp["episode_len_mean"], color="C5", linewidth=1.5, alpha=0.85)
    ax2.set_ylabel("Mean episode length", color="C5")
    ax2.tick_params(axis="y", labelcolor="C5")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(ROOT, "training_per_agent.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    plot_reward_shaped()
    plot_selfplay()
    plot_curriculum()
    plot_comparison()
    plot_per_agent_combined()
