"""Generate training-curve figures for the report.

Outputs (under ``report/``):
  fig_training_all.png  -- 1x4 panel: Phase 1 sparse self-play, Phase 2 reward-shaping
                            vs random, Phase 3 (failed) curriculum v1, Phase 4 LUCKETS
                            final 3-segment curriculum.
  fig_comparison.png    -- single overlay of the three trained agents on win-rate axes.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(ROOT)

# ─── Run paths and identifiers ───────────────────────────────────────────────
RUN_SPARSE_SP = ("PPO_SP/PPO_Soccer_67194_00000_0_2026-03-30_14-51-03",
                 "67194 (Mar 30 14:51, 2.4M steps, sparse PPO + 3 frozen-snapshot opponents)")
RUN_SHAPED    = ("PPO_reward_shaped/PPO_Soccer_a6c6d_00000_0_2026-03-30_21-19-23",
                 "a6c6d (Mar 30 21:19, 10.0M steps, RewardShaperWrapper, frozen random opponent)")
RUN_CURR_V1   = ("PPO_curriculum/PPO_Soccer_7b8e4_00000_0_2026-04-01_19-21-16",
                 "7b8e4 (Apr 1 19:21, 15.0M steps, curriculum v1 — single-opponent design)")
RUN_LUCKETS   = ("reward_shaped_curriculum/PPO_curriculum (3-segment chain)",
                 "c33b8 + 6eb9a + 02fdb (Apr 21–22, 35.8M steps, redesigned 4-stage curriculum)")

P_SPARSE_SP = os.path.join(PROJECT, "ray_results", RUN_SPARSE_SP[0], "progress.csv")
P_SHAPED    = os.path.join(PROJECT, "ray_results", RUN_SHAPED[0],    "progress.csv")
P_CURR_V1   = os.path.join(PROJECT, "ray_results", RUN_CURR_V1[0],   "progress.csv")
P_LUCKETS   = os.path.join(ROOT, "curriculum_full.csv")

STAGE_NAMES  = {0: "S0\nRandom", 1: "S1\nHybrid", 2: "S2\nBaseline", 3: "S3\nSelf-play"}
STAGE_COLORS = {0: "#d4f1d4", 1: "#fffac8", 2: "#ffddb3", 3: "#cde8fb"}

plt.rcParams.update({
    "font.size":         10,
    "axes.titlesize":    10.5,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   8.5,
    "figure.dpi":        150,
})


def smooth(y, w=7):
    y = np.asarray(y, dtype=float)
    if len(y) < w:
        return y
    k = np.ones(w) / w
    yp = np.pad(y, (w // 2, w // 2), mode="edge")
    return np.convolve(yp, k, mode="valid")[: len(y)]


def add_run_id(ax, text):
    """No-op: run identifiers are listed in the figure caption to keep panels clean."""
    return


# ─── Figure 1: 2×2 training-curve summary ────────────────────────────────────
def fig_training_all():
    fig, axes2d = plt.subplots(2, 2, figsize=(12.0, 6.4))
    axes = axes2d.flatten()

    # ── (a) Phase 1 — sparse self-play ───────────────────────────────────────
    df = pd.read_csv(P_SPARSE_SP).sort_values("training_iteration").reset_index(drop=True)
    ts = df["timesteps_total"] / 1e6
    ax = axes[0]
    ax.plot(ts, smooth(df["episode_reward_mean"], w=11),
            color="#c0392b", linewidth=1.7)
    ax.axhline(0.0, color="grey", linestyle=":", linewidth=0.9, alpha=0.7)
    ax.set_xlabel("Environment steps (M)")
    ax.set_ylabel("Mean episode reward")
    ax.set_title("(a) Phase 1: sparse PPO self-play\n(Agent 2 — no shaping, no curriculum)")
    ax.set_xlim(0, ts.max() * 1.02)
    ax.grid(alpha=0.3)
    ax.text(0.5, 0.92, "Reward never crosses 0 — policy collapses",
            transform=ax.transAxes, ha="center", fontsize=8, color="#c0392b",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.9))

    # ── (b) Phase 2 — reward-shaping vs frozen random ─────────────────────────
    df = pd.read_csv(P_SHAPED).sort_values("training_iteration").reset_index(drop=True)
    ts = df["timesteps_total"] / 1e6
    wr = df["custom_metrics/win_rate_mean"]
    ax = axes[1]
    ax.plot(ts, smooth(wr), color="#2166ac", linewidth=1.8)
    ax.fill_between(ts,
                    smooth(df["custom_metrics/win_rate_min"]),
                    smooth(df["custom_metrics/win_rate_max"]),
                    color="#2166ac", alpha=0.12)
    ax.axhline(0.9, color="red", linestyle=":", linewidth=1.0, label="0.9 threshold")
    ax.set_xlabel("Environment steps (M)")
    ax.set_ylabel("Win rate (vs. random)")
    ax.set_title("(b) Phase 2: reward-shaping vs.\\ random\n(Agent 1)")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(0, ts.max() * 1.02)
    ax.grid(alpha=0.3)
    first90 = ts[wr >= 0.90].iloc[0] if (wr >= 0.90).any() else None
    if first90 is not None:
        ax.annotate(f"≥0.9 @ {first90:.2f}M",
                    xy=(first90, 0.90), xytext=(first90 + 1.2, 0.55),
                    arrowprops=dict(arrowstyle="->", color="red", lw=0.9),
                    fontsize=7.5, color="red")
    ax.legend(loc="lower right", fontsize=8)

    # ── (c) Phase 3 — curriculum v1 (FAILED) ─────────────────────────────────
    df = pd.read_csv(P_CURR_V1).sort_values("training_iteration").reset_index(drop=True)
    ts = df["timesteps_total"] / 1e6
    wr = df["custom_metrics/win_rate_mean"]
    stage = df["custom_metrics/curriculum_stage_max"].fillna(0)
    ax = axes[2]
    ax.plot(ts, smooth(wr), color="#e67e22", linewidth=1.8, label="win rate")
    ax.axhline(0.80, color="red", linestyle=":", linewidth=1.0, alpha=0.8,
               label="0.8 promote gate")
    ax.set_xlabel("Environment steps (M)")
    ax.set_ylabel("Win rate (vs. \\texttt{opponent})")
    ax.set_title("(c) Phase 3: curriculum v1 — FAILED\n(single-opponent design)")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(0, ts.max() * 1.02)
    ax.grid(alpha=0.3)
    # twin axis: stage stuck at 0
    ax2 = ax.twinx()
    ax2.step(ts, stage, where="post", color="#7f8c8d", linewidth=1.2,
             linestyle="--", label="curriculum stage")
    ax2.set_ylim(-0.2, 3.2)
    ax2.set_yticks([0, 1, 2, 3])
    ax2.set_ylabel("Curriculum stage", color="#7f8c8d")
    ax2.tick_params(axis="y", labelcolor="#7f8c8d")
    ax.text(0.5, 0.20,
            "Win rate $\\to$ 1, but stage stuck at 0:\nopponent weights never swapped",
            transform=ax.transAxes, ha="center", fontsize=8, color="#34495e",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9))
    ax.legend(loc="lower right", fontsize=8)

    # ── (d) Phase 4 — LUCKETS final, redesigned curriculum ───────────────────
    dfc = pd.read_csv(P_LUCKETS)
    ts = dfc["ts_global"] / 1e6
    wr = dfc["custom_metrics/win_rate_mean"]
    stages = dfc["custom_metrics/curriculum_stage_max"].fillna(0).astype(int)
    ax = axes[3]
    # colored stage backgrounds
    prev_s, prev_x = stages.iloc[0], ts.iloc[0]
    for i in range(1, len(ts)):
        s = stages.iloc[i]
        if s != prev_s or i == len(ts) - 1:
            x_end = ts.iloc[i]
            ax.axvspan(prev_x, x_end, color=STAGE_COLORS[prev_s], alpha=0.55, zorder=0)
            mid = (prev_x + x_end) / 2
            ax.text(mid, 1.04, STAGE_NAMES[prev_s], ha="center", va="bottom",
                    fontsize=7.0, color="dimgray",
                    transform=ax.get_xaxis_transform())
            prev_s, prev_x = s, ts.iloc[i]
    ax.plot(ts, smooth(wr, w=11), color="#1a9850", linewidth=1.7, zorder=3)
    ax.fill_between(ts,
                    smooth(dfc["custom_metrics/win_rate_min"].fillna(0), w=11),
                    smooth(dfc["custom_metrics/win_rate_max"].fillna(1), w=11),
                    color="#1a9850", alpha=0.10, zorder=2)
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.9, alpha=0.7,
               label="0.5 (self-play ceiling)")
    ax.axhline(0.80, color="red", linestyle=":", linewidth=1.0,
               label="0.8 promote gate")
    for boundary in (15.0, 20.8):
        ax.axvline(boundary, color="black", linestyle="--",
                   linewidth=0.9, alpha=0.5)
    ax.set_xlabel("Environment steps (M, stitched)")
    ax.set_ylabel("Win rate (vs.\\ current opponent)")
    ax.set_title("(d) Phase 4: redesigned curriculum\n(LUCKETS-AGENT — submitted)")
    ax.set_ylim(-0.02, 1.10)
    ax.set_xlim(0, ts.max() * 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)

    fig.tight_layout()
    out = os.path.join(ROOT, "fig_training_all.png")
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# ─── Figure 2: comparison overlay (kept for completeness, unused in v2) ──────
def fig_comparison():
    df1 = pd.read_csv(P_SHAPED).sort_values("training_iteration").reset_index(drop=True)
    df2 = pd.read_csv(P_SPARSE_SP).sort_values("training_iteration").reset_index(drop=True)
    dfc = pd.read_csv(P_LUCKETS)

    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    ax.plot(df1["timesteps_total"]/1e6, smooth(df1["custom_metrics/win_rate_mean"]),
            color="#2166ac", linewidth=1.8, label="Agent 1: shaping vs.\\ random")
    wr2 = (df2["episode_reward_mean"] > 0).astype(float).rolling(20, min_periods=1).mean()
    ax.plot(df2["timesteps_total"]/1e6, wr2,
            color="#c0392b", linewidth=1.8, linestyle="--",
            label="Agent 2: sparse self-play (proxy)")
    ax.plot(dfc["ts_global"]/1e6, smooth(dfc["custom_metrics/win_rate_mean"], w=11),
            color="#1a9850", linewidth=1.8, label="Agent 3 / LUCKETS: shaping + curriculum")
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.9, alpha=0.6)
    ax.set_xlabel("Environment steps (M)")
    ax.set_ylabel("Win rate (vs.\\ contemporaneous opponent)")
    ax.set_title("Win-rate comparison")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="center right", fontsize=8.5)
    fig.tight_layout()
    out = os.path.join(ROOT, "fig_comparison.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    fig_training_all()
    fig_comparison()
