"""Build the cross-agent comparison plot for the report.

Three runs:
  - reward-shaped vs. frozen random (Agent 1)
  - pure self-play, no shaping (Agent 2)
  - reward-shaped + 4-stage curriculum (Agent 3 / Luckets)

Outputs ``report/comparison.png``.
"""
import os
import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(ROOT)

RUNS = {
    "Agent 1: reward-shaped (vs random)":
        os.path.join(PROJECT, "ray_results/PPO_reward_shaped/"
                     "PPO_Soccer_a6c6d_00000_0_2026-03-30_21-19-23/progress.csv"),
    "Agent 2: self-play (no shaping)":
        os.path.join(PROJECT, "ray_results/PPO_SP/"
                     "PPO_Soccer_67194_00000_0_2026-03-30_14-51-03/progress.csv"),
    "Agent 3: shaping + curriculum (Luckets)":
        os.path.join(PROJECT, "ray_results/reward_shaped_curriculum/PPO_curriculum/"
                     "PPO_Soccer_02fdb_00000_0_2026-04-22_00-09-09/progress.csv"),
}

COLORS = {
    "Agent 1: reward-shaped (vs random)":         "C0",
    "Agent 2: self-play (no shaping)":            "C3",
    "Agent 3: shaping + curriculum (Luckets)":    "C2",
}

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
})

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

for label, path in RUNS.items():
    if not os.path.isfile(path):
        print("missing:", path)
        continue
    df = pd.read_csv(path).sort_values("training_iteration").reset_index(drop=True)
    ts_m = df["timesteps_total"].astype(float) / 1e6
    color = COLORS[label]

    # Left: episode_reward_mean (sparse + shaped, what RLlib optimises)
    axes[0].plot(ts_m, df["episode_reward_mean"], label=label,
                 color=color, linewidth=1.8)

    # Right: win-rate where available; for self-play use sign of episode_reward
    if "custom_metrics/win_rate_mean" in df.columns:
        wr = df["custom_metrics/win_rate_mean"]
    else:
        # selfplay run has no win-rate metric — derive a proxy from
        # episode_reward_mean: >0 means the trained policy beats its frozen
        # opponents on average.  Smooth with rolling mean.
        wr = (df["episode_reward_mean"] > 0).astype(float).rolling(20, min_periods=1).mean()
    axes[1].plot(ts_m, wr, label=label, color=color, linewidth=1.8)

axes[0].set_xlabel("Environment steps (millions)")
axes[0].set_ylabel("Mean episode reward (sparse + shaped)")
axes[0].set_title("Training reward")
axes[0].grid(alpha=0.3)
axes[0].legend(loc="lower right")

axes[1].set_xlabel("Environment steps (millions)")
axes[1].set_ylabel("Win rate (blue vs orange)")
axes[1].set_title("Win rate vs.\\ contemporaneous opponent")
axes[1].set_ylim(-0.02, 1.02)
axes[1].axhline(0.5, color="grey", linestyle=":", linewidth=1.0, alpha=0.7)
axes[1].grid(alpha=0.3)
axes[1].legend(loc="lower right")

fig.tight_layout()
out = os.path.join(ROOT, "comparison.png")
fig.savefig(out, dpi=160, bbox_inches="tight")
print("Wrote", out)
