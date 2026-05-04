"""Stitch the three curriculum runs (c33b8 -> 6eb9a -> 02fdb) into one
continuous trace.  Each resume restarts ``training_iteration`` and
``timesteps_total`` at 0/24k, so we offset by the cumulative end of the
previous segment.

Outputs ``report/curriculum_full.csv`` and a per-segment summary.
"""
import os
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(ROOT)

CHAIN = [
    ("seg1_random_to_hybrid",
     "ray_results/reward_shaped_curriculum/PPO_curriculum/"
     "PPO_Soccer_c33b8_00000_0_2026-04-21_15-03-20/progress.csv"),
    ("seg2_hybrid_to_baseline",
     "ray_results/reward_shaped_curriculum/PPO_curriculum/"
     "PPO_Soccer_6eb9a_00000_0_2026-04-21_22-24-47/progress.csv"),
    ("seg3_baseline_to_selfplay",
     "ray_results/reward_shaped_curriculum/PPO_curriculum/"
     "PPO_Soccer_02fdb_00000_0_2026-04-22_00-09-09/progress.csv"),
]

dfs = []
iter_offset = 0
ts_offset = 0
print("segment                       rows  iter_local  ts_local         iter_global  ts_global")
for label, rel in CHAIN:
    p = os.path.join(PROJECT, rel)
    df = pd.read_csv(p).sort_values("training_iteration").reset_index(drop=True)
    df["segment"] = label
    df["iter_global"] = df["training_iteration"] + iter_offset
    df["ts_global"]   = df["timesteps_total"]    + ts_offset
    print(f"{label:<28s}  {len(df):4d}  "
          f"{int(df['training_iteration'].iloc[0]):>4d}->{int(df['training_iteration'].iloc[-1]):>4d}  "
          f"{int(df['timesteps_total'].iloc[0]):>10,}->{int(df['timesteps_total'].iloc[-1]):>10,}  "
          f"{int(df['iter_global'].iloc[0]):>4d}->{int(df['iter_global'].iloc[-1]):>4d}  "
          f"{int(df['ts_global'].iloc[0]):>11,}->{int(df['ts_global'].iloc[-1]):>11,}")
    dfs.append(df)
    iter_offset = int(df["iter_global"].iloc[-1])
    ts_offset   = int(df["ts_global"].iloc[-1])

full = pd.concat(dfs, ignore_index=True)
out = os.path.join(ROOT, "curriculum_full.csv")
full.to_csv(out, index=False)
print(f"\nWrote {out} -- {len(full)} rows, {iter_offset} total iters, {ts_offset:,} total steps")
