"""Headless head-to-head evaluation between two agent modules.

Usage:
    python report/eval_match.py <m1> <m2> [--matches 10] [--max-steps 3000]

Loads two agent modules, runs N full episodes (no rendering), and prints a
JSON dict with wins/losses/draws (from team 0's perspective) plus per-match
rewards.  We only count *decisive* episodes (one team strictly outscores the
other on the *sparse goal* signal); episodes that time out 0--0 are draws.

Notes
-----
- Each invocation runs in its own process (Ray+Unity state is messy to share).
- The CEIA baseline agent prints a lot via Ray; stderr is left untouched so
  its output goes to the calling shell -- the JSON we emit is the *last* line
  on stdout, so the driver script can parse it.
"""
import argparse
import importlib
import json
import os
import sys

import numpy as np
import soccer_twos
from soccer_twos.utils import get_agent_class


GOAL_THRESHOLD = 0.9  # |reward| >= 0.9 in any single step => a goal happened


def play_one_match(env, agent1, agent2, max_steps):
    obs = env.reset()
    t0_reward = 0.0
    t1_reward = 0.0
    t0_goals = 0
    t1_goals = 0
    steps = 0
    while True:
        a1 = agent1.act({0: obs[0], 1: obs[1]})
        a2 = agent2.act({0: obs[2], 1: obs[3]})
        actions = {0: a1[0], 1: a1[1], 2: a2[0], 3: a2[1]}
        obs, reward, done, info = env.step(actions)
        # detect goals via large reward magnitudes
        for pid in (0, 1):
            if reward[pid] >= GOAL_THRESHOLD:
                t0_goals += 1
            elif reward[pid] <= -GOAL_THRESHOLD:
                t1_goals += 1
        for pid in (2, 3):
            if reward[pid] >= GOAL_THRESHOLD:
                t1_goals += 1
            elif reward[pid] <= -GOAL_THRESHOLD:
                t0_goals += 1
        t0_reward += reward[0] + reward[1]
        t1_reward += reward[2] + reward[3]
        steps += 1
        if max(done.values()) or steps >= max_steps:
            break
    # de-dup: each goal triggers reward to all four agents (1 per teammate,
    # -1 per opponent), so we counted each goal 4 times.
    t0_goals //= 4
    t1_goals //= 4
    return {
        "team0_reward": float(t0_reward),
        "team1_reward": float(t1_reward),
        "team0_goals": int(t0_goals),
        "team1_goals": int(t1_goals),
        "steps": int(steps),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("m1")
    p.add_argument("m2")
    p.add_argument("--matches", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=3000)
    p.add_argument("--base-port", type=int, default=None)
    args = p.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    m1 = importlib.import_module(args.m1)
    m2 = importlib.import_module(args.m2)

    env = soccer_twos.make(base_port=args.base_port)
    a1 = get_agent_class(m1)(env)
    a2 = get_agent_class(m2)(env)
    env.close()

    env = soccer_twos.make(
        watch=False,
        base_port=args.base_port,
        blue_team_name=getattr(a1, "name", args.m1),
        orange_team_name=getattr(a2, "name", args.m2),
    )

    matches = []
    wins = losses = draws = 0
    for i in range(args.matches):
        r = play_one_match(env, a1, a2, args.max_steps)
        if r["team0_goals"] > r["team1_goals"]:
            wins += 1
            outcome = "win"
        elif r["team0_goals"] < r["team1_goals"]:
            losses += 1
            outcome = "loss"
        else:
            draws += 1
            outcome = "draw"
        r["outcome"] = outcome
        matches.append(r)
        # progress to stderr so driver can watch
        print(f"[{args.m1} vs {args.m2}] match {i+1}/{args.matches}: "
              f"{r['team0_goals']}-{r['team1_goals']} ({outcome}, {r['steps']} steps)",
              file=sys.stderr, flush=True)

    env.close()

    summary = {
        "m1": args.m1, "m2": args.m2,
        "matches": args.matches,
        "wins": wins, "losses": losses, "draws": draws,
        "details": matches,
    }
    # last line = parseable JSON
    print("@@RESULT@@" + json.dumps(summary))


if __name__ == "__main__":
    main()
