"""
Copy a Ray checkpoint into an agent package directory.

Usage:
    python package_agent.py <checkpoint_path> [agent_dir]

Example:
    python package_agent.py ./ray_results/PPO_reward_shaped/PPO_Soccer_XXXXX/checkpoint_000100/checkpoint-100 reward_shaped_agent
"""
import os
import sys
import shutil


def main():
    if len(sys.argv) < 2:
        print("Usage: python package_agent.py <checkpoint_path> [agent_dir]")
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    agent_dir = sys.argv[2] if len(sys.argv) > 2 else "reward_shaped_agent"

    if not os.path.isfile(checkpoint_path):
        print(f"Error: checkpoint file not found at {checkpoint_path}")
        sys.exit(1)

    checkpoint_src_dir = os.path.dirname(checkpoint_path)
    trial_dir = os.path.dirname(checkpoint_src_dir)
    params_path = os.path.join(trial_dir, "params.pkl")

    dest_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), agent_dir, "checkpoint"
    )
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    os.makedirs(dest_dir)

    for fname in os.listdir(checkpoint_src_dir):
        src = os.path.join(checkpoint_src_dir, fname)
        if os.path.isfile(src):
            dst = os.path.join(dest_dir, fname)
            print(f"  {src} -> {dst}")
            shutil.copy2(src, dst)

    if os.path.isfile(params_path):
        dst = os.path.join(dest_dir, "params.pkl")
        print(f"  {params_path} -> {dst}")
        shutil.copy2(params_path, dst)
    else:
        print(f"  Warning: params.pkl not found at {params_path}")

    print(f"\nPackaged into: {dest_dir}")
    print(f"Test with:  python -m soccer_twos.watch -m1 {agent_dir} -m2 example_player_agent")


if __name__ == "__main__":
    main()
