"""
PPO Training with Dense Reward Shaping — Simplified.

Both blue-team agents use the learned "default" policy.
Both orange-team agents use a frozen random-weight policy (fixed opponent).
No self-play ladder updates — the agent first needs to learn basic ball
interaction against a static opponent.

RewardShaperWrapper adds: ball-proximity-delta + ball-to-goal-progress.
"""
import numpy as np
import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from utils import create_rllib_env


NUM_ENVS_PER_WORKER = 3


def policy_mapping_fn(agent_id, *args, **kwargs):
    """Blue team (0,1) = learner, Orange team (2,3) = frozen random."""
    if agent_id in [0, 1]:
        return "default"
    return "opponent"


class WinRateCallback(DefaultCallbacks):
    """Logs win/loss/draw from the sparse Unity reward at episode end."""

    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        raw_rewards = {}
        for (agent_id, _), r in episode.agent_rewards.items():
            raw_rewards[agent_id] = raw_rewards.get(agent_id, 0.0) + r

        blue = raw_rewards.get(0, 0.0) + raw_rewards.get(1, 0.0)
        orange = raw_rewards.get(2, 0.0) + raw_rewards.get(3, 0.0)

        if blue > orange:
            episode.custom_metrics["win_rate"] = 1.0
        elif blue < orange:
            episode.custom_metrics["win_rate"] = 0.0
        else:
            episode.custom_metrics["win_rate"] = 0.5


if __name__ == "__main__":
    ray.init(include_dashboard=False)

    tune.registry.register_env("Soccer", create_rllib_env)
    temp_env = create_rllib_env({"reward_shaping": True})
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    analysis = tune.run(
        "PPO",
        name="PPO_reward_shaped",
        config={
            "num_gpus": 0,
            "num_workers": 4,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level": "INFO",
            "framework": "torch",
            "callbacks": WinRateCallback,
            "multiagent": {
                "policies": {
                    "default": (None, obs_space, act_space, {}),
                    "opponent": (None, obs_space, act_space, {}),
                },
                "policy_mapping_fn": tune.function(policy_mapping_fn),
                "policies_to_train": ["default"],
            },
            "env": "Soccer",
            "env_config": {
                "num_envs_per_worker": NUM_ENVS_PER_WORKER,
                "reward_shaping": True,
            },
            "model": {
                "vf_share_layers": True,
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
            },
            "rollout_fragment_length": 1000,
            "train_batch_size": 12000,
            "sgd_minibatch_size": 512,
            "num_sgd_iter": 10,
            "lr": 3e-4,
            "lambda": 0.95,
            "gamma": 0.995,
            "clip_param": 0.2,
            "entropy_coeff": 0.01,
            "vf_loss_coeff": 0.5,
            "batch_mode": "complete_episodes",
        },
        stop={
            "timesteps_total": 10_000_000,
            "time_total_s": 14400,
        },
        checkpoint_freq=50,
        checkpoint_at_end=True,
        local_dir="./ray_results",
    )

    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    print("Best trial:", best_trial)
    best_checkpoint = analysis.get_best_checkpoint(
        trial=best_trial, metric="episode_reward_mean", mode="max"
    )
    print("Best checkpoint:", best_checkpoint)
    print("Done training")
