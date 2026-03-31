from random import uniform as randfloat

import gym
import numpy as np
from ray.rllib import MultiAgentEnv
import soccer_twos


class RLLibWrapper(gym.core.Wrapper, MultiAgentEnv):
    """
    A RLLib wrapper so our env can inherit from MultiAgentEnv.
    """

    pass


class RewardShaperWrapper(gym.core.Wrapper):
    """
    Simplified dense reward shaping.

    Only two signals that the agent can actually observe:
      1. Ball proximity delta: reward for closing distance to ball
      2. Ball-to-goal progress: reward for the ball moving toward opponent goal

    Note: ball_touched is NOT available in the info dict from this Unity binary,
    so we cannot use a strike/touch reward.
    """

    PROXIMITY_DELTA_COEFF = 0.02
    BALL_PROGRESS_COEFF = 0.05

    def __init__(self, env):
        super().__init__(env)
        self._prev_ball_x = None
        self._prev_dist = {}

    def reset(self):
        obs = self.env.reset()
        self._prev_ball_x = None
        self._prev_dist = {}
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        if not isinstance(reward, dict):
            return obs, reward, done, info

        any_id = next(iter(reward.keys()))
        if any_id not in info or "ball_info" not in info.get(any_id, {}):
            return obs, reward, done, info

        ball_pos = info[any_id]["ball_info"]["position"]
        ball_x = ball_pos[0]

        for pid in reward:
            if pid not in info or "player_info" not in info[pid]:
                continue

            p_pos = info[pid]["player_info"]["position"]
            dist = np.sqrt((p_pos[0] - ball_pos[0])**2 + (p_pos[1] - ball_pos[1])**2)

            shaped = 0.0

            if pid in self._prev_dist:
                shaped += (self._prev_dist[pid] - dist) * self.PROXIMITY_DELTA_COEFF
            self._prev_dist[pid] = dist

            if self._prev_ball_x is not None:
                direction = 1.0 if pid < 2 else -1.0
                shaped += (ball_x - self._prev_ball_x) * direction * self.BALL_PROGRESS_COEFF

            reward[pid] += shaped

        self._prev_ball_x = ball_x
        return obs, reward, done, info


def create_rllib_env(env_config: dict = {}):
    """
    Creates a RLLib environment and prepares it to be instantiated by Ray workers.
    Args:
        env_config: configuration for the environment.
            You may specify the following keys:
            - variation: one of soccer_twos.EnvType. Defaults to EnvType.multiagent_player.
            - opponent_policy: a Callable for your agent to train against. Defaults to a random policy.
            - reward_shaping: if True, wraps env with RewardShaperWrapper.
    """
    if hasattr(env_config, "worker_index"):
        env_config["worker_id"] = (
            env_config.worker_index * env_config.get("num_envs_per_worker", 1)
            + env_config.vector_index
        )
    use_reward_shaping = env_config.get("reward_shaping", False)
    make_kwargs = {k: v for k, v in env_config.items() if k != "reward_shaping"}
    env = soccer_twos.make(**make_kwargs)
    if use_reward_shaping:
        env = RewardShaperWrapper(env)
    if "multiagent" in env_config and not env_config["multiagent"]:
        return env
    return RLLibWrapper(env)


def sample_vec(range_dict):
    return [
        randfloat(range_dict["x"][0], range_dict["x"][1]),
        randfloat(range_dict["y"][0], range_dict["y"][1]),
    ]


def sample_val(range_tpl):
    return randfloat(range_tpl[0], range_tpl[1])


def sample_pos_vel(range_dict):
    _s = {}
    if "position" in range_dict:
        _s["position"] = sample_vec(range_dict["position"])
    if "velocity" in range_dict:
        _s["velocity"] = sample_vec(range_dict["velocity"])
    return _s


def sample_player(range_dict):
    _s = sample_pos_vel(range_dict)
    if "rotation_y" in range_dict:
        _s["rotation_y"] = sample_val(range_dict["rotation_y"])
    return _s
