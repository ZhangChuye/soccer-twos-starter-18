"""
PPO Training with Dense Reward Shaping + Curriculum Learning.

4-stage curriculum for the orange team:
  Stage 0: Both orange agents = random (untrained) policy
  Stage 1: Hybrid — one baseline + one random (agent 2 = baseline, 3 = random)
  Stage 2: Both orange agents = CEIA baseline checkpoint
  Stage 3: Self-play — both orange agents = frozen snapshot of learner

Promotion rule: win_rate >= WIN_RATE_THRESHOLD for CONSECUTIVE_REQUIRED iterations in a row.

Remote workers read curriculum stage from CURRICULUM_STAGE_FILE so policy_mapping_fn stays in sync.

RewardShaperWrapper adds: ball-proximity-delta + ball-to-goal-progress.

Resume from checkpoint (optional):
  - Set ``RESTORE_CHECKPOINT`` or env ``CURRICULUM_RESTORE`` to the checkpoint *file*.
  - Set ``RESTORE_CURRICULUM_STAGE`` / ``CURRICULUM_STAGE`` to 0--3 for ``policy_mapping_fn``.
  - **Weights-only restore**: Tune's full ``restore=`` fails when filter/policy dict layout
    differs (common after changing multi-agent policies). We load **model weights only**
    after iteration 1 starts — see ``RESTORE_POLICY_REMAP`` if old runs used ``opponent``.
"""
import os
import pickle

import numpy as np
import torch
import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from utils import create_rllib_env


NUM_ENVS_PER_WORKER = 3

BASELINE_CHECKPOINT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "ceia_baseline_agent",
    "ray_results",
    "PPO_selfplay_twos",
    "PPO_Soccer_f475e_00000_0_2021-09-19_15-54-02",
    "checkpoint_002449",
    "checkpoint-2449",
)

CURRICULUM_STAGE_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    ".curriculum_stage",
)

STAGE_NAMES = {
    0: "Random",
    1: "Hybrid",
    2: "Baseline",
    3: "Self-play",
}
WIN_RATE_THRESHOLD = 0.80
CONSECUTIVE_REQUIRED = 3
MAX_STAGE = 3

# Resume: checkpoint file path, or None. Env CURRICULUM_RESTORE overrides if set.
RESTORE_CHECKPOINT = None
# When resuming, curriculum stage 0--3 (required for correct opp_* mapping). Env CURRICULUM_STAGE overrides.
RESTORE_CURRICULUM_STAGE = None
# If an old checkpoint uses different policy IDs, map them here, e.g. {"opponent": "opp_rand"}
RESTORE_POLICY_REMAP = {}


def _read_curriculum_stage():
    """Read current stage (used by policy_mapping_fn on all workers)."""
    try:
        with open(CURRICULUM_STAGE_FILE, "r") as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return 0


def _write_curriculum_stage(stage: int):
    with open(CURRICULUM_STAGE_FILE, "w") as f:
        f.write(str(int(stage)))


def policy_mapping_fn(agent_id, *args, **kwargs):
    """Orange agent 2 = baseline slot in hybrid; agent 3 = random slot in hybrid."""
    if agent_id in (0, 1):
        return "default"
    stage = _read_curriculum_stage()
    if stage == 0:
        return "opp_rand"
    if stage == 1:
        return "opp_base" if agent_id == 2 else "opp_rand"
    if stage == 2:
        return "opp_base"
    return "opp_self"


def _restore_checkpoint_weights_only(trainer, checkpoint_path):
    """
    Load neural net weights from an RLlib checkpoint file without restoring filters
    (avoids AssertionError in sync_filters when policy names/count changed).
    """
    with open(checkpoint_path, "rb") as f:
        data = pickle.load(f)
    worker_state = pickle.loads(data["worker"])
    state = worker_state["state"]
    weights = {}
    remap = RESTORE_POLICY_REMAP.copy()
    if os.environ.get("CURRICULUM_POLICY_REMAP"):
        for pair in os.environ["CURRICULUM_POLICY_REMAP"].split(","):
            if "=" in pair:
                old, new = pair.split("=", 1)
                remap[old.strip()] = new.strip()

    for pid, policy_state in state.items():
        target_pid = remap.get(pid, pid)
        policy = trainer.get_policy(target_pid)
        if policy is None:
            print(f"[Resume] checkpoint has '{pid}' → skip (no policy '{target_pid}')")
            continue
        weights[target_pid] = {
            k: np.asarray(v)
            for k, v in policy_state.items()
            if k != "_optimizer_variables"
        }

    if not weights:
        raise RuntimeError(
            "[Resume] No matching policies to load; check RESTORE_POLICY_REMAP / CURRICULUM_POLICY_REMAP"
        )

    trainer.set_weights(weights)
    policy_ids = list(weights.keys())
    local_weights = trainer.workers.local_worker().get_weights(policy_ids)
    for rw in trainer.workers.remote_workers():
        rw.set_weights.remote(local_weights)
    print(f"[Resume] Weight-only load OK for policies: {policy_ids}")


def _load_baseline_weights(checkpoint_path):
    """Extract the 'default' policy model weights from the baseline checkpoint."""
    with open(checkpoint_path, "rb") as f:
        ckpt = pickle.load(f)
    worker = pickle.loads(ckpt["worker"])
    policy_state = worker["state"]["default"]
    weights = {
        k: torch.tensor(v).numpy()
        for k, v in policy_state.items()
        if k != "_optimizer_variables"
    }
    return weights


class CurriculumCallback(DefaultCallbacks):
    """Tracks win rate and promotes through curriculum stages."""

    def __init__(self):
        super().__init__()
        self.stage = _read_curriculum_stage()
        self.consecutive_wins = 0
        self.baseline_weights = None
        self._weights_restore_path = (
            os.environ.get("CURRICULUM_RESUME_WEIGHTS")
            or os.environ.get("CURRICULUM_RESTORE")
            or ""
        )
        self._weights_restored = False

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

        episode.custom_metrics["curriculum_stage"] = float(_read_curriculum_stage())

    def on_train_result(self, *, trainer, result, **kwargs):
        if self._weights_restore_path and not self._weights_restored:
            path = self._weights_restore_path
            print(
                "[Resume] Loading weights only from checkpoint (Tune restore skipped — "
                "filter/policy layout mismatch safe path)"
            )
            _restore_checkpoint_weights_only(trainer, path)
            self._weights_restored = True
            self._weights_restore_path = ""
            for key in ("CURRICULUM_RESUME_WEIGHTS", "CURRICULUM_RESTORE"):
                os.environ.pop(key, None)

        win_rate = result.get("custom_metrics", {}).get("win_rate_mean")
        if win_rate is None:
            return

        it = result["training_iteration"]
        ts = result["timesteps_total"]
        pol_rew = result.get("policy_reward_mean", {}).get("default", 0)
        ep_len = result.get("episode_len_mean", 0)
        entropy = result.get("info", {}).get("learner", {}).get(
            "default", {}).get("learner_stats", {}).get("entropy", 0)

        print(f"[iter {it:>4}] stage={self.stage}({STAGE_NAMES[self.stage]:>9}) "
              f"wr={win_rate:.2f} streak={self.consecutive_wins}/{CONSECUTIVE_REQUIRED} "
              f"pol_rew={pol_rew:+.3f} ep_len={ep_len:.0f} ent={entropy:.2f} ts={ts:,}")

        if win_rate >= WIN_RATE_THRESHOLD:
            self.consecutive_wins += 1
        else:
            self.consecutive_wins = 0

        if self.consecutive_wins >= CONSECUTIVE_REQUIRED and self.stage < MAX_STAGE:
            self.stage += 1
            self.consecutive_wins = 0
            _write_curriculum_stage(self.stage)
            print(f"[Curriculum] === PROMOTED to Stage {self.stage} ({STAGE_NAMES[self.stage]}) ===")

            if self.stage == 1:
                self._set_opp_base_to_baseline(trainer)
            elif self.stage == 2:
                self._sync_opp_base(trainer)
                print("[Curriculum] Mapping: both orange → opp_base (already loaded)")
            elif self.stage == 3:
                self._set_opp_self_from_learner(trainer)

        if self.stage == MAX_STAGE and self.consecutive_wins >= CONSECUTIVE_REQUIRED:
            print("[Curriculum] Self-play: updating opp_self snapshot")
            self._set_opp_self_from_learner(trainer)
            self.consecutive_wins = 0

        _write_curriculum_stage(self.stage)

    @staticmethod
    def _sync_policies(trainer, policy_ids):
        local_weights = trainer.workers.local_worker().get_weights(policy_ids)
        for w in trainer.workers.remote_workers():
            w.set_weights.remote(local_weights)

    def _set_opp_base_to_baseline(self, trainer):
        if self.baseline_weights is None:
            print(f"[Curriculum] Loading baseline from {BASELINE_CHECKPOINT}")
            self.baseline_weights = _load_baseline_weights(BASELINE_CHECKPOINT)
        trainer.set_weights({"opp_base": self.baseline_weights})
        self._sync_policies(trainer, ["opp_base"])
        print("[Curriculum] opp_base = BASELINE weights (synced to all workers)")

    def _sync_opp_base(self, trainer):
        self._sync_policies(trainer, ["opp_base"])

    def _set_opp_self_from_learner(self, trainer):
        trainer.set_weights({"opp_self": trainer.get_weights(["default"])["default"]})
        self._sync_policies(trainer, ["opp_self"])
        print("[Curriculum] opp_self = learner snapshot (synced to all workers)")


def _resolve_restore():
    ckpt = RESTORE_CHECKPOINT or os.environ.get("CURRICULUM_RESTORE")
    if ckpt:
        ckpt = os.path.abspath(os.path.expanduser(ckpt.strip()))
    stage = RESTORE_CURRICULUM_STAGE
    env_stage = os.environ.get("CURRICULUM_STAGE", "").strip()
    if env_stage != "":
        stage = int(env_stage)
    elif ckpt is not None and stage is None:
        stage = 0
    return ckpt, stage


if __name__ == "__main__":
    restore_ckpt, resume_stage = _resolve_restore()
    if restore_ckpt:
        if resume_stage is None:
            resume_stage = 0
        if not os.path.isfile(restore_ckpt):
            raise FileNotFoundError(f"RESTORE_CHECKPOINT not found: {restore_ckpt}")
        _write_curriculum_stage(resume_stage)
        os.environ["CURRICULUM_RESUME_WEIGHTS"] = restore_ckpt
        print(
            f"[Resume] weight-only from={restore_ckpt}\n"
            f"[Resume] curriculum_stage={resume_stage} ({STAGE_NAMES[resume_stage]})"
        )
    else:
        _write_curriculum_stage(0)

    ray.init(include_dashboard=False, _temp_dir="/data/chuye/tmp/ray")

    tune.registry.register_env("Soccer", create_rllib_env)
    temp_env = create_rllib_env({"reward_shaping": True})
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    run_kw = dict(
        name="PPO_curriculum",
        config={
            "num_gpus": 0,
            "num_workers": 12,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level": "WARN",
            "framework": "torch",
            "callbacks": CurriculumCallback,
            "multiagent": {
                "policies": {
                    "default": (None, obs_space, act_space, {}),
                    "opp_rand": (None, obs_space, act_space, {}),
                    "opp_base": (None, obs_space, act_space, {}),
                    "opp_self": (None, obs_space, act_space, {}),
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
            "batch_mode": "truncate_episodes",
        },
        stop={
            "timesteps_total": 15_000_000,
            "time_total_s": 86400,  # 24h max
        },
        checkpoint_freq=50,
        checkpoint_at_end=True,
        local_dir="./ray_results/reward_shaped_curriculum",
    )

    analysis = tune.run("PPO", **run_kw)

    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    print("Best trial:", best_trial)
    best_checkpoint = analysis.get_best_checkpoint(
        trial=best_trial, metric="episode_reward_mean", mode="max"
    )
    print("Best checkpoint:", best_checkpoint)
    print("Done training")
