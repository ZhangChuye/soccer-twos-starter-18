# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

Python 3.8 is required. Create and activate the conda environment:

```bash
conda create --name soccertwos python=3.8 -y
conda activate soccertwos
pip install pip==23.3.2 setuptools==65.5.0 wheel==0.38.4
pip cache purge
pip install -r requirements.txt
pip install protobuf==3.20.3 pydantic==1.10.13
```

## Key Commands

**Watch agents play (visualize):**
```bash
python -m soccer_twos.watch -m LUCKETS_AGENT
python -m soccer_twos.watch -m1 LUCKETS_AGENT -m2 ceia_baseline_agent
python -m soccer_twos.watch -m1 LUCKETS_AGENT -m2 example_player_agent
```

**Train:**
```bash
python train_reward_shaped_curriculum.py   # main: reward shaping + 4-stage curriculum
python train_reward_shaped.py              # reward shaping only, vs frozen random
python train_ray_selfplay.py               # self-play ladder (no reward shaping)
python train_ray_curriculum.py             # curriculum with position/velocity resets
```

**Package a checkpoint into an agent folder:**
```bash
python package_agent.py <checkpoint_path> [agent_dir]
# e.g. python package_agent.py ./ray_results/PPO_reward_shaped/PPO_Soccer_XXXXX/checkpoint_000100/checkpoint-100 reward_shaped_agent
```

**Plot training curves:**
```bash
python plot_training_curve.py --progress ray_results/.../progress.csv --packaged-iter 600
```

**Zip for Gradescope submission:**
```bash
zip -r LUCKETS_AGENT.zip LUCKETS_AGENT
```

**PACE cluster (SLURM):**
```bash
sbatch scripts/soccerstwos_job.batch
```

## Architecture

### Environment Layer
`utils.py` defines two key classes:
- `RLLibWrapper` — thin wrapper making `soccer_twos.make()` compatible with Ray's `MultiAgentEnv`
- `RewardShaperWrapper` — adds dense reward signals on top of the sparse Unity reward:
  - **Ball proximity delta**: reward proportional to decrease in agent-to-ball distance (`PROXIMITY_DELTA_COEFF = 0.02`)
  - **Ball-to-goal progress**: reward for ball moving toward opponent goal, directional by team (`BALL_PROGRESS_COEFF = 0.05`)

`create_rllib_env(env_config)` is the factory registered with Ray Tune. Pass `{"reward_shaping": True}` to enable the wrapper.

### Agent Interface
All submitted agents must:
- Inherit from `soccer_twos.AgentInterface`
- Implement `act(observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]`
- Load their checkpoint from a `checkpoint/` subdirectory relative to `__file__`
- Export as `Agent` from `__init__.py`

The checkpoint-loading pattern used in `LUCKETS_AGENT/agent.py` handles two cases: full `agent.restore()` (normal path) and a weights-only fallback via `_restore_weights_only()` that skips optimizer state — necessary when policy names/count changed between training and inference.

### Training Pipelines

**`train_reward_shaped_curriculum.py`** (main, submitted agent):
- 4 curriculum stages controlled by `CurriculumCallback` and a `.curriculum_stage` file shared across Ray workers:
  - Stage 0 `Random`: orange team = random policy (`opp_rand`)
  - Stage 1 `Hybrid`: orange agent 2 = CEIA baseline (`opp_base`), agent 3 = random
  - Stage 2 `Baseline`: both orange = CEIA baseline
  - Stage 3 `Self-play`: both orange = frozen learner snapshot (`opp_self`)
- Promotion requires `win_rate >= 0.80` for 3 consecutive iterations
- Reward shaping is enabled; uses CEIA baseline checkpoint at `ceia_baseline_agent/ray_results/...`

**`train_reward_shaped.py`**: Simplified version — blue team learns, orange team is frozen random. No curriculum.

**`train_ray_selfplay.py`**: Pure self-play with 3 frozen opponent policy snapshots (`opponent_1/2/3`). Updates opponents when `episode_reward_mean > 0.5`.

**`train_ray_curriculum.py`**: Position/velocity curriculum via `curriculum.yaml` — gradually expands the initial ball and player spawn ranges across 5 tasks. No reward shaping.

### Multiagent Policy Mapping
Agent IDs 0,1 = blue team (always `"default"` policy, the one being trained). Agent IDs 2,3 = orange team (mapped to opponent policies by `policy_mapping_fn`). Only `"default"` is in `policies_to_train`.

### Ray Results & Checkpoints
Training saves to `./ray_results/<run_name>/`. Checkpoints are at `checkpoint_NNNNNN/checkpoint-NNNNNN`. The `params.pkl` lives one level up (in the trial directory), not inside the checkpoint directory — `package_agent.py` copies both to `<agent_dir>/checkpoint/`.

### Agent Folders
Each agent folder (e.g., `LUCKETS_AGENT/`, `curriculum_agent/`, `final_selfplay/`) has:
- `__init__.py` exporting `Agent`
- `agent.py` with the `AgentInterface` subclass
- `checkpoint/` with `checkpoint-NNN`, `checkpoint-NNN.tune_metadata`, `.is_checkpoint`, `params.pkl`

`LUCKETS_AGENT` is the submitted agent (reward shaping + 4-stage curriculum, checkpoint-600).

## Reward Shaping Design Notes

Dense rewards should be much smaller than the sparse game reward (±1 for goal). The `dev_log.md` captures the key principle: shaping based on small state *changes* (deltas) works better than absolute state values, and the ball progress signal should use a single shared previous ball position rather than per-player tracking.
