#!/usr/bin/env bash
# Drive all head-to-head matchups between submitted/baseline agents.
# Results land in report/eval_results/<m1>_vs_<m2>.json
set -u
cd "$(dirname "$0")/.."
mkdir -p report/eval_results

source /data/chuye/miniconda3/etc/profile.d/conda.sh
conda activate soccertwos

MATCHES=${MATCHES:-10}
MAX_STEPS=${MAX_STEPS:-2500}

PAIRS=(
  "reward_shaped_agent example_player_agent"
  "reward_shaped_agent ceia_baseline_agent"
  "selfplay_agent      example_player_agent"
  "selfplay_agent      ceia_baseline_agent"
  "LUCKETS_AGENT       example_player_agent"
  "LUCKETS_AGENT       ceia_baseline_agent"
  "LUCKETS_AGENT       reward_shaped_agent"
  "LUCKETS_AGENT       selfplay_agent"
  "reward_shaped_agent selfplay_agent"
)

for pair in "${PAIRS[@]}"; do
  read -r m1 m2 <<<"$pair"
  out="report/eval_results/${m1}_vs_${m2}.json"
  if [[ -f "$out" ]]; then
    echo "[skip] $m1 vs $m2  (already have $out)"
    continue
  fi
  echo "============================================================"
  echo "[run]  $m1 vs $m2  matches=$MATCHES"
  echo "============================================================"
  log="report/eval_results/${m1}_vs_${m2}.log"
  python report/eval_match.py "$m1" "$m2" \
    --matches "$MATCHES" --max-steps "$MAX_STEPS" 2>"$log" | \
    tee -a "$log" | grep "^@@RESULT@@" | sed 's/^@@RESULT@@//' > "$out" || {
      echo "[fail] $m1 vs $m2  -- see $log"
      continue
    }
  python -c "
import json
d = json.load(open('$out'))
print(f\"  -> wins/losses/draws (team0={d['m1']}): {d['wins']}/{d['losses']}/{d['draws']}\")
"
done

echo
echo "All matchups done.  Results in report/eval_results/"
