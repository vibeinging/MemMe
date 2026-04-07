#!/bin/bash
# Parallel benchmark runner: 3 conversations per round, 4 rounds
set -e

cd "$(dirname "$0")"

API_KEY="${OPENAI_API_KEY:?Please set OPENAI_API_KEY}"
RERANK_KEY="${RERANK_API_KEY:?Please set RERANK_API_KEY}"
OUTDIR="results_v20_full"
mkdir -p "$OUTDIR"

run_conv() {
  local conv_idx=$1
  echo "[$(date +%H:%M:%S)] Starting conv $conv_idx"
  MEMME_CACHE_DIR="cache" python3 run_benchmark_engine.py \
    --api-key "$API_KEY" \
    --llm-api-key "$API_KEY" \
    --base-url "https://api.openai.com/v1/embeddings" \
    --chat-base-url "https://api.openai.com/v1/chat/completions" \
    --llm-base-url "https://api.openai.com/v1/chat/completions" \
    --chat-model "gpt-4o-mini" \
    --judge-model "gpt-4o-mini" \
    --engine-llm-model "gpt-4o-mini" \
    --embed-model "text-embedding-3-small" \
    --embed-dims 1536 \
    --output-dir "$OUTDIR/conv_${conv_idx}" \
    --max-llm-concurrent 2 \
    --rerank-api-key "$RERANK_KEY" \
    --rerank-base-url "https://dashscope.aliyuncs.com" \
    --rerank-model "gte-rerank-v2" \
    --top-k 30 \
    --data-path locomo10.json \
    --conversations "$conv_idx" \
    > "$OUTDIR/conv_${conv_idx}.log" 2>&1
  echo "[$(date +%H:%M:%S)] Finished conv $conv_idx (exit=$?)"
}

run_round() {
  local round_num=$1
  shift
  local convs=("$@")
  echo "========================================="
  echo "  Round $round_num: conversations ${convs[*]} (parallel)"
  echo "========================================="
  PIDS=()
  for i in "${convs[@]}"; do
    mkdir -p "$OUTDIR/conv_${i}"
    run_conv $i &
    PIDS+=($!)
  done
  FAIL=0
  for pid in "${PIDS[@]}"; do
    wait $pid || FAIL=$((FAIL+1))
  done
  echo "[$(date +%H:%M:%S)] Round $round_num done. Failures: $FAIL"
}

run_round 1 0 1 2
run_round 2 3 4 5
run_round 3 6 7 8
run_round 4 9

# Merge all JSONL results
echo ""
echo "Merging results..."
MERGED="$OUTDIR/run_merged.jsonl"
> "$MERGED"
for i in $(seq 0 9); do
  for f in "$OUTDIR/conv_${i}"/run_*.jsonl; do
    [ -f "$f" ] && cat "$f" >> "$MERGED"
  done
done
TOTAL=$(wc -l < "$MERGED")
echo "Merged $TOTAL questions into $MERGED"

# Print summary
echo ""
python3 run_benchmark_engine.py --results-only "$MERGED"

echo ""
echo "Done! Results in $MERGED"
