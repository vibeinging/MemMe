#!/bin/bash
# Parallel benchmark runner: 5 conversations per round, 2 rounds
set -e

cd /Users/Four/PersonalProjects/MemMe/benchmarks/locomo

API_KEY="${OPENAI_API_KEY:?Please set OPENAI_API_KEY}"
RERANK_KEY="${RERANK_API_KEY:?Please set RERANK_API_KEY}"
OUTDIR="results_v18_full"
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
    --max-llm-concurrent 3 \
    --rerank-api-key "$RERANK_KEY" \
    --rerank-base-url "https://dashscope.aliyuncs.com" \
    --rerank-model "gte-rerank-v2" \
    --top-k 30 \
    --data-path locomo10.json \
    --conversations "$conv_idx" \
    > "$OUTDIR/conv_${conv_idx}.log" 2>&1
  echo "[$(date +%H:%M:%S)] Finished conv $conv_idx (exit=$?)"
}

echo "========================================="
echo "  Round 1: conversations 0-4 (parallel)"
echo "========================================="
PIDS=()
for i in 0 1 2 3 4; do
  mkdir -p "$OUTDIR/conv_${i}"
  run_conv $i &
  PIDS+=($!)
done

# Wait for round 1
FAIL=0
for pid in "${PIDS[@]}"; do
  wait $pid || FAIL=$((FAIL+1))
done
echo "[$(date +%H:%M:%S)] Round 1 done. Failures: $FAIL"

echo "========================================="
echo "  Round 2: conversations 5-9 (parallel)"
echo "========================================="
PIDS=()
for i in 5 6 7 8 9; do
  mkdir -p "$OUTDIR/conv_${i}"
  run_conv $i &
  PIDS+=($!)
done

# Wait for round 2
FAIL=0
for pid in "${PIDS[@]}"; do
  wait $pid || FAIL=$((FAIL+1))
done
echo "[$(date +%H:%M:%S)] Round 2 done. Failures: $FAIL"

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
