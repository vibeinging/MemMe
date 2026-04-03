#!/bin/bash
cd /Users/Four/PersonalProjects/MemMe/benchmarks/locomo

API_KEY="${OPENAI_API_KEY:?Please set OPENAI_API_KEY}"
RERANK_KEY="${RERANK_API_KEY:?Please set RERANK_API_KEY}"

run_conv() {
  local idx=$1
  local name=$2
  mkdir -p "results_v12_test/conv_${idx}"
  echo "[$(date +%H:%M:%S)] Starting $name (index $idx)"
  python3 run_benchmark_engine.py \
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
    --output-dir "results_v12_test/conv_${idx}" \
    --max-llm-concurrent 3 \
    --rerank-api-key "$RERANK_KEY" \
    --rerank-base-url "https://dashscope.aliyuncs.com" \
    --rerank-model "gte-rerank-v2" \
    --top-k 30 \
    --data-path locomo10.json \
    --conversations "$idx" \
    > "results_v12_test/conv_${idx}.log" 2>&1
  echo "[$(date +%H:%M:%S)] Finished $name (exit=$?)"
}

run_conv 1 "conv-30" &
run_conv 3 "conv-42" &
run_conv 6 "conv-47" &

wait
echo "All done."
