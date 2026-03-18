#!/bin/bash
# Full LOCOMO benchmark - all 10 conversations, fair protocol
# Schedule: run at 4:30 AM
# Usage: ./run_full_benchmark.sh

set -e
cd "$(dirname "$0")"

# Use conda python explicitly (system python has urllib3/LibreSSL incompatibility)
PYTHON="/opt/anaconda3/bin/python3"

LOG="benchmark_$(date +%Y%m%d_%H%M%S).log"

echo "=== MemMe LOCOMO Full Benchmark ===" | tee "$LOG"
echo "Start: $(date)" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# Run all 10 conversations sequentially (avoid API rate limits)
"$PYTHON" run_benchmark_v2.py \
    --conversations 0,1,2,3,4,5,6,7,8,9 \
    --judge-runs 1 \
    --top-k 10 \
    --output-dir results_fair_full \
    2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "End: $(date)" | tee -a "$LOG"
echo "=== Done ===" | tee -a "$LOG"
