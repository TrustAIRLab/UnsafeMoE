#!/bin/bash
set -euo pipefail

python main.py \
  --llm_model DeepSeek-V2-Lite-Chat \
  --forbidden_dataset AdvBench \
  --jailbreak_method F_SOUR \
  --begin_num 0 --end_num 10 \
  --max_changes 100 --max_iters 5
