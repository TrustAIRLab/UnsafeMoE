# UnsafeMoE (F-SOUR)

This repository is for the paper ["Sparse Models, Sparse Safety: Unsafe Routes in Mixture-of-Experts LLMs"](https://arxiv.org/abs/2602.08621). 
It contains all details for reproduce our proposed **F-SOUR** method.

## 0. Create and activate a conda environment:
```
conda create -n unsafe_moe python=3.8
conda activate unsafe_moe
bash prepare.sh
```

## 1. What is included
- `main.py`: main entry point
- `control_router.py`: routing helpers
- `harmful_questions/advbench_subset.csv`: AdvBench subset
- `Dataset_Jailbreak/`: output folder (auto-created if missing)
- `prepare.sh`: dependency bootstrap
- `run.sh`: example run script

## 2. OpenAI key
The shadow judge uses the OpenAI API. Please set one of:
```
export OPENAI_API_KEY=YOUR_KEY
```
or pass `--openai_api_key`.

## 3. Example usage
```
python main.py \
  --llm_model DeepSeek-V2-Lite-Chat \
  --forbidden_dataset AdvBench \
  --begin_num 0 --end_num 10 \
  --max_changes 100 --max_iters 5
```

## 4. Notes
- AdvBench uses `harmful_questions/advbench_subset.csv` by default.
  If you move it, pass `--advbench_csv`.
- JBB is loaded from HuggingFace: `JailbreakBench/JBB-Behaviors`.
- Model weights are not included; use `--model_path` to point to local weights.

## Citation
```bibtex
@article{JHLBZ26,
author = {Yukun Jiang and Hai Huang and Mingjie Li and Michael Backes and Yang Zhang},
title = {{Sparse Models, Sparse Safety: Unsafe Routes in Mixture-of-Experts LLMs}},
journal = {{CoRR abs/2602.08621}},
year = {2026}
}
```
