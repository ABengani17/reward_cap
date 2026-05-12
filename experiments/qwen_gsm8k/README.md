# experiments/qwen_gsm8k

GRPO scaffold for Qwen 2.5-1.5B-Instruct on GSM8K under two reward composition schemes. This is wiring only — the judge is a deterministic stub and the script is not exercised by `make repro`. You need a GPU and a real judge client to get meaningful numbers.

```bash
pip install -r experiments/qwen_gsm8k/requirements.txt   # torch, trl, transformers, datasets

# Baseline (no constraint projection):
python experiments/qwen_gsm8k/run.py \
    --composition weighted_sum \
    --seed 0 --max-steps 500 \
    --out results/qwen_gsm8k_weighted_seed0

# rewardcap (gate + caps + DifferentialCap on the LLM judge):
python experiments/qwen_gsm8k/run.py \
    --composition rewardcap \
    --delta 0.05 --seed 0 --max-steps 500 \
    --out results/qwen_gsm8k_rewardcap_seed0
```

Each run writes `metrics.json` with task accuracy, reward-hacking rate, and a monitorability proxy.

## What's running

- **Model.** Qwen 2.5-1.5B-Instruct via `trl.GRPOTrainer`.
- **Dataset.** GSM8K, prompted to wrap the final answer in `<answer>...</answer>`.
- **Reward stack** (`rewards.py`):
  - `exact_match_reward` — programmatic check on the parsed answer. CoT-blind.
  - `llm_judge_aware_reward` — judge given the full completion (CoT + answer). CoT-reading.
  - `llm_judge_blind_reward` — same judge given only the parsed answer. Companion for the differential cap.
  - `length_reward` — bounded soft length penalty.
  - `format_reward` — boolean gate.

Under `--composition weighted_sum` the aware judge is summed in directly. Under `--composition rewardcap`, the format check is a gate, the aware judge is wrapped in `DifferentialCap(fn_aware, fn_blind, cap=0.3, delta=0.05)`, and the cap budget respects R3 in `docs/theory.md`.

## Metrics

- `task_accuracy` — fraction of held-out prompts whose parsed answer matches the GSM8K solution.
- `reward_hacking_rate` — fraction where the aware judge scores > 0.7 but `exact_match_reward < 0.5`. The headline metric.
- `monitorability_proxy` — accuracy of a simple held-out classifier predicting "this sample is hacking" from a CoT feature. Lower is better.

## Plugging in a real judge

`rewards.py:_stub_judge` is the deterministic stand-in so the script imports without an API key. Replace it with an OpenAI / Anthropic / vLLM client. Both the aware and blind judges call through the same function with different inputs: full completion vs. parsed answer.

`run.ipynb` is a Colab-compatible companion. It clones the repo, installs requirements, runs a short (50-step) sanity training, and prints metrics. T4 or better.
