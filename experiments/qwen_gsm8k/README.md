# experiments/qwen_gsm8k

A realistic-scale GRPO scaffold: Qwen 2.5-1.5B-Instruct on GSM8K under two reward composition schemes.

```bash
pip install -r experiments/qwen_gsm8k/requirements.txt   # adds torch, trl, transformers, datasets

# Baseline (no constraint projection):
python experiments/qwen_gsm8k/run.py \
    --composition weighted_sum \
    --seed 0 \
    --max-steps 500 \
    --out results/qwen_gsm8k_weighted_seed0

# rewardcap (gate + caps + DifferentialCap on the LLM judge):
python experiments/qwen_gsm8k/run.py \
    --composition rewardcap \
    --delta 0.05 \
    --seed 0 \
    --max-steps 500 \
    --out results/qwen_gsm8k_rewardcap_seed0
```

Each run writes `metrics.json` with task accuracy, reward-hacking rate, and a monitorability proxy.

## What's running

- **Model**: Qwen 2.5-1.5B-Instruct via `trl.GRPOTrainer`.
- **Dataset**: GSM8K, prompted to wrap the final answer in `<answer>...</answer>`.
- **Reward stack** (see `rewards.py`):
  - `exact_match_reward` — programmatic check on the parsed answer. CoT-blind.
  - `llm_judge_aware_reward` — LLM judge given the full completion (CoT + answer). CoT-reading.
  - `llm_judge_blind_reward` — same judge given only the parsed answer. CoT-blind. Companion for the differential cap.
  - `length_reward` — bounded soft length penalty.
  - `format_reward` — boolean gate.

Under `--composition weighted_sum` the aware judge is summed in directly. Under `--composition rewardcap` the format check is a gate, the aware judge is wrapped in a `DifferentialCap(fn_aware=aware, fn_blind=blind, cap=0.3, delta=0.05)`, and the cap budget respects `R3` from `docs/theory.md`.

## Metrics

- **task_accuracy** — fraction of held-out prompts whose parsed answer matches the GSM8K solution.
- **reward_hacking_rate** — fraction of held-out prompts where the aware judge scores above 0.7 but `exact_match_reward < 0.5`. The headline metric for hacking under a CoT-aware reward stack.
- **monitorability_proxy** — accuracy of a held-out classifier predicting "this sample is hacking" from a simple CoT feature. Stand-in for a real held-out classifier; replace with a trained probe in production. Lower means the CoT signal of hacking has been suppressed (less monitorable).

## What is stubbed

`rewards.py:_stub_judge` is a deterministic stand-in so the script is testable without an API key. Plug in a real judge (OpenAI / Anthropic / vLLM endpoint) by replacing `_stub_judge` with your client. Both the aware and blind judges call through the same function with different inputs (full completion vs. parsed answer); a real run uses the same model with the same prompt template.

## We did not run this

The scaffold is not executed in `make repro` — it needs a GPU and several hundred GSM8K rollouts to produce meaningful numbers. The point of the scaffold is to make the `--composition` switch a single-line change for any practitioner who has a GPU. When you run it, dump the resulting `metrics.json` files next to this README and commit them.

## Colab

`run.ipynb` is a Colab-compatible companion. It clones the repo, installs the requirements, runs a short (50-step) sanity training, and prints metrics. It expects a T4 or better.
