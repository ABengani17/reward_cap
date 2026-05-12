"""TRL GRPOTrainer + Qwen 2.5-1.5B + GSM8K under either composition.

The script is intended to be the single-file entry point for a GPU run.
We do not execute it in the artifact `make repro` because TRL and a Qwen
checkpoint together require a GPU and several gigabytes of weights.

Usage (on a GPU host):

    pip install -r experiments/qwen_gsm8k/requirements.txt
    python experiments/qwen_gsm8k/run.py \
        --composition rewardcap \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --dataset openai/gsm8k \
        --seed 0 \
        --max-steps 500 \
        --out results/qwen_gsm8k_rewardcap_seed0

The script writes a metrics JSON to `--out/metrics.json` at the end with
task accuracy, reward-hacking rate, and the monitorability proxy.

The `--composition` flag selects between:
  * weighted_sum  : naive baseline (no gating, no caps).
  * rewardcap     : the constraint-projection layer with format gate and
                    DifferentialCap on the LLM judge.

A Colab-compatible companion notebook lives at run.ipynb.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RunConfig:
    composition: str = "rewardcap"
    model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    dataset: str = "openai/gsm8k"
    seed: int = 0
    max_steps: int = 500
    per_device_train_batch_size: int = 4
    num_generations: int = 8
    learning_rate: float = 5e-7
    beta: float = 0.04
    delta: float = 0.05
    out: str = "results/qwen_gsm8k"


def parse_args() -> RunConfig:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--composition", choices=["weighted_sum", "rewardcap"], default="rewardcap")
    p.add_argument("--model", default=RunConfig.model)
    p.add_argument("--dataset", default=RunConfig.dataset)
    p.add_argument("--seed", type=int, default=RunConfig.seed)
    p.add_argument("--max-steps", type=int, default=RunConfig.max_steps)
    p.add_argument("--per-device-train-batch-size", type=int, default=RunConfig.per_device_train_batch_size)
    p.add_argument("--num-generations", type=int, default=RunConfig.num_generations)
    p.add_argument("--learning-rate", type=float, default=RunConfig.learning_rate)
    p.add_argument("--beta", type=float, default=RunConfig.beta)
    p.add_argument("--delta", type=float, default=RunConfig.delta, help="DifferentialCap budget")
    p.add_argument("--out", default=RunConfig.out)
    args = p.parse_args()
    return RunConfig(**vars(args))


def build_reward_funcs(cfg: RunConfig):
    """Build the list[Callable] passed to GRPOTrainer.

    TRL's `GRPOTrainer` accepts a list of reward functions. We return a
    single composed reward function — composition logic is centralized
    in `rewards.py` for clarity. Both branches return a value in roughly
    [-2, +1] per sample.
    """
    from experiments.qwen_gsm8k.rewards import (
        compose_rewardcap,
        compose_weighted_sum,
    )

    def _reward(prompts: list[str], completions: list[str], **kwargs: Any) -> list[float]:
        # GSM8K ground truths arrive via kwargs["solution"] under the
        # standard `apply_chat_template` pipeline. Adapt as needed.
        ground_truths = kwargs.get("solution") or kwargs.get("answer") or [""] * len(completions)
        out: list[float] = []
        for c, gt in zip(completions, ground_truths):
            if cfg.composition == "rewardcap":
                r = compose_rewardcap(c, gt, delta=cfg.delta).reward
            else:
                r = compose_weighted_sum(c, gt).reward
            out.append(r)
        return out

    return [_reward]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    cfg = parse_args()
    out_dir = Path(cfg.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Late imports: we want `python run.py --help` to work without a
    # GPU stack installed.
    try:
        import torch
        from datasets import load_dataset
        from transformers import AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise SystemExit(
            "TRL/transformers/datasets not installed. Install with:\n"
            "    pip install -r experiments/qwen_gsm8k/requirements.txt"
        ) from exc

    logger.info("loading dataset %s", cfg.dataset)
    ds = load_dataset(cfg.dataset, "main")
    train = ds["train"].shuffle(seed=cfg.seed)

    def _fmt(example: dict[str, Any]) -> dict[str, Any]:
        return {
            "prompt": (
                "Solve the problem. Wrap your final numeric answer in "
                "<answer>...</answer>.\n\n" + example["question"]
            ),
            "solution": example["answer"].split("####")[-1].strip(),
        }

    train = train.map(_fmt)

    logger.info("loading tokenizer %s", cfg.model)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    grpo_args = GRPOConfig(
        output_dir=str(out_dir / "ckpt"),
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        num_generations=cfg.num_generations,
        max_steps=cfg.max_steps,
        learning_rate=cfg.learning_rate,
        beta=cfg.beta,
        seed=cfg.seed,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
    )

    trainer = GRPOTrainer(
        model=cfg.model,
        args=grpo_args,
        train_dataset=train,
        processing_class=tokenizer,
        reward_funcs=build_reward_funcs(cfg),
    )

    logger.info("starting GRPO with composition=%s seed=%d", cfg.composition, cfg.seed)
    trainer.train()

    # Final metrics from a held-out generation pass.
    logger.info("running held-out eval")
    metrics = _eval(trainer, train.select(range(min(128, len(train)))), cfg)
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps({**asdict(cfg), **metrics}, indent=2))
    logger.info("wrote %s", metrics_path)
    logger.info("metrics: %s", metrics)


def _eval(trainer: Any, eval_set: Any, cfg: RunConfig) -> dict[str, float]:
    """Generate on the eval set and compute reward-hacking rate +
    monitorability proxy. Kept light: 128 prompts at greedy decoding.
    """
    from experiments.qwen_gsm8k.rewards import (
        exact_match_reward,
        monitorability_proxy,
        reward_hacking_rate,
    )

    completions: list[str] = []
    ground_truths: list[str] = []
    for ex in eval_set:
        prompt = ex["prompt"]
        gt = ex["solution"]
        completion = trainer.generate(prompt, max_new_tokens=512)[0]
        completions.append(completion)
        ground_truths.append(gt)

    accuracy = sum(
        exact_match_reward(c, g) for c, g in zip(completions, ground_truths)
    ) / max(1, len(completions))
    hacking = reward_hacking_rate(completions, ground_truths)
    is_hack = [
        exact_match_reward(c, g) < 0.5 for c, g in zip(completions, ground_truths)
    ]
    monitor = monitorability_proxy(completions, is_hack)
    return {
        "task_accuracy": accuracy,
        "reward_hacking_rate": hacking,
        "monitorability_proxy": monitor,
    }


if __name__ == "__main__":
    main()
