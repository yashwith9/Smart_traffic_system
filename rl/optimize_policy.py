"""
rl/optimize_policy.py

Automatic train-and-evaluate loop to improve RL policy quality.

What it does:
1) Trains multiple candidate RL models with different reward-shaping weights.
2) Benchmarks each candidate against baseline policies.
3) Selects best candidate by a composite score.
4) Saves the best model to a target path and writes optimization summary CSV.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from dataclasses import dataclass
from typing import Dict, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rl.evaluate import (
    EvalConfig,
    evaluate_policy_longest_queue,
    evaluate_policy_rl,
    evaluate_policy_round_robin,
)
from rl.train_rl import TrafficQLearner, TrainConfig


@dataclass
class CandidateWeights:
    queue_weight: float
    queue_delta_weight: float
    fairness_weight: float
    wait_age_weight: float
    switch_penalty_weight: float


def composite_score(metrics: Dict[str, float]) -> float:
    """
    Higher score is better.
    Emphasize lower avg_queue and higher throughput,
    while still considering max_queue and fairness.
    """
    return (
        -1.0 * metrics["avg_queue"]
        - 0.12 * metrics["max_queue"]
        + 12.0 * metrics["throughput"]
        + 4.0 * metrics["fairness"]
    )


def rl_beats_baselines(rl_metrics: Dict[str, float], baseline_metrics: Dict[str, Dict[str, float]]) -> bool:
    return (
        rl_metrics["avg_queue"] < baseline_metrics["longest_queue"]["avg_queue"]
        and rl_metrics["avg_queue"] < baseline_metrics["round_robin"]["avg_queue"]
        and rl_metrics["throughput"] > baseline_metrics["longest_queue"]["throughput"]
        and rl_metrics["throughput"] > baseline_metrics["round_robin"]["throughput"]
    )


def default_weight_candidates() -> List[CandidateWeights]:
    return [
        CandidateWeights(1.00, 1.20, 0.20, 0.15, 0.10),
        CandidateWeights(0.95, 1.35, 0.25, 0.20, 0.10),
        CandidateWeights(1.10, 1.50, 0.30, 0.25, 0.15),
        CandidateWeights(0.90, 1.60, 0.20, 0.25, 0.20),
        CandidateWeights(1.00, 1.80, 0.35, 0.30, 0.25),
        CandidateWeights(1.20, 1.40, 0.30, 0.20, 0.20),
        CandidateWeights(1.05, 1.55, 0.15, 0.35, 0.30),
        CandidateWeights(0.85, 1.95, 0.25, 0.30, 0.25),
    ]


def write_summary_csv(rows: List[Dict[str, float]], output_path: str) -> None:
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fieldnames = [
        "candidate_id",
        "episodes",
        "queue_weight",
        "queue_delta_weight",
        "fairness_weight",
        "wait_age_weight",
        "switch_penalty_weight",
        "avg_queue",
        "max_queue",
        "throughput",
        "fairness",
        "composite_score",
        "beats_baselines",
        "model_path",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize RL policy by auto train+benchmark loop")
    parser.add_argument("--episodes", type=int, default=4500, help="Training episodes per candidate")
    parser.add_argument("--eval-episodes", type=int, default=80, help="Benchmark evaluation episodes")
    parser.add_argument("--eval-steps", type=int, default=120, help="Steps per benchmark episode")
    parser.add_argument("--min-green-steps", type=int, default=3, help="Minimum green-hold steps")
    parser.add_argument("--yellow-steps", type=int, default=1, help="Yellow transition steps")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument(
        "--best-model-output",
        type=str,
        default="rl/q_table_best.pkl",
        help="Path to save the selected best model",
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default="rl/optimization_summary.csv",
        help="CSV path to save all candidate benchmark metrics",
    )
    parser.add_argument(
        "--candidates-dir",
        type=str,
        default="rl/candidates",
        help="Directory for per-candidate model files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.candidates_dir, exist_ok=True)

    eval_cfg = EvalConfig(
        episodes=args.eval_episodes,
        steps_per_episode=args.eval_steps,
        min_green_steps=args.min_green_steps,
        yellow_steps=args.yellow_steps,
        seed=args.seed,
    )

    baseline_metrics = {
        "longest_queue": evaluate_policy_longest_queue(eval_cfg),
        "round_robin": evaluate_policy_round_robin(eval_cfg),
    }

    print("\nBaseline metrics:")
    for name, m in baseline_metrics.items():
        print(
            f"- {name}: avg_queue={m['avg_queue']:.3f}, max_queue={m['max_queue']:.3f}, "
            f"throughput={m['throughput']:.3f}, fairness={m['fairness']:.3f}"
        )

    candidates = default_weight_candidates()
    rows: List[Dict[str, float]] = []

    best_idx = -1
    best_score = float("-inf")
    best_model_path = ""
    best_rl_metrics: Dict[str, float] = {}

    for idx, weights in enumerate(candidates, start=1):
        model_path = os.path.join(args.candidates_dir, f"q_table_candidate_{idx}.pkl")

        train_cfg = TrainConfig(
            episodes=args.episodes,
            seed=args.seed + idx,
            queue_weight=weights.queue_weight,
            queue_delta_weight=weights.queue_delta_weight,
            fairness_weight=weights.fairness_weight,
            wait_age_weight=weights.wait_age_weight,
            switch_penalty_weight=weights.switch_penalty_weight,
            min_green_steps=args.min_green_steps,
            yellow_steps=args.yellow_steps,
        )

        print(
            f"\n[Candidate {idx}] Training with "
            f"queue_w={weights.queue_weight}, "
            f"delta_w={weights.queue_delta_weight}, "
            f"fairness_w={weights.fairness_weight}, "
            f"wait_age_w={weights.wait_age_weight}, "
            f"switch_w={weights.switch_penalty_weight}"
        )

        learner = TrafficQLearner(train_cfg)
        learner.train()
        learner.export_q_table(model_path)

        rl_metrics = evaluate_policy_rl(eval_cfg, model_path=model_path)
        score = composite_score(rl_metrics)
        beats = rl_beats_baselines(rl_metrics, baseline_metrics)

        print(
            f"[Candidate {idx}] avg_queue={rl_metrics['avg_queue']:.3f}, "
            f"max_queue={rl_metrics['max_queue']:.3f}, "
            f"throughput={rl_metrics['throughput']:.3f}, "
            f"fairness={rl_metrics['fairness']:.3f}, "
            f"score={score:.3f}, beats_baselines={beats}"
        )

        row: Dict[str, float] = {
            "candidate_id": float(idx),
            "episodes": float(args.episodes),
            "queue_weight": weights.queue_weight,
            "queue_delta_weight": weights.queue_delta_weight,
            "fairness_weight": weights.fairness_weight,
            "wait_age_weight": weights.wait_age_weight,
            "switch_penalty_weight": weights.switch_penalty_weight,
            "avg_queue": rl_metrics["avg_queue"],
            "max_queue": rl_metrics["max_queue"],
            "throughput": rl_metrics["throughput"],
            "fairness": rl_metrics["fairness"],
            "composite_score": score,
            "beats_baselines": 1.0 if beats else 0.0,
            "model_path": model_path,
        }
        rows.append(row)

        if score > best_score:
            best_score = score
            best_idx = idx
            best_model_path = model_path
            best_rl_metrics = rl_metrics

    write_summary_csv(rows, args.summary_csv)

    if best_idx < 0:
        print("No candidate models were generated.")
        return

    out_dir = os.path.dirname(args.best_model_output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    shutil.copyfile(best_model_path, args.best_model_output)

    final_beats = rl_beats_baselines(best_rl_metrics, baseline_metrics)

    print("\n=== Optimization Summary ===")
    print(f"Best candidate: {best_idx}")
    print(f"Best model copied to: {args.best_model_output}")
    print(f"Summary CSV: {args.summary_csv}")
    print(
        f"Best RL metrics -> avg_queue={best_rl_metrics['avg_queue']:.3f}, "
        f"max_queue={best_rl_metrics['max_queue']:.3f}, "
        f"throughput={best_rl_metrics['throughput']:.3f}, "
        f"fairness={best_rl_metrics['fairness']:.3f}, "
        f"score={best_score:.3f}"
    )
    print(f"Best RL beats both baselines on avg_queue and throughput: {final_beats}")


if __name__ == "__main__":
    main()
