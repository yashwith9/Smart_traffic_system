"""
rl/ablation_sweep.py

Fast environment ablation loop for tabular RL.

Workflow:
1) Short-run sweep over environment/control parameters.
2) Rank candidates by composite score.
3) Retrain top-K candidates with longer episodes.
4) Benchmark finalists against baselines.
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
from rl.optimize_policy import composite_score, rl_beats_baselines
from rl.train_rl import TrafficQLearner, TrainConfig


@dataclass
class EnvCandidate:
    service_capacity_per_step: int
    arrival_max: int
    min_green_steps: int
    yellow_steps: int
    invalid_action_penalty_weight: float


def default_candidates() -> List[EnvCandidate]:
    return [
        EnvCandidate(4, 2, 3, 1, 0.75),
        EnvCandidate(5, 2, 3, 1, 0.75),
        EnvCandidate(4, 1, 3, 1, 0.75),
        EnvCandidate(4, 2, 2, 1, 0.75),
        EnvCandidate(4, 2, 4, 1, 0.75),
        EnvCandidate(4, 2, 3, 0, 0.75),
        EnvCandidate(4, 2, 3, 1, 0.50),
        EnvCandidate(4, 2, 3, 1, 1.00),
        EnvCandidate(5, 1, 2, 1, 0.50),
        EnvCandidate(5, 1, 3, 0, 0.50),
        EnvCandidate(3, 2, 2, 1, 1.00),
        EnvCandidate(3, 2, 3, 1, 1.00),
    ]


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def write_csv(rows: List[Dict[str, float]], output_path: str, fieldnames: List[str]) -> None:
    ensure_dir(os.path.dirname(output_path))
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fast ablation sweep for tabular RL traffic control")

    parser.add_argument("--short-episodes", type=int, default=900, help="Training episodes per candidate in sweep phase")
    parser.add_argument("--long-episodes", type=int, default=3500, help="Training episodes per finalist in retrain phase")
    parser.add_argument("--eval-episodes-short", type=int, default=30, help="Evaluation episodes in sweep phase")
    parser.add_argument("--eval-episodes-long", type=int, default=80, help="Evaluation episodes in finalist phase")
    parser.add_argument("--eval-steps", type=int, default=120, help="Steps per evaluation episode")
    parser.add_argument("--top-k", type=int, default=3, help="Number of finalists to retrain")
    parser.add_argument("--seed", type=int, default=42, help="Base seed")

    parser.add_argument("--sweep-csv", type=str, default="rl/ablation_sweep_results.csv", help="CSV output for sweep phase")
    parser.add_argument("--final-csv", type=str, default="rl/ablation_final_results.csv", help="CSV output for finalist phase")
    parser.add_argument("--best-model-output", type=str, default="rl/q_table_best_ablation.pkl", help="Best model output path")
    parser.add_argument("--models-dir", type=str, default="rl/ablation_models", help="Folder for generated models")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.models_dir)
    ensure_dir(os.path.dirname(args.best_model_output))

    candidates = default_candidates()

    sweep_rows: List[Dict[str, float]] = []
    ranked: List[Dict[str, object]] = []

    sweep_fieldnames = [
        "candidate_id",
        "service_capacity_per_step",
        "arrival_min",
        "arrival_max",
        "min_green_steps",
        "yellow_steps",
        "invalid_action_penalty_weight",
        "episodes",
        "avg_queue",
        "max_queue",
        "throughput",
        "fairness",
        "composite_score",
        "beats_baselines",
        "model_path",
    ]

    final_fieldnames = sweep_fieldnames.copy()

    for idx, cand in enumerate(candidates, start=1):
        model_path = os.path.join(args.models_dir, f"sweep_candidate_{idx}.pkl")

        train_cfg = TrainConfig(
            episodes=args.short_episodes,
            seed=args.seed + idx,
            service_capacity_per_step=cand.service_capacity_per_step,
            arrival_min=0,
            arrival_max=cand.arrival_max,
            min_green_steps=cand.min_green_steps,
            yellow_steps=cand.yellow_steps,
            invalid_action_penalty_weight=cand.invalid_action_penalty_weight,
            queue_weight=1.05,
            queue_delta_weight=1.55,
            fairness_weight=0.15,
            wait_age_weight=0.35,
            switch_penalty_weight=0.30,
        )

        eval_cfg = EvalConfig(
            episodes=args.eval_episodes_short,
            steps_per_episode=args.eval_steps,
            service_capacity_per_step=cand.service_capacity_per_step,
            arrival_min=0,
            arrival_max=cand.arrival_max,
            min_green_steps=cand.min_green_steps,
            yellow_steps=cand.yellow_steps,
            seed=args.seed,
        )

        baseline_metrics = {
            "longest_queue": evaluate_policy_longest_queue(eval_cfg),
            "round_robin": evaluate_policy_round_robin(eval_cfg),
        }

        learner = TrafficQLearner(train_cfg)
        learner.train()
        learner.export_q_table(model_path)

        rl_metrics = evaluate_policy_rl(eval_cfg, model_path)
        score = composite_score(rl_metrics)
        beats = rl_beats_baselines(rl_metrics, baseline_metrics)

        row: Dict[str, float] = {
            "candidate_id": float(idx),
            "service_capacity_per_step": float(cand.service_capacity_per_step),
            "arrival_min": 0.0,
            "arrival_max": float(cand.arrival_max),
            "min_green_steps": float(cand.min_green_steps),
            "yellow_steps": float(cand.yellow_steps),
            "invalid_action_penalty_weight": float(cand.invalid_action_penalty_weight),
            "episodes": float(args.short_episodes),
            "avg_queue": rl_metrics["avg_queue"],
            "max_queue": rl_metrics["max_queue"],
            "throughput": rl_metrics["throughput"],
            "fairness": rl_metrics["fairness"],
            "composite_score": score,
            "beats_baselines": 1.0 if beats else 0.0,
            "model_path": model_path,
        }
        sweep_rows.append(row)
        ranked.append(
            {
                "candidate_id": idx,
                "candidate": cand,
                "score": score,
            }
        )

        print(
            f"[Sweep {idx:02d}] svc={cand.service_capacity_per_step} arr=[0,{cand.arrival_max}] "
            f"min_green={cand.min_green_steps} yellow={cand.yellow_steps} "
            f"invalid_pen={cand.invalid_action_penalty_weight} -> "
            f"avg_q={rl_metrics['avg_queue']:.3f}, tp={rl_metrics['throughput']:.3f}, score={score:.3f}"
        )

    write_csv(sweep_rows, args.sweep_csv, sweep_fieldnames)

    ranked.sort(key=lambda item: float(item["score"]), reverse=True)
    finalists = ranked[: max(1, min(args.top_k, len(ranked)))]

    final_rows: List[Dict[str, float]] = []
    best_score = float("-inf")
    best_model = ""
    best_metrics: Dict[str, float] = {}
    best_baselines: Dict[str, Dict[str, float]] = {}

    for rank, item in enumerate(finalists, start=1):
        idx = int(item["candidate_id"])
        cand = item["candidate"]
        assert isinstance(cand, EnvCandidate)

        model_path = os.path.join(args.models_dir, f"final_candidate_{idx}.pkl")

        train_cfg = TrainConfig(
            episodes=args.long_episodes,
            seed=args.seed + 100 + idx,
            service_capacity_per_step=cand.service_capacity_per_step,
            arrival_min=0,
            arrival_max=cand.arrival_max,
            min_green_steps=cand.min_green_steps,
            yellow_steps=cand.yellow_steps,
            invalid_action_penalty_weight=cand.invalid_action_penalty_weight,
            queue_weight=1.05,
            queue_delta_weight=1.55,
            fairness_weight=0.15,
            wait_age_weight=0.35,
            switch_penalty_weight=0.30,
        )

        eval_cfg = EvalConfig(
            episodes=args.eval_episodes_long,
            steps_per_episode=args.eval_steps,
            service_capacity_per_step=cand.service_capacity_per_step,
            arrival_min=0,
            arrival_max=cand.arrival_max,
            min_green_steps=cand.min_green_steps,
            yellow_steps=cand.yellow_steps,
            seed=args.seed,
        )

        baseline_metrics = {
            "longest_queue": evaluate_policy_longest_queue(eval_cfg),
            "round_robin": evaluate_policy_round_robin(eval_cfg),
        }

        learner = TrafficQLearner(train_cfg)
        learner.train()
        learner.export_q_table(model_path)

        rl_metrics = evaluate_policy_rl(eval_cfg, model_path)
        score = composite_score(rl_metrics)
        beats = rl_beats_baselines(rl_metrics, baseline_metrics)

        row: Dict[str, float] = {
            "candidate_id": float(idx),
            "service_capacity_per_step": float(cand.service_capacity_per_step),
            "arrival_min": 0.0,
            "arrival_max": float(cand.arrival_max),
            "min_green_steps": float(cand.min_green_steps),
            "yellow_steps": float(cand.yellow_steps),
            "invalid_action_penalty_weight": float(cand.invalid_action_penalty_weight),
            "episodes": float(args.long_episodes),
            "avg_queue": rl_metrics["avg_queue"],
            "max_queue": rl_metrics["max_queue"],
            "throughput": rl_metrics["throughput"],
            "fairness": rl_metrics["fairness"],
            "composite_score": score,
            "beats_baselines": 1.0 if beats else 0.0,
            "model_path": model_path,
        }
        final_rows.append(row)

        print(
            f"[Final {rank}] candidate={idx} -> avg_q={rl_metrics['avg_queue']:.3f}, "
            f"tp={rl_metrics['throughput']:.3f}, fairness={rl_metrics['fairness']:.3f}, score={score:.3f}, beats={beats}"
        )

        if score > best_score:
            best_score = score
            best_model = model_path
            best_metrics = rl_metrics
            best_baselines = baseline_metrics

    write_csv(final_rows, args.final_csv, final_fieldnames)

    if best_model:
        shutil.copyfile(best_model, args.best_model_output)

    beats_final = bool(best_baselines) and rl_beats_baselines(best_metrics, best_baselines)

    print("\n=== Ablation Summary ===")
    print(f"Sweep CSV: {args.sweep_csv}")
    print(f"Final CSV: {args.final_csv}")
    print(f"Best model: {args.best_model_output}")
    print(
        f"Best RL metrics: avg_queue={best_metrics.get('avg_queue', 0.0):.3f}, "
        f"max_queue={best_metrics.get('max_queue', 0.0):.3f}, "
        f"throughput={best_metrics.get('throughput', 0.0):.3f}, "
        f"fairness={best_metrics.get('fairness', 0.0):.3f}, score={best_score:.3f}"
    )
    print(f"Best RL beats both baselines (avg_queue + throughput): {beats_final}")


if __name__ == "__main__":
    main()
