"""
rl/tune_dqn.py

Focused DQN tuning pass:
1) Sweep a small hyperparameter set on one seed.
2) Pick best config by avg_queue (then throughput/fairness tie-break).
3) Retrain best config across 3 seeds.
4) Evaluate and save per-seed + aggregate CSV outputs.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rl.evaluate import EvalConfig, evaluate_policy_dqn, evaluate_policy_longest_queue, evaluate_policy_round_robin
from rl.train_dqn import DQNConfig, DQNTrafficLearner


@dataclass
class DQNCandidate:
    learning_rate: float
    batch_size: int
    discount_factor: float
    epsilon_decay: float
    hidden_size: int
    target_update_steps: int


def default_candidates() -> List[DQNCandidate]:
    return [
        DQNCandidate(1e-3, 128, 0.97, 0.995, 64, 250),
        DQNCandidate(7e-4, 128, 0.98, 0.996, 64, 250),
        DQNCandidate(5e-4, 128, 0.99, 0.997, 128, 300),
        DQNCandidate(5e-4, 64, 0.99, 0.997, 128, 300),
    ]


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def write_csv(rows: List[Dict[str, object]], output_path: str, fieldnames: List[str]) -> None:
    ensure_dir(os.path.dirname(output_path))
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Focused DQN tuning and 3-seed validation")

    parser.add_argument("--sweep-episodes", type=int, default=700, help="Episodes per sweep candidate")
    parser.add_argument("--final-episodes", type=int, default=1400, help="Episodes per final-seed training")
    parser.add_argument("--steps", type=int, default=120, help="Max steps per episode")
    parser.add_argument("--eval-episodes", type=int, default=80, help="Benchmark episodes")

    parser.add_argument("--arrival-min", type=int, default=0, help="Min arrivals per lane")
    parser.add_argument("--arrival-max", type=int, default=1, help="Max arrivals per lane")
    parser.add_argument("--service-capacity", type=int, default=5, help="Service capacity per step")
    parser.add_argument("--min-green-steps", type=int, default=3, help="Minimum green hold")
    parser.add_argument("--yellow-steps", type=int, default=0, help="Yellow transition steps")

    parser.add_argument("--base-seed", type=int, default=42, help="Base seed")
    parser.add_argument("--sweep-csv", type=str, default="rl/dqn_tuning_sweep.csv", help="Sweep result CSV")
    parser.add_argument("--final-csv", type=str, default="rl/dqn_tuning_final.csv", help="Final seed result CSV")
    parser.add_argument("--best-model-output", type=str, default="rl/dqn_model_tuned_best.pt", help="Best tuned model path")
    parser.add_argument("--models-dir", type=str, default="rl/dqn_tuned_models", help="Directory for tuned models")

    return parser.parse_args()


def _rank_tuple(metrics: Dict[str, float]) -> Tuple[float, float, float]:
    # Primary: lower avg_queue. Tie-break: higher throughput, higher fairness.
    return (metrics["avg_queue"], -metrics["throughput"], -metrics["fairness"])


def main() -> None:
    args = parse_args()
    ensure_dir(args.models_dir)

    candidates = default_candidates()

    eval_cfg = EvalConfig(
        episodes=args.eval_episodes,
        steps_per_episode=args.steps,
        service_capacity_per_step=args.service_capacity,
        arrival_min=args.arrival_min,
        arrival_max=args.arrival_max,
        min_green_steps=args.min_green_steps,
        yellow_steps=args.yellow_steps,
        seed=args.base_seed,
    )

    baseline_lq = evaluate_policy_longest_queue(eval_cfg)
    baseline_rr = evaluate_policy_round_robin(eval_cfg)

    sweep_rows: List[Dict[str, object]] = []
    best_idx = -1
    best_metrics: Dict[str, float] = {}

    for idx, cand in enumerate(candidates, start=1):
        model_path = os.path.join(args.models_dir, f"sweep_candidate_{idx}.pt")

        cfg = DQNConfig(
            episodes=args.sweep_episodes,
            max_steps_per_episode=args.steps,
            learning_rate=cand.learning_rate,
            batch_size=cand.batch_size,
            discount_factor=cand.discount_factor,
            epsilon_decay=cand.epsilon_decay,
            hidden_size=cand.hidden_size,
            target_update_steps=cand.target_update_steps,
            arrival_min=args.arrival_min,
            arrival_max=args.arrival_max,
            service_capacity_per_step=args.service_capacity,
            min_green_steps=args.min_green_steps,
            yellow_steps=args.yellow_steps,
            seed=args.base_seed,
        )

        learner = DQNTrafficLearner(cfg)
        learner.train()
        learner.export_model(model_path)

        metrics = evaluate_policy_dqn(eval_cfg, model_path)

        row = {
            "candidate_id": idx,
            "episodes": args.sweep_episodes,
            "learning_rate": cand.learning_rate,
            "batch_size": cand.batch_size,
            "discount_factor": cand.discount_factor,
            "epsilon_decay": cand.epsilon_decay,
            "hidden_size": cand.hidden_size,
            "target_update_steps": cand.target_update_steps,
            "avg_queue": metrics["avg_queue"],
            "max_queue": metrics["max_queue"],
            "throughput": metrics["throughput"],
            "fairness": metrics["fairness"],
            "beats_longest_queue_on_both": (
                metrics["avg_queue"] < baseline_lq["avg_queue"]
                and metrics["throughput"] > baseline_lq["throughput"]
            ),
            "beats_round_robin_on_both": (
                metrics["avg_queue"] < baseline_rr["avg_queue"]
                and metrics["throughput"] > baseline_rr["throughput"]
            ),
            "model_path": model_path,
        }
        sweep_rows.append(row)

        if best_idx < 0 or _rank_tuple(metrics) < _rank_tuple(best_metrics):
            best_idx = idx
            best_metrics = metrics

        print(
            f"[Sweep {idx}] lr={cand.learning_rate} bs={cand.batch_size} gamma={cand.discount_factor} "
            f"eps_decay={cand.epsilon_decay} hidden={cand.hidden_size} -> "
            f"avg_q={metrics['avg_queue']:.4f}, tp={metrics['throughput']:.4f}, fair={metrics['fairness']:.4f}"
        )

    write_csv(
        sweep_rows,
        args.sweep_csv,
        [
            "candidate_id",
            "episodes",
            "learning_rate",
            "batch_size",
            "discount_factor",
            "epsilon_decay",
            "hidden_size",
            "target_update_steps",
            "avg_queue",
            "max_queue",
            "throughput",
            "fairness",
            "beats_longest_queue_on_both",
            "beats_round_robin_on_both",
            "model_path",
        ],
    )

    chosen = candidates[best_idx - 1]
    print(f"\nSelected candidate: {best_idx}")

    final_rows: List[Dict[str, object]] = []
    seeds = [args.base_seed, args.base_seed + 1, args.base_seed + 2]

    best_seed_score: Tuple[float, float, float] | None = None
    best_seed_model_path = ""

    for seed in seeds:
        model_path = os.path.join(args.models_dir, f"dqn_tuned_seed_{seed}.pt")

        cfg = DQNConfig(
            episodes=args.final_episodes,
            max_steps_per_episode=args.steps,
            learning_rate=chosen.learning_rate,
            batch_size=chosen.batch_size,
            discount_factor=chosen.discount_factor,
            epsilon_decay=chosen.epsilon_decay,
            hidden_size=chosen.hidden_size,
            target_update_steps=chosen.target_update_steps,
            arrival_min=args.arrival_min,
            arrival_max=args.arrival_max,
            service_capacity_per_step=args.service_capacity,
            min_green_steps=args.min_green_steps,
            yellow_steps=args.yellow_steps,
            seed=seed,
        )

        learner = DQNTrafficLearner(cfg)
        learner.train()
        learner.export_model(model_path)

        metrics = evaluate_policy_dqn(eval_cfg, model_path)

        beats_lq_both = metrics["avg_queue"] < baseline_lq["avg_queue"] and metrics["throughput"] > baseline_lq["throughput"]
        beats_rr_both = metrics["avg_queue"] < baseline_rr["avg_queue"] and metrics["throughput"] > baseline_rr["throughput"]

        benchmark_csv = os.path.join("rl", f"benchmark_dqn_tuned_seed_{seed}.csv")
        write_csv(
            [
                {"policy": "rl", **metrics},
                {"policy": "longest_queue", **baseline_lq},
                {"policy": "round_robin", **baseline_rr},
            ],
            benchmark_csv,
            ["policy", "avg_queue", "max_queue", "throughput", "fairness"],
        )

        row = {
            "seed": seed,
            "episodes": args.final_episodes,
            "learning_rate": chosen.learning_rate,
            "batch_size": chosen.batch_size,
            "discount_factor": chosen.discount_factor,
            "epsilon_decay": chosen.epsilon_decay,
            "hidden_size": chosen.hidden_size,
            "target_update_steps": chosen.target_update_steps,
            "avg_queue": metrics["avg_queue"],
            "max_queue": metrics["max_queue"],
            "throughput": metrics["throughput"],
            "fairness": metrics["fairness"],
            "beats_longest_queue_on_both": beats_lq_both,
            "beats_round_robin_on_both": beats_rr_both,
            "model_path": model_path,
            "benchmark_csv": benchmark_csv,
        }
        final_rows.append(row)

        score = _rank_tuple(metrics)
        if best_seed_score is None or score < best_seed_score:
            best_seed_score = score
            best_seed_model_path = model_path

        print(
            f"[Final seed {seed}] avg_q={metrics['avg_queue']:.4f}, tp={metrics['throughput']:.4f}, "
            f"fair={metrics['fairness']:.4f}, beat_lq_both={beats_lq_both}, beat_rr_both={beats_rr_both}"
        )

    write_csv(
        final_rows,
        args.final_csv,
        [
            "seed",
            "episodes",
            "learning_rate",
            "batch_size",
            "discount_factor",
            "epsilon_decay",
            "hidden_size",
            "target_update_steps",
            "avg_queue",
            "max_queue",
            "throughput",
            "fairness",
            "beats_longest_queue_on_both",
            "beats_round_robin_on_both",
            "model_path",
            "benchmark_csv",
        ],
    )

    if best_seed_model_path:
        shutil.copyfile(best_seed_model_path, args.best_model_output)

    beat_lq_count = sum(1 for r in final_rows if bool(r["beats_longest_queue_on_both"]))
    beat_rr_count = sum(1 for r in final_rows if bool(r["beats_round_robin_on_both"]))

    print("\n=== DQN Tuning Summary ===")
    print(f"Sweep CSV: {args.sweep_csv}")
    print(f"Final CSV: {args.final_csv}")
    print(f"Best tuned model copied to: {args.best_model_output}")
    print(
        f"Beat longest_queue on both avg_queue+throughput: {beat_lq_count}/{len(final_rows)} seeds"
    )
    print(
        f"Beat round_robin on both avg_queue+throughput: {beat_rr_count}/{len(final_rows)} seeds"
    )


if __name__ == "__main__":
    main()
