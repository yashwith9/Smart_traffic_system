"""
rl/evaluate.py

Benchmark RL policy against baseline traffic-signal policies in simulation.

Policies:
- rl: trained Q-table policy
- longest-queue: choose lane with maximum queued vehicles
- round-robin: cycle lanes 0 -> 1 -> 2 -> 3

Metrics reported:
- avg_queue: average total queued vehicles across steps
- max_queue: maximum total queue observed
- throughput: served vehicles per step (higher is better)
- fairness: 1 - coefficient of variation of per-lane service share (closer to 1 is better)
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rl.infer import InferenceConfig, TrafficSignalInference
from rl.infer_dqn import DQNInferenceConfig, TrafficDQNInference
from rl.sim_env import SimConfig, TrafficSimEnv


@dataclass
class EvalConfig:
    episodes: int = 50
    steps_per_episode: int = 100
    lane_count: int = 4
    max_lane_count: int = 20
    service_capacity_per_step: int = 4
    arrival_min: int = 0
    arrival_max: int = 2
    min_green_steps: int = 3
    yellow_steps: int = 1
    max_wait_age: int = 30
    seed: int = 42


def seeded_initial_state(rng: random.Random, lane_count: int, max_lane_count: int) -> List[int]:
    return [rng.randint(0, max_lane_count) for _ in range(lane_count)]


def choose_longest_queue(lane_counts: Sequence[int]) -> int:
    return max(range(len(lane_counts)), key=lambda i: lane_counts[i])


def choose_round_robin(step_index: int, lane_count: int) -> int:
    return step_index % lane_count


def fairness_score(service_per_lane: Sequence[int]) -> float:
    total = sum(service_per_lane)
    if total <= 0:
        return 0.0

    shares = [x / total for x in service_per_lane]
    mean_share = sum(shares) / len(shares)
    if mean_share <= 0:
        return 0.0

    variance = sum((s - mean_share) ** 2 for s in shares) / len(shares)
    std_dev = variance ** 0.5
    coeff_var = std_dev / mean_share
    score = 1.0 - coeff_var
    return max(0.0, min(1.0, score))


def step_environment(
    lane_counts: Sequence[int],
    action: int,
    rng: random.Random,
    max_lane_count: int,
    service_capacity_per_step: int,
) -> tuple[List[int], int]:
    """
    Backward-compatible helper for tests.
    """
    sim = TrafficSimEnv(
        SimConfig(
            lane_count=len(lane_counts),
            max_lane_count=max_lane_count,
            service_capacity_per_step=service_capacity_per_step,
            min_green_steps=0,
            yellow_steps=0,
        )
    )
    sim.lane_counts = list(lane_counts)
    sim.current_green = int(action)
    sim.steps_since_switch = 999
    info = sim.step(int(action), rng)
    return info["lane_counts"], int(info["served_total"])


def _make_env(cfg: EvalConfig) -> TrafficSimEnv:
    return TrafficSimEnv(
        SimConfig(
            lane_count=cfg.lane_count,
            max_lane_count=cfg.max_lane_count,
            service_capacity_per_step=cfg.service_capacity_per_step,
            arrival_min=cfg.arrival_min,
            arrival_max=cfg.arrival_max,
            min_green_steps=cfg.min_green_steps,
            yellow_steps=cfg.yellow_steps,
            max_wait_age=cfg.max_wait_age,
        )
    )


def _evaluate_policy(
    cfg: EvalConfig,
    action_fn: Callable[[TrafficSimEnv, int], int],
) -> Dict[str, float]:
    total_queue_sum = 0.0
    queue_obs_count = 0
    max_total_queue = 0
    total_served = 0
    service_per_lane = [0] * cfg.lane_count

    for episode in range(cfg.episodes):
        rng = random.Random(cfg.seed + episode)
        env = _make_env(cfg)
        env.reset(rng)

        for step in range(cfg.steps_per_episode):
            requested_action = action_fn(env, step)
            info = env.step(requested_action, rng)

            total_queue = sum(info["lane_counts"])
            total_queue_sum += float(total_queue)
            queue_obs_count += 1
            max_total_queue = max(max_total_queue, total_queue)

            served_total = int(info["served_total"])
            served_by_lane = info["served_per_lane"]
            total_served += served_total
            for lane_idx, served in enumerate(served_by_lane):
                service_per_lane[lane_idx] += int(served)

    total_steps = cfg.episodes * cfg.steps_per_episode
    return {
        "avg_queue": total_queue_sum / max(1, queue_obs_count),
        "max_queue": float(max_total_queue),
        "throughput": total_served / max(1, total_steps),
        "fairness": fairness_score(service_per_lane),
    }


def evaluate_policy_rl(cfg: EvalConfig, model_path: str) -> Dict[str, float]:
    infer = TrafficSignalInference(InferenceConfig(q_table_path=model_path))
    infer.load()

    def rl_action_fn(env: TrafficSimEnv, _: int) -> int:
        return infer.decide_with_context(
            env.lane_counts,
            env.waiting_ages,
            env.current_green,
            steps_since_switch=env.steps_since_switch,
            in_yellow=env.yellow_remaining > 0,
            can_switch=env.can_switch(),
            valid_actions=env.valid_actions(),
        )

    return _evaluate_policy(cfg, rl_action_fn)


def evaluate_policy_dqn(cfg: EvalConfig, model_path: str) -> Dict[str, float]:
    infer = TrafficDQNInference(DQNInferenceConfig(model_path=model_path))
    infer.load()

    def dqn_action_fn(env: TrafficSimEnv, _: int) -> int:
        return infer.decide_with_context(
            raw_counts=env.lane_counts,
            waiting_ages=env.waiting_ages,
            previous_action=env.current_green,
            steps_since_switch=env.steps_since_switch,
            in_yellow=env.yellow_remaining > 0,
            can_switch=env.can_switch(),
            valid_actions=env.valid_actions(),
        )

    return _evaluate_policy(cfg, dqn_action_fn)


def evaluate_policy_longest_queue(cfg: EvalConfig) -> Dict[str, float]:
    def action_fn(env: TrafficSimEnv, _: int) -> int:
        return choose_longest_queue(env.lane_counts)

    return _evaluate_policy(cfg, action_fn)


def evaluate_policy_round_robin(cfg: EvalConfig) -> Dict[str, float]:
    def action_fn(_: TrafficSimEnv, step: int) -> int:
        return choose_round_robin(step, cfg.lane_count)

    return _evaluate_policy(cfg, action_fn)


def save_results_csv(results: Dict[str, Dict[str, float]], output_path: str) -> None:
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["policy", "avg_queue", "max_queue", "throughput", "fairness"])
        for policy_name, metrics in results.items():
            writer.writerow(
                [
                    policy_name,
                    f"{metrics['avg_queue']:.4f}",
                    f"{metrics['max_queue']:.4f}",
                    f"{metrics['throughput']:.4f}",
                    f"{metrics['fairness']:.4f}",
                ]
            )


def print_results(results: Dict[str, Dict[str, float]]) -> None:
    print("\n=== Traffic Policy Benchmark Results ===")
    print(f"{'Policy':<16} {'AvgQueue':>10} {'MaxQueue':>10} {'Throughput':>12} {'Fairness':>10}")
    for policy_name, metrics in results.items():
        print(
            f"{policy_name:<16} "
            f"{metrics['avg_queue']:>10.3f} "
            f"{metrics['max_queue']:>10.3f} "
            f"{metrics['throughput']:>12.3f} "
            f"{metrics['fairness']:>10.3f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RL policy against baseline policies")
    parser.add_argument("--model", type=str, default="rl/q_table.pkl", help="Path to trained Q-table")
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["qtable", "dqn"],
        default="qtable",
        help="Model type for --model path",
    )
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes for evaluation")
    parser.add_argument("--steps", type=int, default=100, help="Steps per episode")
    parser.add_argument("--arrival-min", type=int, default=0, help="Minimum random arrivals per lane per step")
    parser.add_argument("--arrival-max", type=int, default=2, help="Maximum random arrivals per lane per step")
    parser.add_argument("--min-green-steps", type=int, default=3, help="Minimum green-hold steps")
    parser.add_argument("--yellow-steps", type=int, default=1, help="Yellow transition steps when switching")
    parser.add_argument("--seed", type=int, default=42, help="Random seed base")
    parser.add_argument(
        "--output-csv",
        type=str,
        default="rl/benchmark_results.csv",
        help="CSV file path to save benchmark results",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = EvalConfig(
        episodes=args.episodes,
        steps_per_episode=args.steps,
        arrival_min=args.arrival_min,
        arrival_max=args.arrival_max,
        min_green_steps=args.min_green_steps,
        yellow_steps=args.yellow_steps,
        seed=args.seed,
    )

    results: Dict[str, Dict[str, float]] = {}

    try:
        if args.model_type == "dqn":
            results["rl"] = evaluate_policy_dqn(cfg, model_path=args.model)
        else:
            results["rl"] = evaluate_policy_rl(cfg, model_path=args.model)
    except FileNotFoundError:
        print(f"Model file not found: {args.model}")
        if args.model_type == "dqn":
            print("Train first using: python rl/train_dqn.py")
        else:
            print("Train first using: python rl/train_rl.py")
        return

    results["longest_queue"] = evaluate_policy_longest_queue(cfg)
    results["round_robin"] = evaluate_policy_round_robin(cfg)

    print_results(results)
    save_results_csv(results, args.output_csv)
    print(f"\nSaved benchmark CSV to: {args.output_csv}")


if __name__ == "__main__":
    main()
