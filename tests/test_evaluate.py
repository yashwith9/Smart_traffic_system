from __future__ import annotations

import csv
import os
import tempfile
import unittest

from rl.evaluate import (
    EvalConfig,
    choose_longest_queue,
    choose_round_robin,
    evaluate_policy_longest_queue,
    fairness_score,
    save_results_csv,
    step_environment,
)


class TestEvaluateModule(unittest.TestCase):
    def test_choose_longest_queue(self) -> None:
        self.assertEqual(choose_longest_queue([2, 8, 5, 1]), 1)

    def test_round_robin_cycles(self) -> None:
        self.assertEqual(choose_round_robin(0, 4), 0)
        self.assertEqual(choose_round_robin(1, 4), 1)
        self.assertEqual(choose_round_robin(4, 4), 0)

    def test_fairness_score_range(self) -> None:
        balanced = fairness_score([10, 10, 10, 10])
        skewed = fairness_score([40, 0, 0, 0])
        self.assertGreaterEqual(balanced, 0.0)
        self.assertLessEqual(balanced, 1.0)
        self.assertGreaterEqual(skewed, 0.0)
        self.assertLessEqual(skewed, 1.0)
        self.assertGreaterEqual(balanced, skewed)

    def test_step_environment_respects_limits(self) -> None:
        import random

        rng = random.Random(123)
        state, served = step_environment(
            lane_counts=[20, 20, 20, 20],
            action=0,
            rng=rng,
            max_lane_count=20,
            service_capacity_per_step=4,
        )
        self.assertGreaterEqual(served, 0)
        self.assertLessEqual(served, 4)
        self.assertEqual(len(state), 4)
        self.assertTrue(all(0 <= x <= 20 for x in state))

    def test_evaluate_longest_queue_outputs_metrics(self) -> None:
        cfg = EvalConfig(episodes=2, steps_per_episode=5, seed=7)
        metrics = evaluate_policy_longest_queue(cfg)
        for key in ["avg_queue", "max_queue", "throughput", "fairness"]:
            self.assertIn(key, metrics)
            self.assertGreaterEqual(metrics[key], 0.0)

    def test_save_results_csv(self) -> None:
        results = {
            "rl": {
                "avg_queue": 10.0,
                "max_queue": 20.0,
                "throughput": 2.5,
                "fairness": 0.9,
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            out_csv = os.path.join(tmpdir, "results.csv")
            save_results_csv(results, out_csv)

            self.assertTrue(os.path.exists(out_csv))

            with open(out_csv, "r", encoding="utf-8") as f:
                rows = list(csv.reader(f))

            self.assertEqual(rows[0], ["policy", "avg_queue", "max_queue", "throughput", "fairness"])
            self.assertEqual(rows[1][0], "rl")


if __name__ == "__main__":
    unittest.main()
