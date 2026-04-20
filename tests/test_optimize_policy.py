from __future__ import annotations

import unittest

from rl.optimize_policy import composite_score, rl_beats_baselines


class TestOptimizePolicy(unittest.TestCase):
    def test_composite_score_prefers_better_metrics(self) -> None:
        strong = {"avg_queue": 30.0, "max_queue": 55.0, "throughput": 4.2, "fairness": 0.95}
        weak = {"avg_queue": 50.0, "max_queue": 75.0, "throughput": 3.0, "fairness": 0.80}
        self.assertGreater(composite_score(strong), composite_score(weak))

    def test_rl_beats_baselines_logic(self) -> None:
        baseline = {
            "longest_queue": {"avg_queue": 41.0, "throughput": 3.9},
            "round_robin": {"avg_queue": 42.0, "throughput": 3.8},
        }

        rl_good = {"avg_queue": 39.0, "throughput": 4.0}
        rl_bad = {"avg_queue": 43.0, "throughput": 4.1}

        self.assertTrue(rl_beats_baselines(rl_good, baseline))
        self.assertFalse(rl_beats_baselines(rl_bad, baseline))


if __name__ == "__main__":
    unittest.main()
