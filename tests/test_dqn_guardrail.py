from __future__ import annotations

import os
import unittest

from rl.evaluate import (
    EvalConfig,
    evaluate_policy_dqn,
    evaluate_policy_round_robin,
)


class TestDQNGuardrail(unittest.TestCase):
    def test_tuned_dqn_beats_round_robin_guardrail(self) -> None:
        model_path = os.getenv("SMART_TRAFFIC_DQN_MODEL_PATH", "rl/dqn_model_tuned_best.pt")
        if not os.path.exists(model_path):
            self.skipTest(f"Guardrail model not found: {model_path}")

        cfg = EvalConfig(
            episodes=40,
            steps_per_episode=120,
            service_capacity_per_step=5,
            arrival_min=0,
            arrival_max=1,
            min_green_steps=3,
            yellow_steps=0,
            seed=42,
        )

        dqn_metrics = evaluate_policy_dqn(cfg, model_path=model_path)
        rr_metrics = evaluate_policy_round_robin(cfg)

        # Guardrail thresholds chosen from tuned-model smoke benchmarks.
        self.assertLess(dqn_metrics["avg_queue"], rr_metrics["avg_queue"] - 0.8)
        self.assertGreater(dqn_metrics["throughput"], rr_metrics["throughput"] + 0.005)


if __name__ == "__main__":
    unittest.main()
