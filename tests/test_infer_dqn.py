from __future__ import annotations

import os
import tempfile
import unittest

import torch

from rl.infer_dqn import DQNInferenceConfig, TrafficDQNInference
from rl.train_dqn import QNetwork


class TestTrafficDQNInference(unittest.TestCase):
    def test_decide_with_context_respects_valid_actions(self) -> None:
        lane_count = 4
        input_dim = lane_count * 3 + 3
        hidden_size = 8

        net = QNetwork(input_dim, hidden_size, lane_count)
        for param in net.parameters():
            param.data.zero_()

        # With zeroed weights, output equals final-layer bias values.
        with torch.no_grad():
            net.net[-1].bias[:] = torch.tensor([0.5, 3.0, 2.0, 1.0])

        payload = {
            "state_dict": net.state_dict(),
            "meta": {
                "lane_count": lane_count,
                "max_lane_count": 20,
                "max_wait_age": 30,
                "hidden_size": hidden_size,
                "min_green_steps": 3,
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "dqn_model.pt")
            torch.save(payload, model_path)

            infer = TrafficDQNInference(DQNInferenceConfig(model_path=model_path))
            infer.load()

            action = infer.decide_with_context(
                raw_counts=[3, 6, 0, 4],
                waiting_ages=[0, 0, 0, 0],
                previous_action=0,
                steps_since_switch=2,
                in_yellow=False,
                can_switch=False,
                valid_actions=[0, 2, 3],
            )

            # Action 1 has best raw Q but is invalid; next best valid action is 2.
            self.assertEqual(action, 2)


if __name__ == "__main__":
    unittest.main()
