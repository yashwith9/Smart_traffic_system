from __future__ import annotations

import os
import pickle
import tempfile
import unittest

from rl.infer import InferenceConfig, TrafficSignalInference, parse_state


class TestTrafficSignalInference(unittest.TestCase):
    def test_parse_state(self) -> None:
        self.assertEqual(parse_state("4,7,2,9"), [4, 7, 2, 9])

    def test_fallback_chooses_busiest_lane(self) -> None:
        infer = TrafficSignalInference()
        action = infer.decide([1, 9, 4, 3])
        self.assertEqual(action, 1)

    def test_loaded_q_table_decision(self) -> None:
        model_data = {
            "q_table": {(1, 2, 0, 1): [0.5, 1.2, 0.8, 1.2]},
            "meta": {"lane_count": 4, "bucket_size": 3, "max_lane_count": 20},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "q_table.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(model_data, f)

            infer = TrafficSignalInference(InferenceConfig(q_table_path=model_path))
            infer.load()

            # [3,6,0,4] -> buckets [1,2,0,1], tie for actions 1 and 3 => choose min -> 1
            action = infer.decide([3, 6, 0, 4])
            self.assertEqual(action, 1)

    def test_decide_with_context_respects_valid_actions(self) -> None:
        model_data = {
            "q_table": {(1, 2, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0): [0.5, 9.0, 7.0, 6.0]},
            "meta": {
                "lane_count": 4,
                "bucket_size": 3,
                "age_bucket_size": 3,
                "phase_bucket_size": 2,
                "max_lane_count": 20,
                "max_wait_age": 30,
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "q_table.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(model_data, f)

            infer = TrafficSignalInference(InferenceConfig(q_table_path=model_path))
            infer.load()

            action = infer.decide_with_context(
                raw_counts=[3, 6, 0, 4],
                waiting_ages=[0, 0, 0, 0],
                previous_action=0,
                steps_since_switch=2,
                in_yellow=False,
                can_switch=False,
                valid_actions=[0],
            )
            self.assertEqual(action, 0)


if __name__ == "__main__":
    unittest.main()
