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


if __name__ == "__main__":
    unittest.main()
