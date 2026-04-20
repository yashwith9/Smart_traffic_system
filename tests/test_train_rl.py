from __future__ import annotations

import unittest

from rl.train_rl import TrafficQLearner, TrainConfig


class TestTrainRLReward(unittest.TestCase):
    def test_reward_penalizes_switching(self) -> None:
        cfg = TrainConfig(switch_penalty_weight=2.0)
        learner = TrafficQLearner(cfg)

        current_counts = [10, 4, 3, 2]
        next_counts = [8, 5, 4, 2]

        reward_keep = learner.compute_reward(current_counts, next_counts, action=1, previous_action=1)
        reward_switch = learner.compute_reward(current_counts, next_counts, action=2, previous_action=1)

        self.assertLess(reward_switch, reward_keep)

    def test_step_environment_respects_lane_limits(self) -> None:
        cfg = TrainConfig(max_lane_count=20, service_capacity_per_step=4)
        learner = TrafficQLearner(cfg)

        next_counts, _ = learner.step_environment([20, 20, 20, 20], action=0, previous_action=0)
        self.assertEqual(len(next_counts), 4)
        self.assertTrue(all(0 <= x <= 20 for x in next_counts))

    def test_choose_action_respects_valid_actions(self) -> None:
        cfg = TrainConfig()
        learner = TrafficQLearner(cfg)
        state = learner.discretize_state([3, 6, 9, 12], [1, 1, 1, 1], previous_action=0)
        learner.q_table[state] = [0.1, 10.0, 9.0, 8.0]

        action = learner.choose_action(state, epsilon=0.0, valid_actions=[0, 2, 3])
        self.assertEqual(action, 2)


if __name__ == "__main__":
    unittest.main()
