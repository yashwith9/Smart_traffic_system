from __future__ import annotations

import os
import unittest

from config.settings import AppConfig


class TestAppConfig(unittest.TestCase):
    def test_from_env_defaults(self) -> None:
        for key in [
            "SMART_TRAFFIC_MODEL_TYPE",
            "SMART_TRAFFIC_MODEL_PATH",
            "SMART_TRAFFIC_DQN_MODEL_PATH",
            "SMART_TRAFFIC_SERIAL_PORT",
            "SMART_TRAFFIC_SERIAL_BAUD",
            "SMART_TRAFFIC_SERIAL_TIMEOUT",
            "SMART_TRAFFIC_LOG_LEVEL",
        ]:
            os.environ.pop(key, None)

        cfg = AppConfig.from_env()
        self.assertEqual(cfg.model_type, "dqn")
        self.assertEqual(cfg.model_path, "rl/q_table.pkl")
        self.assertEqual(cfg.dqn_model_path, "rl/dqn_model_tuned_best.pt")
        self.assertEqual(cfg.serial_port, "COM5")
        self.assertEqual(cfg.serial_baud, 115200)
        self.assertEqual(cfg.serial_timeout, 1.0)
        self.assertEqual(cfg.log_level, "INFO")

    def test_from_env_overrides(self) -> None:
        os.environ["SMART_TRAFFIC_MODEL_TYPE"] = "qtable"
        os.environ["SMART_TRAFFIC_MODEL_PATH"] = "custom/model.pkl"
        os.environ["SMART_TRAFFIC_DQN_MODEL_PATH"] = "custom/model.pt"
        os.environ["SMART_TRAFFIC_SERIAL_PORT"] = "COM9"
        os.environ["SMART_TRAFFIC_SERIAL_BAUD"] = "9600"
        os.environ["SMART_TRAFFIC_SERIAL_TIMEOUT"] = "2.5"
        os.environ["SMART_TRAFFIC_LOG_LEVEL"] = "debug"

        cfg = AppConfig.from_env()
        self.assertEqual(cfg.model_type, "qtable")
        self.assertEqual(cfg.model_path, "custom/model.pkl")
        self.assertEqual(cfg.dqn_model_path, "custom/model.pt")
        self.assertEqual(cfg.serial_port, "COM9")
        self.assertEqual(cfg.serial_baud, 9600)
        self.assertEqual(cfg.serial_timeout, 2.5)
        self.assertEqual(cfg.log_level, "DEBUG")

    def test_invalid_model_type_falls_back_to_dqn(self) -> None:
        os.environ["SMART_TRAFFIC_MODEL_TYPE"] = "invalid"
        cfg = AppConfig.from_env()
        self.assertEqual(cfg.model_type, "dqn")


if __name__ == "__main__":
    unittest.main()
