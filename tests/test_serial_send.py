from __future__ import annotations

import unittest

from integration.serial_send import SerialConfig, SerialSender


class TestSerialSender(unittest.TestCase):
    def test_mock_connect_and_send(self) -> None:
        sender = SerialSender(SerialConfig(mock=True))
        self.assertTrue(sender.connect())
        sender.send_action(2)
        sender.close()

    def test_invalid_action_raises(self) -> None:
        sender = SerialSender(SerialConfig(mock=True))
        sender.connect()
        with self.assertRaises(ValueError):
            sender.send_action(9)


if __name__ == "__main__":
    unittest.main()
