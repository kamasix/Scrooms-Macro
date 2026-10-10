"""Local input receipts using mocked input and temporary files only."""
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from shrooms_macro.app import Application
from shrooms_macro.engine import Action
from shrooms_macro.platform_win import InputUnavailable


class InputTraceChecks(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.app = Application.__new__(Application)
        self.app.profile = SimpleNamespace(root=self.root)
        self.app.engine = SimpleNamespace(phase="sell_first", caught=1)
        self.app.desktop = MagicMock()
        self.app.target = 101
        self.app.input_cancel = threading.Event()
        self.app.input_trace = []

    def outcomes(self):
        saved = json.loads((self.root / "input-trace.json").read_text(encoding="utf-8"))
        return [event["outcome"] for event in saved["events"]]

    def test_accepted_press_records_request_and_receipt(self):
        self.app.desktop.key.return_value = True
        self.app._action(Action("inventory"))
        self.assertEqual(self.outcomes(), ["requested", "accepted"])
        self.assertFalse((self.root / "input-trace.tmp").exists())

    def test_cancel_before_press_is_not_reported_as_accepted(self):
        self.app.desktop.key.return_value = False
        self.app._action(Action("inventory"))
        self.assertEqual(self.outcomes(), ["requested", "cancelled"])

    def test_failed_press_is_recorded_and_still_stops_the_worker(self):
        self.app.desktop.key.side_effect = InputUnavailable("Mock input rejection")
        with self.assertRaises(InputUnavailable):
            self.app._action(Action("inventory"))
        self.assertEqual(self.outcomes(), ["requested", "failed"])
        self.assertEqual(self.app.input_trace[-1]["error"], "Mock input rejection")

    def test_history_is_bounded_without_images_or_window_identifiers(self):
        self.app.input_trace = [{"action": "inventory", "outcome": "old"} for _ in range(120)]
        self.app._trace_input(Action("inventory"), "requested")
        saved = json.loads((self.root / "input-trace.json").read_text(encoding="utf-8"))
        self.assertEqual(len(saved["events"]), 120)
        self.assertEqual(saved["events"][-1]["outcome"], "requested")
        self.assertEqual(set(saved["events"][-1]), {"time", "action", "outcome", "phase", "caught"})

    def test_unwritable_trace_does_not_prevent_the_key_press(self):
        self.app.desktop.key.return_value = True
        with patch("shrooms_macro.app.Path.write_text", side_effect=OSError("Mock read-only folder")), \
                patch("shrooms_macro.app.logging.exception") as logging:
            self.app._action(Action("inventory"))
        self.app.desktop.key.assert_called_once()
        self.assertEqual(logging.call_count, 2)


if __name__ == "__main__":
    unittest.main()
