"""Pointer ownership checks using mocked Win32 APIs; no desktop input/capture."""

import sys
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from shrooms_macro import platform_win as platform


@unittest.skipUnless(sys.platform == "win32", "Windows backend")
class PointerGuardTests(unittest.TestCase):
    TARGET = 101

    def setUp(self):
        self.desktop = platform.Desktop()
        self.api = MagicMock()
        self.position = (100, 100)

        def cursor_position(pointer):
            pointer._obj.x, pointer._obj.y = self.position
            return True

        self.api.GetCursorPos.side_effect = cursor_position
        self.api.WindowFromPoint.return_value = 202
        self.api.GetAncestor.return_value = self.TARGET
        self.patch_api = patch.object(platform, "_user32", self.api)
        self.patch_api.start()
        self.addCleanup(self.patch_api.stop)
        self.desktop.is_available = MagicMock(return_value=True)
        self.desktop.is_foreground = MagicMock(return_value=True)
        self.desktop.client_rect = MagicMock(return_value=platform.Rect(10, 20, 500, 400))
        self.desktop._send_raw = MagicMock(return_value=1)

    def test_other_root_refuses_press_without_sending_input(self):
        self.api.GetAncestor.return_value = 303
        with self.assertRaises(platform.InputUnavailable):
            self.desktop.mouse_down(self.TARGET)
        self.desktop._send_raw.assert_not_called()
        self.assertFalse(self.desktop._left_held)

    def test_child_window_of_selected_game_accepts_press(self):
        self.desktop.mouse_down(self.TARGET)
        self.assertTrue(self.desktop._left_held)
        self.assertEqual(self.desktop._held_hwnd, self.TARGET)
        items = self.desktop._send_raw.call_args.args[0]
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].mi.dwFlags, 0x0002)
        self.api.GetAncestor.assert_called_with(202, 2)

    def test_invalid_pointer_releases_our_existing_press(self):
        self.desktop._left_held = True
        self.desktop._held_hwnd = self.TARGET
        self.api.GetAncestor.return_value = 303
        with self.assertRaises(platform.InputUnavailable):
            self.desktop.mouse_down(self.TARGET)
        self.assertFalse(self.desktop._left_held)
        self.assertIsNone(self.desktop._held_hwnd)
        self.desktop._send_raw.assert_called_once()
        items = self.desktop._send_raw.call_args.args[0]
        self.assertEqual(items[0].mi.dwFlags, 0x0004)

    def test_game_title_bar_refuses_press(self):
        self.position = (100, 5)
        with self.assertRaises(platform.InputUnavailable):
            self.desktop.mouse_down(self.TARGET)
        self.desktop._send_raw.assert_not_called()

    def _cancel_signal(self):
        signal = MagicMock()
        signal.is_set.return_value = False
        signal.wait.return_value = False
        return signal

    def test_stationary_tap_holds_then_releases_without_moving(self):
        self.desktop.move = MagicMock()
        signal = self._cancel_signal()

        def observe_press(duration):
            self.assertEqual(duration, 0.005)
            self.assertTrue(self.desktop._left_held)
            return False

        signal.wait.side_effect = observe_press
        self.desktop.tap(self.TARGET, cancel=signal)
        flags = [call.args[0][0].mi.dwFlags for call in self.desktop._send_raw.call_args_list]
        self.assertEqual(flags, [0x0002, 0x0004])
        self.assertFalse(self.desktop._left_held)
        self.desktop.move.assert_not_called()

    def test_tap_creates_new_press_after_releasing_owned_button(self):
        self.desktop._left_held = True
        self.desktop._held_hwnd = self.TARGET
        self.desktop.tap(self.TARGET, cancel=self._cancel_signal())
        flags = [call.args[0][0].mi.dwFlags for call in self.desktop._send_raw.call_args_list]
        self.assertEqual(flags, [0x0004, 0x0002, 0x0004])
        self.assertFalse(self.desktop._left_held)

    def test_cancelled_tap_does_not_press(self):
        signal = self._cancel_signal()
        signal.is_set.return_value = True
        self.desktop.tap(self.TARGET, cancel=signal)
        self.desktop._send_raw.assert_not_called()
        signal.wait.assert_not_called()

    def test_tap_on_other_window_refuses_input(self):
        self.api.GetAncestor.return_value = 303
        with self.assertRaises(platform.InputUnavailable):
            self.desktop.tap(self.TARGET, cancel=self._cancel_signal())
        self.desktop._send_raw.assert_not_called()


@unittest.skipUnless(sys.platform == "win32", "Windows backend")
class TapControllerTests(unittest.TestCase):
    def test_stale_overlap_is_rejected_before_dispatch(self):
        from shrooms_macro.app import Application
        from shrooms_macro.engine import Action

        app = Application.__new__(Application)
        app.target = 101
        app.desktop = MagicMock()
        app.input_cancel = threading.Event()
        app.engine = SimpleNamespace(phase="transition", status="Perfect")
        with patch("shrooms_macro.app.time.perf_counter", return_value=1.04):
            app._action(Action("tap"), captured_at=1)
        app.desktop.tap.assert_not_called()
        self.assertEqual(app.engine.phase, "charge")

    def test_stationary_tap_is_restricted_to_perfect_stages(self):
        from shrooms_macro.app import Application
        from shrooms_macro.engine import Action

        app = Application.__new__(Application)
        app.target = 101
        app.desktop = MagicMock()
        app.input_cancel = threading.Event()
        app.engine = SimpleNamespace(phase="catch")
        with self.assertRaises(RuntimeError):
            app._action(Action("tap"))
        app.desktop.tap.assert_not_called()
        app.engine.phase = "transition"
        app._action(Action("tap"))
        app.desktop.tap.assert_called_once_with(101, cancel=app.input_cancel)
        app.desktop.move.assert_not_called()
        app.desktop.click.assert_not_called()

    def test_stop_interrupts_tap_before_acquiring_worker_lock(self):
        from shrooms_macro.app import Application

        app = Application.__new__(Application)
        app.root = MagicMock()
        app.desktop = MagicMock()
        app.ui = MagicMock()
        app.target_label = "Roblox"
        app.input_cancel = threading.Event()
        app.active = threading.Event()
        app.active.set()
        app.pending_start = "start-timer"
        app.generation = 4
        app.engine = SimpleNamespace(status="Active", caught=0, sold_batches=0,
                                     error="", pause=MagicMock())
        test = self

        class WorkerLock:
            def __enter__(self):
                test.assertTrue(app.input_cancel.is_set())

            def __exit__(self, *_exc):
                return False

        app.control_lock = WorkerLock()
        app.stop("Paused")
        self.assertFalse(app.active.is_set())
        self.assertEqual(app.generation, 5)
        self.assertIsNone(app.pending_start)
        app.root.after_cancel.assert_called_once_with("start-timer")
        app.desktop.mouse_up.assert_called_once()
        app.engine.pause.assert_called_once_with("Paused")


if __name__ == "__main__":
    unittest.main()
