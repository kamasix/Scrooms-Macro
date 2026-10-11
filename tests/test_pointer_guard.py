"""Pointer ownership checks using mocked Win32 APIs; no desktop input/capture."""

import sys
import threading
import unittest
from collections import deque
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

    def keyboard_events(self):
        return [(call.args[0][0].ki.wVk, call.args[0][0].ki.dwFlags)
                for call in self.desktop._send_raw.call_args_list]

    def keyboard_packets(self):
        return [(call.args[0][0].ki.wVk, call.args[0][0].ki.wScan,
                 call.args[0][0].ki.dwFlags)
                for call in self.desktop._send_raw.call_args_list]

    def test_inventory_key_has_a_renderable_hold_before_release(self):
        signal = self._cancel_signal()

        def observe_press(duration):
            self.assertEqual(duration, 0.100)
            self.assertEqual(self.desktop._held_key, platform.VK_OEM_3)
            self.assertEqual(self.keyboard_packets(), [(0, 0x29, 0x0008)])
            return False

        signal.wait.side_effect = observe_press
        self.assertIs(self.desktop.key(self.TARGET, platform.VK_OEM_3, cancel=signal), True)
        self.assertEqual(self.keyboard_packets(), [(0, 0x29, 0x0008),
                                                   (0, 0x29, 0x000A)])
        self.assertIsNone(self.desktop._held_key)
        self.assertIsNone(self.desktop._held_key_up_packet)

    def test_generic_virtual_key_packet_and_40ms_hold_are_unchanged(self):
        signal = self._cancel_signal()

        def observe_press(duration):
            self.assertEqual(duration, 0.040)
            self.assertEqual(self.keyboard_packets(), [(0x41, 0, 0)])
            return False

        signal.wait.side_effect = observe_press
        self.assertIs(self.desktop.key(self.TARGET, 0x41, cancel=signal), True)
        self.assertEqual(self.keyboard_packets(), [(0x41, 0, 0), (0x41, 0, 0x0002)])
        self.assertIsNone(self.desktop._held_key_up_packet)

    def test_inventory_release_uses_original_packet_after_mapping_and_focus_change(self):
        signal = self._cancel_signal()
        packet_builder = patch.object(platform, "_keyboard_input",
                                      side_effect=AssertionError("held release was rebuilt"))

        def change_mapping_while_held(duration):
            self.assertEqual(duration, 0.100)
            self.desktop.is_foreground.return_value = False
            packet_builder.start()
            self.addCleanup(packet_builder.stop)
            return False

        signal.wait.side_effect = change_mapping_while_held
        self.desktop.key(self.TARGET, platform.VK_OEM_3, cancel=signal)
        self.assertEqual(self.keyboard_packets(), [(0, 0x29, 0x0008),
                                                   (0, 0x29, 0x000A)])
        self.assertIsNone(self.desktop._held_key)

    def test_cancelled_key_does_not_press_or_wait(self):
        signal = self._cancel_signal()
        signal.is_set.return_value = True
        self.assertIs(self.desktop.key(self.TARGET, platform.VK_OEM_3, cancel=signal), False)
        self.desktop._send_raw.assert_not_called()
        signal.wait.assert_not_called()
        self.assertIsNone(self.desktop._held_key)
        self.assertIsNone(self.desktop._held_key_up_packet)

    def test_cancellation_during_key_hold_releases_after_focus_loss(self):
        signal = self._cancel_signal()

        def cancel_during_press(duration):
            self.assertEqual(duration, 0.100)
            signal.is_set.return_value = True
            self.desktop.is_foreground.return_value = False
            return True

        signal.wait.side_effect = cancel_during_press
        self.assertIs(self.desktop.key(self.TARGET, platform.VK_OEM_3, cancel=signal), True)
        self.assertEqual(self.keyboard_events(), [(0, 0x0008), (0, 0x000A)])
        self.assertIsNone(self.desktop._held_key)

    def test_rejected_key_down_has_no_unowned_release(self):
        self.desktop._send_raw.return_value = 0
        signal = self._cancel_signal()
        with self.assertRaises(platform.InputUnavailable):
            self.desktop.key(self.TARGET, platform.VK_OEM_3, cancel=signal)
        self.assertEqual(self.keyboard_events(), [(0, 0x0008)])
        signal.wait.assert_not_called()
        self.assertIsNone(self.desktop._held_key)
        self.assertIsNone(self.desktop._held_key_up_packet)

    def test_failed_key_release_keeps_ownership_for_shutdown_retry(self):
        self.desktop._send_raw.side_effect = [1, 0]
        with self.assertRaises(platform.InputUnavailable):
            self.desktop.key(self.TARGET, platform.VK_OEM_3,
                             cancel=self._cancel_signal())
        self.assertEqual(self.desktop._held_key, platform.VK_OEM_3)
        packet = self.desktop._held_key_up_packet
        self.assertIsNotNone(packet)
        self.assertIs(self.desktop._send_raw.call_args.args[0][0], packet)
        self.desktop._send_raw.side_effect = None
        self.desktop.close()
        self.assertIsNone(self.desktop._held_key)
        self.assertIs(self.desktop._send_raw.call_args.args[0][0], packet)
        self.assertIsNone(self.desktop._held_key_up_packet)
        self.assertEqual(self.keyboard_packets(), [(0, 0x29, 0x0008),
                                                   (0, 0x29, 0x000A),
                                                   (0, 0x29, 0x000A)])

    def test_key_release_exception_can_be_retried_by_shutdown(self):
        self.desktop._send_raw.side_effect = [1, OSError("mock release failure")]
        with self.assertRaises(platform.InputUnavailable):
            self.desktop.key(self.TARGET, platform.VK_OEM_3,
                             cancel=self._cancel_signal())
        self.assertEqual(self.desktop._held_key, platform.VK_OEM_3)
        self.desktop._send_raw.side_effect = None
        self.desktop.close()
        self.assertIsNone(self.desktop._held_key)

    def test_wait_exception_still_releases_accepted_key(self):
        signal = self._cancel_signal()
        signal.wait.side_effect = RuntimeError("mock cancelled wait")
        with self.assertRaises(RuntimeError):
            self.desktop.key(self.TARGET, platform.VK_OEM_3, cancel=signal)
        self.assertEqual(self.keyboard_events(), [(0, 0x0008), (0, 0x000A)])
        self.assertIsNone(self.desktop._held_key)

    def test_shutdown_can_release_key_while_hold_wait_is_running(self):
        signal = self._cancel_signal()

        def close_during_press(_duration):
            # A different thread must acquire the input lock immediately;
            # the hold wait must not postpone emergency cleanup.
            closer = threading.Thread(target=self.desktop.close)
            closer.start()
            closer.join(timeout=1)
            self.assertFalse(closer.is_alive())
            self.assertIsNone(self.desktop._held_key)
            return True

        signal.wait.side_effect = close_during_press
        self.desktop.key(self.TARGET, platform.VK_OEM_3, cancel=signal)
        self.assertEqual(self.keyboard_events(), [(0, 0x0008), (0, 0x000A)])

    def test_key_foreground_guard_rejects_press_before_input(self):
        self.desktop.is_foreground.return_value = False
        with self.assertRaises(platform.InputUnavailable):
            self.desktop.key(self.TARGET, platform.VK_OEM_3,
                             cancel=self._cancel_signal())
        self.desktop._send_raw.assert_not_called()

    def test_previous_owned_key_is_released_before_a_new_press(self):
        self.desktop._held_key = 0x41
        self.desktop.key(self.TARGET, platform.VK_OEM_3,
                         cancel=self._cancel_signal())
        self.assertEqual(self.keyboard_events(), [(0x41, 0x0002),
                                                  (0, 0x0008), (0, 0x000A)])

    def test_failed_previous_key_release_prevents_another_key_down(self):
        self.desktop._held_key = 0x41
        self.desktop._send_raw.return_value = 0
        with self.assertRaises(platform.InputUnavailable):
            self.desktop.key(self.TARGET, platform.VK_OEM_3,
                             cancel=self._cancel_signal())
        self.assertEqual(self.desktop._held_key, 0x41)
        self.assertTrue(self.keyboard_events())
        self.assertTrue(all(event == (0x41, 0x0002)
                            for event in self.keyboard_events()))

    def test_key_up_without_an_owned_press_sends_no_release(self):
        self.assertTrue(self.desktop.key_up())
        self.desktop._send_raw.assert_not_called()

    def test_key_up_retries_owned_release_without_foreground_requirement(self):
        self.desktop._held_key = platform.VK_OEM_3
        self.desktop._held_key_up_packet = platform._keyboard_input(
            platform.VK_OEM_3, True, scan_code=0x29)
        self.desktop.is_foreground.return_value = False
        self.desktop._send_raw.side_effect = [0, 1]
        self.assertFalse(self.desktop.key_up())
        self.assertEqual(self.desktop._held_key, platform.VK_OEM_3)
        self.assertTrue(self.desktop.key_up())
        self.assertIsNone(self.desktop._held_key)
        self.assertEqual(self.keyboard_packets(), [(0, 0x29, 0x000A),
                                                   (0, 0x29, 0x000A)])

    def mouse_events(self):
        return [call.args[0][0].mi.dwFlags
                for call in self.desktop._send_raw.call_args_list]

    def test_sale_click_holds_for_a_renderable_frame_and_releases(self):
        trace = []
        self.desktop.move = MagicMock(side_effect=lambda *_args: trace.append("move"))
        signal = self._cancel_signal()

        def observe_press(duration):
            trace.append(("wait", duration))
            if duration == 0.080:
                self.assertTrue(self.desktop._left_held)
                self.assertEqual(self.mouse_events(), [0x0002])
            else:
                self.assertFalse(self.desktop._left_held)
                self.assertEqual(self.mouse_events(), [])
            return False

        def send_input(items):
            trace.append(("input", items[0].mi.dwFlags))
            return 1

        self.desktop._send_raw.side_effect = send_input
        signal.wait.side_effect = observe_press
        self.desktop.click(self.TARGET, 230, 170, cancel=signal)
        self.assertEqual(self.desktop.move.call_count, 8)
        self.desktop.move.assert_called_with(self.TARGET, 230, 170)
        self.assertEqual(trace, [item for _ in range(8) for item in ("move", ("wait", 0.010))]
                         + [("wait", 0.060), ("input", 0x0002),
                            ("wait", 0.080), ("input", 0x0004)])
        positions = [call.args[1:] for call in self.desktop.move.call_args_list]
        previous = (90, 80)  # Physical cursor (100, 100), client origin (10, 20).
        deltas = []
        for position in positions:
            self.assertGreater(position[0], previous[0])
            self.assertGreater(position[1], previous[1])
            deltas.append(position[0] - previous[0])
            previous = position
        self.assertLess(deltas[0], deltas[3])
        self.assertLess(deltas[-1], deltas[4])
        self.assertEqual(self.mouse_events(), [0x0002, 0x0004])
        self.assertFalse(self.desktop._left_held)

    def test_cancelled_sale_click_does_not_move_or_press(self):
        self.desktop.move = MagicMock()
        signal = self._cancel_signal()
        signal.is_set.return_value = True
        self.desktop.click(self.TARGET, 230, 170, cancel=signal)
        self.desktop.move.assert_not_called()
        self.desktop._send_raw.assert_not_called()
        signal.wait.assert_not_called()

    def test_sale_click_cancelled_after_move_does_not_press(self):
        signal = self._cancel_signal()
        self.desktop.move = MagicMock(
            side_effect=lambda *_args: setattr(signal.is_set, "return_value", True))
        self.desktop.click(self.TARGET, 230, 170, cancel=signal)
        self.desktop.move.assert_called_once()
        self.desktop._send_raw.assert_not_called()
        signal.wait.assert_not_called()

    def test_sale_click_cancelled_during_hold_releases_after_focus_loss(self):
        self.desktop.move = MagicMock()
        signal = self._cancel_signal()

        def cancel_during_press(duration):
            if duration != 0.080:
                return False
            signal.is_set.return_value = True
            self.desktop.is_foreground.return_value = False
            return True

        signal.wait.side_effect = cancel_during_press
        self.desktop.click(self.TARGET, 230, 170, cancel=signal)
        self.assertEqual(self.mouse_events(), [0x0002, 0x0004])
        self.assertFalse(self.desktop._left_held)

    def test_sale_click_on_other_window_refuses_press(self):
        self.desktop.move = MagicMock()
        self.api.GetAncestor.return_value = 303
        with self.assertRaises(platform.InputUnavailable):
            self.desktop.click(self.TARGET, 230, 170,
                               cancel=self._cancel_signal())
        self.desktop._send_raw.assert_not_called()

    def test_sale_wait_exception_still_releases_owned_press(self):
        self.desktop.move = MagicMock()
        signal = self._cancel_signal()

        def fail_during_press(duration):
            if duration == 0.080:
                raise RuntimeError("mock cancelled wait")
            return False

        signal.wait.side_effect = fail_during_press
        with self.assertRaises(RuntimeError):
            self.desktop.click(self.TARGET, 230, 170, cancel=signal)
        self.assertEqual(self.mouse_events(), [0x0002, 0x0004])
        self.assertFalse(self.desktop._left_held)

    def test_sale_click_failed_release_can_be_retried_by_shutdown(self):
        self.desktop.move = MagicMock()
        self.desktop._send_raw.side_effect = [1, 0]
        with self.assertRaises(platform.InputUnavailable):
            self.desktop.click(self.TARGET, 230, 170,
                               cancel=self._cancel_signal())
        self.assertTrue(self.desktop._left_held)
        self.desktop._send_raw.side_effect = None
        self.desktop.close()
        self.assertFalse(self.desktop._left_held)
        self.assertEqual(self.mouse_events(), [0x0002, 0x0004, 0x0004])

    def test_shutdown_can_release_sale_press_during_hold_wait(self):
        self.desktop.move = MagicMock()
        signal = self._cancel_signal()

        def close_during_press(duration):
            if duration != 0.080:
                return False
            closer = threading.Thread(target=self.desktop.close)
            closer.start()
            closer.join(timeout=1)
            self.assertFalse(closer.is_alive())
            self.assertFalse(self.desktop._left_held)
            return True

        signal.wait.side_effect = close_during_press
        self.desktop.click(self.TARGET, 230, 170, cancel=signal)
        self.assertEqual(self.mouse_events(), [0x0002, 0x0004])

    def test_sale_cancelled_during_smooth_movement_never_presses(self):
        self.desktop.move = MagicMock()
        signal = self._cancel_signal()

        def cancel_after_third_step(duration):
            self.assertEqual(duration, 0.010)
            if self.desktop.move.call_count == 3:
                signal.is_set.return_value = True
                return True
            return False

        signal.wait.side_effect = cancel_after_third_step
        self.desktop.click(self.TARGET, 230, 170, cancel=signal)
        self.assertEqual(self.desktop.move.call_count, 3)
        self.desktop._send_raw.assert_not_called()

    def test_sale_cancelled_during_hover_settle_never_presses(self):
        self.desktop.move = MagicMock()
        signal = self._cancel_signal()

        def cancel_during_hover(duration):
            if duration == 0.060:
                signal.is_set.return_value = True
                return True
            return False

        signal.wait.side_effect = cancel_during_hover
        self.desktop.click(self.TARGET, 230, 170, cancel=signal)
        self.assertEqual(self.desktop.move.call_count, 8)
        self.assertEqual(signal.wait.call_count, 9)
        self.desktop._send_raw.assert_not_called()

    def test_sale_focus_loss_during_movement_stops_before_next_step(self):
        self.desktop.move = MagicMock()
        signal = self._cancel_signal()

        def lose_focus_during_movement(_duration):
            if self.desktop.move.call_count == 2:
                self.desktop.is_foreground.return_value = False
            return False

        signal.wait.side_effect = lose_focus_during_movement
        with self.assertRaises(platform.InputUnavailable):
            self.desktop.click(self.TARGET, 230, 170, cancel=signal)
        self.assertEqual(self.desktop.move.call_count, 2)
        self.desktop._send_raw.assert_not_called()

    def test_sale_focus_loss_during_hover_refuses_final_press(self):
        self.desktop.move = MagicMock()
        signal = self._cancel_signal()

        def lose_focus_during_hover(duration):
            if duration == 0.060:
                self.desktop.is_foreground.return_value = False
            return False

        signal.wait.side_effect = lose_focus_during_hover
        with self.assertRaises(platform.InputUnavailable):
            self.desktop.click(self.TARGET, 230, 170, cancel=signal)
        self.assertEqual(self.desktop.move.call_count, 8)
        self.desktop._send_raw.assert_not_called()

    def test_all_sale_waits_allow_another_thread_to_acquire_input_lock(self):
        self.desktop.move = MagicMock()
        signal = self._cancel_signal()

        def check_unlocked_wait(_duration):
            acquired = threading.Event()

            def acquire_input_lock():
                with self.desktop._input_lock:
                    acquired.set()

            cleanup = threading.Thread(target=acquire_input_lock)
            cleanup.start()
            cleanup.join(timeout=1)
            self.assertFalse(cleanup.is_alive())
            self.assertTrue(acquired.is_set())
            return False

        signal.wait.side_effect = check_unlocked_wait
        self.desktop.click(self.TARGET, 230, 170, cancel=signal)
        self.assertEqual(signal.wait.call_count, 10)


@unittest.skipUnless(sys.platform == "win32", "Windows backend")
class TapControllerTests(unittest.TestCase):
    def test_inventory_and_sale_click_use_the_stop_signal(self):
        from shrooms_macro.app import Application
        from shrooms_macro.engine import Action

        app = Application.__new__(Application)
        app.target = 101
        app.desktop = MagicMock()
        app.input_cancel = threading.Event()
        app.engine = SimpleNamespace(phase="sell_first")
        app._action(Action("inventory"))
        app._action(Action("click", 0.75, 0.6))
        app.desktop.key.assert_called_once_with(101, platform.VK_OEM_3,
                                               cancel=app.input_cancel)
        app.desktop.click.assert_called_once_with(101, 0.75, 0.6,
                                                 cancel=app.input_cancel)

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
        app.profile = SimpleNamespace(data={"auto_sell": True})
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
        app.desktop.key_up.assert_called_once()
        app.engine.pause.assert_called_once_with("Paused")


@unittest.skipUnless(sys.platform == "win32", "Windows backend")
class InputCleanupControllerTests(unittest.TestCase):
    def setUp(self):
        from shrooms_macro.app import Application

        self.app = Application.__new__(Application)
        self.app.root = MagicMock()
        self.app.ui = MagicMock()
        self.app.desktop = MagicMock()
        self.app.desktop.key_up.return_value = True
        self.app.desktop.cursor_over.return_value = True
        self.app.target = 101
        self.app.target_label = "Mock game"
        self.app.profile = SimpleNamespace(data={"auto_sell": True})
        self.app.pending_start = None
        self.app.generation = 4
        self.app.calibrating = False
        self.app.shutdown = threading.Event()
        self.app.active = threading.Event()
        self.app.input_cancel = threading.Event()
        self.app.input_cancel.set()
        self.app.control_lock = threading.RLock()
        self.app.input_release_warning = ""
        self.app.engine = SimpleNamespace(status="Paused", caught=0, sold_batches=0,
                                          error="", pause=MagicMock(), start=MagicMock())
        self.app.engine.pause.side_effect = lambda reason: setattr(self.app.engine, "status", reason)
        logger_patch = patch("shrooms_macro.app.logging.warning")
        self.warning_log = logger_patch.start()
        self.addCleanup(logger_patch.stop)

    def test_failed_key_release_blocks_resume_before_focus_or_actions(self):
        self.app.desktop.key_up.return_value = False
        self.app.finish_start()
        self.assertFalse(self.app.active.is_set())
        self.assertTrue(self.app.input_cancel.is_set())
        self.app.desktop.focus.assert_not_called()
        self.app.engine.start.assert_not_called()
        self.app.engine.pause.assert_called_once()
        warning = self.app.ui.update_status.call_args.args[0]["warning"]
        self.assertIn("did not release the held key", warning)
        self.assertIn("Press Stop", warning)
        self.warning_log.assert_called_once()

    def test_successful_release_retry_allows_resume_and_clears_warning(self):
        self.app.desktop.key_up.side_effect = [False, True]
        self.app.finish_start()
        self.assertFalse(self.app.active.is_set())
        self.app.finish_start()
        self.assertTrue(self.app.active.is_set())
        self.assertFalse(self.app.input_cancel.is_set())
        self.app.desktop.focus.assert_called_once_with(self.app.target)
        self.app.engine.start.assert_called_once()
        self.assertEqual(self.app.snapshot()["warning"], "")

    def test_stop_surfaces_release_failure_and_preserves_warning_in_later_snapshots(self):
        self.app.active.set()
        self.app.desktop.key_up.return_value = False
        self.app.stop("Stopped")
        self.assertFalse(self.app.active.is_set())
        self.assertTrue(self.app.input_cancel.is_set())
        snapshot = self.app.ui.update_status.call_args.args[0]
        self.assertEqual(snapshot["status"], "Stopped")
        self.assertIn("did not release the held key", snapshot["warning"])
        self.assertEqual(self.app.snapshot()["warning"], snapshot["warning"])
        self.warning_log.assert_called_once()

    def test_normal_stop_keeps_status_and_does_not_show_a_release_warning(self):
        self.app.active.set()
        self.app.stop("Paused — F6 resumes")
        snapshot = self.app.ui.update_status.call_args.args[0]
        self.assertFalse(snapshot["running"])
        self.assertEqual(snapshot["status"], "Paused — F6 resumes")
        self.assertEqual(snapshot["warning"], "")
        self.app.desktop.mouse_up.assert_called_once()
        self.app.desktop.key_up.assert_called_once()
        self.warning_log.assert_not_called()


@unittest.skipUnless(sys.platform == "win32", "Windows backend")
class FrameMetricsControllerTests(unittest.TestCase):
    def setUp(self):
        from shrooms_macro.app import Application

        self.app = Application.__new__(Application)
        self.app.root = MagicMock()
        self.app.ui = MagicMock()
        self.app.desktop = MagicMock()
        self.app.desktop.key_up.return_value = True
        self.app.desktop.is_foreground.return_value = True
        self.app.desktop.cursor_over.return_value = True
        self.app.target = 101
        self.app.target_label = "Mock game"
        self.app.profile = SimpleNamespace(data={"auto_sell": True})
        self.app.pending_start = None
        self.app.generation = 4
        self.app.calibrating = False
        self.app.shutdown = threading.Event()
        self.app.active = threading.Event()
        self.app.active.set()
        self.app.input_cancel = threading.Event()
        self.app.control_lock = threading.RLock()
        self.app.detector_lock = threading.RLock()
        self.app.input_release_warning = ""
        self.app.engine = SimpleNamespace(
            phase="transition", status="Perfect", caught=0, sold_batches=0,
            error="", pause=MagicMock(), start=MagicMock(), step=MagicMock())
        self.app.detector = MagicMock()
        self.app.post = MagicMock()
        self.app._reset_frame_metrics()

    def test_history_is_bounded_and_copies_metrics_at_record_time(self):
        for frame in range(250):
            self.app.last_frame_metrics = {"capture_ms": frame, "tap_called": False}
            self.app._record_frame_metrics()
        self.assertIsInstance(self.app.frame_metrics_history, deque)
        self.assertEqual(self.app.frame_metrics_history.maxlen, 240)
        self.assertEqual(len(self.app.frame_metrics_history), 240)
        self.assertEqual(self.app.frame_metrics_history[0]["capture_ms"], 10)
        self.app.last_frame_metrics["tap_called"] = True
        self.assertFalse(self.app.frame_metrics_history[-1]["tap_called"])

    def test_diagnostics_snapshot_does_not_expose_mutable_history(self):
        self.app.last_frame_metrics = {"capture_ms": 4.2, "tap_called": False}
        self.app._record_frame_metrics()
        last, recent = self.app._frame_metrics_snapshot()
        last["capture_ms"] = 100
        recent[0]["tap_called"] = True
        recent.append({"extra": True})
        self.assertEqual(self.app.last_frame_metrics["capture_ms"], 4.2)
        self.assertFalse(self.app.frame_metrics_history[0]["tap_called"])
        self.assertEqual(len(self.app.frame_metrics_history), 1)

    def test_successful_new_run_clears_previous_frame_history(self):
        self.app.last_frame_metrics = {"tap_called": True}
        self.app._record_frame_metrics()
        self.app.finish_start()
        self.assertEqual(self.app.last_frame_metrics, {})
        self.assertEqual(len(self.app.frame_metrics_history), 0)
        self.assertTrue(self.app.active.is_set())
        self.app.engine.start.assert_called_once()

    def _run_one_mocked_frame(self, dispatch_at):
        from shrooms_macro.engine import Action
        from shrooms_macro.vision import BarTrack, Observation

        ring = BarTrack((400, 300, 200, 200), 1.24, 1.2, 1.28, "angle", .95)
        self.app.detector.analyze.return_value = Observation(
            charge=ring, ring_present=True, ring_diagnostic="Perfect and pointer recognized")
        self.app.desktop.capture.return_value = SimpleNamespace(shape=(1080, 1920, 3))
        self.app.engine.step.return_value = [Action("tap")]
        self.app.shutdown.wait = MagicMock(side_effect=lambda _delay: self.app.shutdown.set())
        with patch("shrooms_macro.app.time.perf_counter",
                   side_effect=[0, .001, .003, .004, .005, .006, dispatch_at, .055]):
            self.app.work()
        self.assertEqual(len(self.app.frame_metrics_history), 1)
        return self.app.frame_metrics_history[0]

    def test_worker_records_recognition_and_tap_after_dispatch_without_identifiers(self):
        sample = self._run_one_mocked_frame(.007)
        self.assertTrue(sample["tap_requested"])
        self.assertTrue(sample["tap_called"])
        self.assertAlmostEqual(sample["pointer_angle_rad"], 1.24)
        self.assertAlmostEqual(sample["perfect_center_rad"], 1.24)
        self.assertAlmostEqual(sample["perfect_size_rad"], .08)
        self.assertEqual(sample["capture_ms"], 2)
        self.assertEqual(sample["analysis_ms"], 1)
        self.assertEqual(sample["decision_age_ms"], 4)
        self.assertEqual(sample["dispatch_age_ms"], 5)
        self.assertEqual(sample["phase"], "transition")
        self.assertEqual(sample["status"], "Perfect")
        self.assertEqual(set(sample), {
            "capture_ms", "analysis_ms", "decision_age_ms", "dispatch_age_ms",
            "ring_present", "ring_recognized", "ring_diagnostic", "pointer_angle_rad",
            "perfect_center_rad", "perfect_size_rad", "tap_called", "tap_requested",
            "decision_status", "phase", "status",
        })
        self.app.desktop.tap.assert_called_once_with(101, cancel=self.app.input_cancel)

    def test_worker_records_rejected_stale_tap_and_final_charge_status(self):
        sample = self._run_one_mocked_frame(.050)
        self.assertTrue(sample["tap_requested"])
        self.assertFalse(sample["tap_called"])
        self.assertEqual(sample["dispatch_age_ms"], 48)
        self.assertEqual(sample["phase"], "charge")
        self.assertIn("frame too old", sample["status"])
        self.app.desktop.tap.assert_not_called()


if __name__ == "__main__":
    unittest.main()
