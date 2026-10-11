"""Small ring capture checks using mocked GDI and generated pixels only."""

import ctypes
import json
import math
import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from shrooms_macro import platform_win as platform
from shrooms_macro.vision import Detector


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(sys.platform == "win32", "Windows capture backend")
class RegionCaptureTests(unittest.TestCase):
    def setUp(self):
        self.desktop = platform.Desktop()
        self.desktop.mouse_up = MagicMock()
        self.desktop.is_available = MagicMock(return_value=True)
        self.desktop.is_foreground = MagicMock(return_value=True)
        self.desktop.client_rect = MagicMock(return_value=platform.Rect(-1920, 80, 1920, 1080))
        self.user = MagicMock()
        self.gdi = MagicMock()
        self.user.GetDC.return_value = 11
        self.gdi.CreateCompatibleDC.return_value = 22
        self.gdi.SelectObject.return_value = 44
        self.gdi.BitBlt.return_value = True
        self.gdi.GdiFlush.return_value = True
        self.buffer = None
        self.buffer_shape = None

        def create_dib(_dc, info_pointer, _usage, pixels_pointer, _section, _offset):
            header = info_pointer._obj.bmiHeader
            width, height = header.biWidth, -header.biHeight
            self.buffer_shape = (height, width)
            self.buffer = (ctypes.c_ubyte * (width * height * 4))()
            pixels = np.ctypeslib.as_array(self.buffer).reshape(height, width, 4)
            pixels[:] = (17, 83, 201, 99)
            pixels_pointer._obj.value = ctypes.addressof(self.buffer)
            return 33

        self.gdi.CreateDIBSection.side_effect = create_dib
        user_patch = patch.object(platform, "_user32", self.user)
        gdi_patch = patch.object(platform, "_gdi32", self.gdi)
        user_patch.start()
        gdi_patch.start()
        self.addCleanup(user_patch.stop)
        self.addCleanup(gdi_patch.stop)

    def assert_cleanup(self):
        self.gdi.SelectObject.assert_any_call(22, 44)
        self.gdi.DeleteObject.assert_called_once_with(33)
        self.gdi.DeleteDC.assert_called_once_with(22)
        self.user.ReleaseDC.assert_called_once_with(None, 11)

    def test_region_keeps_negative_monitor_origin_and_client_coordinates(self):
        frame = self.desktop.capture_region(101, (1080, 1920, 3), (400, 250, 8, 6))
        self.assertEqual(frame.shape, (6, 8, 3))
        self.assertEqual(self.buffer_shape, (6, 8))
        self.gdi.BitBlt.assert_called_once_with(22, 0, 0, 8, 6, 11, -1520, 330,
                                               0x00CC0020 | 0x40000000)
        self.assert_cleanup()

    def test_capture_updates_screen_origin_after_window_changes_monitor(self):
        self.desktop.client_rect.return_value = platform.Rect(1920, -1080, 1920, 1080)
        self.desktop.capture_region(101, (1080, 1920, 3), (500, 200, 8, 6))
        self.gdi.BitBlt.assert_called_once_with(22, 0, 0, 8, 6, 11, 2420, -880,
                                               0x00CC0020 | 0x40000000)

    def test_resized_client_requests_full_frame_without_allocating_gdi(self):
        self.desktop.client_rect.return_value = platform.Rect(0, 0, 1280, 720)
        self.assertIsNone(self.desktop.capture_region(101, (1080, 1920, 3), (400, 250, 8, 6)))
        self.user.GetDC.assert_not_called()
        self.gdi.CreateDIBSection.assert_not_called()

    def test_client_resize_during_capture_requests_full_frame_after_cleanup(self):
        self.desktop.client_rect.side_effect = [platform.Rect(-1920, 80, 1920, 1080),
                                                platform.Rect(-1920, 80, 1280, 720)]
        self.assertIsNone(self.desktop.capture_region(101, (1080, 1920, 3), (400, 250, 8, 6)))
        self.assert_cleanup()

    def test_client_move_during_capture_requests_full_frame_after_cleanup(self):
        self.desktop.client_rect.side_effect = [platform.Rect(-1920, 80, 1920, 1080),
                                                platform.Rect(0, 100, 1920, 1080)]
        self.assertIsNone(self.desktop.capture_region(101, (1080, 1920, 3), (400, 250, 8, 6)))
        self.assert_cleanup()

    def test_invalid_regions_request_full_frame_without_allocating_gdi(self):
        for region in (None, (), (1, 2, 3), (-1, 0, 8, 6), (0, 0, 0, 6),
                       (0, 0, 8, -1), (1917, 0, 8, 6), (0, 1077, 8, 6),
                       (0.5, 0, 8, 6), (math.nan, 0, 8, 6)):
            with self.subTest(region=region):
                self.assertIsNone(self.desktop.capture_region(101, (1080, 1920, 3), region))
        self.user.GetDC.assert_not_called()

    def test_background_window_refuses_region_without_capturing(self):
        self.desktop.is_foreground.return_value = False
        with self.assertRaises(platform.InputUnavailable):
            self.desktop.capture_region(101, (1080, 1920, 3), (400, 250, 8, 6))
        self.user.GetDC.assert_not_called()
        self.desktop.mouse_up.assert_called()

    def test_focus_change_during_capture_discards_frame_and_releases_input(self):
        self.desktop.is_foreground.side_effect = [True, False]
        with self.assertRaises(platform.InputUnavailable):
            self.desktop.capture_region(101, (1080, 1920, 3), (400, 250, 8, 6))
        self.assert_cleanup()
        self.desktop.mouse_up.assert_called()

    def test_conversion_is_exact_contiguous_uint8_and_detached_from_gdi_buffer(self):
        frame = self.desktop.capture_region(101, (1080, 1920, 3), (400, 250, 8, 6))
        self.assertEqual(frame.dtype, np.uint8)
        self.assertTrue(frame.flags.c_contiguous)
        np.testing.assert_array_equal(frame, np.tile(np.array((17, 83, 201), dtype=np.uint8), (6, 8, 1)))
        np.ctypeslib.as_array(self.buffer)[:] = 0
        self.assertEqual(tuple(frame[0, 0]), (17, 83, 201))
        self.assert_cleanup()

    def test_failed_bitblt_restores_and_releases_all_gdi_handles(self):
        self.gdi.BitBlt.return_value = False
        with self.assertRaises(platform.InputUnavailable):
            self.desktop.capture_region(101, (1080, 1920, 3), (400, 250, 8, 6))
        self.assert_cleanup()
        self.desktop.mouse_up.assert_called()

    def test_full_capture_keeps_original_client_bounds(self):
        self.desktop.client_rect.return_value = platform.Rect(-40, 70, 8, 6)
        frame = self.desktop.capture(101)
        self.assertEqual(frame.shape, (6, 8, 3))
        self.desktop.is_foreground.assert_not_called()
        self.gdi.BitBlt.assert_called_once_with(22, 0, 0, 8, 6, 11, -40, 70,
                                               0x00CC0020 | 0x40000000)
        self.assert_cleanup()


@unittest.skipUnless(sys.platform == "win32", "Windows input backend")
class DeadlinePressTests(unittest.TestCase):
    def setUp(self):
        self.desktop = platform.Desktop()
        self.desktop.is_available = MagicMock(return_value=True)
        self.desktop.is_foreground = MagicMock(return_value=True)
        self.desktop.cursor_over = MagicMock(return_value=True)
        self.desktop._send_raw = MagicMock(return_value=1)
        self.signal = MagicMock()
        self.signal.is_set.return_value = False
        self.signal.wait.return_value = False

    def flags(self):
        return [call.args[0][0].mi.dwFlags for call in self.desktop._send_raw.call_args_list]

    def test_expiry_during_final_guard_never_sends_button_down(self):
        clock = [1.0]
        guards = [0]

        def guarded_delay(_hwnd):
            guards[0] += 1
            if guards[0] == 2:
                clock[0] = 1.2

        self.desktop._guard = MagicMock(side_effect=guarded_delay)
        with patch.object(platform.time, "perf_counter", side_effect=lambda: clock[0]):
            self.assertIs(self.desktop.tap(101, cancel=self.signal, deadline=1.1), False)
        self.assertEqual(guards[0], 2)
        self.desktop._send_raw.assert_not_called()
        self.assertFalse(self.desktop._left_held)
        self.assertIsNone(self.desktop._held_hwnd)
        self.signal.wait.assert_not_called()

    def test_deadline_boundary_accepts_owned_press_then_matching_release(self):
        with patch.object(platform.time, "perf_counter", return_value=1.1):
            self.assertIs(self.desktop.tap(101, cancel=self.signal, deadline=1.1), True)
        self.assertEqual(self.flags(), [0x0002, 0x0004])
        self.signal.wait.assert_called_once_with(.005)
        self.assertFalse(self.desktop._left_held)
        self.assertIsNone(self.desktop._held_hwnd)

    def test_stop_during_final_guard_never_sends_button_down(self):
        signal = threading.Event()
        guards = [0]

        def stop_during_guard(_hwnd):
            guards[0] += 1
            if guards[0] == 2:
                signal.set()

        self.desktop._guard = MagicMock(side_effect=stop_during_guard)
        with patch.object(platform.time, "perf_counter", return_value=1.0):
            self.assertIs(self.desktop.tap(101, cancel=signal, deadline=1.1), False)
        self.assertEqual(guards[0], 2)
        self.desktop._send_raw.assert_not_called()
        self.assertFalse(self.desktop._left_held)
        self.assertIsNone(self.desktop._held_hwnd)

    def test_untimed_stop_during_final_guard_also_prevents_down(self):
        signal = threading.Event()
        guards = [0]

        def stop_during_guard(_hwnd):
            guards[0] += 1
            if guards[0] == 2:
                signal.set()

        self.desktop._guard = MagicMock(side_effect=stop_during_guard)
        self.assertIs(self.desktop.tap(101, cancel=signal), False)
        self.desktop._send_raw.assert_not_called()
        self.assertFalse(self.desktop._left_held)

    def test_expired_tap_releases_previous_owned_press_without_new_down(self):
        self.desktop._left_held = True
        self.desktop._held_hwnd = 101
        with patch.object(platform.time, "perf_counter", return_value=1.2):
            self.assertIs(self.desktop.tap(101, cancel=self.signal, deadline=1.1), False)
        self.assertEqual(self.flags(), [0x0004])
        self.assertFalse(self.desktop._left_held)

    def test_nonfinite_deadline_rejects_without_input(self):
        for deadline in (math.inf, -math.inf, math.nan):
            with self.subTest(deadline=deadline):
                with self.assertRaises(ValueError):
                    self.desktop.tap(101, cancel=self.signal, deadline=deadline)
        self.desktop._send_raw.assert_not_called()
        self.assertFalse(self.desktop._left_held)

    def test_cancelled_tap_returns_false_without_input(self):
        self.signal.is_set.return_value = True
        self.assertIs(self.desktop.tap(101, cancel=self.signal, deadline=1.1), False)
        self.desktop._send_raw.assert_not_called()
        self.signal.wait.assert_not_called()

    def test_plain_tap_preserves_immediate_press_and_release(self):
        with patch.object(platform.time, "perf_counter", side_effect=AssertionError("untimed press used a deadline")):
            self.assertIs(self.desktop.tap(101, cancel=self.signal), True)
        self.assertEqual(self.flags(), [0x0002, 0x0004])


class CachedRingObservationTests(unittest.TestCase):
    def setUp(self):
        config = json.loads((ROOT / "assets" / "default_config.json").read_text(encoding="utf-8"))
        config["charge_region"] = [0.64, 0.31, 0.22, 0.37]
        self.detector = Detector(config, ROOT / "assets")

    @staticmethod
    def frame(pointer_angle=math.pi):
        frame = np.full((600, 1000, 3), (60, 120, 90), dtype=np.uint8)
        crop = frame[200:400, 650:850]
        yy, xx = np.indices(crop.shape[:2])
        radius = 75
        distance = np.hypot(xx - 100, yy - 100)
        angle = np.arctan2(yy - 100, xx - 100)
        annulus = (distance >= radius * .73) & (distance <= radius)
        crop[annulus] = (8, 25, 8)
        delta = np.abs((angle + math.pi) % math.tau - math.pi)
        crop[annulus & (delta < .48)] = (0, 240, 180)
        crop[annulus & (delta < .075)] = (0, 255, 0)
        pointer_delta = np.abs((angle - pointer_angle + math.pi) % math.tau - math.pi)
        crop[(distance >= radius * .81) & (distance <= radius * .995)
             & (pointer_delta <= .052)] = (255, 255, 255)
        return frame

    def prime(self):
        frame = self.frame()
        observed = self.detector.analyze(frame, 1)
        self.assertIsNotNone(observed.charge)
        region = self.detector.charge_capture_region(frame.shape)
        self.assertIsNotNone(region)
        return frame, region

    def test_region_requires_prior_verified_ring_and_matching_client_size(self):
        self.assertIsNone(self.detector.charge_capture_region((600, 1000, 3)))
        frame, region = self.prime()
        self.assertIsNone(self.detector.charge_capture_region((1080, 1920, 3)))
        self.assertEqual(self.detector.charge_capture_region(frame.shape), region)

    def test_small_capture_matches_full_frame_track_and_client_coordinates(self):
        _, region = self.prime()
        frame = self.frame(.45)
        x, y, width, height = region
        cropped = frame[y:y + height, x:x + width].copy()
        small = self.detector.analyze_charge(cropped, region, 1.02)
        self.assertIsNotNone(small.charge)
        complete = self.detector.analyze(frame, 1.02)
        self.assertEqual(small.charge, complete.charge)
        self.assertAlmostEqual(small.charge.center[0], 750, delta=3)
        self.assertAlmostEqual(small.charge.center[1], 300, delta=3)
        self.assertEqual(small.charge.region, region)

    def test_absent_ring_disables_small_capture_for_full_frame_reacquisition(self):
        frame, region = self.prime()
        width, height = region[2:]
        observation = self.detector.analyze_charge(np.full((height, width, 3), 180, dtype=np.uint8), region, 1.02)
        self.assertIsNone(observation.charge)
        self.assertFalse(observation.ring_present)
        self.assertIsNone(self.detector.charge_capture_region(frame.shape))

    def test_wrong_region_and_crop_size_cannot_authorize_a_charge(self):
        frame, region = self.prime()
        x, y, width, height = region
        crop = frame[y:y + height, x:x + width].copy()
        for pixels, requested in ((crop[:-1], region), (crop, (x + 1, y, width, height)),
                                  (np.empty((0, 0, 3), dtype=np.uint8), region)):
            with self.subTest(shape=pixels.shape, region=requested):
                self.assertIsNone(self.detector.analyze_charge(pixels, requested, 1.02).charge)


if __name__ == "__main__":
    unittest.main()
