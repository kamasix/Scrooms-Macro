"""Generated controls: no game screenshot, desktop capture, or mouse input."""

import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from shrooms_macro.engine import Engine
from shrooms_macro.vision import Detector, TargetTrack, TuneTrack


ROOT = Path(__file__).resolve().parents[1]


def ring_frame(pointer_angle, goal_angle=0.0, height=1080, pointer=True, target=True,
               center=(1074.0, 555.0), radius=93.0):
    frame = np.full((1080, 1920, 3), (60, 120, 90), dtype=np.uint8)
    cx, cy = center
    margin = math.ceil(radius * 1.2 + 3)
    left, top = max(0, math.floor(cx - margin)), max(0, math.floor(cy - margin))
    right, bottom = min(1920, math.ceil(cx + margin)), min(1080, math.ceil(cy + margin))
    crop = frame[top:bottom, left:right]
    yy, xx = np.indices(crop.shape[:2])
    distance = np.hypot(xx + left - cx, yy + top - cy)
    angle = np.arctan2(yy + top - cy, xx + left - cx)
    annulus = (distance >= radius * 0.73) & (distance <= radius)
    crop[annulus] = (8, 25, 8)
    target_delta = np.abs((angle - goal_angle + math.pi) % math.tau - math.pi)
    if target:
        crop[annulus & (target_delta < 0.48)] = (0, 240, 180)
        crop[annulus & (target_delta < 0.075)] = (0, 255, 0)
    if pointer:
        delta = np.abs((angle - pointer_angle + math.pi) % math.tau - math.pi)
        marker = (distance >= radius * 0.81) & (distance <= radius * 0.995) & (delta <= 0.052)
        crop[marker] = (255, 255, 255)
    if height != 1080:
        frame = cv2.resize(frame, (round(1920 * height / 1080), height))
    return frame


class SyntheticRingTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "assets" / "default_config.json").read_text(encoding="utf-8"))
        self.detector = Detector(self.config.copy(), ROOT / "assets")

    def test_detects_green_across_zero_angle_and_multiple_resolutions(self):
        for height in (540, 720, 1080, 1440, 2160):
            with self.subTest(height=height):
                observed = Detector(self.config.copy(), ROOT / "assets").analyze(ring_frame(math.pi, height=height), 1)
                self.assertTrue(observed.ring_present)
                self.assertIsNotNone(observed.charge)
                charge = observed.charge
                self.assertAlmostEqual((charge.zone_angle + math.pi) % math.tau - math.pi, 0, delta=0.03)
                self.assertTrue(charge.zone_low > charge.zone_high)
                self.assertFalse(charge.marker_overlap)

    def test_current_white_pixels_on_green_tap_and_grey_waits(self):
        for angle, expected in ((math.pi, "up"), (0, "tap")):
            with self.subTest(angle=angle):
                detector = Detector(self.config.copy(), ROOT / "assets")
                detector.analyze(ring_frame(math.pi), 0.98)
                frame = ring_frame(angle)
                observation = detector.analyze(frame, 1)
                self.assertIsNotNone(observation.charge)
                engine = Engine(self.config.copy())
                engine.start(0)
                actions = engine.step(observation, frame.shape, 1.004, captured_at=1)
                self.assertEqual([action.kind for action in actions], [expected])

    def test_missing_markers_keep_ring_present_without_blind_click(self):
        for pointer, target in ((True, False), (False, True), (False, False)):
            with self.subTest(pointer=pointer, target=target):
                frame = ring_frame(math.pi, pointer=pointer, target=target)
                observed = Detector(self.config.copy(), ROOT / "assets").analyze(frame, 1)
                self.assertTrue(observed.ring_present)
                self.assertIsNone(observed.charge)
                engine = Engine(self.config.copy())
                engine.start(0)
                self.assertEqual([action.kind for action in engine.step(observed, frame.shape, 1.004, captured_at=1)], ["up"])

    def test_cached_shifted_ring_is_refitted_when_pointer_is_lost(self):
        # A previous legitimate outline can remain plausible after movement.
        frame = ring_frame(math.pi)
        original = self.detector.analyze(frame, 1)
        self.assertIsNotNone(original.charge)
        region, cx, cy, radius = self.detector._ring_geometry
        self.detector._ring_geometry = (region, cx - 14, cy, radius + 4)
        observed = self.detector.analyze(frame, 1.2)
        self.assertTrue(observed.ring_present)
        self.assertIsNotNone(observed.charge)
        self.assertAlmostEqual(observed.charge.center[0], 1074, delta=2)
        self.assertAlmostEqual(observed.charge.center[1], 555, delta=2)

    @staticmethod
    def clipped_config(config):
        # A generated manual rectangle starts inside the synthetic circle.
        # Even its usual padding leaves the left-hand pointer outside the crop.
        config = config.copy()
        config["charge_region"] = [1020 / 1920, 470 / 1080, 160 / 1920, 165 / 1080]
        return config

    def test_clipped_left_pointer_and_perfect_are_recovered_before_clicking(self):
        config = self.clipped_config(self.config)
        for height in (540, 720, 1080, 1440, 2160):
            with self.subTest(height=height):
                detector = Detector(config.copy(), ROOT / "assets")
                # First the omitted left pointer is over the dark ring. Then
                # the omitted Perfect segment is visible while the pointer is
                # on the right. Finally the pointer reaches that left target.
                sequence = ((math.pi, 0.0, "up"), (0.0, math.pi, "up"),
                            (math.pi, math.pi, "tap"))
                for index, (pointer_angle, goal_angle, expected) in enumerate(sequence):
                    frame = ring_frame(pointer_angle, goal_angle, height=height)
                    now = 1 + index * 0.02
                    observation = detector.analyze(frame, now)
                    self.assertTrue(observation.ring_present)
                    self.assertIsNotNone(observation.charge)
                    charge = observation.charge
                    self.assertEqual(charge.marker_overlap, expected == "tap")
                    if pointer_angle == math.pi:
                        # Recovery must read pixels omitted by calibration,
                        # rather than infer a click from the visible green.
                        padded_left = (config["charge_region"][0]
                                       - config["charge_region"][2] * 0.12) * frame.shape[1]
                        self.assertLess(charge.marker_point[0], padded_left)
                        self.assertLess(charge.region[0], charge.marker_point[0])
                    engine = Engine(config.copy())
                    engine.start(0)
                    actions = engine.step(observation, frame.shape, now + 0.004,
                                          captured_at=now)
                    self.assertEqual([action.kind for action in actions], [expected])

    def test_recovered_crop_follows_resolution_changes_without_old_pixel_coordinates(self):
        config = self.clipped_config(self.config)
        detector = Detector(config.copy(), ROOT / "assets")
        for index, height in enumerate((1080, 540, 2160, 720, 1080)):
            with self.subTest(height=height, index=index):
                now = 1 + index * 0.10
                detector.analyze(ring_frame(0.0, math.pi, height=height), now)
                frame = ring_frame(math.pi, math.pi, height=height)
                observation = detector.analyze(frame, now + 0.02)
                self.assertIsNotNone(observation.charge)
                charge = observation.charge
                scale = height / 1080
                self.assertAlmostEqual(charge.center[0], 1074 * scale, delta=6 * scale)
                self.assertAlmostEqual(charge.center[1], 555 * scale, delta=6 * scale)
                self.assertTrue(charge.marker_overlap)
                engine = Engine(config.copy())
                engine.start(0)
                actions = engine.step(observation, frame.shape, now + 0.024,
                                      captured_at=now + 0.02)
                self.assertEqual([action.kind for action in actions], ["tap"])

    def test_recovered_crop_cannot_click_an_expired_missing_green_target(self):
        config = self.clipped_config(self.config)
        detector = Detector(config.copy(), ROOT / "assets")
        known = detector.analyze(ring_frame(0.0, math.pi), 1)
        self.assertIsNotNone(known.charge)
        frame = ring_frame(math.pi, math.pi, target=False)
        observation = detector.analyze(frame, 1.5)
        self.assertTrue(observation.ring_present)
        self.assertIsNone(observation.charge)
        engine = Engine(config.copy())
        engine.start(0)
        actions = engine.step(observation, frame.shape, 1.504, captured_at=1.5)
        self.assertEqual([action.kind for action in actions], ["up"])

    @staticmethod
    def outline_clipped_config(config, center, radius):
        # Even after normal padding, this rectangle exposes less than 45% of
        # the circle. Its center and left pointer are outside the first crop,
        # so recovery cannot depend on already having verified that outline.
        cx, cy = center
        config = config.copy()
        config["charge_region"] = [(cx + radius * 0.45) / 1920,
                                   (cy - radius * 0.95) / 1080,
                                   radius * 1.6 / 1920, radius * 1.9 / 1080]
        return config

    def test_initially_unverified_clipped_outline_is_found_locally_at_multiple_sizes(self):
        for center, radius in (((980.0, 540.0), 70.0), ((1074.0, 555.0), 93.0),
                               ((1150.0, 590.0), 115.0)):
            config = self.outline_clipped_config(self.config, center, radius)
            for height in (540, 720, 1080, 1440, 2160):
                with self.subTest(center=center, radius=radius, height=height):
                    detector = Detector(config.copy(), ROOT / "assets")
                    # The left pointer is over grey before it reaches a small
                    # green goal at the top of the same generated circle.
                    for index, (angle, expected) in enumerate(((math.pi, "up"),
                                                               (-math.pi / 2, "tap"))):
                        frame = ring_frame(angle, -math.pi / 2, height=height,
                                           center=center, radius=radius)
                        now = 1 + index * 0.02
                        observation = detector.analyze(frame, now)
                        self.assertTrue(observation.ring_present)
                        self.assertIsNotNone(observation.charge)
                        charge = observation.charge
                        scale = height / 1080
                        self.assertAlmostEqual(charge.center[0], center[0] * scale,
                                               delta=radius * scale * 0.06)
                        self.assertAlmostEqual(charge.center[1], center[1] * scale,
                                               delta=radius * scale * 0.06)
                        self.assertEqual(charge.marker_overlap, expected == "tap")
                        engine = Engine(config.copy())
                        engine.start(0)
                        actions = engine.step(observation, frame.shape, now + 0.004,
                                              captured_at=now)
                        self.assertEqual([action.kind for action in actions], [expected])

    def test_full_client_discovery_selects_a_verified_distant_ring_over_colored_scenery(self):
        config = self.outline_clipped_config(self.config, (1074.0, 555.0), 93.0)
        for height in (540, 720, 1080, 1440, 2160):
            with self.subTest(height=height):
                # A moved, fully verified ring is found despite unrelated
                # green and white scenery near the old calibration rectangle.
                frame = ring_frame(0.0, 0.0, center=(300.0, 250.0))
                cv2.rectangle(frame, (1080, 520), (1120, 570), (0, 255, 0), -1)
                cv2.circle(frame, (1140, 555), 10, (255, 255, 255), -1)
                if height != 1080:
                    frame = cv2.resize(frame, (round(1920 * height / 1080), height))
                observation = Detector(config.copy(), ROOT / "assets").analyze(frame, 1)
                self.assertTrue(observation.ring_present)
                self.assertIsNotNone(observation.charge)
                scale = height / 1080
                self.assertAlmostEqual(observation.charge.center[0], 300 * scale, delta=4 * scale)
                self.assertAlmostEqual(observation.charge.center[1], 250 * scale, delta=4 * scale)
                self.assertTrue(observation.charge.marker_overlap)

    def test_remote_corners_are_reacquired_and_current_overlap_is_tracked(self):
        for center in ((180.0, 170.0), (1740.0, 170.0), (180.0, 880.0), (1740.0, 880.0)):
            for height in (540, 720, 1080, 1440, 2160):
                with self.subTest(center=center, height=height):
                    detector = Detector(self.config.copy(), ROOT / "assets")
                    for index, (angle, expected) in enumerate(((math.pi, "up"), (0.0, "tap"))):
                        frame = ring_frame(angle, height=height, center=center)
                        now = 1 + index * 0.02
                        observation = detector.analyze(frame, now)
                        self.assertTrue(observation.ring_present)
                        self.assertIsNotNone(observation.charge)
                        scale = height / 1080
                        self.assertAlmostEqual(observation.charge.center[0], center[0] * scale, delta=4 * scale)
                        self.assertAlmostEqual(observation.charge.center[1], center[1] * scale, delta=4 * scale)
                        self.assertEqual(observation.charge.marker_overlap, expected == "tap")
                        engine = Engine(self.config.copy())
                        engine.start(0)
                        actions = engine.step(observation, frame.shape, now + 0.004, captured_at=now)
                        self.assertEqual([action.kind for action in actions], [expected])

    def test_far_movement_replaces_geometry_and_recovered_fast_path_uses_no_hough(self):
        detector = Detector(self.config.copy(), ROOT / "assets")
        initial = detector.analyze(ring_frame(math.pi), 1)
        self.assertIsNotNone(initial.charge)
        for index, center in enumerate(((180.0, 170.0), (1740.0, 880.0))):
            now = 1.35 + index * 0.35
            observation = detector.analyze(ring_frame(math.pi, center=center), now)
            self.assertIsNotNone(observation.charge)
            self.assertAlmostEqual(observation.charge.center[0], center[0], delta=4)
            self.assertAlmostEqual(observation.charge.center[1], center[1], delta=4)
            with patch("shrooms_macro.vision.cv2.HoughCircles", side_effect=AssertionError("Cached valid geometry must stay fast")):
                overlap = detector.analyze(ring_frame(0.0, center=center), now + 0.02)
            self.assertIsNotNone(overlap.charge)
            self.assertTrue(overlap.charge.marker_overlap)

    def test_remote_cache_is_invalidated_on_resize_and_absence(self):
        detector = Detector(self.config.copy(), ROOT / "assets")
        center = (1740.0, 170.0)
        for index, height in enumerate((1080, 540, 2160, 720)):
            with self.subTest(height=height):
                observation = detector.analyze(ring_frame(math.pi, height=height, center=center), 1 + index * 0.02)
                self.assertIsNotNone(observation.charge)
                scale = height / 1080
                self.assertAlmostEqual(observation.charge.center[0], center[0] * scale, delta=4 * scale)
                self.assertAlmostEqual(observation.charge.center[1], center[1] * scale, delta=4 * scale)
        blank = np.full((720, 1280, 3), (60, 120, 90), dtype=np.uint8)
        absent = detector.analyze(blank, 1.5)
        self.assertIsNone(absent.charge)
        self.assertFalse(absent.ring_present)
        self.assertIsNone(detector._ring_recovered_region)
        self.assertIsNone(detector._ring_geometry)
        self.assertIsNone(detector._ring_zone)

    def test_green_white_scenery_and_broad_green_ring_cannot_authorize_a_track(self):
        for background in ("scenery", "broad_ring"):
            with self.subTest(background=background):
                frame = np.full((1080, 1920, 3), (60, 120, 90), dtype=np.uint8)
                if background == "scenery":
                    cv2.rectangle(frame, (150, 140), (180, 180), (0, 255, 0), -1)
                    cv2.circle(frame, (250, 170), 8, (255, 255, 255), -1)
                    cv2.circle(frame, (1740, 880), 90, (0, 255, 0), -1)
                    cv2.rectangle(frame, (1650, 870), (1670, 885), (255, 255, 255), -1)
                else:
                    frame = ring_frame(math.pi, center=(180.0, 170.0), target=False)
                    yy, xx = np.indices(frame.shape[:2])
                    radius = np.hypot(xx - 180, yy - 170)
                    angle = np.arctan2(yy - 170, xx - 180)
                    frame[(radius >= 93 * 0.73) & (radius <= 93) & (np.abs(angle) < 0.8)] = (0, 255, 0)
                observation = Detector(self.config.copy(), ROOT / "assets").analyze(frame, 1)
                self.assertIsNone(observation.charge)

    def test_full_search_is_throttled_and_hough_never_scans_the_full_client(self):
        detector = Detector(self.config.copy(), ROOT / "assets")
        frame = np.full((1080, 1920, 3), (60, 120, 90), dtype=np.uint8)
        detector.analyze(frame, 1)
        detector.analyze(frame, 1.02)
        self.assertEqual(detector._last_full_ring_search[2], 1)
        detector.analyze(frame, 1.35)
        self.assertEqual(detector._last_full_ring_search[2], 1.35)
        original_hough = cv2.HoughCircles

        def bounded_hough(image, *args, **kwargs):
            self.assertLess(image.shape[0] * image.shape[1], frame.shape[0] * frame.shape[1] / 4)
            return original_hough(image, *args, **kwargs)

        with patch("shrooms_macro.vision.cv2.HoughCircles", side_effect=bounded_hough):
            moved = detector.analyze(ring_frame(math.pi, center=(1740.0, 880.0)), 1.7)
        self.assertIsNotNone(moved.charge)

    def test_two_distinct_fully_verified_remote_controls_are_ambiguous(self):
        frame = ring_frame(math.pi, center=(180.0, 170.0))
        second = ring_frame(math.pi, center=(1740.0, 880.0))
        frame[740:1020, 1600:1880] = second[740:1020, 1600:1880]
        detector = Detector(self.config.copy(), ROOT / "assets")
        observed = detector.analyze(frame, 1)
        self.assertIsNone(observed.charge)
        self.assertIsNone(detector._ring_recovered_region)

    def test_recognized_tuning_or_catch_controls_skip_full_ring_reacquisition(self):
        # A plausible old outline without a pointer must not make global
        # recovery interrupt a later minigame already recognized this frame.
        frame = ring_frame(math.pi, pointer=False)
        for mode in ("tuning", "catch"):
            with self.subTest(mode=mode):
                detector = Detector(self.config.copy(), ROOT / "assets")
                tuning = TuneTrack(900, 850, (400, 650, 1050, 300), 0.9, 840) if mode == "tuning" else None
                frequency = TargetTrack(900, 850, 0.9, (400, 800, 1050, 150))
                with patch.object(detector, "_reacquire_ring", side_effect=AssertionError("Other minigames must stay fast")):
                    with patch.object(detector, "_tuning", return_value=tuning), \
                         patch.object(detector, "_progress", return_value=0.5), \
                         patch.object(detector, "_catch_presence", return_value=True), \
                         patch.object(detector, "_catching", return_value=frequency):
                        observed = detector.analyze(frame, 1)
                if mode == "tuning":
                    self.assertIs(observed.tuning, tuning)
                else:
                    self.assertTrue(observed.catch_present)
                    self.assertIs(observed.frequency, frequency)

    def test_locally_discovered_outline_without_markers_cannot_authorize_a_click(self):
        config = self.outline_clipped_config(self.config, (1074.0, 555.0), 93.0)
        for pointer, target in ((True, False), (False, True), (False, False)):
            with self.subTest(pointer=pointer, target=target):
                frame = ring_frame(math.pi, pointer=pointer, target=target)
                observation = Detector(config.copy(), ROOT / "assets").analyze(frame, 1)
                self.assertTrue(observation.ring_present)
                self.assertIsNone(observation.charge)
                engine = Engine(config.copy())
                engine.start(0)
                actions = engine.step(observation, frame.shape, 1.004, captured_at=1)
                self.assertEqual([action.kind for action in actions], ["up"])


class SyntheticSaleTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.assets = Path(self.directory.name)
        self.button = np.zeros((40, 155, 3), dtype=np.uint8)
        cv2.putText(self.button, "SELL ALL", (3, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2, cv2.LINE_AA)
        for name in ("sell", "confirm"):
            self.assertTrue(cv2.imwrite(str(self.assets / f"{name}.png"), self.button))
        self.config = json.loads((ROOT / "assets" / "default_config.json").read_text(encoding="utf-8"))
        self.config["template_regions"] = {}
        self.config["template_source_resolutions"] = {"sell": [1920, 1080], "confirm": [1920, 1080]}
        self.frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        self.frame[690:730, 1440:1595] = self.button

    def test_flat_correlation_peaks_do_not_hide_distant_sale_button(self):
        self.config["template_regions"] = {"sell": [0.05, 0.05, 0.07, 0.04]}
        detector = Detector(self.config, self.assets)
        match_template = cv2.matchTemplate

        def with_flat_peaks(*args, **kwargs):
            result = match_template(*args, **kwargs)
            # Some OpenCV builds return correlation 1.0 on constant black
            # windows. Reproduce that undefined score independently of the
            # installed build so the visible real button must still be found.
            for column in range(5, min(77, result.shape[1]), 6):
                result[min(5, result.shape[0] - 1), column] = 1.0
            return result

        with patch("shrooms_macro.vision.cv2.matchTemplate", side_effect=with_flat_peaks):
            observed = detector.analyze(self.frame, 10.0)
        self.assertIsNotNone(observed.sell)
        self.assertAlmostEqual(observed.sell.x, 1517.5, delta=4)
        self.assertAlmostEqual(observed.sell.y, 710, delta=4)
        self.assertGreater(observed.sell.score, 0.76)

    def test_global_action_buttons_wait_instead_of_replaying_cached_matches(self):
        detector = Detector(self.config, self.assets)
        observed = detector.analyze(self.frame, 10.0)
        self.assertIsNotNone(observed.sell)
        self.assertIsNotNone(observed.confirm)
        with patch.object(detector, "_match_in", wraps=detector._match_in) as scan:
            gone = detector.analyze(np.zeros_like(self.frame), 10.05)
        self.assertIsNone(gone.sell)
        self.assertIsNone(gone.confirm)
        scan.assert_not_called()
        refreshed = detector.analyze(self.frame, 10.25)
        self.assertIsNotNone(refreshed.sell)
        self.assertIsNotNone(refreshed.confirm)

    def test_manual_sale_mode_requires_no_sale_template_but_keeps_tuning_setup(self):
        self.config["auto_sell"] = False
        with tempfile.TemporaryDirectory() as directory:
            assets = Path(directory)
            self.assertTrue(cv2.imwrite(str(assets / "tune_prompt.png"), self.button))
            detector = Detector(self.config, assets)
            self.assertEqual(detector.ready(), [])
            self.config["auto_sell"] = True
            self.assertIn("Sell All template is missing", detector.ready())
            self.config["auto_sell"] = False
            detector.templates.pop("tune_prompt")
            with patch.object(detector, "_refresh_templates"):
                self.assertIn("Tuning prompt template is missing", detector.ready())

    def test_manual_sale_mode_observes_inventory_without_confirmation_matching(self):
        self.config["auto_sell"] = False
        detector = Detector(self.config, self.assets)
        with patch.object(detector, "_match", wraps=detector._match) as match:
            observed = detector.analyze(self.frame, 10)
        self.assertIsNotNone(observed.sell)
        self.assertIsNone(observed.confirm)
        self.assertNotIn("confirm", [call.args[1] for call in match.call_args_list])


if __name__ == "__main__":
    unittest.main()
