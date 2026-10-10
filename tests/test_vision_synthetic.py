"""Generated ring pixels: no game screenshot, desktop capture, or mouse input."""

import json
import math
import unittest
from pathlib import Path

import cv2
import numpy as np

from shrooms_macro.engine import Engine
from shrooms_macro.vision import Detector


ROOT = Path(__file__).resolve().parents[1]


def ring_frame(pointer_angle, goal_angle=0.0, height=1080, pointer=True, target=True):
    frame = np.full((1080, 1920, 3), (60, 120, 90), dtype=np.uint8)
    cx, cy, radius = 1074.0, 555.0, 93.0
    left, top = 960, 440
    crop = frame[top:670, left:1190]
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


if __name__ == "__main__":
    unittest.main()
