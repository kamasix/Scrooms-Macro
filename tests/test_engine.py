import math
import unittest

from shrooms_macro.engine import Engine
from shrooms_macro.vision import Observation, TargetTrack, TuneTrack, BarTrack, Match


class EngineChecks(unittest.TestCase):
    def setUp(self):
        self.config = {"sell_every": 10, "charge_action": "click", "input_latency": 0}
        self.engine = Engine(self.config)
        self.engine.start(0)

    def step(self, obs, now):
        return self.engine.step(obs, (1080, 1920, 3), now)

    def catch(self, progress, now):
        target = TargetTrack(800, 900, .95, (550, 800, 800, 150), 900, 260, 800, progress)
        return self.step(Observation(frequency=target, catch_present=True, catch_progress=progress), now)

    def test_occlusion_does_not_count_complete_progress_as_catch(self):
        self.catch(.97, 1)
        self.catch(.98, 2)
        self.step(Observation(catch_present=True, catch_progress=.99), 2.25)
        self.assertEqual(self.engine.caught, 0)
        self.assertEqual(self.engine.phase, "catch")
        self.step(Observation(), 2.46)
        self.assertEqual(self.engine.caught, 1)
        self.step(Observation(), 2.7)
        self.assertEqual(self.engine.caught, 1)

    def test_disappearing_failed_catch_is_not_counted(self):
        self.catch(.98, 1)
        self.catch(.2, 2)
        self.catch(.01, 3)
        self.step(Observation(), 3.2)
        self.assertEqual(self.engine.caught, 0)

    def test_unrelated_success_or_persistent_notification_is_not_counted(self):
        self.step(Observation(success=True), 1)
        self.assertEqual(self.engine.caught, 0)
        self.catch(.4, 2)
        self.step(Observation(success=True), 3)
        self.assertEqual(self.engine.caught, 1)
        self.step(Observation(success=True), 4)
        self.assertEqual(self.engine.caught, 1)

    def test_resume_with_untracked_visible_catch_does_not_cast(self):
        actions = self.step(Observation(catch_present=True, catch_progress=.5), 1)
        self.assertEqual(actions, [])
        self.assertEqual(self.engine.phase, "catch")

    def test_perfect_angle_wrap_taps_in_place(self):
        self.engine.phase = "charge"
        self.engine.previous_charge = (math.tau - .05, .98)
        ring = BarTrack((100, 100, 180, 180), 0.03, math.tau-.12, .14, "angle", .99)
        self.assertEqual([a.kind for a in self.step(Observation(charge=ring), 1)], ["tap"])
        self.assertEqual(self.engine.phase, "transition")

    def test_release_variant_remains_available(self):
        self.config["charge_action"] = "release"
        self.engine.phase = "charge"
        self.engine.previous_charge = (math.tau - .05, .98)
        ring = BarTrack((100, 100, 180, 180), 0.03, math.tau-.12, .14, "angle", .99)
        self.assertEqual([a.kind for a in self.step(Observation(charge=ring), 1)], ["up"])

    def test_ring_does_not_click_on_grey(self):
        ring = BarTrack((100, 100, 180, 180), 0.5, 4.05, 4.22, "angle", .99)
        self.assertEqual([a.kind for a in self.step(Observation(charge=ring), 1)], ["up"])
        self.assertEqual(self.engine.phase, "charge")

    def test_start_on_existing_ring_does_not_press_on_grey(self):
        ring = BarTrack((100, 100, 180, 180), .5, 4.05, 4.22, "angle", .99)
        actions = self.step(Observation(charge=ring), 1)
        self.assertFalse(any(a.kind in ("tap", "down") for a in actions))

    def test_first_fresh_green_capture_clicks_observed_overlap(self):
        ring = BarTrack((100, 100, 180, 180), 1, .95, 1.05, "angle", .99)
        self.assertEqual([a.kind for a in self.step(Observation(charge=ring), 1)], ["tap"])
        self.assertEqual(self.engine.phase, "transition")

    def test_predicted_future_green_cannot_trigger_from_observed_grey(self):
        # Extrapolation puts .88 into green, but the actual white marker is grey.
        region = (100, 100, 180, 180)
        first = BarTrack(region, .68, .95, 1.05, "angle", .99)
        second = BarTrack(region, .88, .95, 1.05, "angle", .99)
        self.engine.step(Observation(charge=first), (1080, 1920, 3), 1.03, captured_at=1)
        actions = self.engine.step(Observation(charge=second), (1080, 1920, 3), 1.08, captured_at=1.05)
        self.assertEqual([a.kind for a in actions], ["up"])

    def test_recent_green_that_has_already_passed_at_dispatch_does_not_click(self):
        region = (100, 100, 180, 180)
        first = BarTrack(region, .78, .95, 1.05, "angle", .99)
        second = BarTrack(region, .98, .95, 1.05, "angle", .99)
        self.engine.step(Observation(charge=first), (1080, 1920, 3), 1, captured_at=1)
        actions = self.engine.step(Observation(charge=second), (1080, 1920, 3), 1.07, captured_at=1.05)
        self.assertEqual([a.kind for a in actions], ["up"])

    def test_ring_rejects_overlap_if_input_latency_would_land_after_perfect(self):
        self.config["input_latency"] = .008
        region = (100, 100, 180, 180)
        first = BarTrack(region, .90, .95, 1.05, "angle", .99)
        late = BarTrack(region, 1.02, .95, 1.05, "angle", .99)
        self.step(Observation(charge=first), .98)
        # The captured pointer is green and the frame is fresh, but 6 rad/s
        # advances another .048 rad before the click reaches the game.
        actions = self.step(Observation(charge=late), 1)
        self.assertEqual([a.kind for a in actions], ["up"])
        self.assertEqual(self.engine.phase, "charge")

    def test_ring_clicks_at_early_overlap_when_latency_impact_stays_in_perfect(self):
        self.config["input_latency"] = .008
        region = (100, 100, 180, 180)
        first = BarTrack(region, .890, .95, 1.05, "angle", .99)
        early = BarTrack(region, .958, .95, 1.05, "angle", .99)
        self.step(Observation(charge=first), .98)
        # Observed early in green, not on grey; 4 ms frame age + 8 ms input
        # delay places this 3.4 rad/s pointer close to the target center.
        actions = self.engine.step(Observation(charge=early), (1080, 1920, 3),
                                   1.004, captured_at=1)
        self.assertEqual([a.kind for a in actions], ["tap"])

    def test_ring_latency_projection_cannot_trigger_from_fresh_observed_grey(self):
        self.config["input_latency"] = .008
        region = (100, 100, 180, 180)
        self.step(Observation(charge=BarTrack(region, .84, .95, 1.05, "angle", .99)), .98)
        # Projection reaches .98 (inside Perfect), but the observed .94 is
        # outside the narrow green. This must never authorize an early click.
        actions = self.step(Observation(charge=BarTrack(region, .94, .95, 1.05, "angle", .99)), 1)
        self.assertEqual([a.kind for a in actions], ["up"])

    def test_ring_latency_works_in_reverse_rotation_at_the_other_entry_edge(self):
        self.config["input_latency"] = .008
        region = (100, 100, 180, 180)
        self.step(Observation(charge=BarTrack(region, 1.130, .95, 1.05, "angle", .99)), .98)
        actions = self.step(Observation(charge=BarTrack(region, 1.042, .95, 1.05, "angle", .99)), 1)
        self.assertEqual([a.kind for a in actions], ["tap"])

    def test_ring_latency_uses_preserved_duplicate_frame_speed_across_angle_zero(self):
        self.config["input_latency"] = .008
        region = (100, 100, 180, 180)
        for angle, now in ((6.05, .96), (6.13, .98), (6.13, .985)):
            actions = self.step(Observation(charge=BarTrack(region, angle, 6.20, .08, "angle", .99)), now)
            self.assertEqual([a.kind for a in actions], ["up"])
        self.assertAlmostEqual(self.engine._charge_velocity(6.13, .988, True), 4)
        early = BarTrack(region, 6.21, 6.20, .08, "angle", .99)
        actions = self.engine.step(Observation(charge=early), (1080, 1920, 3),
                                   1.004, captured_at=1)
        self.assertEqual([a.kind for a in actions], ["tap"])

    def test_lost_ring_never_becomes_an_unobserved_fresh_press(self):
        ring = BarTrack((100, 100, 180, 180), .5, 4.05, 4.22, "angle", .99)
        self.step(Observation(charge=ring), 1)
        for now in (1.8, 2.8, 4, 7):
            self.assertEqual([a.kind for a in self.step(Observation(), now)], ["up"])
            self.assertEqual(self.engine.phase, "charge")
        self.assertEqual([a.kind for a in self.step(Observation(), 12)], ["up"])
        self.assertEqual(self.engine.error, "")
        self.assertEqual(self.engine.phase, "charge")
        self.assertIn("rechecking", self.engine.status)

    def test_missing_perfect_can_recover_after_the_old_timeout(self):
        missing = Observation(ring_present=True, ring_diagnostic="White pointer not detected")
        for now in (1, 12, 30):
            self.assertEqual([a.kind for a in self.step(missing, now)], ["up"])
            self.assertEqual(self.engine.error, "")
        self.assertIn("White pointer not detected", self.engine.status)
        fresh = BarTrack((100, 100, 180, 180), 1, .95, 1.05, "angle", .99)
        self.assertEqual([a.kind for a in self.step(Observation(charge=fresh), 30.01)], ["tap"])
        self.assertEqual(self.engine.phase, "transition")

    def test_recognized_grey_keeps_tracking_after_the_old_timeout(self):
        grey = BarTrack((100, 100, 180, 180), .5, .95, 1.05, "angle", .99)
        for now in (1, 12, 30):
            self.assertEqual([a.kind for a in self.step(Observation(charge=grey), now)], ["up"])
            self.assertEqual(self.engine.phase, "charge")
            self.assertEqual(self.engine.error, "")
        fresh = BarTrack(grey.region, 1, .95, 1.05, "angle", .99)
        self.assertEqual([a.kind for a in self.step(Observation(charge=fresh), 31)], ["tap"])

    def test_rechecking_still_rejects_stale_perfect_frames(self):
        self.step(Observation(ring_present=True), 1)
        self.step(Observation(), 20)
        green = BarTrack((100, 100, 180, 180), 1, .95, 1.05, "angle", .99)
        actions = self.engine.step(Observation(charge=green), (1080, 1920, 3),
                                   30.3, captured_at=30)
        self.assertEqual([a.kind for a in actions], ["up"])
        self.assertEqual(self.engine.phase, "charge")
        self.assertEqual(self.engine.error, "")

    def test_initial_instrument_press_is_once_and_waits_for_ui(self):
        self.assertEqual([a.kind for a in self.step(Observation(), 1)], ["down"])
        self.assertEqual(self.step(Observation(), 1.2), [])
        self.assertEqual([a.kind for a in self.step(Observation(), 1.6)], ["up"])
        self.assertEqual([a.kind for a in self.step(Observation(), 3)], ["up"])

    def test_unaccepted_perfect_click_reobserves_ring_without_blind_cast(self):
        ring = BarTrack((100, 100, 180, 180), 1, .95, 1.05, "angle", .99)
        self.step(Observation(charge=ring), 1)
        self.assertEqual([a.kind for a in self.step(Observation(), 3)], ["up"])
        self.assertEqual(self.engine.phase, "charge")
        self.assertEqual([a.kind for a in self.step(Observation(), 4)], ["up"])

    def test_new_ring_during_start_waits_even_if_marker_is_unrecognized(self):
        # Presence must protect a pre-existing ring before the first detection.
        observation = Observation(ring_present=True)
        self.assertEqual([a.kind for a in self.step(observation, 1)], ["up"])
        self.assertEqual(self.engine.phase, "charge")

    def test_duplicate_rendered_frames_preserve_angular_velocity(self):
        region = (100, 100, 180, 180)
        self.step(Observation(charge=BarTrack(region, .4, .95, 1.05, "angle", .99)), 1)
        self.step(Observation(charge=BarTrack(region, .6, .95, 1.05, "angle", .99)), 1.05)
        self.step(Observation(charge=BarTrack(region, .6, .95, 1.05, "angle", .99)), 1.06)
        speed = self.engine._charge_velocity(.6, 1.065, True)
        self.assertAlmostEqual(speed, 4.0)

    def test_stale_green_frame_cannot_trigger_click(self):
        ring = BarTrack((100, 100, 180, 180), 1.0, .95, 1.05, "angle", .99)
        actions = self.engine.step(Observation(charge=ring), (1080, 1920, 3), 1.3, captured_at=1)
        self.assertNotIn("tap", [a.kind for a in actions])

    def timed_ring_frame(self, marker, captured_at, age=.003, *, center=1,
                         width=.15, geometry=(120, 120), confidence=.99,
                         observed=True):
        ring = BarTrack((0, 0, 240, 240), marker % math.tau,
                        (center - width / 2) % math.tau,
                        (center + width / 2) % math.tau, "angle", confidence,
                        center=geometry, radius=84, zone_observed=observed,
                        outline_radius=100)
        return self.engine.step(Observation(charge=ring), (1080, 1920, 3),
                                captured_at + age, captured_at=captured_at)

    def assert_timed_green_window(self, action, marker, captured_at, speed,
                                 *, center=1, width=.15, age=.003, latency=.008):
        self.assertEqual(action.kind, "tap")
        self.assertIsNotNone(action.execute_at)
        self.assertLessEqual(captured_at + age, action.not_before)
        self.assertLessEqual(action.not_before, action.execute_at)
        self.assertLessEqual(action.execute_at, action.expires_at)
        self.assertLessEqual(action.execute_at - captured_at - age, .030 + 1e-10)
        self.assertLessEqual(action.expires_at - captured_at, .025 + 1e-10)
        # Every permitted dispatch, including the last possible one, must
        # remain green throughout the configured input-delay interval.
        for dispatch in (action.not_before, action.execute_at, action.expires_at):
            for impact in (dispatch, dispatch + latency):
                position = marker + speed * (impact - captured_at)
                delta = (position - center + math.pi) % math.tau - math.pi
                self.assertLessEqual(abs(delta), width / 2)

    def test_stable_near_entry_model_waits_for_actual_green_before_dispatch(self):
        self.config["input_latency"] = .008
        action = None
        for index in range(50):
            captured, marker = 1 + index * .004, .2 + 5.3 * index * .004
            actions = self.timed_ring_frame(marker, captured)
            action = next((a for a in actions if a.kind == "tap"), None)
            if action is not None:
                break
        self.assertIsNotNone(action)
        self.assertLess(marker, .925)
        self.assert_timed_green_window(action, marker, captured, 5.3)
        self.assertGreater(action.not_before, captured + .003)
        self.assertEqual(self.engine.phase, "transition")

    def test_stable_observed_green_tap_also_has_an_expiry(self):
        self.config["input_latency"] = .008
        # Keep the optional release mode during grey observations to reach a
        # green capture without accepting an earlier click-mode prediction.
        self.config["charge_action"] = "release"
        for index in range(35):
            self.timed_ring_frame(.2 + 5.3 * index * .004, 1 + index * .004)
        self.config["charge_action"] = "click"
        action, = self.timed_ring_frame(.942, 1.14)
        self.assert_timed_green_window(action, .942, 1.14, 5.3)
        # A 20 ms preflight delay must not be allowed to reuse this tap.
        self.assertLess(action.expires_at, 1.163)

    def test_three_positions_do_not_authorize_near_entry_prediction(self):
        self.config["input_latency"] = .008
        for marker, captured in ((.74, 1), (.80, 1.02), (.86, 1.04)):
            actions = self.timed_ring_frame(marker, captured)
            self.assertEqual([a.kind for a in actions], ["up"])
        self.assertFalse(self.engine.charge_timing_reliable)

    def test_stable_far_grey_never_schedules_a_later_revolution(self):
        self.config["input_latency"] = .008
        for marker, captured in ((.2, 1), (.3, 1.02), (.4, 1.04), (.5, 1.06)):
            actions = self.timed_ring_frame(marker, captured)
            self.assertEqual([a.kind for a in actions], ["up"])
        self.assertTrue(self.engine.charge_timing_reliable)
        self.assertEqual(self.engine.phase, "charge")

    def test_stable_late_green_with_no_dispatch_window_is_rejected(self):
        self.config["input_latency"] = .008
        for marker, captured in ((.64, 1), (.76, 1.02), (.88, 1.04)):
            self.assertEqual([a.kind for a in self.timed_ring_frame(marker, captured)], ["up"])
        actions = self.timed_ring_frame(1.00, 1.06, age=.002)
        self.assertEqual([a.kind for a in actions], ["up"])
        self.assertTrue(self.engine.charge_timing_reliable)
        self.assertEqual(self.engine.phase, "charge")

    def test_reverse_prediction_preserves_the_green_window_across_angle_zero(self):
        self.config["input_latency"] = .008
        action = None
        for index in range(50):
            captured, marker = 1 + index * .004, .8 - 7 * index * .004
            actions = self.timed_ring_frame(marker, captured, center=.015)
            action = next((a for a in actions if a.kind == "tap"), None)
            if action is not None:
                break
        self.assertIsNotNone(action)
        self.assert_timed_green_window(action, marker, captured, -7, center=.015)

    def test_duplicate_rendered_frame_retains_motion_without_becoming_new_evidence(self):
        self.config["input_latency"] = .008
        for marker, captured in ((.5, 1), (.62, 1.02), (.74, 1.04)):
            self.timed_ring_frame(marker, captured)
        self.assertEqual([a.kind for a in self.timed_ring_frame(.74, 1.045)], ["up"])
        self.assertEqual(len(self.engine.charge_samples), 3)
        self.assertEqual(self.engine.previous_charge, (.74, 1.04))
        actions = self.timed_ring_frame(.86, 1.06)
        self.assertEqual(len(self.engine.charge_samples), 4)
        self.assertAlmostEqual(self.engine.charge_speeds[-1], 6)
        # These coarse 20 ms samples provide a wide render-time interval.
        # Retaining speed must not authorize a click through that uncertainty.
        self.assertEqual([a.kind for a in actions], ["up"])

    def test_late_duplicate_green_cannot_fall_back_to_an_unbounded_tap(self):
        self.config["input_latency"] = .008
        for marker, captured in ((.5, 1), (.62, 1.02), (.74, 1.04)):
            self.timed_ring_frame(marker, captured)
        # A slow capture first observes green after its usable click window.
        self.assertEqual([a.kind for a in self.timed_ring_frame(.98, 1.08, age=.028)], ["up"])
        # A fresh capture repeats those same pixels; their first-seen position
        # is still 31 ms old, although the new screenshot is only 1 ms old.
        actions = self.timed_ring_frame(.98, 1.11, age=.001)
        self.assertEqual([a.kind for a in actions], ["up"])
        self.assertEqual(self.engine.previous_charge, (.98, 1.08))
        self.assertEqual(self.engine.phase, "charge")
        self.assertIn("reading again", self.engine.status)

    def test_clear_target_or_geometry_change_invalidates_the_old_timing_model(self):
        self.config["input_latency"] = .008
        for changed in ({"center": 1.04}, {"geometry": (132, 120)}, {"width": .22}):
            with self.subTest(changed=changed):
                self.engine.start(0)
                for marker, captured in ((.2, 1), (.35, 1.02), (.5, 1.04), (.65, 1.06)):
                    self.timed_ring_frame(marker, captured)
                actions = self.timed_ring_frame(.87, 1.0893333333333333, **changed)
                self.assertEqual([a.kind for a in actions], ["up"])
                self.assertEqual(len(self.engine.charge_samples), 1)
                self.assertEqual(len(self.engine.charge_targets), 1)

    def test_small_target_centroid_jitter_keeps_a_safe_timing_model(self):
        self.config["input_latency"] = .008
        action = None
        for index in range(50):
            captured, marker = 1 + index * .004, .2 + 5.3 * index * .004
            actions = self.timed_ring_frame(marker, captured, center=1 + .003 * math.sin(index))
            action = next((a for a in actions if a.kind == "tap"), None)
            if action is not None:
                break
        self.assertIsNotNone(action)
        # The underlying target center is 1; the measured +/- .003 rad jitter
        # must be included in the uncertainty allowance rather than widening it.
        self.assert_timed_green_window(action, marker, captured, 5.3)

    def test_unstable_or_unobserved_motion_never_predicts_from_grey(self):
        self.config["input_latency"] = .008
        for marker, captured in ((.4, 1), (.5, 1.02), (.62, 1.04), (.80, 1.06)):
            self.assertEqual([a.kind for a in self.timed_ring_frame(marker, captured)], ["up"])
        self.assertEqual(self.engine.phase, "charge")
        for final_settings in ({"observed": False}, {"confidence": .84}):
            self.engine.start(0)
            for marker, captured in ((.2, 1), (.35, 1.02), (.5, 1.04), (.65, 1.06)):
                self.timed_ring_frame(marker, captured)
            actions = self.timed_ring_frame(.90, 1.0933333333333333, **final_settings)
            self.assertEqual([a.kind for a in actions], ["up"])

    def test_gap_or_recognition_jump_requires_new_motion_evidence(self):
        self.config["input_latency"] = .008
        for captured in (1.4, 1.065):
            self.engine.start(0)
            for marker, stamp in ((.2, 1), (.35, 1.02), (.5, 1.04), (.65, 1.06)):
                self.timed_ring_frame(marker, stamp)
            actions = self.timed_ring_frame(.90, captured)
            self.assertEqual([a.kind for a in actions], ["up"])
            self.assertEqual(len(self.engine.charge_samples), 1)

    def test_input_delay_longer_than_green_dwell_never_schedules_a_tap(self):
        self.config["input_latency"] = .030
        for marker, captured in ((.3, 1), (.46, 1.02), (.62, 1.04), (.78, 1.06), (.94, 1.08)):
            self.assertEqual([a.kind for a in self.timed_ring_frame(marker, captured)], ["up"])
        self.assertTrue(self.engine.charge_timing_reliable)

    def test_start_pause_and_stage_change_discard_charge_prediction_history(self):
        for reset in (lambda: self.engine.start(2), lambda: self.engine.pause(),
                      lambda: self.engine._enter("tuning", 2)):
            self.engine.start(0)
            for marker, captured in ((.2, 1), (.3, 1.02), (.4, 1.04), (.5, 1.06)):
                self.timed_ring_frame(marker, captured)
            self.assertTrue(self.engine.charge_timing_reliable)
            reset()
            self.assertFalse(self.engine.charge_samples)
            self.assertFalse(self.engine.charge_targets)
            self.assertFalse(self.engine.charge_render_samples)
            self.assertIsNone(self.engine.charge_observed_at)
            self.assertIsNone(self.engine.charge_pending_change)
            self.assertFalse(self.engine.charge_has_duplicates)
            self.assertFalse(self.engine.charge_timing_reliable)

    def test_first_crossing_cadence_matrix_has_no_outside_green_dispatches(self):
        # Sample every capture phase across one frame interval, with enough
        # grey observations to establish motion before the first encounter.
        # Even changing captures have uncertain render phase. Slow captures
        # may have no provably safe window; a fast 5.3 rad/s lane covers phases.
        cases = ((30, 5, .003, 0), (60, 4, .020, 0),
                 (60, 8, .003, 0), (60, 8, .020, 0),
                 (120, 4, .003, 64), (240, 5.3, .003, 64))
        for fps, speed, age, minimum_hits in cases:
            hits = 0
            for offset in range(64):
                with self.subTest(fps=fps, speed=speed, age=age, offset=offset):
                    self.engine = Engine({"input_latency": .008, "auto_sell": False})
                    self.engine.start(0)
                    crossing = .20 + offset / fps / 64
                    for index in range(math.ceil((crossing + .04) * fps) + 1):
                        captured = 1 + index / fps
                        marker = 1 + speed * (index / fps - crossing)
                        actions = self.timed_ring_frame(marker, captured, age)
                        tap = next((a for a in actions if a.kind == "tap"), None)
                        if tap is not None:
                            hits += 1
                            self.assert_timed_green_window(tap, marker, captured, speed, age=age)
                            break
            self.assertGreaterEqual(hits, minimum_hits)

    def test_quantized_render_cadence_with_capture_jitter_has_no_unsafe_clicks(self):
        # The real displayed pointer advances at 60 Hz while captures arrive
        # every 4 ms. Its first-seen capture is not its actual appearance time.
        cases = ((8, 0, 80), (8, .0005, 90), (8, .001, 90),
                 (5.3, .001, 127), (-8, .001, 90))
        for speed, jitter, minimum_hits in cases:
            hits = 0
            for offset in range(16):
                for render_phase_index in range(8):
                    with self.subTest(speed=speed, jitter=jitter, offset=offset,
                                      render_phase=render_phase_index):
                        self.engine = Engine({"input_latency": .008, "auto_sell": False})
                        self.engine.start(0)
                        crossing = .20 + offset * .004 / 16
                        render_phase = render_phase_index / 8 / 60
                        for index in range(math.ceil((crossing + .04) / .004) + 1):
                            relative = index * .004 + jitter * math.sin(index * 1.7 + offset)
                            rendered = math.floor((relative - render_phase) * 60 + 1e-9) / 60 + render_phase
                            captured = 1 + relative
                            displayed_marker = 1 + speed * (rendered - crossing)
                            actions = self.timed_ring_frame(displayed_marker, captured)
                            tap = next((a for a in actions if a.kind == "tap"), None)
                            if tap is not None:
                                hits += 1
                                true_marker_at_capture = 1 + speed * (relative - crossing)
                                self.assert_timed_green_window(tap, true_marker_at_capture,
                                                               captured, speed)
                                break
            self.assertGreaterEqual(hits, minimum_hits)

    def test_sub_four_ms_change_retains_its_original_appearance_interval(self):
        # This independent replay previously scheduled a reverse 9 rad/s tap
        # at .124492 s while the actual pointer was still +.079569 rad (grey).
        # A new position was first seen before the velocity estimator's 4 ms
        # minimum; its later processing must not replace that first sighting.
        rows = ((0., 1.3434338225316757),
                (.003763424989941566, 1.1933167610903317),
                (.008132935705261149, 1.1933167610903317),
                (.016786338377770025, 1.1933167610903317),
                (.020824367697639262, 1.0431047775424134),
                (.03384284703827462, 1.0431047775424134),
                (.03869442360262156, .8938598172110199),
                (.046134089818466, .8938598172110199),
                (.05073335371266053, .7436978652607815),
                (.06573338903290392, .7436978652607815),
                (.06979501294780383, .5938385416617085),
                (.08185492598192706, .5938385416617085),
                (.08606738838858623, .44296996146180656),
                (.09764439904847877, .44296996146180656),
                (.10157149490459089, .2925901661009733))
        self.config["input_latency"] = .008
        for captured, marker in rows:
            actions = self.timed_ring_frame(marker, captured, age=.002, center=0)
            self.assertNotIn("tap", [a.kind for a in actions])
        first_interval = self.engine.charge_render_samples[0]
        self.assertEqual(first_interval[:2], (0., .003763424989941566))
        self.assertAlmostEqual(first_interval[2], 1.1933167610903317)

    def test_quantized_centroid_noise_and_sub_four_ms_capture_jitter_stay_green(self):
        hits = 0
        for speed in (-9, -8, 8, 9):
            for capture_dt in (.004, 1 / 240):
                for offset in range(16):
                    for render_phase_index in range(4):
                        self.engine = Engine({"input_latency": .008, "auto_sell": False})
                        self.engine.start(0)
                        base_marker = 1 - math.copysign(1.2, speed)
                        crossing = 1.2 / abs(speed)
                        render_phase = render_phase_index / 4 / 60
                        for index in range(math.ceil((crossing + .04) / capture_dt) + 1):
                            relative = index * capture_dt + .0008 * math.sin(index * 1.7 + offset)
                            rendered_index = math.floor((relative - render_phase) * 60 + 1e-9)
                            rendered = rendered_index / 60 + render_phase
                            # Noise is fixed for each rendered frame, so fast
                            # captures repeat both its position and its jitter.
                            noise = .002 * math.sin(rendered_index * 1.73 + offset)
                            marker = base_marker + speed * rendered + noise
                            captured = 1 + relative
                            actions = self.timed_ring_frame(marker, captured, age=.002)
                            tap = next((a for a in actions if a.kind == "tap"), None)
                            if tap is not None:
                                hits += 1
                                self.assert_timed_green_window(tap, base_marker + speed * relative,
                                                               captured, speed, age=.002)
                                break
        self.assertGreater(hits, 0)

    def test_changing_every_capture_does_not_hide_quantized_render_phase(self):
        # This 120 Hz display/capture replay has no duplicate images. An exact
        # first-seen-time fit previously sent a click whose arrival pointer
        # was -.091725 rad, outside the actual +/- .075 rad target.
        rows = ((0., 1.2582974072376683),
                (.008913552648643224, 1.181978670785549),
                (.01714782087726538, 1.1091549593950853),
                (.02602687280843697, 1.0341020808259054),
                (.03373568174339165, .9569588286659682),
                (.04141089282348726, .8849165144387169),
                (.049127787830591546, .8088582822055408),
                (.05694859679767463, .7344487924824309),
                (.06570801836398467, .6584512463277115),
                (.07446846233890306, .5836801981889191),
                (.08304362673531522, .5073419646838194),
                (.0908768518044506, .4326837488148915),
                (.09915933846158992, .35900889537636677),
                (.10810506821131297, .2821172901496122),
                (.11622222025859408, .20862888456482298))
        self.config["input_latency"] = .008
        for captured, marker in rows:
            actions = self.timed_ring_frame(marker, captured, age=.002, center=0)
            self.assertNotIn("tap", [a.kind for a in actions])
        self.assertFalse(self.engine.charge_has_duplicates)
        self.assertEqual(len(self.engine.charge_render_samples), 14)
        self.assertEqual(self.engine.phase, "charge")

    def test_tuning_holds_for_right_target_and_releases_for_left_target(self):
        track = TuneTrack(1058, 435, (425, 800, 1060, 150), .99, 836)
        actions = self.step(Observation(tuning=track), 1)
        self.assertEqual([a.kind for a in actions], ["down"])
        track = TuneTrack(600, 800, (425, 800, 1060, 150), .99, 836)
        actions = self.step(Observation(tuning=track), 2)
        self.assertEqual([a.kind for a in actions], ["up"])

    def test_catch_steers_upper_dot_into_moving_field(self):
        # Screenshot: dot755, field824..1088. Hold right into the field.
        right = TargetTrack(956, 836, .95, (550, 800, 800, 150), 755, 264, 755, .3)
        self.assertEqual([a.kind for a in self.step(Observation(frequency=right), 1)], ["down"])
        left = TargetTrack(900, 836, .95, (550, 800, 800, 150), 1200, 264, 1200, .4)
        self.assertEqual([a.kind for a in self.step(Observation(frequency=left), 2)], ["up"])

    def test_catch_prediction_never_brakes_away_from_field_when_dot_is_outside(self):
        region = (550, 800, 800, 150)
        first = TargetTrack(900, 836, .95, region, 800, 40, 800, .3)
        moving = TargetTrack(900, 836, .95, region, 850, 40, 850, .3)
        self.assertEqual([a.kind for a in self.step(Observation(frequency=first), 1)], ["down"])
        self.assertEqual([a.kind for a in self.step(Observation(frequency=moving), 1.02)], ["down"])
        inside = TargetTrack(900, 836, .95, region, 895, 40, 895, .3)
        self.assertEqual([a.kind for a in self.step(Observation(frequency=inside), 1.04)], ["up"])

    def test_tuning_brakes_before_target_when_measured_momentum_would_overshoot(self):
        region = (425, 800, 1060, 150)
        first = TuneTrack(800, 700, region, .99, 836)
        second = TuneTrack(800, 770, region, .99, 836)
        self.assertEqual([a.kind for a in self.step(Observation(tuning=first), 1)], ["down"])
        # 70 px in 50 ms is measured rightward momentum. Release before the
        # target so the pointer does not coast past it while input takes effect.
        self.assertEqual([a.kind for a in self.step(Observation(tuning=second), 1.05)], ["up"])
        passed = TuneTrack(800, 810, region, .99, 836)
        self.assertEqual([a.kind for a in self.step(Observation(tuning=passed), 1.1)], ["up"])

    def steering_catch(self, target, control, now, width=100, *, captured_at=None):
        track = TargetTrack(target, 836, .95, (0, 800, 1000, 150), control,
                            width, control, .4, target - width / 2,
                            target + width / 2, 800, 950, 836)
        return self.engine.step(Observation(frequency=track, catch_present=True),
                                (1080, 1920, 3), now, captured_at=captured_at)

    def test_catch_releases_before_right_edge_when_dot_has_rightward_momentum(self):
        self.assertEqual([a.kind for a in self.steering_catch(500, 475, 1)], ["down"])
        self.assertEqual([a.kind for a in self.steering_catch(500, 485, 1.02)], ["up"])

    def test_catch_holds_before_left_edge_when_dot_has_leftward_momentum(self):
        self.assertEqual([a.kind for a in self.steering_catch(500, 525, 1)], ["up"])
        self.assertEqual([a.kind for a in self.steering_catch(500, 515, 1.02)], ["down"])

    def test_catch_matches_moving_target_instead_of_only_global_dot_direction(self):
        self.assertEqual([a.kind for a in self.steering_catch(500, 515, 1)], ["up"])
        # Both move right, but the target moves faster. Holding lets the dot
        # catch its moving center rather than braking global rightward motion.
        self.assertEqual([a.kind for a in self.steering_catch(530, 525, 1.05)], ["down"])

    def test_catch_duplicate_rendered_frame_retains_braking_until_motion_expires(self):
        self.steering_catch(500, 475, 1)
        self.assertEqual([a.kind for a in self.steering_catch(500, 485, 1.02)], ["up"])
        self.assertEqual([a.kind for a in self.steering_catch(500, 485, 1.04)], ["up"])
        # A long observation gap cannot retain that old rightward velocity.
        self.assertEqual([a.kind for a in self.steering_catch(500, 485, 1.4)], ["down"])

    def test_catch_recognition_jump_does_not_become_false_reverse_momentum(self):
        self.steering_catch(500, 475, 1, width=1000)
        self.assertEqual([a.kind for a in self.steering_catch(500, 900, 1.02, width=1000)], ["up"])
        # Returning 405 px in 20 ms is another detection jump, not a reason
        # to accelerate right while the fresh dot is already near the midpoint.
        self.assertEqual([a.kind for a in self.steering_catch(500, 495, 1.04, width=1000)], ["up"])

    def test_catch_stale_frame_releases_and_clears_steering_momentum(self):
        self.steering_catch(500, 475, 1)
        self.assertEqual([a.kind for a in self.steering_catch(500, 485, 1.25, captured_at=1.02)], ["up"])
        self.assertEqual([a.kind for a in self.steering_catch(500, 485, 1.27)], ["down"])

    def test_catch_uses_actual_edges_instead_of_inconsistent_reported_width(self):
        track = TargetTrack(500, 836, .95, (0, 800, 1000, 150), 520,
                            20, 520, .4, 450, 550, 800, 950, 836)
        self.engine.catch_held = True
        self.step(Observation(frequency=track, catch_present=True), 1)
        self.assertEqual([a.kind for a in self.steering_catch(500, 510, 1.02)], ["down"])

    def test_tuning_duplicate_frames_retain_braking_momentum(self):
        region = (0, 800, 1000, 150)
        self.step(Observation(tuning=TuneTrack(500, 480, region, .99, 836)), 1)
        self.assertEqual([a.kind for a in self.step(Observation(tuning=TuneTrack(500, 490, region, .99, 836)), 1.025)], ["up"])
        self.assertEqual([a.kind for a in self.step(Observation(tuning=TuneTrack(500, 490, region, .99, 836)), 1.04)], ["up"])

    def test_centered_catch_brakes_observed_drift_in_either_direction(self):
        for first, expected in ((490, "up"), (510, "down")):
            self.engine.start(0)
            self.steering_catch(500, first, 1)
            actions = self.steering_catch(500, 500, 1.02)
            self.assertEqual([a.kind for a in actions], [expected])

    def test_centered_tuning_brakes_observed_drift_in_either_direction(self):
        region = (0, 800, 1000, 150)
        for first, expected in ((490, "up"), (510, "down")):
            self.engine.start(0)
            self.step(Observation(tuning=TuneTrack(500, first, region, .99, 836)), 1)
            actions = self.step(Observation(tuning=TuneTrack(500, 500, region, .99, 836)), 1.025)
            self.assertEqual([a.kind for a in actions], [expected])

    def test_tuning_instant_reversal_does_not_stall_before_the_target(self):
        # A control with no inertia must still reach the point target; fixed
        # braking lead alone would shuttle between 480 and 486.7 indefinitely.
        region = (0, 800, 1000, 150)
        marker = 400.0
        errors = []
        for index in range(80):
            actions = self.step(Observation(tuning=TuneTrack(500, marker, region, .99, 836)),
                                1 + index / 60)
            errors.append(abs(marker - 500))
            marker += (400 if actions[0].kind == "down" else -400) / 60
        self.assertLessEqual(min(errors), 4)
        self.assertLessEqual(max(errors[-40:]), 14)

    def test_tuning_coasting_after_release_keeps_early_braking(self):
        region = (0, 800, 1000, 150)
        for marker, now in ((470, 1), (480, 1.025), (490, 1.05)):
            self.step(Observation(tuning=TuneTrack(500, marker, region, .99, 836)), now)
        actions = self.step(Observation(tuning=TuneTrack(500, 495, region, .99, 836)), 1.075)
        self.assertEqual([a.kind for a in actions], ["up"])
        self.assertIsNone(self.engine.linear_motion["tuning_response"].horizon_limit)

    def test_tuning_duplicate_capture_does_not_claim_an_instant_response(self):
        region = (0, 800, 1000, 150)
        for marker, now in ((480, 1), (490, 1.025), (490, 1.04)):
            self.step(Observation(tuning=TuneTrack(500, marker, region, .99, 836)), now)
        self.assertIsNone(self.engine.linear_motion["tuning_response"].horizon_limit)

    def test_tuning_newly_observed_coasting_restores_the_default_braking_horizon(self):
        region = (0, 800, 1000, 150)
        for marker, now in ((480, 1), (490, 1.025), (480, 1.05),
                            (490, 1.075), (500, 1.1)):
            self.step(Observation(tuning=TuneTrack(500, marker, region, .99, 836)), now)
        self.assertIsNotNone(self.engine.linear_motion["tuning_response"].horizon_limit)
        # The dot keeps moving right in a fresh frame after release. It now
        # demonstrates coasting, so the short no-inertia fallback must expire.
        actions = self.step(Observation(tuning=TuneTrack(500, 505, region, .99, 836)), 1.125)
        self.assertEqual([a.kind for a in actions], ["up"])
        self.assertIsNone(self.engine.linear_motion["tuning_response"].horizon_limit)

    def test_start_discards_previously_measured_immediate_button_response(self):
        region = (0, 800, 1000, 150)
        for marker, now in ((480, 1), (490, 1.025), (480, 1.05)):
            self.step(Observation(tuning=TuneTrack(500, marker, region, .99, 836)), now)
        self.assertIsNotNone(self.engine.linear_motion["tuning_response"].horizon_limit)
        self.engine.start(1.1)
        for marker, now in ((480, 1.12), (490, 1.145)):
            actions = self.step(Observation(tuning=TuneTrack(500, marker, region, .99, 836)), now)
        self.assertEqual([a.kind for a in actions], ["up"])
        self.assertIsNone(self.engine.linear_motion["tuning_response"].horizon_limit)

    def test_bar_resize_discards_catch_speed_and_response_in_old_pixel_scale(self):
        self.steering_catch(500, 475, 1)
        self.steering_catch(500, 485, 1.02)
        track = TargetTrack(500, 836, .95, (0, 800, 2000, 150), 485,
                            100, 485, .4, 450, 550, 800, 950, 836)
        actions = self.step(Observation(frequency=track, catch_present=True), 1.04)
        self.assertEqual([a.kind for a in actions], ["down"])
        self.assertFalse(self.engine.linear_motion["catch_control"].speeds)
        self.assertIsNone(self.engine.linear_motion["catch_response"].horizon_limit)

    def test_bar_resize_discards_tuning_speed_and_response_in_old_pixel_scale(self):
        for marker, now in ((480, 1), (490, 1.025)):
            self.step(Observation(tuning=TuneTrack(500, marker, (0, 800, 1000, 150), .99, 836)), now)
        actions = self.step(Observation(tuning=TuneTrack(500, 490, (0, 800, 2000, 150), .99, 836)), 1.04)
        self.assertEqual([a.kind for a in actions], ["down"])
        self.assertFalse(self.engine.linear_motion["tuning_control"].speeds)

    def test_catch_target_velocity_uses_observed_edge_center(self):
        # A separate legacy x value may jitter; actual visual edges define
        # both the destination and its measured movement.
        for reported, control, now in ((500, 490, 1), (540, 500, 1.02)):
            track = TargetTrack(reported, 836, .95, (0, 800, 1000, 150), control,
                                100, control, .4, 450, 550, 800, 950, 836)
            actions = self.step(Observation(frequency=track, catch_present=True), now)
        self.assertEqual([a.kind for a in actions], ["up"])
        self.assertFalse(self.engine.linear_motion["catch_target"].speeds)

    def test_tuning_near_midpoint_hysteresis_does_not_toggle_for_pixel_jitter(self):
        region = (0, 800, 1000, 150)
        for index, marker in enumerate((500, 499.5, 500, 500.5, 499.5, 500)):
            actions = self.step(Observation(tuning=TuneTrack(500, marker, region, .99, 836)), 1 + index * .02)
            self.assertEqual([a.kind for a in actions], ["up"])

    def test_tuning_stale_observations_cannot_preserve_a_rightward_press(self):
        region = (0, 800, 1000, 150)
        self.step(Observation(tuning=TuneTrack(800, 700, region, .99, 836)), 1)
        actions = self.engine.step(Observation(tuning=TuneTrack(800, 770, region, .99, 836)),
                                   (1080, 1920, 3), 1.3, captured_at=1.05)
        self.assertEqual([a.kind for a in actions], ["up"])

    def test_start_and_pause_discard_old_catch_momentum(self):
        self.steering_catch(500, 475, 1)
        self.steering_catch(500, 485, 1.02)
        self.engine.start(1.03)
        self.assertEqual([a.kind for a in self.steering_catch(500, 485, 1.04)], ["down"])
        self.steering_catch(500, 495, 1.06)
        self.engine.pause()
        self.engine.start(1.07)
        self.assertEqual([a.kind for a in self.steering_catch(500, 485, 1.08)], ["down"])

    def test_motion_steering_only_holds_or_releases_without_cursor_movement(self):
        for target, control, now in ((500, 475, 1), (500, 485, 1.02), (530, 515, 1.04)):
            actions = self.steering_catch(target, control, now)
            self.assertTrue(all(a.kind in ("up", "down") and a.x == 0 and a.y == 0 for a in actions))

    def test_tuning_releases_button_if_marker_detection_is_lost(self):
        track = TuneTrack(1058, 435, (425, 800, 1060, 150), .99, 836)
        self.step(Observation(tuning=track), 1)
        self.assertEqual(self.step(Observation(), 1.1), [])
        self.assertEqual([a.kind for a in self.step(Observation(), 1.2)], ["up"])

    def test_lost_catch_target_releases_after_short_grace_period(self):
        right = TargetTrack(1100, 900, .95, (550, 800, 800, 150), 800, 264, 800, .4)
        self.step(Observation(frequency=right, catch_present=True), 1)
        self.assertTrue(self.engine.catch_held)
        self.assertEqual(self.step(Observation(catch_present=True), 1.1), [])
        self.assertEqual([a.kind for a in self.step(Observation(catch_present=True), 1.2)], ["up"])
        self.assertFalse(self.engine.catch_held)

    def test_no_gameplay_branch_can_move_cursor(self):
        self.config["charge_action"] = "click"
        observations = [
            Observation(),
            Observation(charge=BarTrack((100, 100, 180, 180), .03, math.tau-.12, .14, "angle", .99)),
            Observation(tuning=TuneTrack(1058, 435, (425, 800, 1060, 150), .99, 836)),
            Observation(frequency=TargetTrack(805, 900, .95, (550, 800, 800, 150), 956, 264, 755, .3)),
        ]
        for obs in observations:
            self.engine.start(0)
            actions = self.step(obs, 1)
            self.assertTrue(all(a.kind in ("up", "down", "tap") for a in actions))
            self.assertTrue(all(a.x == 0 and a.y == 0 for a in actions))

    def test_sell_requires_two_fresh_button_matches_then_closes_inventory(self):
        self.engine.caught = 10
        button = Match(1420, 635, .99, (1380, 620, 80, 30))
        self.step(Observation(), 1)
        self.assertEqual([a.kind for a in self.step(Observation(), 1.4)], ["inventory"])
        self.assertEqual(self.step(Observation(), 2.1), [])
        first = self.step(Observation(sell=button), 2.2)
        self.assertEqual([a.kind for a in first], ["click"])
        self.assertEqual(self.step(Observation(), 3), [])
        second = self.step(Observation(sell=button), 3.2)
        self.assertEqual([a.kind for a in second], ["click"])
        self.assertEqual(self.engine.caught, 10)
        final = self.step(Observation(), 4.1)
        self.assertEqual([a.kind for a in final], ["inventory"])
        self.assertEqual(self.engine.caught, 0)
        self.assertEqual(self.engine.sold_batches, 1)

    def test_sell_always_opens_inventory_despite_a_preopening_button_match(self):
        self.engine.caught = 10
        button = Match(1420, 635, .99, (1380, 620, 80, 30))
        self.assertEqual([a.kind for a in self.step(Observation(sell=button), 1)], ["up"])
        opening = self.step(Observation(sell=button), 1.4)
        self.assertEqual([a.kind for a in opening], ["inventory"])
        self.assertEqual(self.engine.phase, "sell_first")
        # Neither a repeated preopening match nor a short detector gap may
        # reopen inventory or issue the first sale before its opening delay.
        self.assertEqual(self.step(Observation(sell=button), 1.5), [])
        self.assertEqual(self.step(Observation(), 2.1), [])
        self.assertEqual([a.kind for a in self.step(Observation(sell=button), 2.2)], ["click"])
        self.assertEqual(self.step(Observation(), 3), [])
        self.assertEqual([a.kind for a in self.step(Observation(sell=button), 3.2)], ["click"])
        self.assertEqual([a.kind for a in self.step(Observation(), 4.1)], ["inventory"])
        self.assertEqual(self.engine.sold_batches, 1)

    def test_single_confirmed_catch_runs_open_two_fresh_clicks_and_close(self):
        self.config["sell_every"] = 1
        button = Match(1420, 635, .99, (1380, 620, 80, 30))
        self.catch(.4, 1)
        self.assertEqual([a.kind for a in self.step(Observation(success=True), 2)], ["up"])
        self.assertEqual(self.engine.caught, 1)
        # Leave 2.5 s for the catch result animation before requesting inventory.
        self.assertEqual(self.step(Observation(), 3.3), [])
        self.assertEqual([a.kind for a in self.step(Observation(), 4.6)], ["up"])
        opening = self.step(Observation(sell=button), 5.0)
        self.assertEqual([a.kind for a in opening], ["inventory"])
        self.assertEqual(self.step(Observation(), 5.7), [])
        first = self.step(Observation(sell=button), 5.8)
        self.assertEqual([a.kind for a in first], ["click"])
        self.assertEqual(self.step(Observation(), 6.5), [])
        second = self.step(Observation(sell=button), 6.6)
        self.assertEqual([a.kind for a in second], ["click"])
        self.assertEqual([a.kind for a in self.step(Observation(), 7.5)], ["inventory"])
        self.assertEqual(self.engine.caught, 0)
        self.assertEqual(self.engine.sold_batches, 1)

    def test_pending_sale_is_not_blocked_by_a_persistent_catch_confirmation(self):
        self.config["sell_every"] = 1
        self.catch(.4, 1)
        self.assertEqual([a.kind for a in self.step(Observation(success=True), 2)], ["up"])
        self.assertEqual(self.step(Observation(success=True), 4.4), [])
        self.assertEqual([a.kind for a in self.step(Observation(success=True), 4.6)], ["up"])
        self.assertEqual(self.engine.phase, "sell_open")
        self.assertEqual([a.kind for a in self.step(Observation(success=True), 5.0)], ["inventory"])
        self.assertEqual(self.engine.phase, "sell_first")
        self.assertEqual(self.engine.caught, 1)
        self.assertEqual(self.engine.sold_batches, 0)

    def test_catch_cooldown_still_waits_for_persistent_success_if_no_sale_is_due(self):
        self.catch(.4, 1)
        self.step(Observation(success=True), 2)
        self.assertEqual(self.step(Observation(success=True), 5), [])
        self.assertEqual(self.engine.phase, "cooldown")
        self.assertEqual(self.engine.caught, 1)

    def test_missing_sale_confirmation_reports_the_actual_blocked_step(self):
        self.engine.caught = 10
        button = Match(1420, 635, .99, (1380, 620, 80, 30))
        self.step(Observation(), 1)
        self.step(Observation(), 1.4)
        self.step(Observation(sell=button), 2.1)
        self.assertIn("confirmation", self.engine.status.lower())
        self.step(Observation(), 10.2)
        self.assertIn("after the first click", self.engine.error)
        self.assertEqual(self.engine.caught, 10)
        self.assertEqual(self.engine.sold_batches, 0)

    def test_sell_closes_inventory_once_even_if_confirmation_is_still_visible(self):
        self.config["separate_confirmation"] = True
        self.engine.caught = 10
        button = Match(1420, 635, .99, (1380, 620, 80, 30))
        self.step(Observation(), 1)
        self.step(Observation(), 1.4)
        self.step(Observation(sell=button), 2.1)
        self.assertEqual([a.kind for a in self.step(Observation(confirm=button), 3)], ["click"])
        self.assertEqual([a.kind for a in self.step(Observation(confirm=button), 3.9)], ["inventory"])
        self.assertEqual(self.step(Observation(), 4), [])
        self.assertEqual(self.engine.sold_batches, 1)

    def test_separate_confirmation_does_not_click_unconfirmed_sell_again(self):
        self.config["separate_confirmation"] = True
        self.engine.caught = 10
        button = Match(1400, 635, .99, (1380, 620, 80, 30))
        self.step(Observation(), 1)
        self.step(Observation(sell=button), 1.4)
        self.step(Observation(sell=button), 2.1)
        self.assertEqual(self.step(Observation(sell=button), 3), [])
        self.assertEqual(self.engine.caught, 10)
        self.assertEqual([a.kind for a in self.step(Observation(confirm=button), 3.2)], ["click"])

    def test_missing_sell_preserves_counter_on_timeout(self):
        self.engine.caught = 10
        self.step(Observation(), 1)
        self.step(Observation(), 1.4)
        self.step(Observation(), 10)
        self.assertTrue(self.engine.error)
        self.assertIn("after the inventory key", self.engine.error)
        self.assertEqual(self.engine.caught, 10)

    def test_disabled_autosell_does_not_open_inventory_when_count_is_due(self):
        self.config["auto_sell"] = False
        self.engine.caught = 10
        actions = self.step(Observation(), 1)
        self.assertEqual([a.kind for a in actions], ["down"])
        self.assertEqual(self.engine.phase, "charge")
        self.assertEqual(self.engine.caught, 10)

    def test_disabled_autosell_waits_for_manual_inventory_close_without_input(self):
        self.config["auto_sell"] = False
        self.engine.caught = 10
        button = Match(1420, 635, .99, (1380, 620, 80, 30))
        for now in (1, 2, 3):
            actions = self.step(Observation(sell=button), now)
            self.assertEqual([a.kind for a in actions], ["up"])
            self.assertEqual(self.engine.caught, 10)
            self.assertEqual(self.engine.sold_batches, 0)

    def test_disabled_autosell_cancels_every_pending_sale_phase_without_keys_or_clicks(self):
        self.config["auto_sell"] = False
        button = Match(1420, 635, .99, (1380, 620, 80, 30))
        for phase in ("sell_open", "sell_first", "sell_confirm", "sell_finish"):
            with self.subTest(phase=phase):
                self.engine.phase = phase
                self.engine.caught = 10
                actions = self.step(Observation(sell=button, confirm=button), 1)
                self.assertEqual([a.kind for a in actions], ["up"])
                self.assertEqual(self.engine.phase, "idle")
                self.assertEqual(self.engine.caught, 10)
                self.assertEqual(self.engine.sold_batches, 0)


if __name__ == "__main__":
    unittest.main()
