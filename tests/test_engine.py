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
        self.step(Observation(), 12)
        self.assertTrue(self.engine.error)

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
