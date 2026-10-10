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

    def test_tuning_does_not_reverse_before_reaching_the_target(self):
        region = (425, 800, 1060, 150)
        first = TuneTrack(800, 700, region, .99, 836)
        second = TuneTrack(800, 770, region, .99, 836)
        self.assertEqual([a.kind for a in self.step(Observation(tuning=first), 1)], ["down"])
        self.assertEqual([a.kind for a in self.step(Observation(tuning=second), 1.05)], ["down"])
        passed = TuneTrack(800, 810, region, .99, 836)
        self.assertEqual([a.kind for a in self.step(Observation(tuning=passed), 1.1)], ["up"])

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
        self.assertEqual([a.kind for a in self.step(Observation(), 3.3)], ["up"])
        opening = self.step(Observation(sell=button), 3.7)
        self.assertEqual([a.kind for a in opening], ["inventory"])
        self.assertEqual(self.step(Observation(), 4.4), [])
        first = self.step(Observation(sell=button), 4.5)
        self.assertEqual([a.kind for a in first], ["click"])
        self.assertEqual(self.step(Observation(), 5.2), [])
        second = self.step(Observation(sell=button), 5.3)
        self.assertEqual([a.kind for a in second], ["click"])
        self.assertEqual([a.kind for a in self.step(Observation(), 6.2)], ["inventory"])
        self.assertEqual(self.engine.caught, 0)
        self.assertEqual(self.engine.sold_batches, 1)

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
        self.assertEqual(self.engine.caught, 10)


if __name__ == "__main__":
    unittest.main()
