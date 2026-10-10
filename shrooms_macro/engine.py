"""Nonblocking state machine. All decisions use fresh visual observations."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class Action:
    kind: str
    x: float = 0
    y: float = 0


class Engine:
    def __init__(self, config: dict):
        self.config = config
        self.caught = int(config.get("initial_caught", 0))
        self.sold_batches = int(config.get("initial_sold_batches", 0))
        self.phase = "idle"
        self.status = "Ready — equip your instrument and press F6"
        self.error = ""
        self.deadline = 0.0
        self.last_seen = 0.0
        self.last_progress = None
        self.last_progress_time = 0.0
        self.catch_started = None
        self.success_latched = False
        self.previous_charge = None
        self.charge_speeds = deque(maxlen=5)
        self.previous_control = None
        self.previous_target = None
        self.tuning_held = False
        self.catch_held = False
        self.last_target_seen = 0.0
        self.phase_started = 0.0
        self.last_success = -100.0
        self.cast_pending = False

    def start(self, now: float):
        self.phase = "idle"
        self.deadline = now + 0.45
        self.phase_started = now
        self.error = ""
        self.cast_pending = False
        self.catch_started = None
        self.last_progress = None
        self.previous_charge = self.previous_control = self.previous_target = None
        self.charge_speeds.clear()
        self.tuning_held = self.catch_held = False
        self.status = "Reading the game screen…"

    def pause(self, reason: str = "Paused — F6 resumes"):
        self.phase = "idle"
        self.status = reason
        self.catch_started = None
        self.last_progress = None
        self.tuning_held = self.catch_held = False
        self.previous_charge = None
        self.charge_speeds.clear()
        self.cast_pending = False

    def _enter(self, phase: str, now: float, delay: float = 0):
        self.phase, self.phase_started, self.deadline = phase, now, now + delay

    @staticmethod
    def _velocity(previous, position, now, circular=False):
        if previous is None:
            return 0.0
        old_position, old_time = previous
        dt = now - old_time
        if not 0.004 <= dt <= 0.25:
            return 0.0
        delta = position - old_position
        if circular:
            delta = (delta + math.pi) % (2 * math.pi) - math.pi
        return delta / dt

    def _setting(self, key, default, low, high):
        try:
            value = float(self.config.get(key, default))
            return max(low, min(high, value)) if math.isfinite(value) else default
        except (TypeError, ValueError):
            return default

    def _steer_button(self, error, deadband, state_name):
        """Hold moves right; release moves left. Hysteresis limits jitter."""
        held = getattr(self, state_name)
        if error > deadband:
            held = True
        elif error < -deadband:
            held = False
        setattr(self, state_name, held)
        return Action("down" if held else "up")

    def _charge_velocity(self, position, sample_time, circular):
        """Repeated captures of one rendered frame must not erase its speed."""
        if self.previous_charge is None:
            self.previous_charge = (position, sample_time)
            return 0.0
        old_position, old_time = self.previous_charge
        dt = sample_time - old_time
        delta = position - old_position
        if circular:
            delta = (delta + math.pi) % math.tau - math.pi
        if dt > 0.25 or dt <= 0:
            self.previous_charge = (position, sample_time)
            self.charge_speeds.clear()
            return 0.0
        minimum_change = 0.003 if circular else 0.001
        if abs(delta) >= minimum_change and dt >= 0.004:
            speed = delta / dt
            # A recognition jump cannot be used as a timing lead.
            if abs(speed) <= (40 if circular else 6):
                self.charge_speeds.append(speed)
            else:
                self.charge_speeds.clear()
            self.previous_charge = (position, sample_time)
        if not self.charge_speeds:
            return 0.0
        # A median removes occasional one-frame centroid jitter.
        speeds = sorted(self.charge_speeds)
        return speeds[len(speeds) // 2]

    def _caught(self, now):
        self.caught += 1
        self.last_success = now
        self.success_latched = True
        self.catch_started = None
        self.last_progress = None
        self.tuning_held = self.catch_held = False
        self._enter("cooldown", now, 2.5)
        self.status = "Caught! Waiting for the result animation"
        return [Action("up")]

    def step(self, obs, shape: tuple, now: float, captured_at: float | None = None) -> list[Action]:
        sample_time = now if captured_at is None else min(now, captured_at)
        frame_age = max(0.0, now - sample_time)
        if self.error:
            return []
        if not obs.success:
            self.success_latched = False
        # A success label is accepted only after the actual catch stage appeared.
        if (obs.success and not self.success_latched and self.catch_started is not None
                and now - self.catch_started > 0.4 and now - self.last_success > 1):
            return self._caught(now)

        if self.phase.startswith("sell"):
            return self._sell_step(obs, now)
        sale_due = self.caught >= int(self.config.get("sell_every", 1))
        if self.phase == "cooldown":
            # Keep the result-animation delay, but a persistent confirmation
            # must not prevent the inventory key when a sale is already due.
            if now < self.deadline or (obs.success and not sale_due):
                return []
            self._enter("idle", now)

        if sale_due:
            self._enter("sell_open", now, 0.3)
            self.status = "Opening inventory: key requested"
            return [Action("up")]

        if getattr(obs, "catch_present", False) and self.phase != "catch":
            self._enter("catch", now)
            self.catch_started = now
            self.last_seen = now
            self.last_target_seen = now
            self.last_progress = None
            self.previous_control = self.previous_target = None
            self.catch_held = False

        if obs.frequency is not None:
            track = obs.frequency
            if self.phase != "catch":
                self._enter("catch", now)
                self.catch_started = now
                self.last_progress = None
                self.previous_control = self.previous_target = None
                self.catch_held = False
            self.last_seen = now
            self.last_target_seen = now
            if track.progress is not None:
                self.last_progress = track.progress
                self.last_progress_time = now
            if now - self.phase_started > 90:
                self.error = "Catch timed out — press F6 to retry. Check Preview."
                return [Action("up")]
            # The upper white dot is driven by LMB; the moving field is its goal.
            if track.control_x is None:
                self.catch_held = False
                self.status = "Looking for the white dot…"
                return [Action("up")]
            control = track.control_x
            target_speed = self._velocity(self.previous_target, track.x, sample_time)
            control_speed = self._velocity(self.previous_control, control, sample_time)
            self.previous_target = (track.x, sample_time)
            self.previous_control = (control, sample_time)
            w = track.region[2]
            prediction = self._setting("catch_prediction", 0.085, 0, 0.3) + min(frame_age, 0.10)
            raw_error = track.x - control
            field_width = track.control_width
            known_width = field_width is not None and math.isfinite(field_width) and field_width > 0
            lead_limit = min(w * 0.35, field_width * 0.25) if known_width else 0.0
            lead = max(-lead_limit, min(lead_limit, prediction * (target_speed - control_speed)))
            # Velocity estimates may brake inside the goal, but must never
            # direct an outside dot away from the visible field.
            error = raw_error if not known_width or abs(raw_error) >= field_width / 2 else raw_error + lead
            deadband = max(3.0, w * self._setting("catch_deadband", 0.012, 0.001, 0.05))
            if known_width:
                deadband = min(deadband, max(2.0, field_width * 0.1))
            percent = f" · {int(track.progress * 100)}%" if track.progress is not None else ""
            self.status = "Keeping the dot inside the moving field" + percent
            return [self._steer_button(error, deadband, "catch_held")]

        if self.phase == "catch":
            if getattr(obs, "catch_progress", None) is not None:
                self.last_progress = obs.catch_progress
                self.last_progress_time = now
            if getattr(obs, "catch_present", False):
                self.last_seen = now
                self.status = "Looking for the dot and moving field…"
                if now - self.phase_started > 90:
                    self.error = "Catch target not detected — check Preview."
                    return [Action("up")]
                if now - self.last_target_seen >= 0.16:
                    self.catch_held = False
                    return [Action("up")]
                return []
            # Short detector dropouts must not release the button or count a catch.
            if now - self.last_seen < 0.16:
                return []
            if (self.last_progress is not None and self.last_progress >= 0.96
                    and now - self.last_progress_time < 0.6
                    and self.catch_started is not None and now - self.catch_started > 0.4):
                return self._caught(now)
            self._enter("result", now, 1.8)
            self.status = "Waiting for the catch result"
            return [Action("up")]

        if self.phase == "result":
            if now < self.deadline:
                return []
            self.catch_started = None
            self.last_progress = None
            self._enter("idle", now, 0.4)
            self.status = "No catch confirmation — retrying"
            return []

        if obs.tuning is not None:
            track = obs.tuning
            if self.phase != "tuning":
                self._enter("tuning", now)
                self.tuning_held = False
            if now - self.phase_started > 35:
                self.error = "Tuning timed out — check Preview and calibration."
                return [Action("up")]
            w = track.region[2]
            # Frequency tuning has a narrow target: never reverse direction
            # before reaching it based on an assumed movement/inertia model.
            error = track.target_x - track.marker_x
            deadband = max(2.0, w * self._setting("tuning_deadband", 0.004, 0.001, 0.03))
            self.status = "Matching the frequency"
            self.last_seen = now
            return [self._steer_button(error, deadband, "tuning_held")]

        if getattr(obs, "ring_present", False):
            self.cast_pending = False
            if self.phase not in ("charge", "transition"):
                self._enter("charge", now)
                self.previous_charge = None
                self.charge_speeds.clear()
                self.last_seen = now

        if obs.charge is not None and self.phase != "transition":
            self.cast_pending = False
            track = obs.charge
            if self.phase != "charge":
                self._enter("charge", now)
                self.previous_charge = None
                self.charge_speeds.clear()
            circular = track.axis == "angle"
            velocity = self._charge_velocity(track.marker, sample_time, circular)
            if circular:
                actual_delta = (track.marker - track.zone_angle + math.pi) % math.tau - math.pi
                half_zone = track.zone_size * 0.45
                observed_inside = abs(actual_delta) <= half_zone
                # The marker must already be visibly inside narrow Perfect.
                # Account for capture age and the configured input delay so
                # an observed late-edge overlap cannot land in broad green.
                # Keep the projected delta unwrapped: a full revolution must
                # never turn a passed target into a safe current overlap.
                now_delta = actual_delta + velocity * frame_age
                current_inside = abs(now_delta) <= half_zone
                input_delay = self._setting("input_latency", 0.008, 0, 0.12)
                impact_delta = actual_delta + velocity * (frame_age + input_delay)
                impact_inside = abs(impact_delta) <= half_zone
                inside = observed_inside and current_inside and impact_inside
            else:
                margin = (track.zone_high - track.zone_low) * 0.05
                inside = track.zone_low + margin <= track.marker <= track.zone_high - margin
                observed_inside = current_inside = impact_inside = inside
            if observed_inside and frame_age > 0.025:
                self.status = f"Ring: frame too old ({frame_age * 1000:.0f} ms) — reading again"
            elif observed_inside and not current_inside:
                self.status = "Ring: pointer moved past Perfect — reading again"
            elif observed_inside and not impact_inside:
                self.status = "Ring: click would arrive after Perfect — waiting"
            else:
                self.status = "Ring: white pointer outside Perfect"
            self.last_seen = now
            # Never fire a timed click from an abnormally old captured frame.
            if inside and frame_age <= 0.025:
                self._enter("transition", now, 1.5)
                self.previous_charge = None
                self.charge_speeds.clear()
                self.status = "White pointer on Perfect — clicking"
                return [Action("up" if self.config.get("charge_action", "click") == "release" else "tap")]
            if now - self.phase_started > 10:
                self.error = "Perfect timed out — check the ring in Preview."
                return [Action("up")]
            return [Action("down" if self.config.get("charge_action", "click") == "release" else "up")]

        if self.phase == "transition":
            if now < self.deadline:
                return []
            # If a click was not accepted, re-observe the existing ring.
            # Returning to idle would blindly press DOWN on its grey portion.
            self._enter("charge", now)
            self.cast_pending = False
            self.previous_charge = None
            self.charge_speeds.clear()
            self.status = "Ring: looking for the pointer and green segment"
            return [Action("up")]
        if self.phase == "charge":
            if self.cast_pending and now - self.phase_started < 0.55:
                # Preserve the one initial instrument press while its UI opens.
                # This is not permission to issue any further DOWN events.
                return []
            self.cast_pending = False
            detail = getattr(obs, "ring_diagnostic", "")
            self.status = ("Ring: " + detail) if detail else "Ring: looking for the pointer and green segment"
            if now - self.phase_started > 10:
                self.error = "Perfect not detected — check the ring in Preview."
            # Losing the target must never become a fresh cast/click.
            return [Action("up")]
        if self.phase == "tuning":
            if now - self.last_seen < 0.16:
                return []
            self.tuning_held = False
            self._enter("idle", now, 0.8)
            return [Action("up")]
        if obs.sell is not None:
            self._enter("idle", now, 0.7)
            self.status = "Closing the open inventory"
            return [Action("up"), Action("inventory")]
        if now >= self.deadline:
            self._enter("charge", now)
            self.cast_pending = True
            self.last_seen = now
            self.status = "Starting the instrument"
            return [Action("down")]
        return []

    def _sell_step(self, obs, now):
        if now < self.deadline:
            return []
        if now - self.phase_started > 8:
            blocked = {
                "sell_open": "Inventory opening timed out before the key press.",
                "sell_first": "Sell All not detected after the inventory key.",
                "sell_confirm": "Sale confirmation not detected after the first click.",
                "sell_finish": "Inventory closing timed out after confirmation.",
            }.get(self.phase, "Sale sequence timed out.")
            self.error = blocked + " Count preserved — check inventory."
            return [Action("up")]
        if self.phase == "sell_open":
            self._enter("sell_first", now, 0.6)
            self.status = "Waiting for Sell All after inventory key"
            # A preopening template match cannot replace the explicit key
            # press. Read the Sell All button again after inventory opens.
            return [Action("inventory")]
        if self.phase == "sell_first" and obs.sell is not None:
            self._enter("sell_confirm", now, 0.65)
            self.status = "Waiting for sale confirmation"
            return [Action("click", obs.sell.x, obs.sell.y)]
        if self.phase == "sell_confirm":
            # The user's UI confirms via the same Sell All button. If they supply
            # a separate confirmation template, it becomes the only allowed match.
            match = obs.confirm if self.config.get("separate_confirmation", False) else (obs.confirm or obs.sell)
            if match is not None:
                self._enter("sell_finish", now, 0.8)
                self.status = "Sale confirmation clicked — preparing inventory close"
                return [Action("click", match.x, match.y)]
        if self.phase == "sell_finish":
            # Counts describe completed input sequences; the game balance is never read.
            self.caught = 0
            self.sold_batches += 1
            self._enter("cooldown", now, 1.0)
            self.status = "Inventory close key requested — returning to catches"
            # The requested cycle always ends with a second backtick press,
            # even when the button disappears immediately after confirmation.
            return [Action("inventory")]
        return []
