"""Visual recognition of Shrooms controls, using only captured window pixels.

All output coordinates are physical pixels relative to the captured client area.
Regions in configuration are normalized ``[x, y, width, height]`` rectangles.
Perfect can be reacquired elsewhere within the captured game client; other
moving controls remain inside their configured gameplay regions.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


Box = tuple[int, int, int, int]
TAU = math.tau


@dataclass(frozen=True)
class Match:
    x: float
    y: float
    score: float
    box: Box


@dataclass(frozen=True)
class BarTrack:
    region: Box
    marker: float
    zone_low: float
    zone_high: float
    axis: str
    confidence: float
    center: tuple[float, float] | None = None
    radius: float = 0.0
    marker_low: float | None = None
    marker_high: float | None = None
    marker_overlap: bool = False
    zone_observed: bool = True
    outline_radius: float = 0.0

    @property
    def zone_size(self) -> float:
        return (self.zone_high - self.zone_low) % TAU if self.axis == "angle" else self.zone_high - self.zone_low

    @property
    def zone_angle(self) -> float:
        return (self.zone_low + self.zone_size / 2.0) % TAU

    @property
    def zone_center(self) -> tuple[float, float]:
        x, y, w, h = self.region
        if self.axis == "angle":
            cx, cy = self.center or (x + w / 2.0, y + h / 2.0)
            return (cx + self.radius * math.cos(self.zone_angle), cy + self.radius * math.sin(self.zone_angle))
        midpoint = (self.zone_low + self.zone_high) / 2.0
        return (x + midpoint * w, y + h / 2.0) if self.axis == "x" else (x + w / 2.0, y + midpoint * h)

    @property
    def marker_point(self) -> tuple[float, float]:
        x, y, w, h = self.region
        if self.axis == "angle":
            cx, cy = self.center or (x + w / 2.0, y + h / 2.0)
            return (cx + self.radius * math.cos(self.marker), cy + self.radius * math.sin(self.marker))
        return (x + self.marker * w, y + h / 2.0) if self.axis == "x" else (x + w / 2.0, y + self.marker * h)


@dataclass(frozen=True)
class TuneTrack:
    target_x: float
    marker_x: float
    region: Box
    confidence: float
    y: float

    @property
    def x(self) -> float:
        return self.target_x


@dataclass(frozen=True)
class TargetTrack:
    x: float
    y: float
    confidence: float
    region: Box
    control_x: float | None = None
    control_width: float | None = None
    cursor_x: float | None = None
    progress: float | None = None
    goal_left: float | None = None
    goal_right: float | None = None
    goal_top: float | None = None
    goal_bottom: float | None = None
    cursor_y: float | None = None


@dataclass(frozen=True)
class Observation:
    charge: BarTrack | None = None
    frequency: TargetTrack | None = None
    success: bool = False
    sell: Match | None = None
    confirm: Match | None = None
    tuning: TuneTrack | None = None
    catch_present: bool = False
    catch_progress: float | None = None
    ring_present: bool = False
    ring_diagnostic: str = ""


def crop_region(frame: np.ndarray, region: Any) -> tuple[np.ndarray, Box]:
    """Clip a normalized region safely; malformed regions produce an empty crop."""
    empty = np.empty((0, 0, 3), dtype=np.uint8)
    if not isinstance(frame, np.ndarray) or frame.ndim < 2:
        return empty, (0, 0, 0, 0)
    height, width = frame.shape[:2]
    try:
        values = np.asarray(region, dtype=float)
        if values.shape != (4,) or not np.all(np.isfinite(values)) or values[2] <= 0 or values[3] <= 0:
            return empty, (0, 0, 0, 0)
        x, y, w, h = values
        left = max(0, min(width, int(math.floor(x * width))))
        top = max(0, min(height, int(math.floor(y * height))))
        right = max(0, min(width, int(math.ceil((x + w) * width))))
        bottom = max(0, min(height, int(math.ceil((y + h) * height))))
    except (TypeError, ValueError, OverflowError):
        return empty, (0, 0, 0, 0)
    if right <= left or bottom <= top:
        return empty, (left, top, 0, 0)
    return frame[top:bottom, left:right], (left, top, right - left, bottom - top)


def load_image(path: Path) -> np.ndarray | None:
    """OpenCV's normal file reader does not support all Windows Unicode paths."""
    try:
        raw = np.fromfile(path, dtype=np.uint8)
        if raw.size == 0:
            return None
        result = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        return result if result is not None and result.size else None
    except (OSError, ValueError, cv2.error):
        return None


def _number(value: Any, default: float) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _white(image: np.ndarray, value: int = 185) -> np.ndarray:
    b, g, r = cv2.split(image)
    low = cv2.min(cv2.min(b, g), r)
    high = cv2.max(cv2.max(b, g), r)
    return cv2.bitwise_and(cv2.compare(low, value, cv2.CMP_GE), cv2.compare(cv2.subtract(high, low), 65, cv2.CMP_LE))


def _components(mask: np.ndarray, minimum: int = 3) -> list[tuple[Box, int, float, float]]:
    if mask.size == 0 or not np.any(mask):
        return []
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    found = []
    for i in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area >= minimum:
            found.append(((x, y, w, h), area, float(centroids[i, 0]), float(centroids[i, 1])))
    return found


def _runs(values: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(values.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    return list(zip(np.flatnonzero(changes == 1).tolist(), np.flatnonzero(changes == -1).tolist()))


def _circular_interval(angles: np.ndarray) -> tuple[float, float, float]:
    center = math.atan2(float(np.sin(angles).mean()), float(np.cos(angles).mean())) % TAU
    relative = (angles - center + math.pi) % TAU - math.pi
    low, high = np.quantile(relative, [0.03, 0.97])
    return (center + float(low)) % TAU, (center + float(high)) % TAU, center


@dataclass
class _Template:
    image: np.ndarray
    feature: np.ndarray
    mode: str
    stamp: tuple[int, int]


class Detector:
    def __init__(self, config: dict, assets_dir: Path) -> None:
        self.config = config
        self.assets_dir = Path(assets_dir)
        self.templates: dict[str, _Template] = {}
        self._scaled: dict[tuple[str, int, int], np.ndarray] = {}
        self._last_reload = -math.inf
        self._ring_geometry: tuple[tuple[int, int, int, int], float, float, float] | None = None
        self._ring_zone: tuple[Box, float, float, float] | None = None
        self._last_ring_present = False
        self._last_ring_diagnostic = ""
        self._capture_frame_shape: tuple[int, int] | None = None
        self._ring_used_cached_geometry = False
        self._last_geometry_retry = -math.inf
        self._ring_recovered_region: tuple[tuple[int, int], Box, Box] | None = None
        self._last_local_ring_search: tuple[tuple[int, int], Box, float] | None = None
        self._last_full_ring_search: tuple[tuple[int, int], Box, float] | None = None
        self._scan_cache: dict[str, tuple[float, Match | None]] = {}
        self._features: dict[tuple[Any, ...], np.ndarray] = {}
        self._refresh_templates(force=True)

    def _refresh_templates(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_reload < 0.75:
            return
        self._last_reload = now
        for name in ("sell", "confirm", "success", "tune_prompt", "catch_target", "frequency_target"):
            path = self.assets_dir / f"{name}.png"
            try:
                stat = path.stat()
                stamp = (stat.st_mtime_ns, stat.st_size)
            except OSError:
                self.templates.pop(name, None)
                continue
            previous = self.templates.get(name)
            if previous is not None and previous.stamp == stamp:
                continue
            image = load_image(path)
            if image is None or min(image.shape[:2]) < 3:
                self.templates.pop(name, None)
                continue
            # Text/silhouette masks exclude scenery and translucent panel colors.
            feature = (_white(image, 160).astype(np.float32) / 255.0)
            mode = "white"
            if feature.std() < 0.06:
                if name != "frequency_target":
                    self.templates.pop(name, None)
                    continue
                feature = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
                mode = "gray"
            if feature.std() < 0.035:
                self.templates.pop(name, None)
                continue
            self.templates[name] = _Template(image, feature, mode, stamp)
            self._scaled = {key: value for key, value in self._scaled.items() if key[0] != name}
            self._scan_cache.pop(name, None)

    def ready(self) -> list[str]:
        """Return missing setup items, without initiating any desktop actions."""
        self._refresh_templates(force=True)
        missing = []
        for key, label in (("charge_region", "Perfect ring"), ("tuning_region", "tuning control"), ("frequency_region", "catch control")):
            test, _ = crop_region(np.zeros((100, 100, 3), dtype=np.uint8), self.config.get(key))
            if test.size == 0:
                missing.append(f"Set region: {label}")
        if "tune_prompt" not in self.templates:
            missing.append("Tuning prompt template is missing")
        progress_crop, _ = crop_region(np.zeros((100, 100, 3), dtype=np.uint8), self.config.get("progress_region"))
        if "success" not in self.templates and progress_crop.size == 0:
            missing.append("Set the progress bar or a catch confirmation template")
        if self.config.get("auto_sell", True) and "sell" not in self.templates:
            missing.append("Sell All template is missing")
        return missing

    def _frame_feature(self, frame: np.ndarray, mode: str, box: Box) -> np.ndarray:
        key = (mode, *box)
        if key not in self._features:
            x, y, w, h = box
            crop = frame[y:y + h, x:x + w]
            self._features[key] = (_white(crop, 160).astype(np.float32) / 255.0) if mode == "white" else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        return self._features[key]

    def _scales(self, name: str, frame: np.ndarray) -> list[float]:
        resolutions = self.config.get("template_source_resolutions", {})
        source = resolutions.get(name, self.config.get("source_resolution", (1919, 1079))) if isinstance(resolutions, dict) else (1919, 1079)
        try:
            sw, sh = float(source[0]), float(source[1])
            if sw <= 0 or sh <= 0:
                raise ValueError
        except (TypeError, ValueError, IndexError):
            sw, sh = 1919.0, 1079.0
        h, w = frame.shape[:2]
        # Roblox interfaces commonly scale with viewport height. Also cover
        # width scaling and the minor difference in cropped source screenshots.
        bases = {h / sh, w / sw}
        return sorted({round(base * variation, 3) for base in bases for variation in (0.90, 0.96, 1.0, 1.04, 1.10) if 0.08 <= base * variation <= 6.0})

    def _match_in(self, frame: np.ndarray, name: str, box: Box, reduction: float = 1.0) -> Match | None:
        template = self.templates.get(name)
        if template is None:
            return None
        x, y, w, h = box
        if w < 3 or h < 3:
            return None
        haystack = self._frame_feature(frame, template.mode, box)
        if reduction < 1.0:
            haystack = cv2.resize(haystack, (max(3, round(w * reduction)), max(3, round(h * reduction))), interpolation=cv2.INTER_AREA)
        sx, sy = w / haystack.shape[1], h / haystack.shape[0]
        threshold = _number(self.config.get(f"{name}_threshold"), 0.82 if name == "success" else 0.76)
        threshold = max(0.55, min(0.995, threshold))
        best: Match | None = None
        integral = squared_integral = None
        for scale in self._scales(name, frame):
            th, tw = template.feature.shape[:2]
            tw, th = max(3, round(tw * scale / sx)), max(3, round(th * scale / sy))
            if tw > haystack.shape[1] or th > haystack.shape[0]:
                continue
            key = (name, tw, th)
            needle = self._scaled.get(key)
            if needle is None:
                needle = cv2.resize(template.feature, (tw, th), interpolation=cv2.INTER_AREA)
                if needle.std() < 0.035:
                    continue
                if len(self._scaled) >= 240:
                    self._scaled.pop(next(iter(self._scaled)))
                self._scaled[key] = needle
            result = cv2.matchTemplate(haystack, needle, cv2.TM_CCOEFF_NORMED)
            np.nan_to_num(result, copy=False, nan=-1.0, posinf=-1.0, neginf=-1.0)
            # Normalized correlation is undefined on constant image windows.
            # Some OpenCV builds report 1.0 there, hiding the real text match
            # behind a black/white patch. Exclude only those flat windows;
            # retain the existing correlation and foreground checks below.
            if integral is None:
                integral, squared_integral = cv2.integral2(haystack, sdepth=cv2.CV_64F, sqdepth=cv2.CV_64F)
            # Integral arrays are one pixel larger on both axes. These four
            # corners yield exactly (height-th+1, width-tw+1), aligned with
            # each correlation candidate's top-left coordinate.
            sums = integral[th:, tw:] - integral[:-th, tw:] - integral[th:, :-tw] + integral[:-th, :-tw]
            squares = (squared_integral[th:, tw:] - squared_integral[:-th, tw:]
                       - squared_integral[th:, :-tw] + squared_integral[:-th, :-tw])
            area = tw * th
            variance = squares / area - (sums / area) ** 2
            result[variance <= 1e-6] = -1.0
            _, score, _, location = cv2.minMaxLoc(result)
            if score < threshold or (best is not None and score <= best.score):
                continue
            hx, hy = location
            if template.mode == "white":
                patch = haystack[hy:hy + th, hx:hx + tw]
                foreground = needle > 0.45
                if np.count_nonzero(foreground) < 4 or float((patch[foreground] > 0.30).mean()) < 0.62:
                    continue
            bx, by = round(x + hx * sx), round(y + hy * sy)
            bw, bh = max(1, round(tw * sx)), max(1, round(th * sy))
            best = Match(bx + bw / 2.0, by + bh / 2.0, float(score), (bx, by, bw, bh))
        return best

    def _match(self, frame: np.ndarray, name: str, now: float, only_box: Box | None = None) -> Match | None:
        if name not in self.templates:
            return None
        if only_box is not None:
            return self._match_in(frame, name, only_box)
        regions = self.config.get("template_regions", {})
        region = regions.get(name) if isinstance(regions, dict) else None
        local_box = None
        if region is not None:
            try:
                x, y, w, h = [float(v) for v in region]
                _, local_box = crop_region(frame, (x - w * 0.35, y - h * 0.6, w * 1.7, h * 2.2))
            except (TypeError, ValueError):
                pass
            if local_box is not None:
                local = self._match_in(frame, name, local_box)
                if local is not None:
                    self._scan_cache[name] = (now, local)
                    return local
        cached = self._scan_cache.get(name)
        interval = 0.55 if local_box is not None else 0.18
        if cached is not None and 0 <= now - cached[0] < interval:
            # A failed calibrated-area check invalidates a previously found
            # button immediately. Global sale/confirmation matches must also
            # come from this frame: throttle scans by waiting, not replaying
            # an earlier actionable coordinate after a dialog changes.
            return None if local_box is not None or name in ("sell", "confirm") else cached[1]
        h, w = frame.shape[:2]
        reduction = min(1.0, 1100.0 / max(w, h))
        match = self._match_in(frame, name, (0, 0, w, h), reduction)
        self._scan_cache[name] = (now, match)
        return match

    @staticmethod
    def _outline_score(crop: np.ndarray, cx: float, cy: float, radius: float) -> float:
        """A UI ring is a dark annulus with brighter pixels on both sides.

        Sampling its whole circumference verifies the circle independently of
        the green target and white pointer. A solid dark scene object does not
        have the required brighter circular interior.
        """
        h, w = crop.shape[:2]
        angles = np.linspace(0.0, TAU, 120, endpoint=False)
        radii = np.array((0.60, 0.86, 1.14))[:, None]
        mx = (cx + radii * radius * np.cos(angles)).astype(np.float32)
        my = (cy + radii * radius * np.sin(angles)).astype(np.float32)
        valid = ((mx >= 1) & (mx < w - 1) & (my >= 1) & (my < h - 1)).all(axis=0)
        if np.count_nonzero(valid) < 54:
            return 0.0
        values = cv2.remap(crop, mx, my, cv2.INTER_LINEAR).max(axis=2).astype(np.int16)
        middle = values[1]
        ring = (middle < 115) & (values[0] - middle > 12) & (values[2] - middle > 12)
        dark_fraction = float((middle[valid] < 115).mean())
        if dark_fraction < 0.65:
            return 0.0
        return float(ring[valid].mean())

    def _ring_outline(self, crop: np.ndarray, region: Box, *, force: bool = False,
                      radius_reference: float | None = None,
                      radius_bounds: tuple[float, float] | None = None,
                      target_point: tuple[float, float] | None = None) -> float | None:
        _, _, w, h = region
        previous = self._ring_geometry
        self._ring_used_cached_geometry = False
        if not force and previous is not None and previous[0] == region:
            _, cx, cy, radius = self._ring_geometry
            score = self._outline_score(crop, cx, cy, radius)
            if score >= 0.55:
                self._ring_used_cached_geometry = True
                return score
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        reference = min(w, h) if radius_reference is None else radius_reference
        minimum_radius, maximum_radius = (reference * 0.25, reference * 0.59) if radius_bounds is None else radius_bounds
        hough_maximum = reference * 0.57 if radius_bounds is None else maximum_radius
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.0, min(w, h) * 0.20,
                                   param1=80, param2=25, minRadius=max(3, round(minimum_radius)), maxRadius=round(hough_maximum))
        best = None
        if circles is not None:
            for proposed in circles[0]:
                cx, cy, proposed_radius = (float(v) for v in proposed)
                for radius in (proposed_radius, proposed_radius / 0.75, proposed_radius * 0.75):
                    if not minimum_radius <= radius <= maximum_radius:
                        continue
                    if target_point is not None:
                        target_distance = math.hypot(target_point[0] - cx, target_point[1] - cy)
                        if not radius * 0.70 <= target_distance <= radius * 1.05:
                            continue
                    score = self._outline_score(crop, cx, cy, radius)
                    if score >= 0.55 and (best is None or score > best[0]):
                        best = (score, cx, cy, radius)
        if best is None:
            # A failed retry must not discard a currently verified ring or its
            # complete Perfect arc merely because the pointer occludes edges.
            if not force:
                self._ring_geometry = None
                self._ring_zone = None
            return None
        score, cx, cy, radius = best
        if previous is not None:
            same_region = previous[0] == region
            center_shift = math.hypot(cx - previous[1], cy - previous[2])
            radius_shift = abs(radius - previous[3])
            if (not same_region or center_shift > max(1.5, radius * 0.02)
                    or radius_shift > max(1.5, radius * 0.025)):
                # Angular targets belong to their measured circle. A changed
                # center/radius cannot reuse angles from the previous geometry.
                self._ring_zone = None
        self._ring_geometry = (region, cx, cy, radius)
        return score

    def _recover_ring_crop(self, frame: np.ndarray, base_region: Box, region: Box,
                           *, force: bool = False) -> tuple[np.ndarray, Box] | None:
        """Complete a clipped, independently verified circle within its local area.

        A manual rectangle can start inside the ring. Enough of the outline may
        still verify, but a rotating pointer on the omitted side is invisible.
        Follow only that measured circle, rather than searching the full frame.
        """
        if self._ring_geometry is None or self._ring_geometry[0] != region:
            return None
        x, y, width, height = region
        _, cx, cy, radius = self._ring_geometry
        outer = radius * 1.14 + 2.0
        if (not force and cx - outer >= 0 and cy - outer >= 0
                and cx + outer < width and cy + outer < height):
            return None
        frame_height, frame_width = frame.shape[:2]
        global_x, global_y = x + cx, y + cy
        # The clipped fit can be a few pixels off. Leave room for the fresh
        # complete-circle fit so its next frame does not need another recrop.
        margin = radius * 1.30 + 3.0
        left = max(0, math.floor(global_x - margin))
        top = max(0, math.floor(global_y - margin))
        right = min(frame_width, math.ceil(global_x + margin))
        bottom = min(frame_height, math.ceil(global_y + margin))
        recovered = (left, top, right - left, bottom - top)
        if recovered == region or min(recovered[2:]) < 35:
            return None
        # Translation keeps the same physical circle and angular target. A
        # fresh fit below will invalidate that target if the geometry changes.
        self._ring_geometry = (recovered, global_x - left, global_y - top, radius)
        if self._ring_zone is not None and self._ring_zone[0] == region:
            self._ring_zone = (recovered, *self._ring_zone[1:])
        self._ring_recovered_region = ((frame_height, frame_width), base_region, recovered)
        crop = frame[top:bottom, left:right]
        self._ring_outline(crop, recovered, force=True)
        return crop, recovered

    def _find_local_ring_crop(self, frame: np.ndarray, base_region: Box,
                              now: float) -> tuple[np.ndarray, Box] | None:
        """Find a nearby circle when a manual crop hides too much to fit it.

        Search stays around that local rectangle. Its original size supplies
        the radius range: enlarging the search must not exclude the real ring
        by making Hough's minimum radius larger than the game control.
        """
        dimensions = frame.shape[:2]
        previous = self._last_local_ring_search
        if (previous is not None and previous[:2] == (dimensions, base_region)
                and 0 <= now - previous[2] < 0.25):
            return None
        self._last_local_ring_search = (dimensions, base_region, now)
        x, y, width, height = base_region
        if min(width, height) < 35:
            return None
        padding_x, padding_y = math.ceil(width * 0.65), math.ceil(height * 0.65)
        left, top = max(0, x - padding_x), max(0, y - padding_y)
        right = min(dimensions[1], x + width + padding_x)
        bottom = min(dimensions[0], y + height + padding_y)
        search_region = (left, top, right - left, bottom - top)
        if search_region == base_region:
            return None
        crop = frame[top:bottom, left:right]
        if self._ring_outline(crop, search_region, force=True,
                              radius_reference=min(width, height)) is None:
            return None
        # Immediately return to a compact, cached ROI around the verified
        # circle. Normal frames do not repeat the wider fallback or Hough fit.
        return self._recover_ring_crop(frame, base_region, search_region, force=True)

    def _reacquire_ring(self, frame: np.ndarray, base_region: Box, now: float) -> BarTrack | None:
        """Reacquire a moved Perfect inside this client, without full-frame Hough.

        A reduced image supplies compact green candidates. At most twelve
        surrounding patches get a circle fit; a result is accepted only after
        the normal original-pixel outline, green arc and white pointer checks.
        """
        dimensions = frame.shape[:2]
        previous = self._last_full_ring_search
        if (previous is not None and previous[:2] == (dimensions, base_region)
                and 0 <= now - previous[2] < 0.30):
            return None
        self._last_full_ring_search = (dimensions, base_region, now)
        height, width = dimensions
        reduction = min(1.0, 540.0 / height, 1200.0 / width)
        small = frame if reduction == 1 else cv2.resize(frame, (round(width * reduction), round(height * reduction)),
                                                      interpolation=cv2.INTER_AREA)
        small_height, small_width = small.shape[:2]
        sx, sy = width / small_width, height / small_height
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        green = cv2.inRange(hsv, np.array((50, 170, 190)), np.array((95, 255, 255)))
        white = _white(small, 190) > 0
        lime = (hsv[:, :, 0] >= 25) & (hsv[:, :, 0] <= 49) & (hsv[:, :, 1] >= 120) & (hsv[:, :, 2] >= 150)
        shorter = min(small_height, small_width)
        minimum_radius, maximum_radius = max(12.0, shorter * 0.035), shorter * 0.16
        candidates = []
        for box, area, gx, gy in _components(green, 5):
            x, y, w, h = box
            if area > shorter * shorter * 0.006 or max(w, h) > maximum_radius * 0.90:
                continue
            margin = math.ceil(maximum_radius * 2.15)
            left, top = max(0, x - margin), max(0, y - margin)
            right, bottom = min(small_width, x + w + margin), min(small_height, y + h + margin)
            if np.count_nonzero(white[top:bottom, left:right]) < 6:
                continue
            nearby = math.ceil(maximum_radius * 0.45)
            lime_count = np.count_nonzero(lime[max(0, y - nearby):min(small_height, y + h + nearby),
                                              max(0, x - nearby):min(small_width, x + w + nearby)])
            priority = min(6.0, lime_count / max(1, area)) + min(1.0, area / 80.0)
            candidates.append((priority, (left, top, right - left, bottom - top), (gx - left, gy - top)))
        candidates.sort(key=lambda candidate: candidate[0], reverse=True)
        saved = (self._ring_geometry, self._ring_zone, self._ring_recovered_region,
                 self._last_ring_present, self._last_ring_diagnostic, self._ring_used_cached_geometry)
        found = None
        found_state = None
        for _, search_region, target_point in candidates[:12]:
            x, y, w, h = search_region
            self._ring_geometry = self._ring_zone = None
            crop = small[y:y + h, x:x + w]
            if self._ring_outline(crop, search_region, force=True,
                                  radius_bounds=(minimum_radius, maximum_radius), target_point=target_point) is None:
                continue
            _, cx, cy, radius = self._ring_geometry
            global_x, global_y = (x + cx) * sx, (y + cy) * sy
            physical_radius = radius * (sx + sy) * 0.5
            margin = physical_radius * 1.30 + 3.0
            left, top = max(0, math.floor(global_x - margin)), max(0, math.floor(global_y - margin))
            right, bottom = min(width, math.ceil(global_x + margin)), min(height, math.ceil(global_y + margin))
            recovered = (left, top, right - left, bottom - top)
            physical_crop = frame[top:bottom, left:right]
            self._ring_geometry = (recovered, global_x - left, global_y - top, physical_radius)
            # Refit against original pixels; reduced pixels never authorize an
            # input or create a remembered angular Perfect target.
            if self._ring_outline(physical_crop, recovered, force=True) is None:
                continue
            track = self._ring(physical_crop, recovered, now)
            if track is None or not track.zone_observed:
                continue
            state = (self._ring_geometry, self._ring_zone,
                     (dimensions, base_region, recovered), True, "", self._ring_used_cached_geometry)
            if found is not None:
                same_circle = (math.dist(found.center, track.center) <= max(4.0, track.outline_radius * 0.12)
                               and abs(found.outline_radius - track.outline_radius) <= track.outline_radius * 0.10)
                if not same_circle:
                    # Two fully qualifying controls are ambiguous. Keep the
                    # local observation instead of choosing an arbitrary ring.
                    found = found_state = None
                    break
            else:
                found, found_state = track, state
        (self._ring_geometry, self._ring_zone, self._ring_recovered_region,
         self._last_ring_present, self._last_ring_diagnostic, self._ring_used_cached_geometry) = found_state or saved
        return found

    @staticmethod
    def _flanking_green_zone(greens: np.ndarray, distance: np.ndarray, angles: np.ndarray,
                             radius: float, marker: float, marker_low: float,
                             marker_high: float) -> tuple[float, float] | None:
        """Join current green remnants only when the verified pointer splits them.

        Tiny Perfect sectors can be mostly covered on the very first frame.
        Two measured flanks determine the arc; a hidden or one-sided target
        supplies no such evidence and must not be inferred.
        """
        marker_left = (marker_low - marker + math.pi) % TAU - math.pi
        marker_right = (marker_high - marker + math.pi) % TAU - math.pi
        fragments = []
        # Antialiasing can separate a flank into adjacent pixel islands.
        # Dilation groups those islands, but all counts/angles/radii below
        # still come exclusively from the original measured green pixels.
        kernel = min(9, max(3, math.ceil(radius * 0.035) | 1))
        groups = cv2.dilate(greens.astype(np.uint8) * 255,
                            np.ones((kernel, kernel), dtype=np.uint8))
        for box, _, _, _ in _components(groups, 2):
            x, y, w, h = box
            mask = greens[y:y + h, x:x + w]
            area = np.count_nonzero(mask)
            if area < 2:
                continue
            relative = (angles[y:y + h, x:x + w][mask] - marker + math.pi) % TAU - math.pi
            low, high = np.quantile(relative, (0.03, 0.97))
            radial_low, radial_high = np.quantile(distance[y:y + h, x:x + w][mask], (0.05, 0.95))
            if high - low <= 0.30 and radial_high - radial_low >= radius * 0.03:
                fragments.append((area, float(low), float(high), float(radial_low), float(radial_high)))
        best = None
        for left in fragments:
            if not left[2] <= marker_left + 0.04 or left[1] >= marker_left:
                continue
            for right in fragments:
                if not right[1] >= marker_right - 0.04 or right[2] <= marker_right:
                    continue
                gap = right[1] - left[2]
                span = right[2] - left[1]
                area = left[0] + right[0]
                if (area >= max(6, round(radius * 0.12))
                        and 0 < gap <= marker_right - marker_left + 0.08
                        and left[2] >= marker_left - 0.04 and right[1] <= marker_right + 0.04
                        and min(left[4], right[4]) - max(left[3], right[3]) >= max(1.0, radius * 0.015)
                        and 0.015 <= span <= 0.65 and left[1] < 0 < right[2]
                        and (best is None or area > best[0])):
                    best = (area, (marker + left[1]) % TAU, (marker + right[2]) % TAU)
        return (best[1], best[2]) if best is not None else None

    def _ring(self, crop: np.ndarray, region: Box, now: float | None = None) -> BarTrack | None:
        now = time.monotonic() if now is None else now
        self._last_ring_diagnostic = ""

        def unavailable(reason: str) -> None:
            self._last_ring_diagnostic = reason
            return None

        x, y, w, h = region
        if min(w, h) < 35:
            return None
        outline_score = self._ring_outline(crop, region)
        self._last_ring_present = outline_score is not None
        if outline_score is None:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        green = cv2.inRange(hsv, np.array((50, 170, 190)), np.array((95, 255, 255)))
        cached = self._ring_zone
        recent_zone = cached is not None and cached[0] == region and 0 <= now - cached[3] <= 0.25
        green_count = np.count_nonzero(green)
        enough_green = green_count >= max(8, w * h * 0.001)
        _, cx, cy, radius = self._ring_geometry
        yy, xx = np.indices((h, w), dtype=np.float32)
        distance = np.hypot(xx - cx, yy - cy)
        annulus = (distance >= radius * 0.75) & (distance <= radius * 1.03)
        greens = (green > 0) & annulus
        components = _components(greens.astype(np.uint8) * 255, max(6, round(radius * 0.08)))
        observed_zone = None
        if components and (enough_green or recent_zone):
            zone = max(components, key=lambda item: item[1])
            zx, zy, zw, zh = zone[0]
            zone_mask = np.zeros((h, w), dtype=bool)
            zone_mask[zy:zy + zh, zx:zx + zw] = greens[zy:zy + zh, zx:zx + zw]
            angles = np.arctan2(yy[zone_mask] - cy, xx[zone_mask] - cx) % TAU
            low, high, _ = _circular_interval(angles)
            if 0.015 <= (high - low) % TAU <= 0.65:
                observed_zone = (low, high)
        marker_color = self.config.get("charge_marker_color", (245, 245, 245))
        try:
            sample = np.asarray(marker_color, dtype=float)
            if sample.shape != (3,) or not np.all(np.isfinite(sample)):
                raise ValueError
            sample = np.clip(sample, 0, 255).astype(np.uint8)
            if int(sample.max()) - int(sample.min()) <= 65:
                marker_mask = _white(crop, max(140, round(float(sample.min()) * 0.84)))
            else:
                lower = np.clip(sample.astype(np.int16) - 55, 0, 255).astype(np.uint8)
                upper = np.clip(sample.astype(np.int16) + 55, 0, 255).astype(np.uint8)
                marker_mask = cv2.inRange(crop, lower, upper)
        except (TypeError, ValueError):
            marker_mask = _white(crop, 205)
        whites = (marker_mask > 0) & annulus
        candidates = []
        for box, area, mx, my in _components(whites.astype(np.uint8) * 255, max(6, round(radius * 0.07))):
            bx, by, bw, bh = box
            component = whites[by:by + bh, bx:bx + bw]
            ay, ax = np.nonzero(component)
            angles = np.arctan2(ay + by - cy, ax + bx - cx) % TAU
            ml, mh, angle = _circular_interval(angles)
            radial_values = distance[by:by + bh, bx:bx + bw][component]
            radial_low, radial_high = np.quantile(radial_values, (0.05, 0.95))
            span = (mh - ml) % TAU
            if (0.035 <= span <= 0.35 and radius * 0.08 <= radial_high - radial_low <= radius * 0.35
                    and radius * radius * 0.004 <= area <= radius * radius * 0.07):
                candidates.append((area, angle, span, ml, mh))
        if len(candidates) != 1:
            # Ambiguous bright fragments cannot authorize an input. The
            # independently verified ring remains present, so the engine waits.
            return unavailable("White pointer not detected" if not candidates else f"Multiple white pointer candidates ({len(candidates)})")
        # A ring also needs a dark annulus; isolated bright scenery cannot
        # qualify merely by having green and white pixels in this rectangle.
        dark_fraction = float((hsv[:, :, 2][annulus] < 115).mean())
        if dark_fraction < 0.35:
            return unavailable("Dark ring interior not verified")
        _, marker, marker_width, marker_low, marker_high = max(candidates)
        split_zone = None
        # A complete independent target elsewhere on the circle, or a broad
        # green component, cannot be replaced by near-pointer fragments.
        if observed_zone is not None:
            observed_size = (observed_zone[1] - observed_zone[0]) % TAU
            observed_center = (observed_zone[0] + observed_size / 2) % TAU
            target_near_pointer = abs((marker - observed_center + math.pi) % TAU - math.pi) <= (observed_size + marker_width) / 2 + 0.04
        else:
            target_near_pointer = not (enough_green and components)
        if target_near_pointer:
            split_zone = self._flanking_green_zone(greens, distance, np.arctan2(yy - cy, xx - cx),
                                                   radius, marker, marker_low, marker_high)
        if split_zone is not None and observed_zone is not None:
            split_size = (split_zone[1] - split_zone[0]) % TAU
            split_center = (split_zone[0] + split_size / 2) % TAU
            center_shift = abs((observed_center - split_center + math.pi) % TAU - math.pi)
            if center_shift + observed_size / 2 > split_size / 2 + 0.02:
                # It must extend the very same measured fragment, rather
                # than replace another nearby independently visible target.
                split_zone = None
        if split_zone is not None:
            observed_zone = split_zone
        if observed_zone is None and not recent_zone:
            return unavailable("Perfect target not detected on the ring" if enough_green else "Perfect target not detected")
        if recent_zone:
            old_low, old_high = cached[1], cached[2]
            old_size = (old_high - old_low) % TAU
            old_center = (old_low + old_size / 2.0) % TAU
            difference = abs((marker - old_center + math.pi) % TAU - math.pi)
            covering_zone = difference <= (old_size + marker_width) / 2.0 + 0.03
            if observed_zone is not None:
                new_low, new_high = observed_zone
                new_size = (new_high - new_low) % TAU
                new_center = (new_low + new_size / 2.0) % TAU
                center_shift = abs((new_center - old_center + math.pi) % TAU - math.pi)
                # Preserve the complete target while the arrow hides any part
                # of it, rather than chasing the exposed green remainder.
                if covering_zone and center_shift + new_size / 2.0 <= old_size / 2.0 + 0.04:
                    low, high = old_low, old_high
                else:
                    low, high = new_low, new_high
                self._ring_zone = (region, low, high, now)
            else:
                lime = (hsv[:, :, 0] >= 25) & (hsv[:, :, 0] <= 49) & (hsv[:, :, 1] >= 120) & (hsv[:, :, 2] >= 150)
                angle_grid = np.arctan2(yy - cy, xx - cx)
                nearby = np.abs((angle_grid - old_center + math.pi) % TAU - math.pi) <= 0.45
                if not covering_zone or np.count_nonzero(lime & annulus & nearby) < max(8, round(radius * 0.15)):
                    return unavailable("Hidden Perfect target could not be verified")
                low, high = old_low, old_high
        else:
            low, high = observed_zone
            self._ring_zone = (region, low, high, now)
        confidence = min(0.99, 0.78 + dark_fraction * 0.18)
        overlap = (marker - low) % TAU <= (high - low) % TAU
        return BarTrack(region, marker, low, high, "angle", confidence, (x + cx, y + cy), radius * 0.88,
                        marker_low, marker_high, overlap, observed_zone is not None, radius)

    def _linear_charge(self, crop: np.ndarray, region: Box) -> BarTrack | None:
        _, _, w, h = region
        axis = self.config.get("charge_axis", "x" if w >= h else "y")
        axis = axis if axis in ("x", "y") else ("x" if w >= h else "y")
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        green = cv2.inRange(hsv, np.array((35, 110, 130)), np.array((95, 255, 255)))
        length, cross = (w, h) if axis == "x" else (h, w)
        zones = []
        for box, area, cx, cy in _components(green):
            bx, by, bw, bh = box
            along, across = (bw, bh) if axis == "x" else (bh, bw)
            if 0.004 * length <= along <= 0.42 * length and across >= 0.25 * cross:
                zones.append((area, bx if axis == "x" else by, along))
        markers = []
        for box, area, cx, cy in _components(_white(crop, 190)):
            _, _, bw, bh = box
            along, across = (bw, bh) if axis == "x" else (bh, bw)
            if along <= 0.18 * length and across >= 0.25 * cross:
                markers.append((area, cx if axis == "x" else cy))
        if not zones or not markers:
            return None
        _, begin, span = max(zones)
        marker = max(markers)[1]
        return BarTrack(region, marker / max(1, length - 1), begin / length, (begin + span) / length, axis, 0.82)

    @staticmethod
    def _round_marker(crop: np.ndarray, start: float, end: float) -> tuple[float, float] | None:
        h, w = crop.shape[:2]
        top, bottom = max(0, int(h * start)), min(h, int(h * end))
        if bottom <= top:
            return None
        found = []
        for box, area, cx, cy in _components(_white(crop[top:bottom], 215), 8):
            _, _, bw, bh = box
            if 0.55 <= bw / max(1, bh) <= 1.85 and 4 <= bw <= max(8, h * 0.40) and 4 <= bh <= max(8, h * 0.40):
                if area / (bw * bh) >= 0.35:
                    found.append((area, cx, cy + top))
        if not found:
            return None
        _, cx, cy = max(found)
        return cx, cy

    def _tuning(self, frame: np.ndarray, now: float) -> TuneTrack | None:
        crop, region = crop_region(frame, self.config.get("tuning_region"))
        if crop.size == 0 or self._match(frame, "tune_prompt", now) is None:
            return None
        upper = self._round_marker(crop, 0.0, 0.37)
        lower = self._round_marker(crop, 0.62, 0.99)
        if upper is None or lower is None:
            return None
        x, y, _, _ = region
        return TuneTrack(x + lower[0], x + upper[0], region, 0.92, y + upper[1])

    def _progress(self, frame: np.ndarray) -> float | None:
        crop, _ = crop_region(frame, self.config.get("progress_region"))
        if crop.size == 0 or crop.shape[1] < 30 or crop.shape[0] < 5:
            return None
        h, w = crop.shape[:2]
        high, low = crop.max(axis=2), crop.min(axis=2)
        neutral = ((high.astype(np.int16) - low.astype(np.int16)) < 55) & (high > (60 if h <= 12 else 80))
        # Thin neutral border at both ends of a wide rectangular progress bar.
        top_coverage = neutral[:max(2, h // 3)].mean(axis=1).max()
        bottom_coverage = neutral[-max(2, h // 3):].mean(axis=1).max()
        if top_coverage < 0.70 or bottom_coverage < 0.70:
            return None
        bright = _white(crop[max(1, h // 4):max(2, h * 3 // 4)], 190) > 0
        columns = bright.mean(axis=0) >= 0.45
        filled = _runs(columns)
        start_limit = max(6, int(w * 0.025))
        leading = [run for run in filled if run[0] <= start_limit]
        if not leading:
            return 0.0
        start, finish = max(leading, key=lambda run: run[1] - run[0])
        return float(np.clip((finish - start) / max(1, w - 4), 0.0, 1.0))

    def _catch_presence(self, frame: np.ndarray, progress: float | None) -> bool:
        """Detect the catch UI independently of its moving or occluded target."""
        crop, _ = crop_region(frame, self.config.get("frequency_region"))
        if progress is None or crop.size == 0 or crop.shape[0] < 10:
            return False
        h = crop.shape[0]
        rows = (_white(crop, 150) > 0).mean(axis=1)
        span = max(4, h // 5)
        # UI borders are thin lines separated by a mostly translucent area.
        # A white flash or a blank white window is not a finished catch UI.
        middle = rows[span:-span]
        return bool(middle.size and np.median(middle) < 0.85 and rows[:span].max() >= 0.65 and rows[-span:].max() >= 0.65)

    @staticmethod
    def _catch_bar_limits(crop: np.ndarray) -> tuple[int, int] | None:
        h = crop.shape[0]
        rows = (_white(crop, 150) > 0).mean(axis=1)
        top = [run for run in _runs(rows >= 0.65) if run[0] < h * 0.22]
        bottom = [run for run in _runs(rows >= 0.65) if run[1] > h * 0.78]
        if not top or not bottom:
            return None
        first = max(top, key=lambda run: float(rows[run[0]:run[1]].sum()))
        last = max(bottom, key=lambda run: float(rows[run[0]:run[1]].sum()))
        a, b = round((first[0] + first[1] - 1) / 2), round((last[0] + last[1] - 1) / 2)
        return (a, b) if b - a >= h * 0.5 else None

    @staticmethod
    def _field_edges(frame: np.ndarray, region: Box, top: int, bottom: int) -> tuple[float, float] | None:
        """Find the field's long vertical strokes without assuming their hue.

        Compare edges inside the minigame to the scene immediately above and
        below it. This removes world geometry crossing a translucent field.
        Use the band above the decorative mushroom to avoid species-dependent
        shapes and outlines being mistaken for the rectangular goal.
        """
        x, _, width, _ = region
        height = bottom - top
        pad = max(8, round(height * 0.20))
        rim = max(3, round(height * 0.05))
        inset = max(2, round(height * 0.025))
        image_h, image_w = frame.shape[:2]
        sx, ex = max(0, x - 3), min(image_w, x + width + 3)
        sy, ey = max(0, top - pad), min(image_h, bottom + pad)
        sample = frame[sy:ey, sx:ex]
        if sample.size == 0:
            return None
        gradient = cv2.convertScaleAbs(cv2.Sobel(sample, cv2.CV_32F, 1, 0, ksize=3), alpha=0.25)
        b, g, r = cv2.split(gradient)
        gradient = cv2.max(cv2.max(b, g), r)[:, x - sx:x - sx + width]
        inner = gradient[top + inset - sy:bottom - inset - sy]
        upper = gradient[top + inset - sy:top + max(inset + 1, round(height * 0.17)) - sy]
        above = gradient[max(0, top - pad - sy):max(0, top - rim - sy)]
        below = gradient[max(0, bottom + rim - sy):min(len(gradient), bottom + pad - sy)]
        outside_parts = [part for part in (above, below) if part.size]
        if inner.size == 0 or upper.size == 0 or not outside_parts:
            return None
        outside = np.concatenate(outside_parts, axis=0)
        strength = np.maximum(0.0, inner.mean(axis=0) - outside.mean(axis=0) * 0.8)
        full_support = (inner > 18).mean(axis=0)
        upper_support = (upper > 18).mean(axis=0)
        valid = (full_support > 0.18) & (strength > 8) & (upper_support > 0.45)
        kernel = max(3, round(height * 0.017))
        valid = cv2.morphologyEx(valid.astype(np.uint8)[None, :], cv2.MORPH_CLOSE, np.ones((1, kernel), dtype=np.uint8))[0] > 0
        candidates = []
        for a, b in _runs(valid):
            if b - a > max(12, round(height * 0.13)):
                continue
            score = float(strength[a:b].sum()) / math.sqrt(b - a)
            candidates.append(((a + b - 1) / 2.0, score))
        pairs = [(a, b, score_a + score_b) for i, (a, score_a) in enumerate(candidates)
                 for b, score_b in candidates[i + 1:] if width * 0.07 <= b - a <= width * 0.78]
        if not pairs:
            return None
        left, right, _ = max(pairs, key=lambda pair: pair[2])
        return x + left, x + right

    @staticmethod
    def _catch_cursor(crop: np.ndarray, bar_top: int, bar_bottom: int) -> tuple[float, float] | None:
        height = bar_bottom - bar_top
        start = max(0, bar_top + round(height * 0.12))
        end = min(len(crop), bar_top + round(height * 0.44))
        if end - start < 8:
            return None
        band = crop[start:end]
        found = []
        for box, area, cx, cy in _components(_white(band, 225), 8):
            _, _, w, h = box
            if 0.65 <= w / max(1, h) <= 1.55 and height * 0.055 <= w <= height * 0.28 and height * 0.055 <= h <= height * 0.28:
                if area / (w * h) >= 0.5:
                    found.append((area, cx, cy + start))
        if found:
            _, cx, cy = max(found)
            return cx, cy
        # A white boundary can join the white dot into a single component.
        # Its circular rim still identifies the knob independently of the field.
        gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(cv2.GaussianBlur(gray, (3, 3), 0), cv2.HOUGH_GRADIENT, 1.0,
                                   max(8, height * 0.15), param1=80, param2=12,
                                   minRadius=max(3, round(height * 0.045)), maxRadius=max(5, round(height * 0.13)))
        if circles is None:
            return None
        yy, xx = np.indices(gray.shape)
        candidates = []
        for cx, cy, radius in circles[0]:
            distance = np.hypot(xx - cx, yy - cy)
            center = distance <= radius * 0.65
            rim = (distance >= radius * 0.9) & (distance <= radius * 1.15)
            if not np.any(center) or not np.any(rim):
                continue
            light = float(gray[center].mean())
            contrast = light - float(gray[rim].mean())
            if light >= 220 and contrast >= 16:
                candidates.append((contrast, float(cx), float(cy + start)))
        if not candidates:
            return None
        _, cx, cy = max(candidates)
        return cx, cy

    def _catching(self, frame: np.ndarray, now: float, *, progress: float | None = None,
                  verified_presence: bool = False) -> TargetTrack | None:
        crop, region = crop_region(frame, self.config.get("frequency_region"))
        if crop.size == 0 or min(crop.shape[:2]) < 15:
            return None
        if not verified_presence:
            progress = self._progress(frame)
            if progress is None or not self._catch_presence(frame, progress):
                return None
        limits = self._catch_bar_limits(crop)
        if limits is None:
            return None
        bar_top, bar_bottom = limits
        cursor = self._catch_cursor(crop, bar_top, bar_bottom)
        if cursor is None:
            return None
        x, y, w, h = region
        bounds = self._field_edges(frame, region, y + bar_top, y + bar_bottom)
        if bounds is None:
            return None
        left, right = bounds
        return TargetTrack((left + right) / 2.0, y + (bar_top + bar_bottom) / 2.0, 0.92, region,
                           x + cursor[0], right - left, x + cursor[0], progress,
                           left, right, y + bar_top, y + bar_bottom, y + cursor[1])

    def charge_capture_region(self, full_shape: tuple[int, ...]) -> Box | None:
        """Return only an already verified ring crop in full-client coordinates."""
        if (self.config.get("charge_shape", "ring") != "ring"
                or not self._last_ring_present or self._ring_geometry is None):
            return None
        try:
            dimensions = tuple(int(value) for value in full_shape[:2])
            if len(dimensions) != 2 or dimensions != self._capture_frame_shape:
                return None
            height, width = dimensions
            region, cx, cy, radius = self._ring_geometry
            x, y, crop_width, crop_height = region
            if (min(width, height, crop_width, crop_height) <= 0
                    or min(x, y) < 0 or x + crop_width > width or y + crop_height > height
                    or not all(math.isfinite(value) for value in (cx, cy, radius))
                    or radius <= 0):
                return None
            # The normal full-frame path recovers clipped calibration crops.
            # Stay on that path until the entire pointer circumference fits.
            if (cx - radius * 1.03 < 0 or cy - radius * 1.03 < 0
                    or cx + radius * 1.03 >= crop_width
                    or cy + radius * 1.03 >= crop_height):
                return None
        except (TypeError, ValueError, OverflowError):
            return None
        return region

    def analyze_charge(self, crop: np.ndarray, region: Box, now: float) -> Observation:
        """Observe a verified small capture while retaining client coordinates.

        This path intentionally cannot recognize other minigames. Its caller
        reacquires the full client immediately whenever the ring is unavailable.
        """
        if (self.config.get("charge_shape", "ring") != "ring"
                or self._ring_geometry is None or self._ring_geometry[0] != region
                or not isinstance(crop, np.ndarray) or crop.ndim != 3
                or crop.shape[2] < 3 or crop.size == 0):
            return Observation()
        try:
            if len(region) != 4 or crop.shape[:2] != (region[3], region[2]):
                return Observation()
        except (TypeError, IndexError):
            return Observation()
        crop = crop[:, :, :3]
        if crop.dtype != np.uint8:
            crop = np.clip(crop, 0, 255).astype(np.uint8)
        self._last_ring_present = False
        self._last_ring_diagnostic = ""
        charge = self._ring(crop, region, now)
        return Observation(charge=charge, ring_present=self._last_ring_present,
                           ring_diagnostic=self._last_ring_diagnostic)

    def analyze(self, frame: np.ndarray, now: float) -> Observation:
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] < 3 or frame.size == 0:
            return Observation()
        frame = frame[:, :, :3]
        self._capture_frame_shape = frame.shape[:2]
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        self._refresh_templates()
        self._features = {}
        self._last_ring_present = False
        self._last_ring_diagnostic = ""
        normalized = self.config.get("charge_region")
        shape = self.config.get("charge_shape", "ring")
        if shape == "ring":
            try:
                rx, ry, rw, rh = (float(value) for value in normalized)
                # Leave enough room for the complete circumference when a
                # supplied screenshot is cropped more narrowly than its game.
                normalized = (rx - rw * 0.12, ry - rh * 0.12, rw * 1.24, rh * 1.24)
            except (TypeError, ValueError):
                pass
        charge_crop, charge_region = crop_region(frame, normalized)
        base_charge_crop, base_charge_region = charge_crop, charge_region
        recovered = self._ring_recovered_region
        using_recovered = (shape == "ring" and recovered is not None
                           and recovered[:2] == (frame.shape[:2], base_charge_region))
        if using_recovered:
            charge_region = recovered[2]
            x, y, width, height = charge_region
            charge_crop = frame[y:y + height, x:x + width]
        elif recovered is not None:
            self._ring_recovered_region = None
        charge = None
        if charge_crop.size:
            shape = self.config.get("charge_shape", "ring" if charge_region[2] / max(1, charge_region[3]) < 1.6 else "line")
            charge = self._ring(charge_crop, charge_region, now) if shape == "ring" else self._linear_charge(charge_crop, charge_region)
            if shape == "ring" and using_recovered and not self._last_ring_present:
                # A new cast/window layout can move the circle. Revisit the
                # supplied local rectangle once, without widening the search.
                self._ring_recovered_region = None
                charge_crop, charge_region = base_charge_crop, base_charge_region
                charge = self._ring(charge_crop, charge_region, now)
            if shape == "ring" and not self._last_ring_present:
                nearby = self._find_local_ring_crop(frame, base_charge_region, now)
                if nearby is not None:
                    charge_crop, charge_region = nearby
                    charge = self._ring(charge_crop, charge_region, now)
            if shape == "ring" and self._last_ring_present:
                complete = self._recover_ring_crop(frame, base_charge_region, charge_region)
                if complete is not None:
                    charge_crop, charge_region = complete
                    charge = self._ring(charge_crop, charge_region, now)
            if (shape == "ring" and charge is None and self._last_ring_present
                    and self._ring_used_cached_geometry
                    and not self._last_ring_diagnostic.startswith("Multiple white")
                    and (now < self._last_geometry_retry or now - self._last_geometry_retry >= 0.12)):
                # A slightly wrong cached center can pass the broad outline
                # check yet clip a narrow pointer out of the marker annulus.
                # Refit instead of permanently waiting on that stale geometry.
                self._last_geometry_retry = now
                if self._ring_outline(charge_crop, charge_region, force=True) is not None:
                    charge = self._ring(charge_crop, charge_region, now)
        if charge is not None:
            # Ring timing is sensitive to frame rate. No inventory/text scans
            # can change the charge action, so keep this fast path minimal.
            return Observation(charge=charge, ring_present=self._last_ring_present)
        # Active tuning and catching controls must keep their regular cheap
        # path, even if scenery resembles a ring at the old charge position.
        # Full-client reacquisition runs only when neither other UI is present.
        tuning = self._tuning(frame, now)
        if tuning is not None:
            return Observation(tuning=tuning)
        progress = self._progress(frame)
        catch_present = self._catch_presence(frame, progress)
        frequency = None
        if catch_present:
            frequency = self._catching(frame, now, progress=progress, verified_presence=True)
            return Observation(frequency=frequency, catch_present=True, catch_progress=progress)
        if shape == "ring" and charge_crop.size:
            charge = self._reacquire_ring(frame, base_charge_region, now)
            if charge is not None:
                return Observation(charge=charge, ring_present=True)
        if self._last_ring_present:
            return Observation(ring_present=True, ring_diagnostic=self._last_ring_diagnostic)
        # Catch notification samples are user-provided. Never infer a catch from
        # the word Perfect while the starting minigame is still on screen.
        success = charge is None and self._match(frame, "success", now) is not None
        # A visible inventory still pauses instrument input in manual sale
        # mode. Its confirmation button has no purpose while sale is disabled.
        sell = self._match(frame, "sell", now)
        confirm = self._match(frame, "confirm", now) if self.config.get("auto_sell", True) else None
        return Observation(charge, frequency, success, sell, confirm, tuning, catch_present, progress if catch_present else None)

    @staticmethod
    def annotate(frame: np.ndarray, observation: Observation) -> np.ndarray:
        return annotate(frame, observation)


def annotate(frame: np.ndarray, observation: Observation) -> np.ndarray:
    """Draw a diagnostic copy; never alter the capture used for recognition."""
    output = frame.copy()
    if output.size == 0:
        return output

    def box(region: Box, label: str, color: tuple[int, int, int]) -> None:
        x, y, w, h = region
        cv2.rectangle(output, (x, y), (x + w - 1, y + h - 1), color, 1)
        cv2.putText(output, label, (x, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    def dot(x: float, y: float, color: tuple[int, int, int]) -> None:
        cv2.circle(output, (round(x), round(y)), 5, color, 2)

    if observation.charge is not None:
        track = observation.charge
        box(track.region, "Perfect", (40, 255, 70))
        dot(*track.zone_center, (40, 255, 70))
        dot(*track.marker_point, (255, 230, 30))
        if track.center is not None and track.outline_radius > 0:
            cv2.circle(output, (round(track.center[0]), round(track.center[1])), round(track.outline_radius), (160, 190, 255), 1)
    elif observation.ring_present and observation.ring_diagnostic:
        cv2.putText(output, f"Ring: {observation.ring_diagnostic}", (15, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1, cv2.LINE_AA)
    if observation.tuning is not None:
        track = observation.tuning
        box(track.region, "Tuning", (255, 210, 30))
        dot(track.target_x, track.y, (40, 255, 70))
        dot(track.marker_x, track.y, (255, 230, 30))
    if observation.frequency is not None:
        track = observation.frequency
        box(track.region, f"Catch {track.progress or 0:.0%}", (255, 180, 30))
        dot(track.x, track.y, (40, 255, 70))
        if all(value is not None for value in (track.goal_left, track.goal_right, track.goal_top, track.goal_bottom)):
            cv2.rectangle(output, (round(track.goal_left), round(track.goal_top)), (round(track.goal_right), round(track.goal_bottom)), (40, 255, 70), 2)
        if track.cursor_x is not None:
            dot(track.cursor_x, track.cursor_y if track.cursor_y is not None else track.region[1] + track.region[3] * 0.25, (255, 230, 30))
    for label, match in (("Sell", observation.sell), ("Confirm", observation.confirm)):
        if match is not None:
            box(match.box, f"{label} {match.score:.2f}", (0, 200, 255))
    if observation.success:
        cv2.putText(output, "Catch confirmed", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 255, 70), 2, cv2.LINE_AA)
    return output
