"""Portable profile files beside the application, with atomic settings writes."""

from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path


PROFILE_VERSION = 2


def application_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def resource_dir() -> Path:
    return Path(getattr(sys, "_MEIPASS", application_dir())) / "assets"


class Profile:
    def __init__(self, base: Path | None = None):
        self.root = (base or application_dir()) / "profiles" / "default"
        self.templates = self.root / "templates"
        self.templates.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "config.json"
        self.data = json.loads((resource_dir() / "default_config.json").read_text(encoding="utf-8"))
        stored = {}
        if self.path.exists():
            try:
                stored = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(stored, dict):
                    self.data.update(stored)
            except (OSError, ValueError):
                shutil.copy2(self.path, self.root / "config-damaged.json")
        self._migrate(stored)
        for image in resource_dir().glob("*.png"):
            target = self.templates / image.name
            if not target.exists():
                shutil.copy2(image, target)
        self.validate()
        self.save()

    def _migrate(self, stored):
        """Correct the old forced release mode without resetting calibrations."""
        try:
            previous_version = int(stored.get("profile_version", stored.get("version", 1)))
        except (AttributeError, ValueError, TypeError, OverflowError):
            previous_version = 1
        if previous_version < PROFILE_VERSION and self.data.get("charge_action") == "release":
            self.data["charge_action"] = "click"
        if previous_version < PROFILE_VERSION:
            try:
                if math.isclose(float(self.data.get("input_latency", 0.008)), 0.028, rel_tol=0, abs_tol=1e-12):
                    self.data["input_latency"] = 0.008
            except (TypeError, ValueError, OverflowError):
                pass
        self.data["profile_version"] = max(PROFILE_VERSION, previous_version)
        self.data["version"] = self.data["profile_version"]

    def validate(self):
        for key, low, high, default in (
            ("sell_every", 1, 999, 10), ("initial_caught", 0, 999, 0),
            ("initial_sold_batches", 0, 1000000, 0),
        ):
            try:
                self.data[key] = max(low, min(high, int(self.data.get(key, default))))
            except (ValueError, TypeError, OverflowError):
                self.data[key] = default
        self.data["control_mode"] = "hold_release"
        if self.data.get("charge_action") not in ("click", "release"):
            self.data["charge_action"] = "click"
        try:
            latency = float(self.data.get("input_latency", 0.008))
            self.data["input_latency"] = max(0.0, min(0.12, latency)) if math.isfinite(latency) else 0.008
        except (TypeError, ValueError, OverflowError):
            self.data["input_latency"] = 0.008
        self.data["frequency_axis"] = "x"
        self.data["always_on_top"] = bool(self.data.get("always_on_top", True))

    def save(self):
        self.validate()
        temp = self.path.with_suffix(".tmp")
        temp.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")
        temp.replace(self.path)
