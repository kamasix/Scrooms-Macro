"""Profile migration preserves authored regions while correcting old defaults."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from shrooms_macro.config import Profile


class ProfileChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.resources = self.base / "resources"
        self.resources.mkdir()
        original = Path(__file__).resolve().parents[1] / "assets" / "default_config.json"
        (self.resources / "default_config.json").write_bytes(original.read_bytes())
        self.patch = patch("shrooms_macro.config.resource_dir", return_value=self.resources)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def profile(self, stored=None):
        if stored is not None:
            directory = self.base / "profiles" / "default"
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "config.json").write_text(json.dumps(stored), encoding="utf-8")
        return Profile(self.base)

    def test_new_profile_uses_stationary_click_and_current_schema(self):
        profile = self.profile()
        self.assertEqual(profile.data["charge_action"], "click")
        self.assertEqual(profile.data["profile_version"], 2)
        self.assertEqual(profile.data["input_latency"], 0.008)
        self.assertEqual(profile.data["control_mode"], "hold_release")

    def test_v1_stock_release_migrates_without_resetting_authoring_or_counts(self):
        region = [0.1, 0.2, 0.3, 0.4]
        templates = {"sell": [0.5, 0.6, 0.1, 0.1]}
        profile = self.profile({
            "version": 1, "charge_action": "release", "input_latency": 0.028,
            "charge_region": region, "template_regions": templates,
            "sell_every": 14, "initial_caught": 6, "initial_sold_batches": 9,
        })
        self.assertEqual(profile.data["charge_action"], "click")
        self.assertEqual(profile.data["input_latency"], 0.008)
        self.assertEqual(profile.data["charge_region"], region)
        self.assertEqual(profile.data["template_regions"], templates)
        self.assertEqual(profile.data["sell_every"], 14)
        self.assertEqual(profile.data["initial_caught"], 6)
        self.assertEqual(profile.data["initial_sold_batches"], 9)
        saved = json.loads(profile.path.read_text(encoding="utf-8"))
        self.assertEqual(saved["profile_version"], 2)
        self.assertEqual(saved["charge_action"], "click")

    def test_v1_custom_input_latency_is_preserved(self):
        profile = self.profile({"version": 1, "input_latency": 0.017})
        self.assertEqual(profile.data["input_latency"], 0.017)

    def test_v2_technical_release_is_preserved_on_load_and_save(self):
        profile = self.profile({"profile_version": 2, "charge_action": "release", "input_latency": 0.028})
        self.assertEqual(profile.data["charge_action"], "release")
        self.assertEqual(profile.data["input_latency"], 0.028)
        profile.save()
        reloaded = Profile(self.base)
        self.assertEqual(reloaded.data["charge_action"], "release")

    def test_bad_latency_and_unknown_action_fall_back_without_breaking_profile(self):
        profile = self.profile({"profile_version": 2, "charge_action": "invalid", "input_latency": "nan"})
        self.assertEqual(profile.data["charge_action"], "click")
        self.assertEqual(profile.data["input_latency"], 0.008)
        for raw, expected in ((-1, 0.0), (1, 0.12), ("bad", 0.008)):
            profile.data["input_latency"] = raw
            profile.validate()
            self.assertEqual(profile.data["input_latency"], expected)

    def test_existing_calibration_template_is_not_overwritten_by_stock(self):
        (self.resources / "catch_target.png").write_bytes(b"stock")
        template = self.base / "profiles" / "default" / "templates" / "catch_target.png"
        template.parent.mkdir(parents=True)
        template.write_bytes(b"custom")
        self.profile({"version": 1})
        self.assertEqual(template.read_bytes(), b"custom")


if __name__ == "__main__":
    unittest.main()
