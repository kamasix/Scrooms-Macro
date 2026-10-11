"""Auto sell controls using value and widget mocks; no Tk window is created."""

import unittest
from unittest.mock import MagicMock, patch

from shrooms_macro.ui import MacroUI


class Value:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class AutoSellUIChecks(unittest.TestCase):
    def setUp(self):
        self.ui = MacroUI.__new__(MacroUI)
        self.ui.root = MagicMock()
        self.ui.initial = {
            "auto_sell": True, "sell_every": 14, "input_latency": .028,
            "always_on_top": True, "custom_region": [0.1, 0.2, 0.3, 0.4],
        }
        self.settings = MagicMock()
        self.ui.callbacks = {"settings": self.settings}
        self.ui.running = False
        self.ui._caught = 6
        self.ui.auto_sell_var = Value(True)
        self.ui.sell_every_var = Value("14")
        self.ui.perfect_ms_var = Value("28")
        self.ui.topmost_var = Value(True)
        self.ui.count_var = Value("6 / 14")
        self.ui.status_var = Value("Ready")
        self.ui.target_var = Value("")
        self.ui.sold_var = Value("Sale cycles: 0")
        self.ui.warning_var = Value("")
        self.ui.sell_spin = MagicMock()
        self.ui.start_button = MagicMock()
        self.ui._schedule_fit = MagicMock()

    def test_constructor_isolates_edits_until_the_settings_callback_applies_them(self):
        config = dict(self.ui.initial)
        requests = []

        def apply_after_stop(changes):
            # The controller can stop using the old config before applying edits.
            self.assertTrue(config["auto_sell"])
            self.assertEqual(config["sell_every"], 14)
            requests.append(dict(changes))
            config.update(changes)

        with patch("shrooms_macro.ui.tk.StringVar", side_effect=Value), \
                patch("shrooms_macro.ui.tk.BooleanVar", side_effect=Value), \
                patch("shrooms_macro.ui.tk.Frame"), \
                patch("shrooms_macro.ui.tk.Label"), \
                patch.object(MacroUI, "_setup_styles"), \
                patch.object(MacroUI, "_build_header"), \
                patch.object(MacroUI, "_build_progress"), \
                patch.object(MacroUI, "_build_controls"), \
                patch.object(MacroUI, "_build_window_choice"), \
                patch.object(MacroUI, "_build_advanced", autospec=True,
                             side_effect=lambda ui, _parent: setattr(ui, "sell_spin", MagicMock())), \
                patch.object(MacroUI, "_button"), \
                patch.object(MacroUI, "_schedule_fit"):
            ui = MacroUI(MagicMock(), {"settings": apply_after_stop}, config)
            self.assertIsNot(ui.initial, config)
            ui.auto_sell_var.set(False)
            self.assertTrue(config["auto_sell"])
            ui._save_settings()

        self.assertEqual(len(requests), 1)
        self.assertFalse(requests[0]["auto_sell"])
        self.assertEqual(requests[0]["sell_every"], 14)
        self.assertFalse(config["auto_sell"])
        self.assertEqual(config["custom_region"], [.1, .2, .3, .4])

    def test_disable_saves_option_and_preserves_interval_and_other_settings(self):
        self.ui.auto_sell_var.set(False)
        self.ui._save_settings()
        self.settings.assert_called_once_with({
            "auto_sell": False, "sell_every": 14,
            "input_latency": .028, "always_on_top": True,
        })
        self.assertEqual(self.ui.count_var.get(), "6")
        self.ui.sell_spin.configure.assert_called_once_with(state="disabled")
        self.assertEqual(self.ui.initial["custom_region"], [.1, .2, .3, .4])
        self.assertEqual(self.ui.sell_every_var.get(), "14")

    def test_reenable_restores_threshold_without_resetting_count(self):
        self.ui.auto_sell_var.set(False)
        self.ui._save_settings()
        self.ui.auto_sell_var.set(True)
        self.ui._save_settings()
        self.assertEqual(self.ui.count_var.get(), "6 / 14")
        self.ui.sell_spin.configure.assert_called_with(state="normal")
        self.assertEqual(self.ui.initial["sell_every"], 14)

    def test_snapshot_disables_auto_sell_without_saving_again(self):
        self.ui.update_status({"auto_sell": False, "caught": 18})
        self.assertFalse(self.ui.auto_sell_var.get())
        self.assertEqual(self.ui.count_var.get(), "18")
        self.assertFalse(self.ui.initial["auto_sell"])
        self.settings.assert_not_called()

    def test_partial_snapshot_keeps_disabled_count_without_threshold(self):
        self.ui.update_status({"auto_sell": False})
        self.ui.update_status({"caught": 9})
        self.assertEqual(self.ui.count_var.get(), "9")
        self.assertFalse(self.ui.auto_sell_var.get())
        self.ui.update_status({"auto_sell": True})
        self.assertEqual(self.ui.count_var.get(), "9 / 14")

    def test_disabling_does_not_bypass_existing_number_validation(self):
        self.ui.auto_sell_var.set(False)
        self.ui.sell_every_var.set("invalid")
        self.ui.perfect_ms_var.set("invalid")
        self.ui._save_settings()
        self.assertEqual(self.ui.sell_every_var.get(), "14")
        self.assertEqual(self.ui.perfect_ms_var.get(), "28")
        self.assertEqual(self.settings.call_args.args[0]["sell_every"], 14)


if __name__ == "__main__":
    unittest.main()
