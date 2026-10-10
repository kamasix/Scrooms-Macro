"""Tk controller and interruptible worker for the local desktop macro."""

from __future__ import annotations

# This import establishes physical-pixel DPI awareness before Tk creates a window.
from .platform_win import Desktop, InputUnavailable, VK_OEM_3, DPI_AWARENESS

import json
import logging
import queue
import threading
import time
import webbrowser
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

import cv2
from PIL import Image, ImageTk

from .calibration import choose_region
from .config import Profile, application_dir
from .engine import Engine
from .ui import MacroUI
from .vision import Detector


DISCORD_URL = "https://discord.gg/rrcTR6pNnC"


def save_image(path: Path, frame):
    ok, encoded = cv2.imencode(".png", frame)
    if not ok:
        raise OSError("Could not save the image.")
    encoded.tofile(str(path))


class Application:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.profile = Profile()
        logging.basicConfig(filename=str(self.profile.root / "errors.log"), level=logging.WARNING,
                            format="%(asctime)s %(levelname)s %(message)s", encoding="utf-8")
        self.desktop = Desktop()
        self.detector = Detector(self.profile.data, self.profile.templates)
        self.engine = Engine(self.profile.data)
        self.target = 0
        self.target_label = ""
        self.active = threading.Event()
        self.shutdown = threading.Event()
        self.input_cancel = threading.Event()
        self.input_release_warning = ""
        self.control_lock = threading.RLock()
        self.detector_lock = threading.RLock()
        self.generation = 0
        self.calibrating = False
        self.pending_start = None
        self.messages = queue.Queue(maxsize=2)
        self.last_keys = {0x75: False, 0x77: False}
        self.saved_counts = (self.engine.caught, self.engine.sold_batches)
        self.last_frame_metrics = {}
        self.ui = MacroUI(root, {
            "toggle": self.toggle, "stop": self.stop, "refresh": self.windows,
            "select": self.select, "settings": self.settings,
            "calibrate": self.calibrate, "preview": self.preview,
            "diagnostics": self.diagnostics,
            "discord": self.open_discord,
        }, self.profile.data)
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.ui.update_status(self.snapshot())
        self.worker = threading.Thread(target=self.work, name="ScreenWorker", daemon=True)
        self.worker.start()
        self.root.after(25, self.poll)

    def windows(self):
        return [{"hwnd": w.hwnd, "label": f"{w.title} · {w.hwnd}"}
                for w in self.desktop.list_windows()]

    def select(self, hwnd: int):
        self.stop()
        self.target = hwnd
        try:
            r = self.desktop.client_rect(hwnd)
            self.target_label = f"{r.width} × {r.height} · monitor follows the game window"
        except InputUnavailable:
            self.target_label = "Restore Roblox from the taskbar"
        self.ui.update_status(self.snapshot())

    def snapshot(self, warning=""):
        return {"running": self.active.is_set(), "status": self.engine.status,
                "caught": self.engine.caught, "sold_batches": self.engine.sold_batches,
                "target": self.target_label,
                "warning": warning or getattr(self, "input_release_warning", "") or self.engine.error}

    def _release_inputs(self):
        self.desktop.mouse_up()
        self.input_release_warning = (
            "Windows did not release the held key. Press Stop to retry."
            if self.desktop.key_up() is False else "")
        if self.input_release_warning:
            logging.warning(self.input_release_warning)
        return self.input_release_warning

    def post(self, snapshot):
        snapshot["generation"] = self.generation
        try:
            self.messages.put_nowait(snapshot)
        except queue.Full:
            try:
                self.messages.get_nowait()
            except queue.Empty:
                pass
            try:
                self.messages.put_nowait(snapshot)
            except queue.Full:
                pass

    def toggle(self):
        if self.calibrating:
            return
        if self.active.is_set() or self.pending_start is not None:
            self.stop("Paused — F6 resumes")
            return
        if not self.target:
            self.ui.refresh_windows()
        if not self.target:
            self.ui.update_status(self.snapshot("Open Roblox first, then refresh the window list."))
            return
        with self.detector_lock:
            missing = self.detector.ready()
        if missing:
            self.ui.update_status(self.snapshot("Complete calibration: " + ", ".join(missing)))
            return
        if not self.desktop.cursor_over(self.target):
            self.engine.status = "Starting in 3 s — place the cursor over the game"
            self.pending_start = self.root.after(3000, self.finish_start)
            self.ui.update_status(self.snapshot())
            return
        self.finish_start()

    def finish_start(self):
        self.pending_start = None
        if self.shutdown.is_set() or self.calibrating:
            return
        try:
            warning = self._release_inputs()
            if warning:
                raise InputUnavailable(warning)
            self.desktop.focus(self.target)
            if not self.desktop.cursor_over(self.target):
                raise InputUnavailable("Place the cursor over a clear area of Roblox and press F6.")
            with self.control_lock:
                self.generation += 1
                self.engine.start(time.perf_counter())
                self.input_cancel.clear()
                self.active.set()
        except InputUnavailable as error:
            with self.control_lock:
                self.active.clear()
                self.input_cancel.set()
                self.generation += 1
                self.engine.pause("Start cancelled — input could not be prepared")
            self.ui.update_status(self.snapshot(str(error)))

    def stop(self, reason="Stopped — F6 starts"):
        # Interrupt a short click before waiting for the worker's decision lock.
        self.input_cancel.set()
        if self.pending_start is not None:
            self.root.after_cancel(self.pending_start)
            self.pending_start = None
        with self.control_lock:
            self.active.clear()
            self.generation += 1
            self._release_inputs()
            self.engine.pause(reason)
        if hasattr(self, "ui"):
            self.ui.update_status(self.snapshot())

    def settings(self, changes: dict):
        self.stop("Settings saved — F6 starts")
        self.profile.data.update(changes)
        self.profile.save()
        self.detector = Detector(self.profile.data, self.profile.templates)
        self.ui.update_status(self.snapshot())

    def open_discord(self):
        self.stop("Paused — F6 resumes")
        if not webbrowser.open(DISCORD_URL, new=2):
            self.ui.update_status(self.snapshot("Open " + DISCORD_URL + " in your browser."))

    def _action(self, action, captured_at=None):
        if action.kind == "up":
            self.desktop.mouse_up()
        elif action.kind == "move":
            raise RuntimeError("Minigame controls cannot move the cursor.")
        elif action.kind == "down":
            self.desktop.mouse_down(self.target)
        elif action.kind == "tap":
            if self.engine.phase not in ("charge", "transition"):
                raise RuntimeError("Stationary clicks are allowed only during Perfect.")
            dispatch_age = None if captured_at is None else time.perf_counter() - captured_at
            metrics = getattr(self, "last_frame_metrics", None)
            if metrics is not None:
                metrics["dispatch_age_ms"] = None if dispatch_age is None else round(dispatch_age * 1000, 2)
                metrics["tap_called"] = False
            if dispatch_age is not None and dispatch_age > 0.025:
                self.engine.phase = "charge"
                self.engine.status = f"Ring: frame too old ({dispatch_age * 1000:.0f} ms) — reading again"
                return
            if metrics is not None:
                metrics["tap_called"] = True
            self.desktop.tap(self.target, cancel=self.input_cancel)
        elif action.kind == "click":
            if not self.engine.phase.startswith("sell"):
                raise RuntimeError("Clicks that move the cursor are allowed only during a sale.")
            self.desktop.click(self.target, action.x, action.y, cancel=self.input_cancel)
        elif action.kind == "inventory":
            self.desktop.key(self.target, VK_OEM_3, cancel=self.input_cancel)

    def work(self):
        try:
            while not self.shutdown.is_set():
                if not self.active.wait(0.08):
                    continue
                began = time.perf_counter()
                with self.control_lock:
                    generation, target, detector = self.generation, self.target, self.detector
                try:
                    if not self.desktop.is_foreground(target):
                        raise InputUnavailable("Paused: return to Roblox and press F6")
                    capture_started = time.perf_counter()
                    frame = self.desktop.capture(target)
                    capture_finished = time.perf_counter()
                    captured_at = (capture_started + capture_finished) * 0.5
                    analysis_started = time.perf_counter()
                    with self.detector_lock:
                        obs = detector.analyze(frame, captured_at)
                    analysis_finished = time.perf_counter()
                    with self.control_lock:
                        if (not self.active.is_set() or generation != self.generation
                                or self.input_cancel.is_set()):
                            continue
                        decision_time = time.perf_counter()
                        self.last_frame_metrics = {
                            "capture_ms": round((capture_finished - capture_started) * 1000, 2),
                            "analysis_ms": round((analysis_finished - analysis_started) * 1000, 2),
                            "decision_age_ms": round((decision_time - captured_at) * 1000, 2),
                            "ring_present": obs.ring_present,
                            "ring_recognized": obs.charge is not None,
                            "ring_diagnostic": obs.ring_diagnostic,
                            "tap_called": False,
                        }
                        actions = self.engine.step(obs, frame.shape, decision_time,
                                                   captured_at=captured_at)
                        self.last_frame_metrics["tap_requested"] = any(action.kind == "tap" for action in actions)
                        self.last_frame_metrics["decision_status"] = self.engine.status
                        for action in actions:
                            if not self.active.is_set() or self.input_cancel.is_set():
                                break
                            self._action(action, captured_at=captured_at)
                        if self.engine.error:
                            self.active.clear()
                            self._release_inputs()
                        self.post(self.snapshot())
                except InputUnavailable as error:
                    with self.control_lock:
                        if generation != self.generation:
                            continue
                        self.active.clear()
                        self._release_inputs()
                        self.engine.pause(str(error))
                        self.post(self.snapshot())
                except Exception as error:
                    logging.exception("Macro worker error")
                    with self.control_lock:
                        if generation != self.generation:
                            continue
                        self.active.clear()
                        release_warning = self._release_inputs()
                        self.engine.pause("Stopped after an error")
                        warning = str(error) + (" · " + release_warning if release_warning else "")
                        self.post(self.snapshot(warning))
                # Bounded, interruptible timing; no busy waiting or long input sequences.
                interval = 1 / 120 if self.engine.phase in ("charge", "transition") else 1 / 60
                self.shutdown.wait(max(0, interval - (time.perf_counter() - began)))
        finally:
            self._release_inputs()

    def poll(self):
        if self.shutdown.is_set():
            return
        for key in self.last_keys:
            pressed = self.desktop.key_pressed(key)
            if pressed and not self.last_keys[key]:
                if key == 0x77:
                    self.stop("Emergency stop — mouse released")
                elif not self.calibrating:
                    self.toggle()
            self.last_keys[key] = pressed
        latest = None
        while True:
            try:
                latest = self.messages.get_nowait()
            except queue.Empty:
                break
        if latest is not None and latest.get("generation") == self.generation:
            # Ignore a queued active snapshot after a newer Stop request.
            latest["running"] = self.active.is_set()
            latest["status"] = self.engine.status
            self.ui.update_status(latest)
        counts = (self.engine.caught, self.engine.sold_batches)
        if counts != self.saved_counts:
            self.profile.data["initial_caught"], self.profile.data["initial_sold_batches"] = counts
            try:
                self.profile.save()
                self.saved_counts = counts
            except OSError:
                logging.exception("Could not save the counter")
        self.root.after(25, self.poll)

    def capture_after_countdown(self, callback, label):
        if self.calibrating:
            return
        self.stop()
        if not self.target or not self.desktop.is_available(self.target):
            self.ui.update_status(self.snapshot("Select a visible Roblox window."))
            return
        self.calibrating = True
        self.ui.update_status(self.snapshot("In 3 seconds: " + label + ". Show the element in the game."))
        try:
            self.desktop.focus(self.target)
        except InputUnavailable:
            pass

        def capture():
            if self.shutdown.is_set():
                return
            self.root.withdraw()

            def finish():
                try:
                    if not self.desktop.is_foreground(self.target):
                        raise InputUnavailable("Capture cancelled: make Roblox the active window.")
                    frame = self.desktop.capture(self.target)
                    self.root.deiconify()
                    callback(frame)
                except Exception as error:
                    self.root.deiconify()
                    self.ui.update_status(self.snapshot(str(error)))
                finally:
                    self.calibrating = False
            self.root.after(180, finish)
        self.root.after(3000, capture)

    def calibrate(self, kind):
        names = {
            "charge_region": "select the entire Perfect ring",
            "tuning_region": "select both tuning sliders",
            "frequency_region": "select the entire catch field",
            "progress_region": "select the inside of the progress bar",
            "charge_marker": "click the moving white ring pointer",
            "frequency_target": "select the entire white mushroom icon in the catch bar",
            "catch_target": "select the entire white mushroom icon in the catch bar",
            "tune_prompt": "select the Hold to find the correct frequency! text",
            "success": "select a fixed catch confirmation label",
            "sell": "select the Sell All text",
            "confirm": "select the sale confirmation button",
        }
        if kind not in names:
            return

        def choose(frame):
            result = choose_region(self.root, frame, names[kind], point=kind == "charge_marker")
            if result is None:
                return
            config = self.profile.data
            if kind.endswith("_region"):
                config[kind] = result["region"]
            elif kind == "charge_marker":
                config[kind + "_color"] = result["color"]
            else:
                template = "catch_target" if kind == "frequency_target" else kind
                save_image(self.profile.templates / (template + ".png"), result["crop"])
                config.setdefault("template_regions", {})[template] = result["region"]
                config.setdefault("template_source_resolutions", {})[template] = [frame.shape[1], frame.shape[0]]
                if kind == "confirm":
                    config["separate_confirmation"] = True
            self.profile.save()
            self.detector = Detector(config, self.profile.templates)
            self.ui.update_status(self.snapshot("Calibration saved. F6 starts the macro."))
        self.capture_after_countdown(choose, names[kind])

    def preview(self):
        def show(frame):
            with self.detector_lock:
                obs = self.detector.analyze(frame, time.perf_counter())
            annotated = self.detector.annotate(frame, obs)
            window = tk.Toplevel(self.root)
            window.title("Detection preview — no input")
            scale = min(1100 / frame.shape[1], 680 / frame.shape[0], 1)
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            picture = ImageTk.PhotoImage(Image.fromarray(rgb).resize(
                (round(frame.shape[1] * scale), round(frame.shape[0] * scale))))
            label = tk.Label(window, image=picture)
            label.image = picture
            label.pack()
            window.bind("<Escape>", lambda _event: window.destroy())
        self.capture_after_countdown(show, "detection preview")

    def diagnostics(self):
        def save(frame):
            folder = application_dir() / "diagnostics"
            folder.mkdir(exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            with self.detector_lock:
                obs = self.detector.analyze(frame, time.perf_counter())
            with self.control_lock:
                metrics = dict(self.last_frame_metrics)
            save_image(folder / (stamp + "-screen.png"), frame)
            save_image(folder / (stamp + "-detection.png"), self.detector.annotate(frame, obs))
            (folder / (stamp + ".json")).write_text(json.dumps({
                "dpi": DPI_AWARENESS, "size": [frame.shape[1], frame.shape[0]],
                "missing": self.detector.ready(), "phase": self.engine.phase,
                "observation": str(obs),
                "last_worker_frame": metrics,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            self.ui.update_status(self.snapshot("Local snapshots saved in the diagnostics folder."))
        self.capture_after_countdown(save, "image diagnostics")

    def close(self):
        self.stop()
        self.shutdown.set()
        self.desktop.close()
        self.worker.join(timeout=0.5)
        self.root.destroy()


def main():
    root = tk.Tk()
    app = None
    try:
        app = Application(root)
        root.mainloop()
    except Exception as error:
        try:
            messagebox.showerror("Shrooms Macro", f"Could not start the application:\n{error}")
        finally:
            try:
                root.destroy()
            except tk.TclError:
                pass
    finally:
        if app is not None:
            app.active.clear()
            app.shutdown.set()
            app.input_cancel.set()
            app.desktop.close()
