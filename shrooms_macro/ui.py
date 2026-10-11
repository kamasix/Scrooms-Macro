"""Compact desktop controls for the screen-based Shrooms macro."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any, Callable

from . import __version__


class MacroUI:
    """Tkinter view. All methods and callbacks run on the Tk main thread."""

    BG = "#0e1521"
    CARD = "#151f2f"
    FIELD = "#1a2739"
    BORDER = "#28374c"
    TEXT = "#eef5fa"
    MUTED = "#94a6bc"
    GREEN = "#80e3ae"
    RED = "#ff9a9d"
    AMBER = "#f3cf87"

    def __init__(self, root: tk.Tk, callbacks: dict, initial: dict) -> None:
        self.root = root
        self.callbacks: dict[str, Callable[..., Any]] = callbacks
        # Keep edits local until the controller has stopped the worker safely.
        self.initial = dict(initial or {})
        self.running = False
        self._caught = 0
        self._advanced_open = False
        self._selected_hwnd: int | None = None
        self._window_choices: dict[str, int] = {}
        self._fit_job: str | None = None

        root.title(f"Shrooms Macro {__version__}")
        root.configure(bg=self.BG)
        root.geometry("326x350+18+78")
        root.resizable(False, False)
        root.protocol("WM_DELETE_WINDOW", self._close)
        root.attributes("-topmost", bool(self.initial.get("always_on_top", True)))
        self._setup_styles()

        self.status_var = tk.StringVar(value="Ready")
        self.target_var = tk.StringVar(value="")
        self.count_var = tk.StringVar(value="")
        self.sold_var = tk.StringVar(value="Sale cycles: 0")
        self.warning_var = tk.StringVar(value="")
        self.window_var = tk.StringVar(value="Select game window…")
        self.sell_every_var = tk.StringVar(value=str(self._sell_every()))
        self.auto_sell_var = tk.BooleanVar(value=bool(self.initial.get("auto_sell", True)))
        self.perfect_ms_var = tk.StringVar(value=str(self._perfect_ms()))
        self.topmost_var = tk.BooleanVar(value=bool(self.initial.get("always_on_top", True)))

        content = tk.Frame(root, bg=self.BG)
        content.pack(fill="both", expand=True, padx=15, pady=(13, 12))
        self.content = content
        self._build_header(content)
        self._build_progress(content)
        self._build_controls(content)
        self._build_window_choice(content)
        self._build_advanced(content)
        self._sync_sell_controls()

        self.warning_label = tk.Label(
            content, textvariable=self.warning_var, bg=self.BG,
            fg=self.AMBER, font=("Segoe UI", 9), justify="left",
            anchor="w", wraplength=292,
        )
        self.guidance = tk.Label(
            content,
            text="F6  start / pause     •     F8  stop",
            bg=self.BG, fg=self.MUTED, font=("Segoe UI", 9), anchor="center",
        )
        self.guidance.pack(fill="x", pady=(12, 0))
        self.discord_button = self._button(
            content, "Join Discord ↗", lambda: self._call("discord"), compact=True,
        )
        self.discord_button.configure(
            bg=self.BG, fg=self.GREEN, activebackground=self.BG,
            activeforeground="#a2edc4", pady=2,
        )
        self.discord_button.pack(pady=(5, 0))
        self._schedule_fit()
        root.after_idle(self.refresh_windows)

    def _setup_styles(self) -> None:
        style = ttk.Style(self.root)
        # clam honors explicit dark colors on Windows, unlike native themed controls.
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure(
            "Macro.TCombobox", fieldbackground=self.FIELD, background=self.FIELD,
            foreground=self.TEXT, arrowcolor=self.MUTED, bordercolor=self.BORDER,
            lightcolor=self.BORDER, darkcolor=self.BORDER, padding=5,
        )
        style.map(
            "Macro.TCombobox",
            fieldbackground=[("readonly", self.FIELD)],
            foreground=[("readonly", self.TEXT)],
            selectbackground=[("readonly", self.FIELD)],
            selectforeground=[("readonly", self.TEXT)],
        )
        style.configure(
            "Macro.Vertical.TScrollbar", background=self.FIELD,
            troughcolor=self.CARD, bordercolor=self.CARD,
            arrowcolor=self.MUTED, lightcolor=self.FIELD, darkcolor=self.FIELD,
        )
        self.root.option_add("*TCombobox*Listbox.background", self.FIELD)
        self.root.option_add("*TCombobox*Listbox.foreground", self.TEXT)
        self.root.option_add("*TCombobox*Listbox.selectBackground", self.BORDER)
        self.root.option_add("*TCombobox*Listbox.selectForeground", self.TEXT)

    def _label(self, parent: tk.Widget, text: str = "", **kwargs: Any) -> tk.Label:
        return tk.Label(
            parent, text=text, bg=kwargs.pop("bg", self.BG),
            fg=kwargs.pop("fg", self.TEXT), font=kwargs.pop("font", ("Segoe UI", 10)),
            **kwargs,
        )

    def _button(
        self, parent: tk.Widget, text: str, command: Callable[[], Any],
        *, primary: bool = False, compact: bool = False,
    ) -> tk.Button:
        color = self.GREEN if primary else self.FIELD
        return tk.Button(
            parent, text=text, command=command, bg=color,
            fg=self.BG if primary else self.TEXT,
            activebackground="#a2edc4" if primary else self.BORDER,
            activeforeground=self.BG if primary else self.TEXT,
            relief="flat", borderwidth=0, highlightthickness=0,
            cursor="hand2", font=("Segoe UI", 9 if compact else 10, "bold"),
            padx=9 if compact else 15, pady=6 if compact else 9,
            takefocus=True,
        )

    def _build_header(self, parent: tk.Widget) -> None:
        header = tk.Frame(parent, bg=self.BG)
        header.pack(fill="x")
        self._label(header, "●", fg=self.GREEN, font=("Segoe UI", 12)).pack(side="left", padx=(0, 7))
        self._label(header, "SHROOMS", font=("Segoe UI", 13, "bold")).pack(side="left")
        self._label(header, "MACRO", fg=self.MUTED, font=("Segoe UI", 9)).pack(side="left", padx=(7, 0), pady=(3, 0))
        self._label(
            parent, textvariable=self.status_var, fg=self.MUTED, anchor="w",
            font=("Segoe UI", 9), wraplength=292,
        ).pack(fill="x", pady=(6, 12))

    def _build_progress(self, parent: tk.Widget) -> None:
        card = tk.Frame(parent, bg=self.CARD, highlightbackground=self.BORDER, highlightthickness=1)
        card.pack(fill="x")
        inside = tk.Frame(card, bg=self.CARD)
        inside.pack(fill="x", padx=12, pady=10)
        row = tk.Frame(inside, bg=self.CARD)
        row.pack(fill="x")
        self._label(row, "CAUGHT", bg=self.CARD, fg=self.MUTED, font=("Segoe UI", 8, "bold")).pack(side="left")
        self._label(row, textvariable=self.count_var, bg=self.CARD, fg=self.GREEN, font=("Segoe UI", 16, "bold")).pack(side="right")
        self._label(
            inside, textvariable=self.sold_var, bg=self.CARD, fg=self.MUTED,
            anchor="w", font=("Segoe UI", 9),
        ).pack(fill="x", pady=(3, 0))

    def _build_controls(self, parent: tk.Widget) -> None:
        row = tk.Frame(parent, bg=self.BG)
        row.pack(fill="x", pady=(12, 5))
        self.start_button = self._button(row, "▶  Start", lambda: self._call("toggle"), primary=True)
        self.start_button.pack(side="left", fill="x", expand=True)
        self.stop_button = self._button(row, "■  Stop", lambda: self._call("stop"))
        self.stop_button.pack(side="left", padx=(8, 0))
        self.auto_sell_check = tk.Checkbutton(
            parent, text="Auto sell", variable=self.auto_sell_var,
            command=self._save_settings, bg=self.BG, fg=self.TEXT,
            activebackground=self.BG, activeforeground=self.TEXT,
            selectcolor=self.FIELD, relief="flat", font=("Segoe UI", 9),
            anchor="w", cursor="hand2", highlightthickness=0,
        )
        self.auto_sell_check.pack(fill="x", pady=(0, 9))

    def _build_window_choice(self, parent: tk.Widget) -> None:
        self._label(parent, "GAME WINDOW", fg=self.MUTED, font=("Segoe UI", 8, "bold"), anchor="w").pack(fill="x", pady=(0, 5))
        row = tk.Frame(parent, bg=self.BG)
        row.pack(fill="x")
        self.window_combo = ttk.Combobox(
            row, textvariable=self.window_var, state="readonly",
            style="Macro.TCombobox", font=("Segoe UI", 9), width=23,
        )
        self.window_combo.pack(side="left", fill="x", expand=True)
        self.window_combo.bind("<<ComboboxSelected>>", self._select_window)
        self._button(row, "↻", self.refresh_windows, compact=True).pack(side="left", padx=(6, 0))
        self._label(
            parent, textvariable=self.target_var, fg=self.MUTED,
            font=("Segoe UI", 8), anchor="w", wraplength=292,
        ).pack(fill="x", pady=(4, 0))

    def _build_advanced(self, parent: tk.Widget) -> None:
        self.advanced_button = self._button(parent, "▸  Settings & calibration", self._toggle_advanced, compact=True)
        self.advanced_button.configure(bg=self.BG, fg=self.MUTED, anchor="w", padx=0)
        self.advanced_button.pack(fill="x", pady=(10, 0))
        self.advanced = tk.Frame(parent, bg=self.CARD, highlightbackground=self.BORDER, highlightthickness=1)
        self._advanced_canvas = tk.Canvas(
            self.advanced, bg=self.CARD, highlightthickness=0,
            borderwidth=0, width=270, height=260, yscrollincrement=20,
        )
        scrollbar = ttk.Scrollbar(
            self.advanced, orient="vertical", command=self._advanced_canvas.yview,
            style="Macro.Vertical.TScrollbar",
        )
        self._advanced_canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self._advanced_canvas.pack(side="left", fill="both", expand=True)
        inner = tk.Frame(self._advanced_canvas, bg=self.CARD)
        self._advanced_inner = inner
        self._advanced_window = self._advanced_canvas.create_window((10, 10), window=inner, anchor="nw")
        inner.bind("<Configure>", self._advanced_scrollregion)
        self._advanced_canvas.bind("<Configure>", self._advanced_width)

        settings_row = tk.Frame(inner, bg=self.CARD)
        settings_row.pack(fill="x")
        self._label(settings_row, "Sell every", bg=self.CARD, fg=self.MUTED, font=("Segoe UI", 9)).pack(side="left")
        self.sell_spin = tk.Spinbox(
            settings_row, from_=1, to=999, width=4, textvariable=self.sell_every_var,
            bg=self.FIELD, fg=self.TEXT, insertbackground=self.TEXT,
            disabledbackground=self.FIELD, disabledforeground=self.MUTED,
            buttonbackground=self.FIELD, relief="flat", highlightthickness=1,
            highlightbackground=self.BORDER, highlightcolor=self.GREEN,
            font=("Segoe UI", 9), command=self._save_settings,
        )
        self.sell_spin.pack(side="right")
        self.sell_spin.bind("<FocusOut>", lambda _event: self._save_settings())
        self.sell_spin.bind("<Return>", lambda _event: self._save_settings())
        self._label(settings_row, "shrooms", bg=self.CARD, fg=self.MUTED, font=("Segoe UI", 9)).pack(side="right", padx=(4, 7))

        timing_row = tk.Frame(inner, bg=self.CARD)
        timing_row.pack(fill="x", pady=(8, 0))
        self._label(timing_row, "Perfect correction", bg=self.CARD, fg=self.MUTED,
                    font=("Segoe UI", 9)).pack(side="left")
        timing_spin = tk.Spinbox(
            timing_row, from_=0, to=120, width=4, textvariable=self.perfect_ms_var,
            bg=self.FIELD, fg=self.TEXT, insertbackground=self.TEXT,
            buttonbackground=self.FIELD, relief="flat", highlightthickness=1,
            highlightbackground=self.BORDER, highlightcolor=self.GREEN,
            font=("Segoe UI", 9), command=self._save_settings,
        )
        timing_spin.pack(side="right")
        timing_spin.bind("<FocusOut>", lambda _event: self._save_settings())
        timing_spin.bind("<Return>", lambda _event: self._save_settings())
        self._label(timing_row, "ms", bg=self.CARD, fg=self.MUTED,
                    font=("Segoe UI", 9)).pack(side="right", padx=(4, 7))
        self._label(inner, "Increase if Perfect clicks land late.", bg=self.CARD,
                    fg=self.MUTED, font=("Segoe UI", 8)).pack(fill="x", pady=(3, 0))

        tk.Checkbutton(
            inner, text="Always on top", variable=self.topmost_var,
            command=self._save_settings, bg=self.CARD, fg=self.TEXT,
            activebackground=self.CARD, activeforeground=self.TEXT,
            selectcolor=self.FIELD, relief="flat", font=("Segoe UI", 9),
            anchor="w", cursor="hand2", highlightthickness=0,
        ).pack(fill="x", pady=(7, 8))

        self._label(
            inner, "Calibration: select an element in the game.", bg=self.CARD,
            fg=self.MUTED, font=("Segoe UI", 9), anchor="w",
        ).pack(fill="x", pady=(10, 5))

        calibrations = (
            ("charge_region", "Perfect ring", "frequency_region", "Catch field"),
            ("tuning_region", "Tuning", "progress_region", "Progress bar"),
            ("tune_prompt", "Tuning prompt", "charge_marker", "Perfect pointer"),
            ("success", "Catch success", "sell", "Sell All"),
        )
        for left_kind, left_text, right_kind, right_text in calibrations:
            row = tk.Frame(inner, bg=self.CARD)
            row.pack(fill="x", pady=2)
            self._button(row, left_text, lambda kind=left_kind: self._call("calibrate", kind), compact=True).pack(side="left", fill="x", expand=True)
            self._button(row, right_text, lambda kind=right_kind: self._call("calibrate", kind), compact=True).pack(side="left", fill="x", expand=True, padx=(5, 0))
        self._button(inner, "Sale confirmation", lambda: self._call("calibrate", "confirm"), compact=True).pack(fill="x", pady=(2, 7))
        row = tk.Frame(inner, bg=self.CARD)
        row.pack(fill="x")
        self._button(row, "Preview", lambda: self._call("preview"), compact=True).pack(side="left", fill="x", expand=True)
        self._button(row, "Diagnostics", lambda: self._call("diagnostics"), compact=True).pack(side="left", fill="x", expand=True, padx=(5, 0))
        self._bind_advanced_wheel(self.advanced)

    def _advanced_scrollregion(self, _event: Any = None) -> None:
        bounds = self._advanced_canvas.bbox("all")
        if bounds:
            self._advanced_canvas.configure(scrollregion=(0, 0, bounds[2] + 10, bounds[3] + 10))

    def _advanced_width(self, event: Any) -> None:
        self._advanced_canvas.itemconfigure(self._advanced_window, width=max(1, event.width - 20))
        self._advanced_scrollregion()

    def _bind_advanced_wheel(self, widget: tk.Widget) -> None:
        # Bind only descendants of the settings panel, never the whole application.
        widget.bind("<MouseWheel>", self._scroll_advanced, add="+")
        widget.bind("<Button-4>", self._scroll_advanced, add="+")
        widget.bind("<Button-5>", self._scroll_advanced, add="+")
        for child in widget.winfo_children():
            self._bind_advanced_wheel(child)

    def _scroll_advanced(self, event: Any) -> str | None:
        if not self._advanced_open:
            return None
        if getattr(event, "num", None) == 4:
            steps = -1
        elif getattr(event, "num", None) == 5:
            steps = 1
        else:
            delta = getattr(event, "delta", 0)
            if not delta:
                return "break"
            steps = -int(delta / 120) if abs(delta) >= 120 else (-1 if delta > 0 else 1)
        self._advanced_canvas.yview_scroll(steps, "units")
        return "break"

    def _sell_every(self) -> int:
        try:
            return max(1, min(999, int(self.initial.get("sell_every", 1))))
        except (TypeError, ValueError):
            return 1

    def _perfect_ms(self) -> int:
        try:
            return max(0, min(120, round(float(self.initial.get("input_latency", 0.008)) * 1000)))
        except (TypeError, ValueError, OverflowError):
            return 8

    def _sync_sell_controls(self) -> None:
        enabled = bool(self.auto_sell_var.get())
        self.sell_spin.configure(state="normal" if enabled else "disabled")
        self.count_var.set(
            f"{self._caught} / {self._sell_every()}" if enabled else str(self._caught)
        )

    def _call(self, name: str, *args: Any) -> Any:
        callback = self.callbacks.get(name)
        if callback is None:
            return None
        try:
            return callback(*args)
        except Exception as error:
            self._show_warning(f"Could not complete action: {error}")
            return None

    def refresh_windows(self) -> None:
        windows = self._call("refresh")
        if windows is None:
            return
        choices: dict[str, int] = {}
        for entry in windows:
            try:
                hwnd = int(entry["hwnd"])
                label = str(entry["label"])
            except (KeyError, TypeError, ValueError):
                continue
            display = label
            if display in choices:
                display = f"{label} ({hwnd})"
            choices[display] = hwnd
        self._window_choices = choices
        self.window_combo["values"] = tuple(choices)
        selected = next((name for name, hwnd in choices.items() if hwnd == self._selected_hwnd), None)
        if selected:
            self.window_var.set(selected)
        elif choices:
            self.window_var.set(next(iter(choices)))
            self._select_window()
        else:
            self._selected_hwnd = None
            self.window_var.set("No game window found")

    def _select_window(self, _event: Any = None) -> None:
        hwnd = self._window_choices.get(self.window_var.get())
        if hwnd is not None:
            self._selected_hwnd = hwnd
            self._call("select", hwnd)

    def _save_settings(self) -> None:
        try:
            sell_every = max(1, min(999, int(self.sell_every_var.get())))
        except (TypeError, ValueError):
            sell_every = self._sell_every()
        self.sell_every_var.set(str(sell_every))
        try:
            perfect_ms = max(0, min(120, round(float(self.perfect_ms_var.get()))))
        except (TypeError, ValueError, OverflowError):
            perfect_ms = self._perfect_ms()
        self.perfect_ms_var.set(str(perfect_ms))
        settings = {
            "auto_sell": bool(self.auto_sell_var.get()),
            "sell_every": sell_every,
            "input_latency": perfect_ms / 1000,
            "always_on_top": bool(self.topmost_var.get()),
        }
        self.initial.update(settings)
        self._sync_sell_controls()
        self.root.attributes("-topmost", settings["always_on_top"])
        self._call("settings", settings)

    def _toggle_advanced(self) -> None:
        self._advanced_open = not self._advanced_open
        if self._advanced_open:
            self.advanced.pack(fill="x", after=self.advanced_button, pady=(5, 0))
            self.advanced_button.configure(text="▾  Settings & calibration")
        else:
            self.advanced.pack_forget()
            self.advanced_button.configure(text="▸  Settings & calibration")
        self._schedule_fit()

    def _show_warning(self, message: str) -> None:
        self.warning_var.set(message)
        if message:
            self.warning_label.pack(fill="x", before=self.guidance, pady=(8, 0))
        else:
            self.warning_label.pack_forget()
        self._schedule_fit()

    def _schedule_fit(self) -> None:
        if self._fit_job is not None:
            self.root.after_cancel(self._fit_job)
        self._fit_job = self.root.after_idle(self._fit_height)

    def _fit_height(self) -> None:
        self._fit_job = None
        self.root.update_idletasks()
        if self._advanced_open:
            collapsed_height = self.content.winfo_reqheight() + 25 - self.advanced.winfo_reqheight() - 5
            visible_height = max(200, min(430, self.root.winfo_screenheight() - collapsed_height - 120))
            if int(self._advanced_canvas.cget("height")) != visible_height:
                self._advanced_canvas.configure(height=visible_height)
                self.root.update_idletasks()
        height = self.content.winfo_reqheight() + 25
        # Preserve the user's dragged position when expanding or collapsing settings.
        height = min(height, max(300, self.root.winfo_screenheight() - 100))
        self.root.geometry(f"326x{height}")

    def update_status(self, snapshot: dict) -> None:
        """Apply a complete or partial status snapshot from the controller."""
        self.running = bool(snapshot.get("running", self.running))
        self.start_button.configure(
            text="Ⅱ  Pause" if self.running else "▶  Start",
            bg=self.AMBER if self.running else self.GREEN,
            activebackground="#ffe2a8" if self.running else "#a2edc4",
        )
        if "status" in snapshot and str(snapshot["status"]) != self.status_var.get():
            self.status_var.set(str(snapshot["status"]))
            self._schedule_fit()
        if "target" in snapshot and str(snapshot["target"]) != self.target_var.get():
            self.target_var.set(str(snapshot["target"]))
            self._schedule_fit()
        if "auto_sell" in snapshot:
            self.initial["auto_sell"] = bool(snapshot["auto_sell"])
            self.auto_sell_var.set(self.initial["auto_sell"])
        if "caught" in snapshot:
            self._caught = snapshot["caught"]
        if "auto_sell" in snapshot or "caught" in snapshot:
            self._sync_sell_controls()
        if "sold_batches" in snapshot:
            self.sold_var.set(f"Sale cycles: {snapshot['sold_batches']}")
        if "warning" in snapshot and str(snapshot.get("warning") or "") != self.warning_var.get():
            self._show_warning(str(snapshot.get("warning") or ""))

    def _close(self) -> None:
        self._call("stop")
        self.root.destroy()
