"""Select a region or reference point from a static client screenshot."""

from __future__ import annotations

import ctypes
import math
import os
import tkinter as tk
from ctypes import wintypes
from typing import Any

import numpy as np
from PIL import Image, ImageTk


def _monitor_work_area(root: tk.Misc) -> tuple[int, int, int, int]:
    """Use the monitor containing the control panel, including negative origins."""
    fallback = (0, 0, root.winfo_screenwidth(), root.winfo_screenheight())
    if os.name != "nt":
        return fallback
    try:
        class MonitorInfo(ctypes.Structure):
            _fields_ = [
                ("size", wintypes.DWORD),
                ("monitor", wintypes.RECT),
                ("work", wintypes.RECT),
                ("flags", wintypes.DWORD),
            ]

        user32 = ctypes.WinDLL("user32", use_last_error=True)
        user32.MonitorFromWindow.argtypes = (wintypes.HWND, wintypes.DWORD)
        user32.MonitorFromWindow.restype = wintypes.HANDLE
        user32.GetMonitorInfoW.argtypes = (wintypes.HANDLE, ctypes.POINTER(MonitorInfo))
        user32.GetMonitorInfoW.restype = wintypes.BOOL
        monitor = user32.MonitorFromWindow(root.winfo_id(), 2)
        info = MonitorInfo()
        info.size = ctypes.sizeof(info)
        if monitor and user32.GetMonitorInfoW(monitor, ctypes.byref(info)):
            work = info.work
            return work.left, work.top, work.right - work.left, work.bottom - work.top
    except (AttributeError, OSError, tk.TclError):
        pass
    return fallback


def _place_dialog(dialog: tk.Toplevel, x: int, y: int) -> None:
    """Place by absolute coordinates; Tk negative offsets mean right/bottom edges."""
    if os.name == "nt":
        try:
            user32 = ctypes.WinDLL("user32", use_last_error=True)
            user32.GetAncestor.argtypes = (wintypes.HWND, wintypes.UINT)
            user32.GetAncestor.restype = wintypes.HWND
            user32.SetWindowPos.argtypes = (
                wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int, wintypes.UINT,
            )
            user32.SetWindowPos.restype = wintypes.BOOL
            hwnd = user32.GetAncestor(dialog.winfo_id(), 2) or dialog.winfo_id()
            # SWP_NOSIZE | SWP_NOZORDER: alter placement only.
            if user32.SetWindowPos(hwnd, None, x, y, 0, 0, 0x0001 | 0x0004):
                return
        except (AttributeError, OSError, tk.TclError):
            pass
    dialog.geometry(f"+{max(0, x)}+{max(0, y)}")


def choose_region(
    root: tk.Tk, frame_bgr: np.ndarray, title: str, point: bool = False,
) -> dict[str, Any] | None:
    """Return a normalized selection from the supplied BGR image, or ``None``.

    The screenshot is never refreshed and no desktop input is generated. Region
    coordinates are based on the original image dimensions, independent of the
    displayed preview size and monitor scale.
    """
    try:
        source = np.asarray(frame_bgr)
        if source.ndim != 3 or source.shape[2] != 3 or source.size == 0:
            return None
        height, width = source.shape[:2]
        if height < 1 or width < 1 or (not point and (height < 8 or width < 8)):
            return None
        if not np.issubdtype(source.dtype, np.number):
            return None
        if source.dtype != np.uint8:
            source = np.clip(np.nan_to_num(source, nan=0, posinf=255, neginf=0), 0, 255).astype(np.uint8)
        else:
            source = source.copy()
    except (TypeError, ValueError, OverflowError):
        return None

    left, top, monitor_width, monitor_height = _monitor_work_area(root)
    available_width = max(100, min(1200, monitor_width - 64))
    available_height = max(100, min(760, monitor_height - 180))
    scale = min(1.0, available_width / width, available_height / height)
    display_width = max(1, round(width * scale))
    display_height = max(1, round(height * scale))
    image = Image.fromarray(np.ascontiguousarray(source[:, :, ::-1]))
    if image.size != (display_width, display_height):
        image = image.resize((display_width, display_height), Image.Resampling.LANCZOS)

    dialog = tk.Toplevel(root)
    dialog.withdraw()
    dialog.title(title)
    dialog.configure(bg="#0e1521")
    dialog.resizable(False, False)
    if root.winfo_viewable():
        dialog.transient(root)
    dialog.attributes("-topmost", True)

    container = tk.Frame(dialog, bg="#0e1521")
    container.pack(padx=12, pady=12)
    instruction = "Click the reference point." if point else "Drag to select a rectangle."
    tk.Label(
        container, text=instruction, bg="#0e1521", fg="#eef5fa",
        font=("Segoe UI", 10, "bold"), anchor="w",
    ).pack(fill="x")
    detail_var = tk.StringVar(value="Enter saves • Esc cancels")
    tk.Label(
        container, textvariable=detail_var, bg="#0e1521", fg="#94a6bc",
        font=("Segoe UI", 9), anchor="w",
    ).pack(fill="x", pady=(4, 9))
    canvas = tk.Canvas(
        container, width=display_width, height=display_height,
        bg="#151f2f", highlightthickness=0, borderwidth=0, cursor="crosshair",
    )
    canvas.pack()
    photo = ImageTk.PhotoImage(image, master=dialog)
    canvas.create_image(0, 0, image=photo, anchor="nw")
    # Keep the Tk image alive for the complete modal lifetime.
    canvas.image_reference = photo  # type: ignore[attr-defined]

    result: dict[str, Any] | None = None
    selected_bounds: tuple[int, int, int, int] | None = None
    selected_point: tuple[int, int] | None = None
    drag_start: tuple[float, float] | None = None

    buttons = tk.Frame(container, bg="#0e1521")
    buttons.pack(fill="x", pady=(10, 0))

    def cancel(_event: Any = None) -> str:
        dialog.destroy()
        return "break"

    def confirm(_event: Any = None) -> str:
        nonlocal result
        if selected_bounds is None:
            return "break"
        x0, y0, x1, y1 = selected_bounds
        result = {
            "region": [x0 / width, y0 / height, (x1 - x0) / width, (y1 - y0) / height],
            "crop": source[y0:y1, x0:x1].copy(),
        }
        if point and selected_point is not None:
            px, py = selected_point
            sample = source[max(0, py - 1):min(height, py + 2), max(0, px - 1):min(width, px + 2)]
            result["point"] = [px / width, py / height]
            result["color"] = [int(value) for value in np.median(sample.reshape(-1, 3), axis=0)]
        dialog.destroy()
        return "break"

    def button(text: str, command: Any, primary: bool = False) -> tk.Button:
        return tk.Button(
            buttons, text=text, command=command, bg="#80e3ae" if primary else "#1a2739",
            fg="#0e1521" if primary else "#eef5fa", relief="flat", borderwidth=0,
            activebackground="#a2edc4" if primary else "#28374c",
            activeforeground="#0e1521" if primary else "#eef5fa",
            disabledforeground="#6c7f94", cursor="hand2", padx=16, pady=7,
            font=("Segoe UI", 9, "bold"), highlightthickness=0,
        )

    save_button = button("Save", confirm, primary=True)
    save_button.configure(state="disabled")
    save_button.pack(side="right")
    button("Cancel", cancel).pack(side="right", padx=(0, 7))

    def canvas_position(event: Any) -> tuple[float, float]:
        return (
            max(0.0, min(float(display_width), float(event.x))),
            max(0.0, min(float(display_height), float(event.y))),
        )

    def draw_rectangle(end: tuple[float, float]) -> None:
        nonlocal selected_bounds
        if drag_start is None:
            return
        sx, sy = drag_start
        ex, ey = end
        low_x, high_x = sorted((sx, ex))
        low_y, high_y = sorted((sy, ey))
        x0 = max(0, min(width - 1, math.floor(low_x * width / display_width)))
        y0 = max(0, min(height - 1, math.floor(low_y * height / display_height)))
        x1 = max(0, min(width, math.ceil(high_x * width / display_width)))
        y1 = max(0, min(height, math.ceil(high_y * height / display_height)))
        valid = x1 - x0 >= 8 and y1 - y0 >= 8
        selected_bounds = (x0, y0, x1, y1) if valid else None
        canvas.delete("selection")
        canvas.create_rectangle(low_x, low_y, high_x, high_y, outline="#80e3ae" if valid else "#f3cf87", width=2, tags="selection")
        detail_var.set(
            f"{max(0, x1 - x0)} × {max(0, y1 - y0)} px • Enter saves • Esc cancels"
            if valid else "Select at least 8 × 8 px • Esc cancels"
        )
        save_button.configure(state="normal" if valid else "disabled")

    def press(event: Any) -> None:
        nonlocal drag_start, selected_point, selected_bounds
        cx, cy = canvas_position(event)
        if not point:
            drag_start = (cx, cy)
            draw_rectangle(drag_start)
            return
        px = max(0, min(width - 1, int(cx * width / display_width)))
        py = max(0, min(height - 1, int(cy * height / display_height)))
        selected_point = (px, py)
        selected_bounds = (max(0, px - 7), max(0, py - 7), min(width, px + 8), min(height, py + 8))
        canvas.delete("selection")
        canvas.create_oval(cx - 5, cy - 5, cx + 5, cy + 5, outline="#80e3ae", width=2, tags="selection")
        canvas.create_line(cx - 10, cy, cx + 10, cy, fill="#80e3ae", tags="selection")
        canvas.create_line(cx, cy - 10, cx, cy + 10, fill="#80e3ae", tags="selection")
        detail_var.set(f"Point: {px}, {py} px • Enter saves • Esc cancels")
        save_button.configure(state="normal")

    def drag(event: Any) -> None:
        if not point:
            draw_rectangle(canvas_position(event))

    canvas.bind("<ButtonPress-1>", press)
    canvas.bind("<B1-Motion>", drag)
    canvas.bind("<ButtonRelease-1>", drag)
    dialog.bind("<Escape>", cancel)
    dialog.bind("<Return>", confirm)
    dialog.bind("<KP_Enter>", confirm)
    dialog.protocol("WM_DELETE_WINDOW", cancel)
    dialog.update_idletasks()
    requested_width, requested_height = dialog.winfo_reqwidth(), dialog.winfo_reqheight()
    x = left + max(0, (monitor_width - requested_width) // 2)
    y = top + max(20, (monitor_height - requested_height) // 2 - 20)
    dialog.deiconify()
    dialog.update_idletasks()
    _place_dialog(dialog, x, y)
    dialog.grab_set()
    canvas.focus_set()
    dialog.focus_force()
    root.wait_window(dialog)
    return result
