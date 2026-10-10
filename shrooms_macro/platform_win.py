"""Desktop-only Windows capture and input; never accesses Roblox internals.

Import this module before creating any Tk windows. Coordinates accepted by
``move`` and ``click`` are physical pixels relative to the Roblox client area.
"""

from __future__ import annotations

import ctypes
import math
import os
import sys
import threading
from ctypes import wintypes
from dataclasses import dataclass
from typing import Iterable

import numpy as np


class InputUnavailable(RuntimeError):
    """The selected game is unavailable or is no longer the foreground app."""


@dataclass(frozen=True)
class Rect:
    left: int
    top: int
    width: int
    height: int


@dataclass(frozen=True)
class WindowInfo:
    hwnd: int
    title: str
    process_name: str


_IS_WINDOWS = sys.platform == "win32"
_PLAYER_EXES = frozenset(("robloxplayerbeta.exe", "robloxplayer.exe"))
_ULONG_PTR = ctypes.c_size_t
_HGDIOBJ = ctypes.c_void_p
_HBITMAP = ctypes.c_void_p


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", _ULONG_PTR),
    ]


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", _ULONG_PTR),
    ]


class _HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("mi", _MOUSEINPUT), ("ki", _KEYBDINPUT), ("hi", _HARDWAREINPUT)]


class _INPUT(ctypes.Structure):
    _anonymous_ = ("data",)
    _fields_ = [("type", wintypes.DWORD), ("data", _INPUTUNION)]


class _BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD),
        ("biWidth", wintypes.LONG),
        ("biHeight", wintypes.LONG),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]


class _RGBQUAD(ctypes.Structure):
    _fields_ = [
        ("rgbBlue", wintypes.BYTE),
        ("rgbGreen", wintypes.BYTE),
        ("rgbRed", wintypes.BYTE),
        ("rgbReserved", wintypes.BYTE),
    ]


class _BITMAPINFO(ctypes.Structure):
    _fields_ = [("bmiHeader", _BITMAPINFOHEADER), ("bmiColors", _RGBQUAD * 1)]


if _IS_WINDOWS:
    _user32 = ctypes.WinDLL("user32", use_last_error=True)
    _gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def _prototype(dll, name, result, args):
        function = getattr(dll, name)
        function.restype = result
        function.argtypes = args
        return function

    _prototype(_user32, "EnumWindows", wintypes.BOOL, [_WNDENUMPROC, wintypes.LPARAM])
    _prototype(_user32, "IsWindow", wintypes.BOOL, [wintypes.HWND])
    _prototype(_user32, "IsWindowVisible", wintypes.BOOL, [wintypes.HWND])
    _prototype(_user32, "IsIconic", wintypes.BOOL, [wintypes.HWND])
    _prototype(_user32, "GetWindowTextLengthW", ctypes.c_int, [wintypes.HWND])
    _prototype(_user32, "GetWindowTextW", ctypes.c_int,
               [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int])
    _prototype(_user32, "GetWindowThreadProcessId", wintypes.DWORD,
               [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)])
    _prototype(_user32, "GetClientRect", wintypes.BOOL,
               [wintypes.HWND, ctypes.POINTER(wintypes.RECT)])
    _prototype(_user32, "ClientToScreen", wintypes.BOOL,
               [wintypes.HWND, ctypes.POINTER(wintypes.POINT)])
    _prototype(_user32, "GetForegroundWindow", wintypes.HWND, [])
    _prototype(_user32, "SetForegroundWindow", wintypes.BOOL, [wintypes.HWND])
    _prototype(_user32, "ShowWindow", wintypes.BOOL, [wintypes.HWND, ctypes.c_int])
    _prototype(_user32, "GetCursorPos", wintypes.BOOL, [ctypes.POINTER(wintypes.POINT)])
    _prototype(_user32, "WindowFromPoint", wintypes.HWND, [wintypes.POINT])
    _prototype(_user32, "GetAncestor", wintypes.HWND, [wintypes.HWND, wintypes.UINT])
    _prototype(_user32, "GetSystemMetrics", ctypes.c_int, [ctypes.c_int])
    _prototype(_user32, "GetAsyncKeyState", wintypes.SHORT, [ctypes.c_int])
    _prototype(_user32, "SendInput", wintypes.UINT,
               [wintypes.UINT, ctypes.POINTER(_INPUT), ctypes.c_int])
    _prototype(_user32, "GetDC", wintypes.HDC, [wintypes.HWND])
    _prototype(_user32, "ReleaseDC", ctypes.c_int, [wintypes.HWND, wintypes.HDC])
    _prototype(_kernel32, "OpenProcess", wintypes.HANDLE,
               [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD])
    _prototype(_kernel32, "QueryFullProcessImageNameW", wintypes.BOOL,
               [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR,
                ctypes.POINTER(wintypes.DWORD)])
    _prototype(_kernel32, "CloseHandle", wintypes.BOOL, [wintypes.HANDLE])
    _prototype(_gdi32, "CreateCompatibleDC", wintypes.HDC, [wintypes.HDC])
    _prototype(_gdi32, "CreateDIBSection", _HBITMAP,
               [wintypes.HDC, ctypes.POINTER(_BITMAPINFO), wintypes.UINT,
                ctypes.POINTER(ctypes.c_void_p), wintypes.HANDLE, wintypes.DWORD])
    _prototype(_gdi32, "SelectObject", _HGDIOBJ, [wintypes.HDC, _HGDIOBJ])
    _prototype(_gdi32, "BitBlt", wintypes.BOOL,
               [wintypes.HDC, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                wintypes.HDC, ctypes.c_int, ctypes.c_int, wintypes.DWORD])
    _prototype(_gdi32, "GdiFlush", wintypes.BOOL, [])
    _prototype(_gdi32, "DeleteObject", wintypes.BOOL, [_HGDIOBJ])
    _prototype(_gdi32, "DeleteDC", wintypes.BOOL, [wintypes.HDC])


def enable_dpi_awareness() -> str:
    """Request physical-pixel coordinates before any GUI or capture is created."""
    if not _IS_WINDOWS:
        return "unsupported"
    if hasattr(_user32, "SetProcessDpiAwarenessContext"):
        function = _prototype(_user32, "SetProcessDpiAwarenessContext", wintypes.BOOL,
                              [ctypes.c_void_p])
        ctypes.set_last_error(0)
        if function(ctypes.c_void_p(-4)):  # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
            return "per-monitor-v2"
        if ctypes.get_last_error() == 5:  # The host already established awareness.
            return "already-set"
    try:
        shcore = ctypes.WinDLL("shcore", use_last_error=True)
        function = _prototype(shcore, "SetProcessDpiAwareness", ctypes.c_long,
                              [ctypes.c_int])
        result = function(2)  # PROCESS_PER_MONITOR_DPI_AWARE
        if result == 0:
            return "per-monitor"
        if (result & 0xFFFFFFFF) == 0x80070005:
            return "already-set"
    except (AttributeError, OSError):
        pass
    function = _prototype(_user32, "SetProcessDPIAware", wintypes.BOOL, [])
    return "system" if function() else "already-set"


DPI_AWARENESS = enable_dpi_awareness()
VK_OEM_3 = 0xC0


def _mouse_input(flags: int, dx: int = 0, dy: int = 0) -> _INPUT:
    item = _INPUT()
    item.type = 0
    item.mi = _MOUSEINPUT(dx, dy, 0, flags, 0, 0)
    return item


def _keyboard_input(vk: int, release: bool = False,
                    *, scan_code: int | None = None) -> _INPUT:
    item = _INPUT()
    item.type = 1
    flags = (0x0002 if release else 0) | (0x0008 if scan_code is not None else 0)
    item.ki = _KEYBDINPUT(0 if scan_code is not None else vk,
                         0 if scan_code is None else scan_code, flags, 0, 0)
    return item


class Desktop:
    """Capture Roblox Player and send foreground-only desktop mouse/key events.

    A Desktop instance should be used by one automation worker at a time. Calling
    ``mouse_up``, ``key_up`` or ``close`` is always safe, including from cleanup.
    """

    def __init__(self) -> None:
        if not _IS_WINDOWS:
            raise OSError("Shrooms Macro requires Windows.")
        self._closed = False
        self._left_held = False
        self._held_hwnd: int | None = None
        self._held_key: int | None = None
        self._held_key_up_packet: _INPUT | None = None
        self._input_lock = threading.RLock()

    @staticmethod
    def _process_name(hwnd: int) -> str:
        process_id = wintypes.DWORD()
        if not _user32.GetWindowThreadProcessId(hwnd, ctypes.byref(process_id)):
            return ""
        handle = _kernel32.OpenProcess(0x1000, False, process_id.value)
        if not handle:
            return ""
        try:
            length = wintypes.DWORD(32768)
            path = ctypes.create_unicode_buffer(length.value)
            if not _kernel32.QueryFullProcessImageNameW(handle, 0, path,
                                                       ctypes.byref(length)):
                return ""
            return os.path.basename(path.value)
        finally:
            _kernel32.CloseHandle(handle)

    def list_windows(self) -> list[WindowInfo]:
        windows: list[WindowInfo] = []

        @_WNDENUMPROC
        def visit(hwnd, _lparam):
            if not _user32.IsWindowVisible(hwnd):
                return True
            name = self._process_name(hwnd)
            if name.casefold() not in _PLAYER_EXES:
                return True
            length = _user32.GetWindowTextLengthW(hwnd)
            title = ctypes.create_unicode_buffer(max(1, length + 1))
            _user32.GetWindowTextW(hwnd, title, len(title))
            windows.append(WindowInfo(int(hwnd), title.value or "Roblox", name))
            return True

        if not _user32.EnumWindows(visit, 0):
            self.mouse_up()
            raise ctypes.WinError(ctypes.get_last_error())
        return sorted(windows, key=lambda item: (item.title.casefold(), item.hwnd))

    def is_available(self, hwnd: int) -> bool:
        return bool(
            not self._closed
            and hwnd
            and _user32.IsWindow(hwnd)
            and _user32.IsWindowVisible(hwnd)
            and not _user32.IsIconic(hwnd)
            and self._process_name(hwnd).casefold() in _PLAYER_EXES
        )

    def is_foreground(self, hwnd: int) -> bool:
        return bool(hwnd and _user32.IsWindow(hwnd)
                    and int(_user32.GetForegroundWindow() or 0) == int(hwnd))

    def _guard(self, hwnd: int) -> None:
        if not self.is_available(hwnd):
            self.mouse_up()
            raise InputUnavailable("The selected Roblox window is unavailable.")
        if not self.is_foreground(hwnd):
            self.mouse_up()
            raise InputUnavailable("Roblox must be the foreground window.")

    def client_rect(self, hwnd: int) -> Rect:
        if not self.is_available(hwnd):
            self.mouse_up()
            raise InputUnavailable("The selected Roblox window is unavailable.")
        bounds = wintypes.RECT()
        origin = wintypes.POINT(0, 0)
        if not _user32.GetClientRect(hwnd, ctypes.byref(bounds)):
            self.mouse_up()
            raise InputUnavailable("Could not read the Roblox client area.")
        if not _user32.ClientToScreen(hwnd, ctypes.byref(origin)):
            self.mouse_up()
            raise InputUnavailable("Could not locate the Roblox window.")
        width = bounds.right - bounds.left
        height = bounds.bottom - bounds.top
        if width <= 0 or height <= 0:
            self.mouse_up()
            raise InputUnavailable("The Roblox window has no visible client area.")
        return Rect(origin.x, origin.y, width, height)

    def focus(self, hwnd: int) -> None:
        self.mouse_up()
        if (self._closed or not hwnd or not _user32.IsWindow(hwnd)
                or not _user32.IsWindowVisible(hwnd)
                or self._process_name(hwnd).casefold() not in _PLAYER_EXES):
            raise InputUnavailable("The selected Roblox window is unavailable.")
        if _user32.IsIconic(hwnd):
            _user32.ShowWindow(hwnd, 9)  # SW_RESTORE
        _user32.SetForegroundWindow(hwnd)
        self._guard(hwnd)

    @staticmethod
    def _send_raw(items: Iterable[_INPUT]) -> int:
        entries = tuple(items)
        if not entries:
            return 0
        batch = (_INPUT * len(entries))(*entries)
        ctypes.set_last_error(0)
        return int(_user32.SendInput(len(entries), batch, ctypes.sizeof(_INPUT)))

    def _send_target(self, hwnd: int, items: Iterable[_INPUT]) -> None:
        entries = tuple(items)
        with self._input_lock:
            try:
                self._guard(hwnd)
                if self._send_raw(entries) != len(entries):
                    raise InputUnavailable("Windows did not accept the mouse or keyboard event.")
            except Exception:
                self.mouse_up()
                raise

    def capture(self, hwnd: int) -> np.ndarray:
        """Return a contiguous uint8 BGR screen image cropped to the game client.

        Roblox must remain visible and unobscured. Reading the visible desktop
        supports DirectX game rendering without hooks or process access.
        """
        source_dc = memory_dc = bitmap = previous = None
        try:
            bounds = self.client_rect(hwnd)
            source_dc = _user32.GetDC(None)
            if not source_dc:
                raise InputUnavailable("Windows did not provide an image of the Roblox window.")
            memory_dc = _gdi32.CreateCompatibleDC(source_dc)
            if not memory_dc:
                raise InputUnavailable("Could not initialize screen capture.")
            info = _BITMAPINFO()
            info.bmiHeader.biSize = ctypes.sizeof(_BITMAPINFOHEADER)
            info.bmiHeader.biWidth = bounds.width
            info.bmiHeader.biHeight = -bounds.height  # Top-down BGRA rows.
            info.bmiHeader.biPlanes = 1
            info.bmiHeader.biBitCount = 32
            info.bmiHeader.biCompression = 0  # BI_RGB
            pixels = ctypes.c_void_p()
            bitmap = _gdi32.CreateDIBSection(source_dc, ctypes.byref(info), 0,
                                            ctypes.byref(pixels), None, 0)
            if not bitmap or not pixels.value:
                raise InputUnavailable("Could not create the image buffer.")
            previous = _gdi32.SelectObject(memory_dc, bitmap)
            if not previous or previous == ctypes.c_void_p(-1).value:
                previous = None
                raise InputUnavailable("Could not initialize the image buffer.")
            if not _gdi32.BitBlt(memory_dc, 0, 0, bounds.width, bounds.height,
                                source_dc, bounds.left, bounds.top,
                                0x00CC0020 | 0x40000000):
                raise InputUnavailable("Could not capture the Roblox window.")
            if not _gdi32.GdiFlush():
                raise InputUnavailable("Could not finish screen capture.")
            buffer = (ctypes.c_ubyte * (bounds.width * bounds.height * 4)).from_address(
                pixels.value)
            bgra = np.ctypeslib.as_array(buffer).reshape(bounds.height, bounds.width, 4)
            return bgra[:, :, :3].copy()
        except Exception:
            self.mouse_up()
            raise
        finally:
            if previous and memory_dc:
                _gdi32.SelectObject(memory_dc, previous)
            if bitmap:
                _gdi32.DeleteObject(bitmap)
            if memory_dc:
                _gdi32.DeleteDC(memory_dc)
            if source_dc:
                _user32.ReleaseDC(None, source_dc)

    def mouse_position(self) -> tuple[int, int]:
        position = wintypes.POINT()
        if not _user32.GetCursorPos(ctypes.byref(position)):
            self.mouse_up()
            raise InputUnavailable("Could not read the mouse position.")
        return position.x, position.y

    def cursor_over(self, hwnd: int) -> bool:
        """True only when the visible window under the cursor belongs to the game."""
        if not self.is_available(hwnd):
            return False
        point = wintypes.POINT()
        if not _user32.GetCursorPos(ctypes.byref(point)):
            return False
        try:
            bounds = self.client_rect(hwnd)
        except InputUnavailable:
            return False
        if not (bounds.left <= point.x < bounds.left + bounds.width
                and bounds.top <= point.y < bounds.top + bounds.height):
            return False
        under = _user32.WindowFromPoint(point)
        root = _user32.GetAncestor(under, 2) if under else None
        return bool(root and int(root) == int(hwnd))

    def move(self, hwnd: int, x: float, y: float) -> None:
        try:
            self._guard(hwnd)
            if not math.isfinite(x) or not math.isfinite(y):
                raise ValueError("Mouse coordinates must be finite.")
            bounds = self.client_rect(hwnd)
            screen_x = bounds.left + max(0, min(bounds.width - 1, round(x)))
            screen_y = bounds.top + max(0, min(bounds.height - 1, round(y)))
            virtual_left = _user32.GetSystemMetrics(76)  # SM_XVIRTUALSCREEN
            virtual_top = _user32.GetSystemMetrics(77)  # SM_YVIRTUALSCREEN
            virtual_width = _user32.GetSystemMetrics(78)  # SM_CXVIRTUALSCREEN
            virtual_height = _user32.GetSystemMetrics(79)  # SM_CYVIRTUALSCREEN
            if virtual_width <= 1 or virtual_height <= 1:
                raise InputUnavailable("Could not determine the virtual desktop bounds.")
            absolute_x = round((screen_x - virtual_left) * 65535 / (virtual_width - 1))
            absolute_y = round((screen_y - virtual_top) * 65535 / (virtual_height - 1))
            absolute_x = max(0, min(65535, absolute_x))
            absolute_y = max(0, min(65535, absolute_y))
            self._send_target(hwnd, [_mouse_input(0x0001 | 0x8000 | 0x4000,
                                                  absolute_x, absolute_y)])
        except Exception:
            self.mouse_up()
            raise

    def mouse_down(self, hwnd: int) -> None:
        with self._input_lock:
            self._guard(hwnd)
            if not self.cursor_over(hwnd):
                self.mouse_up()
                raise InputUnavailable("Place the cursor over an unobscured part of Roblox and press F6.")
            if self._left_held and self._held_hwnd == hwnd:
                return
            self.mouse_up()
            if self._left_held:
                raise InputUnavailable("Windows did not release the previous mouse press.")
            self._send_target(hwnd, [_mouse_input(0x0002)])
            self._left_held = True
            self._held_hwnd = hwnd

    def mouse_up(self) -> None:
        """Release only a press owned by this instance, without focus requirements."""
        with self._input_lock:
            if not self._left_held:
                return
            try:
                if self._send_raw([_mouse_input(0x0004)]) == 1:
                    self._left_held = False
                    self._held_hwnd = None
            except Exception:
                pass  # Retain ownership so later cleanup can retry the release.

    def _move_sale(self, hwnd: int, x: float, y: float,
                   cancel: threading.Event) -> bool:
        """Ease toward a sale button in eight guarded, cancellable steps."""
        with self._input_lock:
            if cancel.is_set():
                return False
            self._guard(hwnd)
            if not math.isfinite(x) or not math.isfinite(y):
                raise ValueError("Mouse coordinates must be finite.")
            bounds = self.client_rect(hwnd)
            screen_x, screen_y = self.mouse_position()
            start_x = max(0, min(bounds.width - 1, screen_x - bounds.left))
            start_y = max(0, min(bounds.height - 1, screen_y - bounds.top))
            end_x = max(0, min(bounds.width - 1, x))
            end_y = max(0, min(bounds.height - 1, y))
        for step in range(1, 9):
            with self._input_lock:
                if cancel.is_set():
                    return False
                self._guard(hwnd)
                progress = step / 8
                eased = progress * progress * (3 - 2 * progress)
                self.move(hwnd, start_x + (end_x - start_x) * eased,
                          start_y + (end_y - start_y) * eased)
                if cancel.is_set():
                    return False
            if cancel.wait(0.010):
                return False
        return True

    def click(self, hwnd: int, x: float, y: float,
              cancel: threading.Event | None = None) -> None:
        """Ease to a sale button, settle for 60 ms, then press for up to 80 ms."""
        signal = cancel if cancel is not None else threading.Event()
        try:
            with self._input_lock:
                self.mouse_up()
                if self._left_held:
                    raise InputUnavailable("Windows did not release the previous mouse press.")
                if signal.is_set():
                    return
            if not self._move_sale(hwnd, x, y, signal):
                return
            if signal.wait(0.060):
                return
            with self._input_lock:
                if signal.is_set():
                    return
                self.mouse_down(hwnd)
            signal.wait(0.080)
        finally:
            self.mouse_up()
        if self._left_held:
            raise InputUnavailable("Windows did not release the mouse press.")

    def tap(self, hwnd: int, cancel: threading.Event | None = None) -> None:
        """Click at the current cursor position without moving it.

        An existing press owned by this macro is released first so the tap is a
        fresh button-down event. The bounded 5 ms hold keeps the release close
        to the timed press; cancellation/shutdown can release it sooner.
        """
        try:
            with self._input_lock:
                self.mouse_up()
                if self._left_held:
                    raise InputUnavailable("Windows did not release the previous mouse press.")
                if cancel is not None and cancel.is_set():
                    return
                self.mouse_down(hwnd)
            if cancel is not None:
                cancel.wait(0.005)
            else:
                threading.Event().wait(0.005)
        finally:
            self.mouse_up()

    def key_up(self) -> bool:
        """Release our key after focus loss; False retains ownership for retry."""
        with self._input_lock:
            if self._held_key is None:
                return True
            try:
                packet = self._held_key_up_packet
                if packet is None:
                    packet = _keyboard_input(self._held_key, True)
                if self._send_raw([packet]) == 1:
                    self._held_key = None
                    self._held_key_up_packet = None
                    return True
            except Exception:
                pass  # Preserve ownership so shutdown or the next attempt can retry.
            return False

    def key(self, hwnd: int, vk: int, cancel: threading.Event | None = None) -> bool:
        """Press a foreground-only key, then release the exact accepted packet.

        Inventory uses physical grave scan 0x29 for up to 100 ms. Other virtual
        keys keep their 40 ms press. Cancellation releases either one early.
        Return False if cancellation prevents DOWN; return True once Windows
        accepts DOWN and its matching UP, even if cancellation shortens the
        hold. Rejected input or release raises InputUnavailable.
        """
        try:
            with self._input_lock:
                vk = int(vk)
                if not 1 <= vk <= 0xFF:
                    raise ValueError("Invalid Windows virtual-key code.")
                self.mouse_up()
                if self._left_held:
                    raise InputUnavailable("Windows did not release the previous mouse press.")
                if not self.key_up():
                    raise InputUnavailable("Windows did not release the previous key press.")
                if cancel is not None and cancel.is_set():
                    return False
                self._guard(hwnd)
                scan_code = 0x29 if vk == VK_OEM_3 else None
                down = _keyboard_input(vk, scan_code=scan_code)
                up = _keyboard_input(vk, True, scan_code=scan_code)
                if self._send_raw([down]) != 1:
                    raise InputUnavailable("Windows did not accept the keyboard event.")
                self._held_key = vk
                self._held_key_up_packet = up
                duration = 0.100 if vk == VK_OEM_3 else 0.040
            if cancel is not None:
                cancel.wait(duration)
            else:
                threading.Event().wait(duration)
        finally:
            self.mouse_up()
            released = self.key_up()
        if not released:
            raise InputUnavailable("Windows did not release the key press.")
        return True

    def key_pressed(self, vk: int) -> bool:
        return bool(_user32.GetAsyncKeyState(vk) & 0x8000)

    def close(self) -> None:
        with self._input_lock:
            self._closed = True
            self.mouse_up()
            self.key_up()

    def __enter__(self) -> "Desktop":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
