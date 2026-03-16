from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Dict

MODE_MANUAL = "manual"
MODE_BALANCED = "balanced"
MODE_ATTACK = "attack"
MODE_DEFENSE = "defense"

VK_M = 0x4D
VK_N = 0x4E
VK_B = 0x42
VK_V = 0x56
VK_P = 0x50
VK_O = 0x4F
VK_L = 0x4C
VK_H = 0x48

# Drive keys (manual mode)
VK_W = 0x57
VK_A = 0x41
VK_S = 0x53
VK_D = 0x44
VK_UP = 0x26
VK_DOWN = 0x28
VK_LEFT = 0x25
VK_RIGHT = 0x27
VK_SPACE = 0x20
VK_LSHIFT = 0xA0

try:
    _user32 = ctypes.windll.user32
    _user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
    _user32.GetAsyncKeyState.restype = ctypes.c_short
except Exception:
    _user32 = None


@dataclass
class ModeState:
    mode: str = MODE_BALANCED
    temporary_reset: bool = False


class ModeManager:
    def __init__(self, initial_mode: str = MODE_BALANCED):
        self.state = ModeState(mode=initial_mode)
        self._prev: Dict[int, bool] = {
            VK_M: False, VK_N: False, VK_B: False, VK_V: False,
            VK_P: False, VK_O: False, VK_L: False, VK_H: False,
        }
        self._open_setup_pending = False
        self._overlay_toggle_pending = False

    def _is_pressed(self, vk: int) -> bool:
        if _user32 is None:
            return False
        return bool(_user32.GetAsyncKeyState(vk) & 0x8000)

    def update_from_keyboard(self) -> ModeState:
        key_to_mode = {
            VK_M: MODE_MANUAL,
            VK_N: MODE_BALANCED,
            VK_B: MODE_ATTACK,
            VK_V: MODE_DEFENSE,
        }

        for vk, target_mode in key_to_mode.items():
            pressed = self._is_pressed(vk)
            if pressed and not self._prev[vk]:
                if target_mode != self.state.mode:
                    self._announce_mode(target_mode)
                self.state.mode = target_mode
            self._prev[vk] = pressed

        pressed_p = self._is_pressed(VK_P)
        if pressed_p and not self._prev[VK_P]:
            self.state.temporary_reset = True
        self._prev[VK_P] = pressed_p

        pressed_o = self._is_pressed(VK_O)
        if pressed_o and not self._prev[VK_O]:
            self.state.temporary_reset = False
        self._prev[VK_O] = pressed_o

        pressed_l = self._is_pressed(VK_L)
        if pressed_l and not self._prev[VK_L]:
            self._open_setup_pending = True
        self._prev[VK_L] = pressed_l

        pressed_h = self._is_pressed(VK_H)
        if pressed_h and not self._prev[VK_H]:
            self._overlay_toggle_pending = True
        self._prev[VK_H] = pressed_h

        return self.state

    def _announce_mode(self, mode: str) -> None:
        """Print a debug message when the active mode changes."""
        _LABELS = {
            MODE_ATTACK:   "[MODE] Attack Mode Enabled",
            MODE_DEFENSE:  "[MODE] Defense Mode Enabled",
            MODE_BALANCED: "[MODE] Balanced Mode Enabled",
            MODE_MANUAL:   "[MODE] Manual Mode Enabled",
        }
        print(_LABELS.get(mode, f"[MODE] {mode.title()} Mode Enabled"))

    def set_mode(self, mode: str):
        if mode != self.state.mode:
            self._announce_mode(mode)
        self.state.mode = mode

    def set_temporary_reset(self, temporary_reset: bool):
        self.state.temporary_reset = bool(temporary_reset)

    def consume_open_setup_request(self) -> bool:
        pending = self._open_setup_pending
        self._open_setup_pending = False
        return pending

    def consume_overlay_toggle_request(self) -> bool:
        pending = self._overlay_toggle_pending
        self._overlay_toggle_pending = False
        return pending

    def read_human_controls(self) -> Dict[str, float]:
        """Read WASD / arrows / Space / LShift and return car controls."""
        throttle = 0.0
        steer = 0.0
        if self._is_pressed(VK_W) or self._is_pressed(VK_UP):
            throttle += 1.0
        if self._is_pressed(VK_S) or self._is_pressed(VK_DOWN):
            throttle -= 1.0
        if self._is_pressed(VK_A) or self._is_pressed(VK_LEFT):
            steer -= 1.0
        if self._is_pressed(VK_D) or self._is_pressed(VK_RIGHT):
            steer += 1.0
        jump = 1.0 if self._is_pressed(VK_SPACE) else 0.0
        boost = 1.0 if self._is_pressed(VK_LSHIFT) else 0.0
        return {
            "throttle": throttle,
            "steer": steer,
            "jump": jump,
            "boost": boost,
        }
