"""
runtime/ipc.py  —  Lightweight JSON-file IPC between the GUI process and the bot subprocess.

Files:
  _bot_status.json   — written by bot, read by GUI  (live match data)
  _gui_commands.json — written by GUI, read by bot  (mode/algo overrides)

All writes are atomic: write to .tmp then os.replace() so readers never see
a partial file.  All reads return {} on any error so caller code is simple.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

_RUNTIME_DIR = os.path.dirname(os.path.abspath(__file__))

STATUS_FILE   = os.path.join(_RUNTIME_DIR, "_bot_status.json")
COMMANDS_FILE = os.path.join(_RUNTIME_DIR, "_gui_commands.json")


# ── Writers ──────────────────────────────────────────────────────────────────

def write_bot_status(data: Dict[str, Any]) -> None:
    """Bot calls this to publish live match state to the GUI."""
    _atomic_write(STATUS_FILE, data)


def write_gui_commands(data: Dict[str, Any]) -> None:
    """GUI calls this to send control commands to the bot."""
    _atomic_write(COMMANDS_FILE, data)


# ── Readers ───────────────────────────────────────────────────────────────────

def read_bot_status() -> Dict[str, Any]:
    """GUI calls this to read the latest match state from the bot."""
    return _safe_read(STATUS_FILE)


def read_gui_commands() -> Dict[str, Any]:
    """Bot calls this to pick up commands sent from the GUI."""
    return _safe_read(COMMANDS_FILE)


def clear_gui_commands() -> None:
    """Bot calls this after consuming commands so they aren't re-processed."""
    _atomic_write(COMMANDS_FILE, {})


# ── Helpers ───────────────────────────────────────────────────────────────────

def _atomic_write(path: str, data: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
    except Exception:
        pass  # never crash the caller on IPC failures


def _safe_read(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}
