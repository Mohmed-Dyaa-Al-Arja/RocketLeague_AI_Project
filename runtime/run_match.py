"""
Rocket League Search-Only AI runtime.

Algorithms used by the bot:
- A*
- BFS
- UCS
- DFS
- Greedy Best First Search
- Decision Tree
- Beam Search
- IDA*

When the setup menu is left on Auto, the bot runs the search set in parallel
and keeps the strongest candidate while the adaptive RL stack still handles
role selection and control blending.

Keyboard controls:
M = manual user control mode
N = balanced mode
B = attack mode
V = defense mode
P = temporary session model (new from scratch for this match only)
L = stop the current match and reopen the setup menu
"""

from __future__ import annotations

import argparse
import asyncio
import atexit
import json
import logging
import math
import os
import sys
import threading
import time
from typing import Dict, Tuple

from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.matchconfig import match_config as rlbot_match_config_module
from rlbot.matchconfig.match_config import MatchConfig, PlayerConfig
from rlbot.setup_manager import SetupManager
from rlbot.utils.structures.game_data_struct import GameTickPacket

RUNTIME_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(RUNTIME_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

_ALGO_STATS_FILE = os.path.join(PROJECT_ROOT, "model", "algorithm_stats.json")

# ── File-based logger so subprocess output is always visible ─────────
_BOT_LOG_FILE = os.path.join(RUNTIME_DIR, "_bot_subprocess.log")
_blog = logging.getLogger("bot_subprocess")
_blog.setLevel(logging.DEBUG)
_blog.propagate = False
if not _blog.handlers:
    _fh = logging.FileHandler(_BOT_LOG_FILE, mode="a", encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    _blog.addHandler(_fh)
_blog.info("=== run_match.py module loaded (pid=%d, exe=%s) ===", os.getpid(), sys.executable)

try:
    from game_logic.decision_engine import (DecisionEngine, KnowledgeStore,
        classify_situation, is_kickoff_state,
        GameState, detect_game_state,
        GM_SOCCER, GM_HOOPS, GM_DROPSHOT, GM_SNOW, GM_RUMBLE, GM_HEATSEEKER, GM_GRIDIRON,
        GM_KNOCKOUT, GM_SPIKE_RUSH, GM_BOOMER, GM_BEACH_BALL, GM_SPEED_DEMON,
        GM_SUPER_CUBE, GM_SPRING_LOADED, GM_DROPSHOT_RUMBLE, GM_WINTER_BREAKAWAY)
    from core.all_algorithms import predict_trajectory, find_intercept
    from game_logic.mode_manager import MODE_MANUAL, ModeManager
    from game_logic.strategy_layer import StrategyLayer
    from core.ball_prediction import predict_ball_path, find_best_intercept, detect_open_goal
    from core.opponent_analyzer import OpponentAnalyzer
    from core.aerial_controller import AerialController
    from core.boost_manager import BoostManager
    from game_logic.strategy_manager import StrategyManager
    from game_logic.team_rotation import TeamRotationManager
    from core.shot_selector import ShotSelector
    from core.fake_shot_ai import FakeShotAI
    from core.kickoff_ai import KickoffAI
    from core.dribbling_ai import DribblingAI
    from core.wall_play_ai import WallPlayAI
    from game_logic.match_strategy_analyzer import MatchStrategyAnalyzer
    from game_logic.team_communication import get_default_team_state
    from game_logic.kickoff_detector import KickoffDetector
    from game_logic.kickoff_roles import KickoffRoleManager
    from game_logic.passing_ai import PassingAI
    from game_logic.positioning_ai import PositioningAI
    from game_logic.game_awareness import GameAwareness
    from core.kickoff_prediction import find_kickoff_intercept
    from core.opponent_prediction import OpponentPredictor
    from core.shot_accuracy import ShotAccuracyEvaluator
    from learning.match_learning import MatchLearning
    from strategy.strategy_book import StrategyBook
    from strategy.strategy_selector import StrategySelector
    from game_logic.ability_discovery import AbilityDiscovery, RUMBLE_POWERUP_NAMES
    from game_logic.ability_strategy import AbilityStrategy
    from game_logic.team_ability_awareness import TeamAbilityAwareness
    from game_logic.team_ability_strategy import TeamAbilityStrategy
    from game_logic.team_controller import (TeamController,
        TEAM_MODE_DEFAULT_RL, TEAM_MODE_CUSTOM_AI, TEAM_MODE_SPECIFIC)
    from game_logic.team_power_strategy import TeamPowerStrategy, StrategicDecision
    from game_logic.match_state_detector import MatchStateDetector, MatchState
    from runtime.status_manager import StatusManager
    import runtime.ipc as ipc
    from gui.control_panel import ControlPanel
    from visualization.overlay_renderer import OverlayRenderer
    _blog.info("All project imports successful")
except Exception as _import_exc:
    _blog.error("IMPORT FAILED: %s", _import_exc, exc_info=True)
    raise

BOT_CFG = os.path.join(RUNTIME_DIR, "rl_ai_bot.cfg")

# Global so atexit can access it for saving knowledge
_active_bot_instance = None
_starting_mode = "balanced"
_starting_temporary_reset = False
_pending_match_settings: dict[str, str] = {}
_MATCH_SETTINGS_FILE = os.path.join(RUNTIME_DIR, "_match_settings.json")
_reopen_settings_requested = threading.Event()
_OVERLAY_POSITION_MAP = {
    "Top Right": "top_right",
    "Bottom Right": "bottom_right",
    "Hidden by Default": "hidden",
}
_OVERLAY_SIZE_MAP = {
    "Auto": "auto",
    "Small": "small",
    "Medium": "medium",
}
_CUSTOM_RLBOT_UPK_MAPS = {
    "IceDome": "IceDome",
}

# Pre-initialized engine: created BEFORE connecting to the game so algorithms
# are ready the instant the first tick arrives.
_pre_initialized_engine: DecisionEngine | None = None

# ── Headless / no-GUI mode flag ───────────────────────────────────────────────
# Set to True when launched by the new standalone GUI as a subprocess.
# In this mode we skip ControlPanel (Tkinter) entirely and use IPC files.
_NO_GUI: bool = "--no-gui" in sys.argv

# ── IPC status writer ─────────────────────────────────────────────────────────
_IPC_STATUS_INTERVAL = 10           # write status every N ticks (~0.17 s at 60 Hz)
_ipc_tick_counter: int = 0
_ipc_episode_start_reward: float = 0.0


def _update_algo_stats(algo: str, event: str) -> None:
    """Increment a stat counter in model/algorithm_stats.json without crashing."""
    try:
        with open(_ALGO_STATS_FILE, "r", encoding="utf-8") as _f:
            _data = json.load(_f)
        if algo in _data:
            _data[algo][event] = int(_data[algo].get(event, 0)) + 1
            _data[algo]["attempts"] = int(_data[algo].get("attempts", 0)) + 1
        with open(_ALGO_STATS_FILE, "w", encoding="utf-8") as _f:
            json.dump(_data, _f, indent=2)
    except Exception:
        pass


def _write_ipc_status(bot) -> None:
    """Publish live bot state to _bot_status.json for the GUI to display."""
    try:
        stats = bot.decision_engine.adaptive.reward_calc.get_episode_stats()
        strategy_info = bot._strategy_layer.state_summary() if hasattr(bot, "_strategy_layer") else {}
        ipc.write_bot_status({
            "alive":          True,
            "current_mode":   getattr(bot, "_current_mode_str", "balanced"),
            "current_algo":   getattr(bot, "last_algorithm", "A*"),
            "current_game_state": getattr(bot, "_current_game_state", ""),
            "path_cost":      round(getattr(bot, "last_path_cost", 0.0), 1),
            "tick":           getattr(bot, "_tick", 0),
            "my_score":       getattr(bot, "_prev_my_score", 0),
            "opp_score":      getattr(bot, "_prev_opp_score", 0),
            "goal_limit":     getattr(bot, "_goal_limit", 7),
            "algo_usage":     dict(getattr(bot, "_algo_use_counts", {})),
            "threat_level":   round(bot.decision_engine.adaptive.get_threat_level(), 3),
            "episode_reward": round(stats.get("total", 0.0), 2),
            "reward_breakdown": {k: round(v, 2) for k, v in stats.items() if k != "total"},
            "strategy":       strategy_info.get("strategy", ""),
            "search_advice":  strategy_info.get("search_advice", ""),
            "rl_advice":      strategy_info.get("rl_advice", ""),
            "rl_role":        getattr(bot.decision_engine.adaptive, "_current_role", ""),
            "situation":      getattr(bot, "_last_situation", ""),
            "ball_pos":       list(getattr(bot, "_last_ball_pos_ipc", (0.0, 0.0, 0.0))),
            "ball_vel":       list(getattr(bot, "_last_ball_vel_ipc", (0.0, 0.0, 0.0))),
            "car_pos":        list(getattr(bot, "_last_car_pos_ipc",  (0.0, 0.0, 0.0))),
            "opp_style":      getattr(bot._opp_analyzer, "get_style", lambda: "")() if hasattr(bot, "_opp_analyzer") else "",
            "arena":          getattr(bot, "_current_arena", "DFHStadium"),
            "active_strategies": getattr(bot, "_active_strategies", {}),
            "strategy_switch_reason": (
                bot._strategy_selector.last_switch_reason
                if hasattr(bot, "_strategy_selector") else ""
            ),
            # Ability discovery fields
            "ability_name":       getattr(bot, "_current_ability", ""),
            "ability_experimenting": getattr(bot, "_ability_experiment_mode", False),
            "ability_type": (
                bot._ability_discovery.get_ability_info(bot._current_ability).get("type", "")
                if getattr(bot, "_current_ability", "") and hasattr(bot, "_ability_discovery")
                else ""
            ),
            "ability_use_case": (
                bot._ability_discovery.get_ability_info(bot._current_ability).get("best_use_case", "")
                if getattr(bot, "_current_ability", "") and hasattr(bot, "_ability_discovery")
                else ""
            ),
            "ability_success_rate": (
                f"{bot._ability_discovery.get_ability_info(bot._current_ability).get('success_rate', 0.5):.0%}"
                if getattr(bot, "_current_ability", "") and hasattr(bot, "_ability_discovery")
                else ""
            ),
            # Team fields
            "bot_role":              getattr(bot, "_bot_role", "support"),
            "team_mode":             (bot._team_controller.team_mode
                                       if hasattr(bot, "_team_controller") else ""),
            "team_rotation_reason":  (bot._team_controller.last_rotation_reason
                                       if hasattr(bot, "_team_controller") else ""),
            "team_directive":        getattr(bot, "_team_directive_reason", ""),
            "power_action":          getattr(bot, "_power_decision_action", "standard"),
            "power_combo":           getattr(bot, "_power_decision_combo",  "standard"),
            "power_threat_score": (
                round(bot._team_power_strategy.intelligence.threat_score, 3)
                if hasattr(bot, "_team_power_strategy") else 0.0
            ),
            "power_opportunity_score": (
                round(bot._team_power_strategy.intelligence.opportunity_score, 3)
                if hasattr(bot, "_team_power_strategy") else 0.0
            ),
            **( bot._team_ability_awareness.summary()
                if hasattr(bot, "_team_ability_awareness") else
                {"teammate_abilities": {}, "opponent_abilities": {}} ),
        })
        # Write model status alongside bot status
        if hasattr(bot, "_status_mgr"):
            bot._status_mgr.write_model_status(
                current_model=getattr(bot, "_current_model_type", "persistent"),
                training_state="training" if getattr(bot, "_learning_enabled", True) else "idle",
                learning_rate=float(getattr(
                    bot.decision_engine.adaptive, "_learning_rate", 0.001)
                    if hasattr(bot, "decision_engine") else 0.001),
                last_reward=float(getattr(bot.decision_engine.adaptive.reward_calc,
                    "episode_reward", 0.0)
                    if hasattr(bot, "decision_engine") else 0.0),
                active_models=list(getattr(bot.decision_engine.adaptive,
                    "active_rl_models", [])
                    if hasattr(bot, "decision_engine") else []),
                active_search=list(getattr(bot.decision_engine,
                    "active_search_algorithms", ["A*"])
                    if hasattr(bot, "decision_engine") else ["A*"]),
                team_strategy=getattr(bot, "_team_strategy", "balanced"),
            )
    except Exception:
        pass


def _poll_ipc_commands(bot) -> None:
    """Check _bot_commands.json for mode or algorithm overrides from the GUI."""
    try:
        cmds = ipc.read_gui_commands()
        if not cmds:
            return
        mode_override = cmds.get("mode")
        if mode_override and hasattr(bot, "mode_manager"):
            bot.mode_manager.set_mode(str(mode_override))
        model_sel = cmds.get("active_rl_models")
        if model_sel and isinstance(model_sel, list):
            bot.decision_engine.adaptive.set_active_models(set(model_sel))
        search_sel = cmds.get("active_search_algorithms")
        if search_sel and isinstance(search_sel, list):
            bot.decision_engine.active_search_algorithms = search_sel
        # ── Match control commands ─────────────────────────────────────────
        command = cmds.get("command")
        if command in ("restart_match", "start_match", "next_match"):
            try:
                bot._prev_my_score = 0
                bot._prev_opp_score = 0
                bot.decision_engine.adaptive.reward_calc.reset()
                bot._possession_us = 0
                bot._possession_them = 0
                if hasattr(bot, "_match_analyzer"):
                    bot._match_analyzer.__init__()
            except Exception:
                pass
        elif command == "stop_bot":
            try:
                import signal as _sig
                _sig.raise_signal(_sig.SIGTERM)
            except Exception:
                raise SystemExit(0)
        ipc.clear_gui_commands()
    except Exception:
        pass


def _patch_rlbot_custom_map_support() -> None:
    if getattr(rlbot_match_config_module.MatchConfig, "_search_ai_custom_map_patch", False):
        return

    original_create_flatbuffer = rlbot_match_config_module.MatchConfig.create_flatbuffer

    def create_flatbuffer_with_custom_maps(self):
        if self.game_map not in _CUSTOM_RLBOT_UPK_MAPS:
            return original_create_flatbuffer(self)

        builder = rlbot_match_config_module.Builder(1000)
        name_dict = {}
        player_config_offsets = [pc.write_to_flatbuffer(builder, name_dict) for pc in self.player_configs]
        rlbot_match_config_module.MatchSettingsFlat.MatchSettingsStartPlayerConfigurationsVector(
            builder, len(player_config_offsets)
        )
        for i in reversed(range(0, len(player_config_offsets))):
            builder.PrependUOffsetTRelative(player_config_offsets[i])
        player_list_offset = builder.EndVector(len(player_config_offsets))
        mutator_settings_offset = self.mutators.write_to_flatbuffer(builder)

        upk_offset = builder.CreateString(_CUSTOM_RLBOT_UPK_MAPS[self.game_map])

        rlbot_match_config_module.MatchSettingsFlat.MatchSettingsStart(builder)
        rlbot_match_config_module.MatchSettingsFlat.MatchSettingsAddPlayerConfigurations(builder, player_list_offset)
        rlbot_match_config_module.MatchSettingsFlat.MatchSettingsAddGameMode(
            builder,
            rlbot_match_config_module.index_or_zero(
                rlbot_match_config_module.game_mode_types,
                self.game_mode,
            ),
        )
        rlbot_match_config_module.MatchSettingsFlat.MatchSettingsAddGameMap(builder, -1)
        rlbot_match_config_module.MatchSettingsFlat.MatchSettingsAddGameMapUpk(builder, upk_offset)
        rlbot_match_config_module.MatchSettingsFlat.MatchSettingsAddSkipReplays(builder, self.skip_replays)
        rlbot_match_config_module.MatchSettingsFlat.MatchSettingsAddInstantStart(builder, self.instant_start)
        rlbot_match_config_module.MatchSettingsFlat.MatchSettingsAddMutatorSettings(builder, mutator_settings_offset)
        rlbot_match_config_module.MatchSettingsFlat.MatchSettingsAddExistingMatchBehavior(
            builder,
            rlbot_match_config_module.index_or_zero(
                rlbot_match_config_module.existing_match_behavior_types,
                self.existing_match_behavior,
            ),
        )
        rlbot_match_config_module.MatchSettingsFlat.MatchSettingsAddEnableLockstep(builder, self.enable_lockstep)
        builder.Finish(rlbot_match_config_module.MatchSettingsFlat.MatchSettingsEnd(builder))
        return builder

    rlbot_match_config_module.MatchConfig.create_flatbuffer = create_flatbuffer_with_custom_maps
    rlbot_match_config_module.MatchConfig._search_ai_custom_map_patch = True


_patch_rlbot_custom_map_support()


def _save_match_settings_for_subprocess(settings: dict, starting_mode: str, temporary_reset: bool) -> None:
    """Persist match settings so the bot subprocess can read them."""
    payload = {
        "starting_mode": starting_mode,
        "temporary_reset": temporary_reset,
        "settings": settings,
    }
    with open(_MATCH_SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def _load_match_settings_from_file() -> dict | None:
    """Load match settings written by the main process."""
    if not os.path.exists(_MATCH_SETTINGS_FILE):
        return None
    try:
        with open(_MATCH_SETTINGS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _request_settings_reopen():
    _reopen_settings_requested.set()


def _load_launcher_settings():
    """Read launcher_settings.cfg from the runtime directory."""
    import configparser
    cfg = configparser.ConfigParser()
    settings_path = os.path.join(RUNTIME_DIR, 'launcher_settings.cfg')
    if os.path.exists(settings_path):
        cfg.read(settings_path, encoding='utf-8')
    return cfg


def _save_launcher_settings(cfg) -> None:
    settings_path = os.path.join(RUNTIME_DIR, 'launcher_settings.cfg')
    with open(settings_path, 'w', encoding='utf-8') as handle:
        cfg.write(handle)


def _load_ui_preferences() -> dict[str, str]:
    cfg = _load_launcher_settings()
    launcher_platform = cfg.get('launcher', 'platform', fallback='epic').strip().lower()
    launcher_mode = cfg.get('launcher', 'launch_mode', fallback='direct').strip().lower()
    return {
        "overlay_menu_position": cfg.get('ui', 'overlay_menu_position', fallback='Top Right'),
        "overlay_menu_size": cfg.get('ui', 'overlay_menu_size', fallback='Auto'),
        "launcher_platform": "Steam" if launcher_platform == 'steam' else "Epic",
        "launcher_launch_mode": {
            'manual': "Manual Only (Do Not Launch Game)",
            'auto': "Auto via RLBot",
            'offline': "Direct EXE (No Launcher)",
            'direct': "Direct EXE (No Launcher)",
        }.get(launcher_mode, "Direct EXE (No Launcher)"),
    }


def _save_ui_preferences(settings: dict) -> None:
    cfg = _load_launcher_settings()
    if not cfg.has_section('ui'):
        cfg.add_section('ui')
    if not cfg.has_section('launcher'):
        cfg.add_section('launcher')
    cfg.set('ui', 'overlay_menu_position', str(settings.get('overlay_menu_position', 'Top Right')))
    cfg.set('ui', 'overlay_menu_size', str(settings.get('overlay_menu_size', 'Auto')))
    platform = str(settings.get('launcher_platform', 'Epic')).strip().lower()
    launch_mode_label = str(settings.get('launcher_launch_mode', 'Direct EXE (No Launcher)')).strip()
    launch_mode = {
        "Manual Only (Do Not Launch Game)": "manual",
        "Auto via RLBot": "auto",
        "Direct EXE (No Launcher)": "direct",
    }.get(launch_mode_label, 'direct')
    cfg.set('launcher', 'platform', 'steam' if platform == 'steam' else 'epic')
    cfg.set('launcher', 'launch_mode', launch_mode)
    cfg.set('launcher', 'use_login_tricks', 'true' if launch_mode == 'auto' else 'false')
    cfg.set('launcher', 'disable_epic_login_tricks', 'false' if launch_mode == 'auto' else 'true')
    _save_launcher_settings(cfg)


class _IPCControlPanel:
    """Lightweight no-Tkinter control-panel stub used when running as a subprocess
    (``--no-gui`` flag).  All state is exchanged via JSON IPC files instead.
    Every method that normally touches Tkinter becomes a no-op or returns a safe
    default so the rest of the bot code doesn't need any guard clauses."""

    def __init__(self) -> None:
        self._mode: str = "balanced"

    # ── consume methods (polled by the bot tick loop) ─────────────────────
    def consume_mode_override(self):
        try:
            cmds = ipc.read_gui_commands()
            return cmds.get("mode") if cmds else None
        except Exception:
            return None

    def consume_model_override(self):
        try:
            cmds = ipc.read_gui_commands()
            return cmds.get("model_override") if cmds else None
        except Exception:
            return None

    def consume_model_selection(self):
        try:
            cmds = ipc.read_gui_commands()
            return cmds.get("active_rl_models") if cmds else None
        except Exception:
            return None

    def consume_open_setup_request(self) -> bool:
        return False

    # ── setters (informational; store locally so IPC status can read them) ─
    def set_mode(self, mode: str) -> None:
        self._mode = mode

    def set_algorithm(self, algo: str) -> None:
        pass

    def set_path_cost(self, cost: float) -> None:
        pass

    def set_runtime_status(self, msg: str) -> None:
        pass

    def set_surface_info(self, *args, **kwargs) -> None:
        pass

    def set_game_mode(self, mode: str) -> None:
        pass

    def set_planner_mode(self, mode: str) -> None:
        pass

    def set_setup_hint(self, hint: str) -> None:
        pass

    def set_temporary_reset(self, val: bool) -> None:
        pass

    def set_threat_level(self, level, *args, **kwargs) -> None:
        pass

    def record_timing(self, algo: str, elapsed_ms: float) -> None:
        pass

    def usage_percentage(self) -> dict:
        return {}

    def usage_counts(self) -> dict:
        return {}

    # ── lifecycle ─────────────────────────────────────────────────────────
    def wait_for_startup(self, timeout: float = 3.0) -> bool:
        return True

    def startup_error(self):
        return None

    def hide(self) -> None:
        pass

    def show(self) -> None:
        pass

    def close(self) -> None:
        pass


class SearchOnlyBot(BaseAgent):
    def initialize_agent(self):
        global _active_bot_instance, _pre_initialized_engine, _starting_temporary_reset, _pending_match_settings
        _active_bot_instance = self
        _blog.info("initialize_agent START (pid=%d, index=%s, spawn_id=%s)",
                    os.getpid(), getattr(self, 'index', '?'), getattr(self, 'spawn_id', '?'))

        # ── Load settings written by the main process (cross-process) ──
        saved = _load_match_settings_from_file()
        _blog.info("Sidecar loaded: %s", saved is not None)
        if saved is not None:
            _starting_mode_eff = saved.get("starting_mode", _starting_mode)
            _starting_temporary_reset = saved.get("temporary_reset", _starting_temporary_reset)
            _pending_match_settings = saved.get("settings", _pending_match_settings)
        else:
            _starting_mode_eff = _starting_mode
        _blog.info("Effective mode=%s, temp_reset=%s, settings_keys=%s",
                    _starting_mode_eff, _starting_temporary_reset, list(_pending_match_settings.keys()))

        self.mode_manager = ModeManager(initial_mode=_starting_mode_eff)
        self.mode_manager.set_temporary_reset(_starting_temporary_reset)

        if _NO_GUI:
            # Running as subprocess of the new GUI — use a lightweight stub ControlPanel
            self.control_panel = _IPCControlPanel()
        else:
            self.control_panel = ControlPanel(initial_mode=_starting_mode_eff)
            self.mode_manager.set_temporary_reset(_starting_temporary_reset)
            self.control_panel.set_temporary_reset(_starting_temporary_reset)
            self.control_panel.set_setup_hint("L or New Match button = stop current game and reopen setup")
            gui_ok = self.control_panel.wait_for_startup(timeout=3.0)
            gui_err = self.control_panel.startup_error()
            _blog.info("ControlPanel GUI started=%s, error=%s", gui_ok, gui_err)
            print(f"[BOT] ControlPanel GUI started={gui_ok}, error={gui_err}")

        # Reuse the engine that was pre-initialized before connecting.
        if _pre_initialized_engine is not None:
            self.decision_engine = _pre_initialized_engine
            _pre_initialized_engine = None          # consumed
        else:
            self.decision_engine = DecisionEngine(PROJECT_ROOT)
            # Apply settings that were set in the main process.
            if _pending_match_settings:
                _apply_settings_to_engine(self.decision_engine, _pending_match_settings)
                print(f"[BOT] Applied match settings from sidecar: mode={_pending_match_settings.get('game_mode','?')}")
        self.control_panel.set_planner_mode("ensemble")
        self._strategy_layer = StrategyLayer()
        self._current_game_state = ""
        self._current_mode_str = "balanced"

        # ── New AI subsystems ─────────────────────────────────────────────
        self._aerial_ctrl    = AerialController()
        self._boost_mgr      = BoostManager()
        self._strategy_mgr   = StrategyManager()
        self._team_rotation  = TeamRotationManager()
        self._opp_analyzer   = OpponentAnalyzer()
        self._shot_selector   = ShotSelector()
        self._fake_shot_ai    = FakeShotAI()
        self._kickoff_ai      = KickoffAI()
        self._dribbling_ai    = DribblingAI()
        self._wall_play_ai    = WallPlayAI()
        self._match_analyzer  = MatchStrategyAnalyzer()
        self._team_state      = get_default_team_state()
        # New AI subsystems
        self._kickoff_detector = KickoffDetector()
        self._kickoff_roles    = KickoffRoleManager()
        self._passing_ai       = PassingAI()
        self._positioning_ai   = PositioningAI()
        self._game_awareness   = GameAwareness()
        self._opp_predictor    = OpponentPredictor()
        self._shot_accuracy    = ShotAccuracyEvaluator()
        self._match_learning   = MatchLearning()
        # Strategy Book subsystem
        self._strategy_book    = StrategyBook()
        _arena_ns = _pending_match_settings.get("arena", "DFHStadium")
        self._strategy_selector = StrategySelector(self._strategy_book)
        self._active_strategies = self._strategy_selector.select_pre_match(_arena_ns)
        self._current_arena     = _arena_ns
        self._strategy_adapt_tick = 0
        # Ability Discovery subsystem
        self._ability_discovery = AbilityDiscovery()
        self._ability_strategy  = AbilityStrategy(self._ability_discovery)
        self._current_ability: str = ""
        self._ability_experiment_mode: bool = False
        self._ability_trial_pending: bool = False
        self._pre_ball_vel_snapshot: float = 0.0
        self._pre_opp_dist_snapshot: float = 9999.0
        # Team Ability Awareness + coordination
        self._team_ability_awareness = TeamAbilityAwareness()
        self._team_ability_strategy  = TeamAbilityStrategy(
            self._team_ability_awareness, self._ability_discovery)
        # Team Controller (role assignment & rotation)
        self._team_controller = TeamController()
        _team_mode_cfg = _pending_match_settings.get("team_mode", TEAM_MODE_DEFAULT_RL)
        self._team_controller.set_team_mode(_team_mode_cfg)
        _team_models_cfg = _pending_match_settings.get("team_models", {})
        for _role, _model in _team_models_cfg.items():
            self._team_controller.set_team_model(_role, _model)
        self._bot_role: str = "support"
        self._team_directive_reason: str = ""
        # Team Power Strategy (comprehensive synthesis layer)
        self._team_power_strategy = TeamPowerStrategy(
            self._team_controller,
            self._team_ability_awareness,
            self._ability_discovery,
        )
        self._power_decision_action: str = "standard"
        self._power_decision_combo: str  = "standard"
        self._status_mgr = StatusManager(PROJECT_ROOT)
        self._match_detector = MatchStateDetector()
        self._match_detector.register_callback(self._on_match_state_change)
        # Ball prediction cache (refreshed every 10 ticks, requirement #9)
        self._ball_traj_cache: list = []
        self._ball_traj_tick: int   = -999
        self._intercept_cache       = None
        self._shoot_aim_cache       = None
        # Last known positions for IPC status
        self._last_ball_pos_ipc = (0.0, 0.0, 0.0)
        self._last_ball_vel_ipc = (0.0, 0.0, 0.0)
        self._last_car_pos_ipc  = (0.0, 0.0, 0.0)
        # ── End new subsystems ────────────────────────────────────────────

        # Persistent vs temporary (session-only) learning model.
        self._persistent_knowledge = self.decision_engine.knowledge
        self._session_knowledge = None
        self._prev_temporary_reset = False

        # Planning cache (reduces CPU load on normal PCs).
        self._tick = 0
        self._cached_decision = None
        self._last_plan_tick = -999
        self._last_plan_ball_xy = (0.0, 0.0)
        self._last_plan_mode = ""
        self._last_plan_reset = False
        self._last_plan_situation = ""

        self._last_checkpoint_time = 0.0

        self.overlay = OverlayRenderer(self.renderer)
        overlay_pref = str(_pending_match_settings.get("overlay_menu_position", "Top Right"))
        self._overlay_menu_position = _OVERLAY_POSITION_MAP.get(overlay_pref, "top_right")
        overlay_size_pref = str(_pending_match_settings.get("overlay_menu_size", "Auto"))
        self._overlay_menu_size = _OVERLAY_SIZE_MAP.get(overlay_size_pref, "auto")
        self._overlay_menu_visible = self._overlay_menu_position != "hidden"
        if self._overlay_menu_position == "hidden":
            self._overlay_menu_position = "top_right"

        # Load boost pad positions from field info
        self._boost_pad_positions = []
        try:
            field_info = self.get_field_info()
            for i in range(field_info.num_boosts):
                pad = field_info.boost_pads[i]
                self._boost_pad_positions.append({
                    'pos': (pad.location.x, pad.location.y),
                    'is_large': pad.is_full_boost,
                })
            self._boost_mgr.set_pads(self._boost_pad_positions)
            # Auto-detect goal positions, field dimensions from FieldInfo
            if field_info.num_goals > 0:
                self.decision_engine.set_goal_positions(field_info)
        except Exception:
            pass

        # Detect game mode from match settings (flatbuffer)
        self._detected_game_mode = GM_SOCCER
        try:
            ms = self.get_match_settings()
            self._detected_game_mode = self.decision_engine.detect_game_mode_from_settings(ms)
            # Refine further with mutator / map inspection
            self._detected_game_mode = self.decision_engine.refine_mode_from_mutators(ms)
        except Exception:
            pass

        self.last_planned_path = []
        self.last_path_cost = 0.0
        self.last_algorithm = "A*"
        self._algo_use_counts: dict[str, int] = {}
        self._last_target = (0.0, 0.0)
        self._prev_my_score = 0
        self._prev_opp_score = 0
        self._goal_limit = self._read_goal_limit()
        self._last_save_time = 0.0

        # Dodge state machine
        self._dodge_timer = 0       # ticks since dodge started
        self._dodge_active = False   # currently in a dodge sequence
        self._dodge_pitch = 0.0
        self._dodge_yaw = 0.0

        # Demo/bump tracking
        self._prev_opp_demolished = False

        # Possession tracking (who is closer to ball more often)
        self._possession_us = 0
        self._possession_them = 0
        self._possession_check_tick = 0

        # Save detection: ball was heading to our goal, now it's not
        self._ball_was_threatening = False
        self._last_situation = "free_ball"

        # Ground friction detection: track previous speed to compute acceleration
        self._prev_car_speed = 0.0
        self._prev_throttle = 0.0

        # Force-save knowledge on process exit
        atexit.register(self._atexit_save)

    def _atexit_save(self):
        try:
            # Always save the persistent model on exit.
            self._persistent_knowledge.save()
            self._persistent_knowledge.save_checkpoint("exit")
            self.decision_engine.save_rl_data()
        except Exception:
            pass
        try:
            total = self._possession_us + self._possession_them
            poss_pct = self._possession_us / total if total > 0 else 0.5
            self._match_learning.record_match(
                goals_scored=self._prev_my_score,
                goals_conceded=self._prev_opp_score,
                kickoff_successes=0,
                shot_successes=self._prev_my_score,
                possession_pct=poss_pct,
            )
        except Exception:
            pass
        try:
            self._strategy_selector.record_outcome(
                arena=self._current_arena,
                goals_scored=self._prev_my_score,
                goals_conceded=self._prev_opp_score,
            )
        except Exception:
            pass
        try:
            if getattr(self, "_current_ability", ""):
                success = self._prev_my_score > self._prev_opp_score
                self._ability_strategy.record_outcome(self._current_ability, success)
        except Exception:
            pass
        try:
            if hasattr(self, "_team_power_strategy"):
                if self._prev_my_score > self._prev_opp_score:
                    self._team_power_strategy.notify_goal_scored()
                elif self._prev_opp_score > self._prev_my_score:
                    self._team_power_strategy.notify_goal_conceded()
        except Exception:
            pass

    @staticmethod
    @staticmethod
    def _read_goal_limit() -> int:
        """Read max_score from _match_settings.json; returns 7 if unlimited/unset."""
        try:
            import re as _re
            path = os.path.join(RUNTIME_DIR, "_match_settings.json")
            with open(path, "r", encoding="utf-8") as _f:
                _raw = json.load(_f).get("settings", {}).get("max_score", "Unlimited")
            if isinstance(_raw, int):
                return _raw if _raw > 0 else 7
            _s = str(_raw).lower().strip()
            if not _s or "unlimited" in _s or _s in ("none", "0"):
                return 7
            _m = _re.search(r"\d+", _s)
            return int(_m.group()) if _m else 7
        except Exception:
            return 7

    @staticmethod
    def _speed_xy(car) -> float:
        vel = car.physics.velocity
        return math.hypot(vel.x, vel.y)

    @staticmethod
    def _mode_text(mode: str) -> str:
        return {
            "manual": "manual",
            "balanced": "attack/defense",
            "attack": "attack",
            "defense": "defense",
        }.get(mode, mode)

    def _resolve_bot_index(self, packet: GameTickPacket) -> int:
        """Follow the actual car index for this bot after RLBot reassigns spawn slots."""
        index = self.index
        if index < packet.num_cars:
            try:
                if getattr(self, "spawn_id", -1) == packet.game_cars[index].spawn_id:
                    return index
            except Exception:
                pass

        spawn_id = getattr(self, "spawn_id", -1)
        if spawn_id not in (-1, None):
            for car_index in range(packet.num_cars):
                try:
                    if packet.game_cars[car_index].spawn_id == spawn_id:
                        if getattr(self, "_resolved_index", None) != car_index:
                            print(f"[INFO] Bot car index remapped from {self.index} to {car_index} (spawn_id={spawn_id})")
                        self._resolved_index = car_index
                        return car_index
                except Exception:
                    continue

        resolved = getattr(self, "_resolved_index", None)
        if resolved is not None and resolved < packet.num_cars:
            return resolved
        return min(index, max(0, packet.num_cars - 1)) if packet.num_cars > 0 else index

    def _get_opponent(self, packet: GameTickPacket) -> Tuple[float, float]:
        """Find closest opponent car position."""
        my_index = self._resolve_bot_index(packet)
        my_team = packet.game_cars[my_index].team
        best_dist = float("inf")
        best_pos = (0.0, 0.0)
        ball = packet.game_ball
        ball_xy = (ball.physics.location.x, ball.physics.location.y)
        for i in range(packet.num_cars):
            if i == my_index:
                continue
            opp = packet.game_cars[i]
            if opp.team != my_team:
                pos = (opp.physics.location.x, opp.physics.location.y)
                d = math.hypot(pos[0] - ball_xy[0], pos[1] - ball_xy[1])
                if d < best_dist:
                    best_dist = d
                    best_pos = pos
        return best_pos

    def _get_opponent_vel(self, packet: GameTickPacket) -> Tuple[float, float]:
        """Find velocity of closest opponent."""
        my_index = self._resolve_bot_index(packet)
        my_team = packet.game_cars[my_index].team
        best_dist = float("inf")
        best_vel = (0.0, 0.0)
        ball = packet.game_ball
        ball_xy = (ball.physics.location.x, ball.physics.location.y)
        for i in range(packet.num_cars):
            if i == my_index:
                continue
            opp = packet.game_cars[i]
            if opp.team != my_team:
                pos = (opp.physics.location.x, opp.physics.location.y)
                d = math.hypot(pos[0] - ball_xy[0], pos[1] - ball_xy[1])
                if d < best_dist:
                    best_dist = d
                    best_vel = (opp.physics.velocity.x, opp.physics.velocity.y)
        return best_vel

    def _find_opponent_keeper(self, packet: GameTickPacket):
        """Find opponent player closest to their own goal (likely goalkeeper)."""
        my_index = self._resolve_bot_index(packet)
        my_team = packet.game_cars[my_index].team
        push_dir = 1.0 if my_team == 0 else -1.0
        opp_goal_y = 5120.0 * push_dir
        best_dist = float('inf')
        best_pos = None
        for i in range(packet.num_cars):
            if i == my_index:
                continue
            c = packet.game_cars[i]
            if c.team != my_team:
                pos = (c.physics.location.x, c.physics.location.y)
                d = abs(pos[1] - opp_goal_y)
                if d < best_dist:
                    best_dist = d
                    best_pos = pos
        # Only useful if keeper is actually near their goal
        if best_pos and best_dist < 2500:
            return best_pos
        return None

    def _on_match_state_change(self, state: "MatchState") -> None:
        """Called by MatchStateDetector on every state transition."""
        try:
            from game_logic.match_state_detector import MatchState as _MS
            if state == _MS.GOAL_SCORED:
                self._status_mgr.on_goal_scored()
                self._prev_my_score  = getattr(self, "_prev_my_score",  0) + 1
            elif state == _MS.GOAL_CONCEDED:
                self._status_mgr.on_goal_conceded()
                self._prev_opp_score = getattr(self, "_prev_opp_score", 0) + 1
            elif state == _MS.MATCH_END:
                my  = getattr(self, "_prev_my_score",  0)
                opp = getattr(self, "_prev_opp_score", 0)
                self._status_mgr.on_episode_end(won=(my > opp))
                self._match_detector.reset_for_new_match()
        except Exception:
            pass

    def _render(self, packet: GameTickPacket, mode: str, temporary_reset: bool):
        me = packet.game_cars[self._resolve_bot_index(packet)]
        ball = packet.game_ball

        car_pos = (me.physics.location.x, me.physics.location.y, me.physics.location.z)
        ball_pos = (ball.physics.location.x, ball.physics.location.y, ball.physics.location.z)
        opponent_goal_y = 5120.0 if me.team == 0 else -5120.0

        usage = self.control_panel.usage_percentage()

        # Gather RL adaptive info
        rl_role = getattr(self.decision_engine.adaptive, '_current_role', '')
        opp_pred = self.decision_engine.adaptive.get_opponent_prediction()
        rl_trend = self.decision_engine.adaptive.get_trend()
        human_active = self.decision_engine.adaptive.policy_human.has_learned()

        self.renderer.begin_rendering("search_ai_overlay")
        self.overlay.render(
            my_team=me.team,
            car_pos=car_pos,
            ball_pos=ball_pos,
            opponent_goal_y=opponent_goal_y,
            mode=self._mode_text(mode),
            algorithm=self.last_algorithm,
            usage_pct=usage,
            path_cost=self.last_path_cost,
            path_points=self.last_planned_path,
            temporary_reset=temporary_reset,
            target=self._last_target,
            situation=self._last_situation,
            rl_role=rl_role,
            opp_prediction=opp_pred,
            rl_trend=rl_trend,
            human_policy_active=human_active,
            show_menu=self._overlay_menu_visible,
            menu_position=self._overlay_menu_position,
            menu_size=self._overlay_menu_size,
            game_mode=self.decision_engine.settings.game_mode,
            goal_width=self.decision_engine.settings.goal_width,
            predicted_trajectory=getattr(self, "_ball_traj_cache", None),
            intercept_point=getattr(self, "_intercept_cache", None),
            shoot_aim=getattr(self, "_shoot_aim_cache", None),
            strategy=getattr(self.decision_engine, "settings", None) and getattr(self.decision_engine.settings, "strategy_preset", "") or "",
            match_ended=bool(getattr(packet.game_info, "is_match_ended", False)),
            my_score=self._prev_my_score,
            opp_score=self._prev_opp_score,
            goal_limit=self._goal_limit,
        )
        self.renderer.end_rendering()

    def _update_score_events(self, packet: GameTickPacket, mode: str, temporary_reset: bool):
        teams = packet.teams
        if len(teams) < 2:
            return

        my_team = packet.game_cars[self._resolve_bot_index(packet)].team
        my_score = teams[0].score if my_team == 0 else teams[1].score
        opp_score = teams[1].score if my_team == 0 else teams[0].score

        # We scored!
        if my_score > self._prev_my_score:
            self.decision_engine.knowledge.record_goal_path(
                path=self.last_planned_path,
                mode="attack" if mode == "attack" else "defense" if mode == "defense" else "attack",
                algorithm=self.last_algorithm,
            )
            self.decision_engine.knowledge.save()
            self.decision_engine.knowledge.save_checkpoint("goal")
            # Signal adaptive RL system
            self.decision_engine.adaptive.signal_goal_scored()
            # Notify match analyzer and update algo stats
            try:
                self._match_analyzer.notify_goal()
            except Exception:
                pass
            _update_algo_stats(self.last_algorithm, "goals_scored")

        # Opponent scored — punish the active algorithm
        if opp_score > self._prev_opp_score:
            self.decision_engine.knowledge.punish_conceded_goal(self.last_algorithm)
            self.decision_engine.knowledge.save()
            self.decision_engine.knowledge.save_checkpoint("conceded")
            # Signal adaptive RL system
            self.decision_engine.adaptive.signal_goal_conceded()
            try:
                self._match_analyzer.notify_conceded()
            except Exception:
                pass

        self._prev_my_score = my_score
        self._prev_opp_score = opp_score

    def _update_demo_reward(self, packet: GameTickPacket):
        """Check if opponent got demolished — reward our algorithm with context-aware signal."""
        my_index = self._resolve_bot_index(packet)
        my_team = packet.game_cars[my_index].team
        for i in range(packet.num_cars):
            if i == my_index:
                continue
            opp = packet.game_cars[i]
            if opp.team != my_team:
                is_demolished = opp.is_demolished
                if is_demolished and not self._prev_opp_demolished:
                    self.decision_engine.knowledge.reward_demo(self.last_algorithm)
                    # Context-aware demo reward: choose richest applicable signal
                    try:
                        ball = packet.game_ball
                        car = packet.game_cars[my_index]
                        opp_y = opp.physics.location.y
                        ball_xy = (ball.physics.location.x, ball.physics.location.y)
                        opp_xy = (opp.physics.location.x, opp.physics.location.y)
                        push_dir = 1.0 if my_team == 0 else -1.0
                        defender_near_goal = abs(opp_y) > 3500 and (opp_y * push_dir) < 0
                        dist_opp_to_ball = math.hypot(opp_xy[0] - ball_xy[0], opp_xy[1] - ball_xy[1])
                        near_ball = dist_opp_to_ball < 800
                        rc = self.decision_engine.adaptive.reward_calc
                        if defender_near_goal:
                            rc.signal_demo_defender_near_goal()
                        elif near_ball:
                            rc.signal_demo_plus_ball()
                        else:
                            rc.signal_demo_basic()
                    except Exception:
                        self.decision_engine.adaptive.signal_demo()
                self._prev_opp_demolished = is_demolished
                break  # only track closest opponent

    def _update_possession(self, packet: GameTickPacket):
        """Track ball possession and punish when opponent controls ball too much."""
        car = packet.game_cars[self._resolve_bot_index(packet)]
        ball = packet.game_ball
        car_xy = (car.physics.location.x, car.physics.location.y)
        ball_xy = (ball.physics.location.x, ball.physics.location.y)
        opp_pos = self._get_opponent(packet)

        my_dist = math.hypot(ball_xy[0] - car_xy[0], ball_xy[1] - car_xy[1])
        opp_dist = math.hypot(ball_xy[0] - opp_pos[0], ball_xy[1] - opp_pos[1])

        if my_dist < opp_dist:
            self._possession_us += 1
        else:
            self._possession_them += 1

        self._possession_check_tick += 1
        # Every ~300 ticks (~5 seconds), check possession balance
        if self._possession_check_tick >= 300:
            total = self._possession_us + self._possession_them
            if total > 0 and self._possession_them > self._possession_us * 1.8:
                # Opponent has ball way more than us — punish
                self.decision_engine.knowledge.punish_possession_lost(self.last_algorithm)
            self._possession_us = 0
            self._possession_them = 0
            self._possession_check_tick = 0

    def _update_save_detection(self, packet: GameTickPacket):
        """Detect when we clear a ball that was heading to our goal."""
        car = packet.game_cars[self._resolve_bot_index(packet)]
        ball = packet.game_ball
        push_dir = 1.0 if car.team == 0 else -1.0
        ball_vel_y = ball.physics.velocity.y
        ball_y = ball.physics.location.y

        ball_threatening = (
            (ball_y * push_dir) < -1000
            and (ball_vel_y * push_dir) < -400
        )

        if self._ball_was_threatening and not ball_threatening:
            # Ball was heading to our goal and now it's not — we made a save/clear
            car_xy = (car.physics.location.x, car.physics.location.y)
            ball_xy = (ball.physics.location.x, ball.physics.location.y)
            dist = math.hypot(ball_xy[0] - car_xy[0], ball_xy[1] - car_xy[1])
            if dist < 1500:  # we were close enough to have caused the clear
                self.decision_engine.knowledge.reward_save(self.last_algorithm)
                # Use richer signal: full save if very close (< 600 u), else clear
                rc = self.decision_engine.adaptive.reward_calc
                if dist < 600:
                    rc.signal_save()
                else:
                    rc.signal_clear_ball()
                self.decision_engine.adaptive.signal_save_made()

        self._ball_was_threatening = ball_threatening

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        try:
            return self._get_output_inner(packet)
        except Exception as exc:
            if not getattr(self, '_error_logged', False):
                import traceback
                _blog.error("get_output crashed: %s\n%s", exc, traceback.format_exc())
                print(f"[BOT ERROR] get_output crashed: {exc}")
                traceback.print_exc()
                self._error_logged = True
            return SimpleControllerState()

    def _get_output_inner(self, packet: GameTickPacket) -> SimpleControllerState:
        if self._tick == 0:
            _blog.info("First get_output tick! num_cars=%d", packet.num_cars)
            # Hide external GUI during gameplay — overlay handles in-game info
            self.control_panel.hide()
        # Poll IPC commands when running headlessly (--no-gui mode)
        if _NO_GUI:
            _poll_ipc_commands(self)
        # Update match state detector every tick
        try:
            _my_team = self._resolve_bot_index(packet)
            _car_team = packet.game_cars[_my_team].team if _my_team < len(packet.game_cars) else 0
            self._match_detector.update(packet, _car_team)
        except Exception:
            pass
        state = self.mode_manager.update_from_keyboard()
        panel_mode = self.control_panel.consume_mode_override()
        if panel_mode is not None:
            self.mode_manager.set_mode(panel_mode)
            state.mode = panel_mode

        panel_model = self.control_panel.consume_model_override()
        if panel_model is not None:
            self.mode_manager.set_temporary_reset(panel_model)
            state.temporary_reset = panel_model

        panel_model_sel = self.control_panel.consume_model_selection()
        if panel_model_sel is not None:
            self.decision_engine.adaptive.set_active_models(panel_model_sel)
            count = len(panel_model_sel) if panel_model_sel else 8
            names = ", ".join(sorted(panel_model_sel)) if panel_model_sel else "ALL"
            print(f"[MODEL] Active RL models updated ({count}): {names}")
            self.control_panel.set_setup_hint(f"RL models active: {count}/8 — {names[:60]}")

        if self.mode_manager.consume_overlay_toggle_request():
            self._overlay_menu_visible = not self._overlay_menu_visible
            overlay_state = "shown" if self._overlay_menu_visible else "hidden"
            self.control_panel.set_setup_hint(f"Overlay menu {overlay_state}. Press H to toggle it.")
            print(f"[OVERLAY] In-game menu {overlay_state}.")

        reopen_requested = (
            self.mode_manager.consume_open_setup_request()
            or self.control_panel.consume_open_setup_request()
        )
        if reopen_requested:
            self.control_panel.show()
            self.control_panel.set_runtime_status("Leaving match and reopening setup menu...")
            self.control_panel.set_setup_hint("Setup reopen requested. Wait a moment...")
            _request_settings_reopen()

        mode = state.mode
        temporary_reset = state.temporary_reset
        self._current_mode_str = mode
        self.control_panel.set_mode(mode)
        self.control_panel.set_temporary_reset(temporary_reset)

        # P key: start a fresh session-only model. O key: restore the persistent model.
        if temporary_reset != self._prev_temporary_reset:
            if temporary_reset:
                self._session_knowledge = KnowledgeStore(knowledge_path=None, persist=False)
                self.decision_engine.set_knowledge_store(self._session_knowledge)
                self.control_panel.set_runtime_status("Session model active: fresh temporary learning branch")
                self.control_panel.set_setup_hint("Session model active. Press O to restore persistent learning.")
                print("[MODEL] Switched to SESSION model (temporary learning branch).")
            else:
                self.decision_engine.set_knowledge_store(self._persistent_knowledge)
                self._persistent_knowledge.save()
                self.control_panel.set_runtime_status("Persistent model active: saved evolving knowledge restored")
                self.control_panel.set_setup_hint("Persistent model active. Press P for a fresh session branch.")
                print("[MODEL] Switched to PERSISTENT model (saved knowledge restored).")
            self._cached_decision = None
            self._prev_temporary_reset = temporary_reset

        bot_index = self._resolve_bot_index(packet)
        car = packet.game_cars[bot_index]
        ball = packet.game_ball
        ball_speed_3d = math.sqrt(
            ball.physics.velocity.x ** 2 +
            ball.physics.velocity.y ** 2 +
            ball.physics.velocity.z ** 2
        )
        self.decision_engine.settings.update_surface_state(
            now_seconds=packet.game_info.seconds_elapsed,
            car_pos=(car.physics.location.x, car.physics.location.y, car.physics.location.z),
            car_speed=self._speed_xy(car),
            car_on_ground=car.has_wheel_contact,
            ball_pos=(ball.physics.location.x, ball.physics.location.y, ball.physics.location.z),
            ball_speed=ball_speed_3d,
        )
        gs_surface = self.decision_engine.settings
        self.control_panel.set_runtime_status(
            f"Bot index={bot_index} spawn_id={getattr(self, 'spawn_id', -1)} mode={mode}"
        )
        self.control_panel.set_surface_info(
            f"Car:{gs_surface.car_surface} {gs_surface.car_surface_time:.1f}s {gs_surface.car_surface_max_speed:.0f}u/s | "
            f"Ball:{gs_surface.ball_surface} {gs_surface.ball_surface_time:.1f}s {gs_surface.ball_surface_max_speed:.0f}u/s"
        )
        self.control_panel.set_game_mode(gs_surface.game_mode)

        self._tick += 1

        # ── Game state detection + strategy update ────────────────────────
        try:
            _bot_idx_gs = self._resolve_bot_index(packet)
            _car_gs = packet.game_cars[_bot_idx_gs]
            _ball_gs = packet.game_ball
            _opp_gs = self._get_opponent(packet)
            _push_dir_gs = 1.0 if _car_gs.team == 0 else -1.0
            _score_diff_gs = self.decision_engine.settings.score_diff
            _time_gs = packet.game_info.seconds_elapsed
            _car_pos_gs = (_car_gs.physics.location.x, _car_gs.physics.location.y)
            _ball_pos_gs = (_ball_gs.physics.location.x, _ball_gs.physics.location.y)
            _ball_vel_gs = (_ball_gs.physics.velocity.x, _ball_gs.physics.velocity.y)
            _car_vel_gs = (_car_gs.physics.velocity.x, _car_gs.physics.velocity.y)
            self._current_game_state = detect_game_state(
                car_pos=_car_pos_gs,
                ball_pos=_ball_pos_gs,
                opp_pos=_opp_gs,
                boost=_car_gs.boost,
                score_diff=_score_diff_gs,
                time_remaining=max(0.0, 300.0 - _time_gs),
                ball_vel=_ball_vel_gs,
                car_vel=_car_vel_gs,
                ball_z=_ball_gs.physics.location.z,
                push_dir=_push_dir_gs,
                car_on_ground=_car_gs.has_wheel_contact,
                opp_demolished=self._prev_opp_demolished,
                has_teammates=False,
                my_team=_car_gs.team,
            )
            # Update strategy adaptor
            self._strategy_layer.adapt(
                game_state=self._current_game_state,
                score_diff=_score_diff_gs,
                time_remaining=max(0.0, 300.0 - _time_gs),
                boost=_car_gs.boost,
                dist_to_ball=math.hypot(_car_pos_gs[0] - _ball_pos_gs[0], _car_pos_gs[1] - _ball_pos_gs[1]),
            )
        except Exception:
            pass

        # ── IPC status write (--no-gui mode, every N ticks) ──────────────
        if _NO_GUI and self._tick % _IPC_STATUS_INTERVAL == 0:
            _write_ipc_status(self)

        self._update_score_events(packet, mode, temporary_reset)
        self._update_demo_reward(packet)
        self._update_possession(packet)
        self._update_save_detection(packet)

        # ── New subsystem updates ─────────────────────────────────────────
        try:
            _bot_idx_ns = self._resolve_bot_index(packet)
            _car_ns = packet.game_cars[_bot_idx_ns]
            _ball_ns = packet.game_ball
            _car_pos_ns = (_car_ns.physics.location.x, _car_ns.physics.location.y)
            _car_pos3d_ns = (_car_ns.physics.location.x,
                             _car_ns.physics.location.y,
                             _car_ns.physics.location.z)
            _car_vel3d_ns = (_car_ns.physics.velocity.x,
                             _car_ns.physics.velocity.y,
                             _car_ns.physics.velocity.z)
            _ball_pos3d_ns = (_ball_ns.physics.location.x,
                              _ball_ns.physics.location.y,
                              _ball_ns.physics.location.z)
            _ball_vel_ns   = (_ball_ns.physics.velocity.x,
                              _ball_ns.physics.velocity.y,
                              _ball_ns.physics.velocity.z)
            _ball_vel3d_ns = _ball_vel_ns + (_ball_ns.physics.velocity.z,)
            _car_boost_ns  = float(_car_ns.boost)
            _car_speed_ns  = self._speed_xy(_car_ns)
            _opp_pos_ns    = self._get_opponent(packet)
            _opp_vel_ns    = self._get_opponent_vel(packet)
            _push_dir_ns   = 1.0 if _car_ns.team == 0 else -1.0
            _own_goal_y_ns = -5120.0 if _car_ns.team == 0 else 5120.0

            # ── Ball prediction cache (refreshed every 10 ticks) ─────────
            if (self._tick - self._ball_traj_tick) >= 10:
                self._ball_traj_cache = predict_ball_path(
                    _ball_pos3d_ns,
                    (_ball_ns.physics.velocity.x,
                     _ball_ns.physics.velocity.y,
                     _ball_ns.physics.velocity.z),
                    steps=120,   # 2 seconds @ 60 Hz
                )
                self._intercept_cache = find_best_intercept(
                    self._ball_traj_cache, _car_pos_ns, _car_speed_ns
                )
                # Shoot aim direction (toward opponent goal)
                _opp_goal_y_ns = 5120.0 if _car_ns.team == 0 else -5120.0
                self._shoot_aim_cache = (0.0, _opp_goal_y_ns)
                self._ball_traj_tick = self._tick

            # ── IPC position cache (for _write_ipc_status) ───────────────
            self._last_ball_pos_ipc = _ball_pos3d_ns
            self._last_ball_vel_ipc = _ball_vel_ns
            self._last_car_pos_ipc  = (_car_ns.physics.location.x,
                                       _car_ns.physics.location.y,
                                       _car_ns.physics.location.z)

            # ── Opponent Analyzer ─────────────────────────────────────────
            self._opp_analyzer.update(
                opp_pos=_opp_pos_ns,
                opp_vel=_opp_vel_ns,
                ball_pos=(_ball_ns.physics.location.x, _ball_ns.physics.location.y),
                push_dir=_push_dir_ns,
                opp_demolished=self._prev_opp_demolished,
            )

            # ── Boost Manager ──────────────────────────────────────────────
            self._boost_mgr.update(
                current_boost=_car_boost_ns,
                mode=mode,
                car_pos=_car_pos_ns,
            )

            # ── Strategy Manager ──────────────────────────────────────────
            _opp_goal_y_sm = 5120.0 if _car_ns.team == 0 else -5120.0
            _ball_dist_own = math.hypot(
                _ball_ns.physics.location.x,
                _ball_ns.physics.location.y - _own_goal_y_ns,
            )
            _opp_style = self._opp_analyzer.get_style()
            self._strategy_manager_recommendation = self._strategy_mgr.decide(
                ball_pos=(_ball_ns.physics.location.x, _ball_ns.physics.location.y),
                push_dir=_push_dir_ns,
                score_diff=self.decision_engine.settings.score_diff,
                time_remaining=max(0.0, 300.0 - packet.game_info.seconds_elapsed),
                our_boost=_car_boost_ns,
                opp_style=_opp_style,
                ball_dist_from_our_goal=_ball_dist_own,
            )

            # ── Team Rotation (multi-car matches) ─────────────────────────
            _teammate_positions_ns = []
            for _ti in range(packet.num_cars):
                if _ti == _bot_idx_ns:
                    continue
                if packet.game_cars[_ti].team == _car_ns.team:
                    _teammate_positions_ns.append((
                        packet.game_cars[_ti].physics.location.x,
                        packet.game_cars[_ti].physics.location.y,
                    ))
            if _teammate_positions_ns:
                self._team_rotation.update(
                    my_pos=_car_pos_ns,
                    ball_pos=(_ball_ns.physics.location.x, _ball_ns.physics.location.y),
                    teammate_positions=_teammate_positions_ns,
                    push_dir=_push_dir_ns,
                    own_goal_y=_own_goal_y_ns,
                )

            # ── Match Strategy Analyzer ───────────────────────────────────
            _opp_goal_y_ma = 5120.0 if _car_ns.team == 0 else -5120.0
            _we_have_ball = (
                math.hypot(_ball_ns.physics.location.x - _car_ns.physics.location.x,
                           _ball_ns.physics.location.y - _car_ns.physics.location.y) < 350.0
            )
            self._match_analyzer.update(
                ball_pos=(_ball_ns.physics.location.x, _ball_ns.physics.location.y),
                car_pos=_car_pos_ns,
                opp_pos=_opp_pos_ns,
                opp_goal_y=_opp_goal_y_ma,
                own_goal_y=_own_goal_y_ns,
                we_have_ball=_we_have_ball,
            )

            # ── Team State (multi-bot role assignment) ────────────────────
            _bot_id_ns = getattr(self, "index", 0)
            self._team_state.register(_bot_id_ns, _own_goal_y_ns, _opp_goal_y_ma)
            self._team_state.update(
                bot_id=_bot_id_ns,
                position=_car_pos_ns,
                ball_position=(_ball_ns.physics.location.x, _ball_ns.physics.location.y),
                boost=_car_boost_ns,
                has_ball=_we_have_ball,
            )

            # ── Kickoff AI ────────────────────────────────────────────────
            if getattr(packet.game_info, "is_kickoff_pause", False):
                if not self._kickoff_ai.active:
                    self._kickoff_ai.notify_kickoff(_car_pos_ns, _push_dir_ns)

            # ── Kickoff Detector ──────────────────────────────────────────
            self._kickoff_detector.update(
                ball_pos=_ball_pos3d_ns,
                ball_vel=_ball_vel_ns,
                is_kickoff_pause=getattr(packet.game_info, "is_kickoff_pause", False),
            )
            if self._kickoff_detector.just_started:
                _bot_pos_map = {getattr(self, "index", 0): _car_pos_ns}
                for _ki in range(packet.num_cars):
                    if packet.game_cars[_ki].team == _car_ns.team and _ki != _bot_idx_ns:
                        _bot_pos_map[_ki] = (
                            packet.game_cars[_ki].physics.location.x,
                            packet.game_cars[_ki].physics.location.y,
                        )
                self._kickoff_roles.update(
                    bot_positions=_bot_pos_map,
                    ball_pos=(_ball_pos3d_ns[0], _ball_pos3d_ns[1]),
                )

            # ── Opponent Predictor ────────────────────────────────────────
            _opp_vel3d_ns = (_opp_vel_ns[0], _opp_vel_ns[1], 0.0)
            self._opp_predictor.update(
                opp_pos=(_opp_pos_ns[0], _opp_pos_ns[1], 0.0),
                opp_vel=_opp_vel3d_ns,
                ball_pos=_ball_pos3d_ns,
            )

            # ── Game Awareness ────────────────────────────────────────────
            _opp_goal_y_ga = 5120.0 if _car_ns.team == 0 else -5120.0
            _ga_state = self._game_awareness.update(
                ball_pos=_ball_pos3d_ns,
                car_pos=_car_pos3d_ns,
                opp_pos=(_opp_pos_ns[0], _opp_pos_ns[1], 0.0),
                own_goal_y=_own_goal_y_ns,
                opp_goal_y=_opp_goal_y_ga,
                is_kickoff=self._kickoff_detector.is_kickoff,
                we_have_ball=_we_have_ball,
            )
            # Expose game-awareness state on the bot for IPC and overlay
            self._current_game_state = _ga_state

            # ── Strategy Selector (in-match adaptation, every 60 ticks) ──
            self._strategy_adapt_tick += 1
            if self._strategy_adapt_tick >= 60:
                self._strategy_adapt_tick = 0
                _total_ticks = self._possession_us + self._possession_them
                _poss_pct = (
                    self._possession_us / _total_ticks
                    if _total_ticks > 0 else 0.5
                )
                _opp_style_ns = (
                    self._opp_analyzer.get_style()
                    if hasattr(self._opp_analyzer, "get_style") else "unknown"
                )
                _score_diff_ns = self._prev_my_score - self._prev_opp_score
                self._active_strategies = self._strategy_selector.adapt_in_match(
                    arena=self._current_arena,
                    score_diff=_score_diff_ns,
                    possession_pct=_poss_pct,
                    opponent_style=_opp_style_ns,
                    is_kickoff=self._kickoff_detector.is_kickoff,
                    game_state=_ga_state,
                    tick=self._tick,
                )

            # ── Ability Discovery subsystem ───────────────────────────────
            try:
                _raw_powerup = getattr(_car_ns, "powerup", None) or \
                               getattr(_car_ns, "powerup_active", None) or ""
                _ball_speed_ns = math.sqrt(
                    _ball_ns.physics.velocity.x ** 2 +
                    _ball_ns.physics.velocity.y ** 2 +
                    _ball_ns.physics.velocity.z ** 2
                )
                self._ability_discovery.update(
                    raw_powerup=str(_raw_powerup) if _raw_powerup else None,
                    ball_pos=(_ball_ns.physics.location.x,
                               _ball_ns.physics.location.y),
                    car_pos=_car_pos_ns,
                    opp_pos=_opp_pos_ns,
                    ball_vel_magnitude=_ball_speed_ns,
                )
                self._current_ability = self._ability_discovery.current_ability or ""
                self._ability_experiment_mode = self._ability_discovery._experiment_mode

                # Team Ability Awareness update
                self._team_ability_awareness.update(
                    game_cars=packet.game_cars,
                    num_cars=packet.num_cars,
                    bot_index=_bot_idx_ns,
                    bot_team=_car_ns.team,
                )

                # Team Controller role assignment
                _teammate_positions_tc: dict = {}
                for _ti in range(packet.num_cars):
                    _tc = packet.game_cars[_ti]
                    if _tc.team == _car_ns.team:
                        _teammate_positions_tc[_ti] = (
                            _tc.physics.location.x,
                            _tc.physics.location.y,
                        )
                self._team_controller.update(
                    ball_pos=(_ball_ns.physics.location.x,
                               _ball_ns.physics.location.y),
                    car_positions=_teammate_positions_tc,
                    my_index=_bot_idx_ns,
                    my_team=_car_ns.team,
                    tick=self._tick,
                )
                self._bot_role = self._team_controller.get_role(_bot_idx_ns)

                # Team Ability Strategy directive
                _team_dir = self._team_ability_strategy.get_team_directive(
                    bot_index=_bot_idx_ns,
                    my_ability=self._current_ability or None,
                    game_state=_ga_state,
                    ball_pos=(_ball_ns.physics.location.x,
                               _ball_ns.physics.location.y),
                    car_pos=_car_pos_ns,
                    score_diff=self._prev_my_score - self._prev_opp_score,
                    tick=self._tick,
                )
                self._team_directive_reason = _team_dir.reason

                # Team Power Strategy — comprehensive synthesis decision
                _opp_positions_tp: Dict[int, Tuple[float, float]] = {}
                for _ci in range(packet.num_cars):
                    _oc = packet.game_cars[_ci]
                    if _oc.team != _car_ns.team and _ci != _bot_idx_ns:
                        _opp_positions_tp[_ci] = (
                            _oc.physics.location.x,
                            _oc.physics.location.y,
                        )
                _power_dec = self._team_power_strategy.get_strategic_decision(
                    bot_index=_bot_idx_ns,
                    game_state=_ga_state,
                    ball_pos=(_ball_ns.physics.location.x,
                               _ball_ns.physics.location.y),
                    car_pos=_car_pos_ns,
                    opp_positions=_opp_positions_tp,
                    score_diff=self._prev_my_score - self._prev_opp_score,
                    tick=self._tick,
                )
                self._power_decision_action = _power_dec.action
                self._power_decision_combo  = _power_dec.combo_name
            except Exception:
                pass

        except Exception:
            pass
        # ── End new subsystem updates ─────────────────────────────────────

        # ── Manual mode ──
        if mode == MODE_MANUAL:
            car_pos = (car.physics.location.x, car.physics.location.y)
            ball_pos = (ball.physics.location.x, ball.physics.location.y)
            ball_vel_m = (ball.physics.velocity.x, ball.physics.velocity.y)
            speed = self._speed_xy(car)
            opp_pos_m = self._get_opponent(packet)
            opp_vel_m = self._get_opponent_vel(packet)
            if hasattr(self, '_manual_tick'):
                self._manual_tick += 1
            else:
                self._manual_tick = 0

            # Classify situation even in manual mode (for opponent learning)
            situation_m = classify_situation(car_pos, ball_pos, ball_vel_m, opp_pos_m, car.team)
            self._last_situation = situation_m

            # Record opponent state for learning (always, even manual)
            self.decision_engine.knowledge.record_opponent_state(opp_pos_m, opp_vel_m, ball_pos)

            # Update RL reward tracking + opponent model during manual mode
            # so the system learns from how the user plays and from opponent
            push_dir_m = 1.0 if car.team == 0 else -1.0
            self.decision_engine.adaptive.update_rewards_only(
                car_pos, ball_pos, ball_vel_m, opp_pos_m, opp_vel_m,
                speed, car.boost, self.decision_engine.settings.score_diff,
                push_dir_m, situation_m,
                self.decision_engine._adaptive_context_tag(),
            )

            # Read keyboard → real car controls
            hc = self.mode_manager.read_human_controls()
            out = SimpleControllerState()
            out.throttle = hc["throttle"]
            out.steer = hc["steer"]
            out.jump = bool(hc["jump"])
            out.boost = bool(hc["boost"])
            out.handbrake = abs(hc["steer"]) > 0.5 and speed > 800.0

            # Record state + controls for BOTH legacy and RL policy learning
            if self._manual_tick % 3 == 0:
                self.decision_engine.knowledge.record_human_frame(
                    car_pos, ball_pos, speed, car.boost, car.team,
                    throttle=hc["throttle"],
                    steer=hc["steer"],
                    boost_input=hc["boost"],
                    jump_input=hc["jump"],
                )
                # Feed human demonstration to adaptive RL policy gradient
                self.decision_engine.record_human_demo(
                    car_pos, ball_pos, speed, car.boost, opp_pos_m, car.team,
                    throttle=hc["throttle"], steer=hc["steer"],
                    boost_in=hc["boost"], jump_in=hc["jump"],
                )
            if self._manual_tick % 300 == 0:
                if not temporary_reset:
                    self.decision_engine.knowledge.save()
                    self.decision_engine.save_rl_data()
            if self._tick % 600 == 1:
                print(f"[BOT] tick={self._tick} mode=manual throttle={out.throttle:.1f} "
                      f"steer={out.steer:.1f} boost={out.boost} idx={bot_index}")
            self._render(packet, mode, temporary_reset)
            return out

        # ── Extract game state ──
        car_xy = (car.physics.location.x, car.physics.location.y)
        ball_xy = (ball.physics.location.x, ball.physics.location.y)
        ball_vel = (ball.physics.velocity.x, ball.physics.velocity.y)
        ball_z = ball.physics.location.z
        car_speed = self._speed_xy(car)
        car_boost_amount = car.boost
        opp_pos = self._get_opponent(packet)
        opp_vel = self._get_opponent_vel(packet)
        dist_to_ball = math.hypot(ball_xy[0] - car_xy[0], ball_xy[1] - car_xy[1])

        # ── Classify situation ──
        situation = classify_situation(car_xy, ball_xy, ball_vel, opp_pos, car.team)
        self._last_situation = situation

        # ── Record opponent for learning ──
        self.decision_engine.knowledge.record_opponent_state(opp_pos, opp_vel, ball_xy)

        # ── Adaptive RL-based mode switching ──
        # The adaptive system (Q-Learning + SARSA + Actor-Critic + Ensemble)
        # decides the effective role based on game state and learning.
        # User mode (attack/defense/balanced) is respected as a bias.
        effective_mode = self.decision_engine.adaptive_decide_role(
            situation=situation,
            car_pos=car_xy,
            ball_pos=ball_xy,
            ball_vel=ball_vel,
            opp_pos=opp_pos,
            opp_vel=opp_vel,
            car_speed=car_speed,
            car_boost=car_boost_amount,
            my_team=car.team,
            user_mode=mode,
        )

        # ── Explicit attack / defense commands always override the adaptive layer ──
        # B key forces attack; V key forces defense regardless of RL-suggested role.
        if mode == "attack":
            effective_mode = "attack"
        elif mode == "defense":
            effective_mode = "defense"

        # ── Ball trajectory prediction (physics-based, mode-adaptive) ──
        ball_vz = ball.physics.velocity.z
        gs = self.decision_engine.settings
        trajectory = predict_trajectory(
            ball_xy[0], ball_xy[1], ball_z,
            ball_vel[0], ball_vel[1], ball_vz,
            total_time=3.0,
            gravity=gs.gravity,
            restitution=gs.bounce,
            puck_mode=gs.puck_mode,
        )
        self.decision_engine.set_ball_trajectory(trajectory)

        # ── Per-frame game mode heuristic (backup if settings unavailable) ──
        self.decision_engine.detect_game_mode_from_packet(
            num_tiles=packet.num_tiles,
            world_gravity=packet.game_info.world_gravity_z,
            ball_z=ball_z,
        )
        self.decision_engine.detect_game_mode(packet.game_info)

        # ── Ground friction detection: compute actual acceleration vs expected ──
        if car.has_wheel_contact and self._tick > 1:
            dt = 1.0 / 60.0  # ~60 fps
            actual_accel = (car_speed - self._prev_car_speed) / dt
            self.decision_engine.detect_ground_type(
                car_on_ground=True,
                car_speed=car_speed,
                throttle=self._prev_throttle,
                actual_accel=actual_accel,
            )

        # ── Boost pad state from packet ──
        pads = []
        for i, pos_info in enumerate(self._boost_pad_positions):
            if i < packet.num_boost:
                pads.append({
                    'pos': pos_info['pos'],
                    'is_large': pos_info['is_large'],
                    'active': packet.game_boosts[i].is_active,
                })
        self.decision_engine.set_boost_pads(pads)

        # ── Opponent keeper detection ──
        self.decision_engine.set_keeper_pos(
            self._find_opponent_keeper(packet))

        # ── Teammate positions (for team play / anti-double-commit) ──
        teammates = []
        my_index = self._resolve_bot_index(packet)
        for i in range(packet.num_cars):
            if i == my_index:
                continue
            c = packet.game_cars[i]
            if c.team == car.team:
                teammates.append((c.physics.location.x, c.physics.location.y))
        self.decision_engine.set_teammates(teammates)

        # ── Score state (trailing / overtime awareness) ──
        teams = packet.teams
        if len(teams) >= 2:
            my_score = teams[0].score if car.team == 0 else teams[1].score
            opp_score = teams[1].score if car.team == 0 else teams[0].score
            is_ot = getattr(packet.game_info, 'is_overtime', False)
            self.decision_engine.set_score_state(my_score, opp_score, is_ot)

        # ── Dropshot tile tracking (opponent half) ──
        if gs.game_mode == GM_DROPSHOT and packet.num_tiles > 0:
            damaged = []
            for ti in range(packet.num_tiles):
                ts = packet.dropshot_tiles[ti].tile_state
                if ts >= 2:  # Damaged=2, Open=3
                    damaged.append((ti, ts))
            self.decision_engine.set_dropshot_tiles(damaged)

        # ── Kickoff spawn detection ──
        _kickoff_active = (
            getattr(packet.game_info, 'is_kickoff_pause', False)
            or is_kickoff_state(ball_xy, ball_vel)
        )
        if _kickoff_active:
            self.decision_engine.detect_kickoff_spawn(car_xy, car.team)
            effective_mode = "attack"

        # ── Score-aware strategy: adjust effective_mode based on goal deficit/lead ──
        # Only override if not already set by an explicit B/V key or kickoff.
        if mode not in ("attack", "defense"):
            _score_diff_sa = self._prev_my_score - self._prev_opp_score
            _goals_to_win_sa = max(0, self._goal_limit - self._prev_my_score)
            if _goals_to_win_sa == 1 and effective_mode != "defense":
                # One goal from winning — protect the lead
                effective_mode = "defense"
            elif _score_diff_sa <= -2 and effective_mode != "attack":
                # Losing by 2 or more — push harder
                effective_mode = "attack"

        # ── Choose algorithm & plan ──
        def _needs_replan() -> bool:
            if self._cached_decision is None:
                return True
            if effective_mode != self._last_plan_mode:
                return True
            if temporary_reset != self._last_plan_reset:
                return True
            if situation != self._last_plan_situation:
                return True
            ball_moved = math.hypot(ball_xy[0] - self._last_plan_ball_xy[0], ball_xy[1] - self._last_plan_ball_xy[1])
            if ball_moved > 200.0:
                return True

            # Replan every tick when close or defending; every 2 ticks otherwise.
            if situation in ("defending", "opp_has_ball") or dist_to_ball < 2000.0:
                return True
            return (self._tick - self._last_plan_tick) >= 2

        if _needs_replan():
            t0 = time.perf_counter()
            decision = self.decision_engine.plan_parallel_ensemble(
                mode=effective_mode,
                my_team=car.team,
                car_pos=car_xy,
                ball_pos=ball_xy,
                ball_vel=ball_vel,
                car_speed=car_speed,
                opp_pos=opp_pos,
                reset_mode=False,
                situation=situation,
                car_boost=car_boost_amount,
                ball_z=ball_z,
            )
            self.control_panel.set_planner_mode("ensemble")
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            self._cached_decision = decision
            self._last_plan_tick = self._tick
            self._last_plan_ball_xy = ball_xy
            self._last_plan_mode = effective_mode
            self._last_plan_reset = temporary_reset
            self._last_plan_situation = situation
        else:
            decision = self._cached_decision
            elapsed_ms = 0.0

        self.last_algorithm = decision.algorithm
        self._algo_use_counts[decision.algorithm] = self._algo_use_counts.get(decision.algorithm, 0) + 1
        self.last_planned_path = decision.path_result.world_path
        self.last_path_cost = decision.path_result.cost
        self._last_target = decision.target

        self.control_panel.set_algorithm(decision.algorithm)
        self.control_panel.set_path_cost(decision.path_result.cost)
        self.control_panel.record_timing(decision.algorithm, elapsed_ms)
        self.control_panel.set_threat_level(
            self.decision_engine.adaptive.get_threat_level()
        )

        # ── Look-ahead on path (further ahead = smoother driving) ──
        path = decision.path_result.world_path
        if not path:
            next_point = decision.target
        else:
            best_i = 0
            best_d = float("inf")
            for i, p in enumerate(path[:20]):
                d = math.hypot(p[0] - car_xy[0], p[1] - car_xy[1])
                if d < best_d:
                    best_d = d
                    best_i = i
            # Look further ahead for smooth arcs; less when very close to target
            look_ahead = 7 if len(path) > 10 else 4
            next_point = path[min(len(path) - 1, best_i + look_ahead)]

        # ── Build controls ──
        controls = self.decision_engine.build_controls(
            car_yaw=car.physics.rotation.yaw,
            car_pos=car_xy,
            target=next_point,
            speed=car_speed,
            dist_to_ball=dist_to_ball,
            car_on_ground=car.has_wheel_contact,
            ball_z=ball_z,
            car_boost_amount=car_boost_amount,
            situation=situation,
        )

        # ── Blend human experience (legacy) ──
        controls = self.decision_engine.blend_human_experience(
            controls, car_xy, ball_xy, car_speed, car.team,
        )

        # ── Adaptive RL blend: policy gradient + actor-critic adjustments ──
        controls = self.decision_engine.adaptive_blend_controls(
            controls, car_xy, ball_xy, car_speed, car_boost_amount,
            opp_pos, car.team,
        )

        # ── Attack mode: use boost aggressively when chasing the ball ───────────
        if mode == "attack" and dist_to_ball < 2000 and car_boost_amount > 15:
            controls["boost"] = True

        # ── Defense mode: keep throttle on to maintain defensive cover ─────────
        if mode == "defense":
            # Don't over-commit: reduce boost when already between ball and goal
            car_y_abs = car_xy[1]
            ball_y_abs = ball_xy[1]
            own_goal_y = -5120.0 if car.team == 0 else 5120.0
            between_ball_and_goal = (
                (own_goal_y < car_y_abs < ball_y_abs)
                or (ball_y_abs < car_y_abs < own_goal_y)
            )
            if between_ball_and_goal:
                controls["boost"] = False  # don't waste boost when already positioned

        # ── Aerial Controller override ──────────────────────────────────────
        out = SimpleControllerState()
        if not self._dodge_active:
            try:
                _aerial_out = self._aerial_ctrl.update(
                    ball_pos=(
                        packet.game_ball.physics.location.x,
                        packet.game_ball.physics.location.y,
                        packet.game_ball.physics.location.z,
                    ),
                    car_pos=(
                        car.physics.location.x,
                        car.physics.location.y,
                        car.physics.location.z,
                    ),
                    car_vel=(
                        car.physics.velocity.x,
                        car.physics.velocity.y,
                        car.physics.velocity.z,
                    ),
                    car_boost=float(car.boost),
                    car_on_ground=car.has_wheel_contact,
                )
                if _aerial_out.get("active"):
                    out.jump  = bool(_aerial_out["jump"])
                    out.boost = bool(_aerial_out["boost"])
                    out.pitch = float(_aerial_out["pitch"])
                    out.yaw   = float(_aerial_out["yaw"])
                    out.roll  = float(_aerial_out["roll"])
                    out.throttle = 1.0
                    return out
            except Exception:
                pass

        # ── Dodge state machine (multi-frame) ──
        if self._dodge_active:
            self._dodge_timer += 1
            if self._dodge_timer <= 4:
                # Phase 1: first jump (hold for 4 frames for height)
                out.jump = True
                out.pitch = 0.0
                out.yaw = 0.0
            elif self._dodge_timer <= 7:
                # Phase 2: release jump (3 frames)
                out.jump = False
                out.pitch = 0.0
            elif self._dodge_timer <= 10:
                # Phase 3: second jump + direction = dodge
                out.jump = True
                out.pitch = self._dodge_pitch
                out.yaw = self._dodge_yaw
            else:
                # Done
                self._dodge_active = False
                self._dodge_timer = 0

            # Keep throttle and boost during dodge
            out.throttle = controls["throttle"]
            out.boost = bool(controls["boost"])
            out.steer = controls["steer"]
        else:
            # Check if controls want a dodge
            wants_jump = bool(controls.get("jump", 0))
            if wants_jump and car.has_wheel_contact:
                self._dodge_active = True
                self._dodge_timer = 0
                self._dodge_pitch = controls["pitch"]
                self._dodge_yaw = 0.0
                out.jump = True
                out.pitch = 0.0
                out.throttle = controls["throttle"]
                out.boost = bool(controls["boost"])
                out.steer = controls["steer"]
            else:
                out.throttle = controls["throttle"]
                out.steer = controls["steer"]
                out.yaw = controls["yaw"]
                out.pitch = controls["pitch"]
                out.boost = bool(controls["boost"])
                out.handbrake = bool(controls["handbrake"])
                out.jump = False

        # ── Fast recovery: orient car for clean landing ──
        if not self._dodge_active and not car.has_wheel_contact:
            car_z = car.physics.location.z
            if car_z > 100:
                car_roll = car.physics.rotation.roll
                car_pitch_angle = car.physics.rotation.pitch
                # Counter-steer roll to land wheels-down
                if abs(car_roll) > 0.3:
                    out.roll = max(-1.0, min(1.0, -car_roll * 2.0))
                # Nose-down if pitched up too much
                if car_pitch_angle > 0.4:
                    out.pitch = -0.5

        # ── Kickoff boost override ──
        if _kickoff_active:
            out.boost = True
            out.throttle = 1.0

        # ── Auto-save knowledge every 5 seconds ──
        t = packet.game_info.seconds_elapsed
        if (not temporary_reset) and (t - self._last_save_time > 5.0):
            self.decision_engine.knowledge.save()
            self.decision_engine.save_rl_data()
            self._last_save_time = t

        # Checkpoint the persistent model periodically.
        if (not temporary_reset) and (t - self._last_checkpoint_time > 30.0):
            self.decision_engine.knowledge.save_checkpoint("auto")
            self._last_checkpoint_time = t

        # Periodic diagnostic — confirm the bot is alive and sending controls
        if self._tick % 600 == 1:
            print(f"[BOT] tick={self._tick} mode={mode} throttle={out.throttle:.1f} "
                  f"steer={out.steer:.1f} boost={out.boost} idx={bot_index}")

        # Store for next-frame friction detection
        self._prev_car_speed = car_speed
        self._prev_throttle = out.throttle

        self._render(packet, mode, temporary_reset)
        return out

    def retire(self):
        self.control_panel.show()
        self._persistent_knowledge.save()
        self._persistent_knowledge.save_checkpoint("retire")
        self.decision_engine.save_rl_data()
        self.decision_engine.adaptive.advanced.on_match_end()


def build_match(settings: dict | None = None) -> MatchConfig:
    """Build a MatchConfig from the settings dict collected by the GUI."""
    if settings is None:
        settings = {}

    cfg = MatchConfig()
    cfg.game_mode = settings.get("game_mode", "Soccer")
    cfg.game_map  = settings.get("arena", "DFHStadium")

    cfg.enable_rendering        = True
    cfg.enable_state_setting    = True
    cfg.networking_role         = "none"
    cfg.network_address         = "127.0.0.1"
    cfg.existing_match_behavior = "Restart If Different"
    cfg.skip_replays            = True

    # ── All mutators wired directly from GUI selections ──────────────────
    m = cfg.mutators
    _VALID_MAX_SCORE = {"Unlimited", "1 Goal", "3 Goals", "5 Goals"}
    m.match_length    = settings.get("match_length",    "5 Minutes")
    _raw_ms = settings.get("max_score", "Unlimited")
    m.max_score       = _raw_ms if _raw_ms in _VALID_MAX_SCORE else "Unlimited"
    m.overtime        = settings.get("overtime",        "Unlimited")
    m.series_length   = settings.get("series_length",   "Unlimited")
    m.game_speed      = settings.get("game_speed",      "Default")
    m.boost_amount    = settings.get("boost_amount",    "Default")
    m.boost_strength  = settings.get("boost_strength",  "1x")
    m.rumble          = settings.get("rumble",          "None")
    m.gravity         = settings.get("gravity",         "Default")
    m.demolish        = settings.get("demolish",        "Default")
    m.respawn_time    = settings.get("respawn_time",    "3 Seconds")
    m.ball_max_speed  = settings.get("ball_max_speed",  "Default")
    m.ball_type       = settings.get("ball_type",       "Default")
    m.ball_weight     = settings.get("ball_weight",     "Default")
    m.ball_size       = settings.get("ball_size",       "Default")
    m.ball_bounciness = settings.get("ball_bounciness", "Default")

    # ── Players ──────────────────────────────────────────────────────────
    n = int(settings.get("team_size", "1v1")[0])   # "2v2" → 2, etc.
    _BOT_SKILL = {"Rookie": 0.0, "Pro": 0.5, "All-Star": 1.0}
    psyonix_skill = _BOT_SKILL.get(settings.get("psyonix_difficulty", "All-Star"), 1.0)

    # Our AI bot (always on team 0)
    ai = PlayerConfig()
    ai.bot = True
    ai.rlbot_controlled = True
    ai.team = 0
    ai.name = "medo dyaa"
    ai.config_path = BOT_CFG
    cfg.player_configs = [ai]

    # Psyonix teammates on team 0 if team size > 1
    for i in range(n - 1):
        ally = PlayerConfig()
        ally.bot = True
        ally.rlbot_controlled = False
        ally.bot_skill = psyonix_skill
        ally.team = 0
        ally.name = f"Ally {i + 1}"
        cfg.player_configs.append(ally)

    # Psyonix opponents on team 1
    for i in range(n):
        opp = PlayerConfig()
        opp.bot = True
        opp.rlbot_controlled = False
        opp.bot_skill = psyonix_skill
        opp.team = 1
        opp.name = f"Psyonix {i + 1}"
        cfg.player_configs.append(opp)

    # Optional live human (joins team 1 as the user's car)
    if settings.get("play_as_human", False):
        human = PlayerConfig()
        human.bot = False
        human.human_index = 0
        human.team = 1
        human.name = ""
        cfg.player_configs.append(human)

    return cfg


def _apply_settings_to_engine(engine: DecisionEngine, settings: dict):
    """Apply the full GUI settings onto the runtime decision engine."""
    _MODE_MAP = {
        "Soccer": GM_SOCCER, "Hoops": GM_HOOPS, "Hockey": GM_SNOW,
        "Dropshot": GM_DROPSHOT, "Rumble": GM_RUMBLE,
        "Heatseeker": GM_HEATSEEKER, "Gridiron": GM_GRIDIRON,
    }
    gs = engine.settings

    # Game mode (override to Rumble if a rumble mutator is active)
    gs.game_mode = _MODE_MAP.get(settings.get("game_mode", "Soccer"), GM_SOCCER)
    if settings.get("rumble", "None") not in ("None",):
        gs.game_mode = GM_RUMBLE
    gs.apply_mode()

    # Gravity mutator → physics gravity value
    _GRAV = {"Default": -650.0, "Low": -325.0, "High": -1300.0, "Super High": -2600.0}
    gs.gravity = _GRAV.get(settings.get("gravity", "Default"), -650.0)
    if gs.gravity > -650.0:                              # lower gravity → more air time
        gs.aerial_bias   += 0.15
        gs.max_aerial_z   = max(gs.max_aerial_z, 650.0)
    elif gs.gravity < -650.0:                            # higher gravity → stay grounded
        gs.aerial_bias   -= 0.10
        gs.max_aerial_z  *= 0.80

    # Ball type → physics adjustments
    ball_t = settings.get("ball_type", "Default")
    if ball_t == "Puck":
        gs.puck_mode     = True
        gs.max_aerial_z  = min(gs.max_aerial_z, 200.0)
    elif ball_t == "Basketball":
        gs.aerial_bias  += 0.30
        gs.max_aerial_z  = max(gs.max_aerial_z, 700.0)
    elif ball_t == "Cube":
        gs.bounce        = 0.7

    # Ball size → affect approach distance bias
    _SIZE_BOOST = {"Small": 0.90, "Default": 1.0, "Large": 1.10, "Gigantic": 1.25}
    gs.boost_aggression *= _SIZE_BOOST.get(settings.get("ball_size", "Default"), 1.0)

    # Game speed → throttle/boost scaling
    if settings.get("game_speed") == "Slo-Mo":
        gs.boost_aggression *= 0.50
        gs.throttle_mult    *= 0.60

    # Difficulty → engine tuning multipliers
    diff = settings.get("difficulty", "Medium")
    if diff == "Easy":
        gs.throttle_mult    *= 0.80
        gs.steer_mult       *= 0.90
        gs.boost_aggression *= 0.70
        gs.max_aerial_z     *= 0.85
    elif diff == "Hard":
        gs.throttle_mult    *= 1.05
        gs.steer_mult       *= 1.05
        gs.boost_aggression *= 1.20
        gs.max_aerial_z     *= 1.10

    # RL tuning sliders from the dedicated RL tab.
    gs.throttle_mult *= _safe_float(settings.get("movement_speed", 1.0), 1.0)
    gs.steer_mult *= _safe_float(settings.get("steering_sensitivity", 1.0), 1.0)
    gs.boost_aggression *= _safe_float(settings.get("boost_aggression_scale", 1.0), 1.0)
    aerial_commitment = _safe_float(settings.get("aerial_commitment", 1.0), 1.0)
    gs.max_aerial_z *= aerial_commitment
    if aerial_commitment > 1.0:
        gs.aerial_bias += min(0.35, (aerial_commitment - 1.0) * 0.5)
    elif aerial_commitment < 1.0:
        gs.aerial_bias -= min(0.25, (1.0 - aerial_commitment) * 0.4)

    # Always run the full ensemble. The user does not pick a single algorithm.
    engine.forced_algorithm = ""

    # ── Search algorithm filter ────────────────────────────────────────────────
    # Resolve which algorithms are active, from the most specific source available.
    _ALL_SEARCH = ["A*", "BFS", "UCS", "DFS", "Greedy", "Decision Tree", "Beam Search", "IDA*"]

    # Dashboard (DashboardApp) sends a list; old SettingsGUI sends comma-string.
    search_list = settings.get("search_algorithms")
    if isinstance(search_list, list) and search_list:
        engine.active_search_algorithms = [a for a in search_list if a in _ALL_SEARCH] or _ALL_SEARCH
    else:
        algo_str = settings.get("active_search_algorithms", "").strip()
        if algo_str:
            active = [a.strip() for a in algo_str.split(",") if a.strip() in _ALL_SEARCH]
            engine.active_search_algorithms = active if active else _ALL_SEARCH
        else:
            # Fall back to the preset name to look up the algo list.
            # _SettingsGUI is defined later in this same module — safe to call at runtime.
            preset_name = settings.get("search_algo_preset", "").strip()
            preset_entry = _SettingsGUI._SEARCH_ALGO_PRESETS.get(preset_name)
            engine.active_search_algorithms = preset_entry[0] if preset_entry else _ALL_SEARCH

    _ALL_RL = {"q_learning", "actor_critic", "online_learner", "dqn", "ppo", "a2c", "monte_carlo", "model_based"}
    rl_list = settings.get("rl_models")
    if isinstance(rl_list, list) and rl_list:
        active_rl = {m for m in rl_list if m in _ALL_RL}
        engine.adaptive.set_active_models(active_rl if active_rl else _ALL_RL)
    else:
        rl_str = settings.get("active_rl_models", "").strip()
        if rl_str:
            active_rl2 = {name.strip() for name in rl_str.split(",") if name.strip() in _ALL_RL}
            engine.adaptive.set_active_models(active_rl2 if active_rl2 else _ALL_RL)
        else:
            engine.adaptive.set_active_models(_ALL_RL)

    _ALL_ADV = {"anomaly", "multi_task", "passive_aggressive", "maml", "deep_rl", "causal"}
    adv_str = settings.get("active_advanced_models", "").strip()
    if adv_str:
        active_adv = {name.strip() for name in adv_str.split(",") if name.strip() in _ALL_ADV}
        engine.adaptive.set_active_advanced_components(active_adv if active_adv else _ALL_ADV)
    else:
        engine.adaptive.set_active_advanced_components(_ALL_ADV)


def _is_rl_running() -> bool:
    """Check if Rocket League is already running."""
    try:
        import subprocess
        output = subprocess.check_output(
            ["tasklist", "/FI", "IMAGENAME eq RocketLeague.exe"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return "RocketLeague.exe" in output
    except Exception:
        return False


def _find_rl_exe() -> str:
    """Return the path to RocketLeague.exe by scanning running processes, common
    install dirs (Epic + Steam), and the Windows registry.  Returns "" if not found."""
    import subprocess, os

    # 1. Check a currently-running RL process (most reliable source of truth).
    try:
        out = subprocess.check_output(
            ["wmic", "process", "where", 'Name="RocketLeague.exe"', "get",
             "ExecutablePath", "/value"],
            text=True, stderr=subprocess.DEVNULL,
        )
        for line in out.splitlines():
            if "ExecutablePath=" in line:
                path = line.split("=", 1)[1].strip()
                if path and os.path.isfile(path):
                    return path
    except Exception:
        pass

    # 2. Common install paths for Epic Games Launcher and Steam on Windows.
    _CANDIDATES = [
        r"C:\Program Files\Epic Games\rocketleague\Binaries\Win64\RocketLeague.exe",
        r"C:\Program Files (x86)\Epic Games\rocketleague\Binaries\Win64\RocketLeague.exe",
        r"C:\Epic Games\rocketleague\Binaries\Win64\RocketLeague.exe",
        r"D:\Epic Games\rocketleague\Binaries\Win64\RocketLeague.exe",
        r"E:\Epic Games\rocketleague\Binaries\Win64\RocketLeague.exe",
        r"F:\Epic Games\rocketleague\Binaries\Win64\RocketLeague.exe",
        r"C:\Program Files (x86)\Steam\steamapps\common\rocketleague\Binaries\Win64\RocketLeague.exe",
        r"C:\Program Files\Steam\steamapps\common\rocketleague\Binaries\Win64\RocketLeague.exe",
        r"D:\Steam\steamapps\common\rocketleague\Binaries\Win64\RocketLeague.exe",
        r"D:\SteamLibrary\steamapps\common\rocketleague\Binaries\Win64\RocketLeague.exe",
        r"E:\Steam\steamapps\common\rocketleague\Binaries\Win64\RocketLeague.exe",
        r"E:\SteamLibrary\steamapps\common\rocketleague\Binaries\Win64\RocketLeague.exe",
    ]
    for path in _CANDIDATES:
        if os.path.isfile(path):
            return path

    # 3. Search Windows registry for Epic Games install location.
    try:
        import winreg
        for root_key in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
            try:
                key = winreg.OpenKey(
                    root_key,
                    r"SOFTWARE\EpicGames\Unreal Engine\rocketleague",
                )
                install_dir, _ = winreg.QueryValueEx(key, "InstalledDirectory")
                candidate = os.path.join(
                    install_dir, "Binaries", "Win64", "RocketLeague.exe"
                )
                if os.path.isfile(candidate):
                    return candidate
            except Exception:
                pass
    except Exception:
        pass

    return ""


def _has_internet(host: str = "8.8.8.8", port: int = 53, timeout: float = 2.0) -> bool:
    """Return True when an outbound TCP connection to *host:port* succeeds within *timeout* s."""
    import socket
    try:
        socket.setdefaulttimeout(timeout)
        with socket.create_connection((host, port)):
            return True
    except Exception:
        return False



    try:
        return int(str(value).strip())
    except Exception:
        return default


def _safe_int(value, default: int) -> int:
    try:
        return int(str(value).strip())
    except Exception:
        return default


def _safe_float(value, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _apply_looks_settings(settings: dict):
    import configparser

    looks_path = os.path.join(RUNTIME_DIR, "rl_ai_looks.cfg")
    cfg = configparser.ConfigParser()
    if os.path.exists(looks_path):
        cfg.read(looks_path, encoding="utf-8")

    blue = "Bot Loadout"
    orange = "Bot Loadout Orange"
    for section in (blue, orange):
        if not cfg.has_section(section):
            cfg.add_section(section)

    car_preset_map = {
        "Octane": {
            "car_id": 23, "decal_id": 307, "wheels_id": 1568,
            "boost_id": 32, "paint_finish_id": 1978,
            "custom_finish_id": 1978, "goal_explosion_id": 1901,
        },
        "Dominus": {
            "car_id": 403, "decal_id": 508, "wheels_id": 1568,
            "boost_id": 32, "paint_finish_id": 1978,
            "custom_finish_id": 1978, "goal_explosion_id": 1901,
        },
        "Breakout": {
            "car_id": 22, "decal_id": 300, "wheels_id": 1568,
            "boost_id": 32, "paint_finish_id": 1978,
            "custom_finish_id": 1978, "goal_explosion_id": 1901,
        },
        "Merc": {
            "car_id": 30, "decal_id": 350, "wheels_id": 1568,
            "boost_id": 32, "paint_finish_id": 1978,
            "custom_finish_id": 1978, "goal_explosion_id": 1901,
        },
        "Paladin": {
            "car_id": 24, "decal_id": 314, "wheels_id": 1568,
            "boost_id": 32, "paint_finish_id": 1978,
            "custom_finish_id": 1978, "goal_explosion_id": 1901,
        },
    }
    decal_preset_map = {
        "Default Style": {
            "Octane": 307,
            "Dominus": 508,
            "Breakout": 300,
            "Merc": 350,
            "Paladin": 314,
        },
        "Flames": {
            "Octane": 302,
            "Dominus": 504,
            "Breakout": 295,
            "Merc": 345,
            "Paladin": 309,
        },
        "Wings": {
            "Octane": 308,
            "Dominus": 510,
            "Breakout": 301,
            "Merc": 350,
            "Paladin": 315,
        },
    }
    wheel_preset_map = {
        "Bender": 360,
        "Almas": 364,
        "Mountaineer": 375,
        "Vortex": 381,
    }
    boost_preset_map = {
        "Gold Rush": 32,
        "Flamethrower Red": 41,
        "Thermal Yellow": 58,
        "Standard": 63,
    }
    goal_explosion_map = {
        "Classic": 1903,
        "Fireworks": 1905,
        "Hellfire": 1908,
        "Dueling Dragons": 2044,
        "Ballistic": 2791,
        "Atomizer": 2817,
    }
    finish_preset_map = {
        "Brushed Metal": {
            "paint_finish_id": 1978,
            "custom_finish_id": 1978,
        },
        "Glossy": {
            "paint_finish_id": 1978,
            "custom_finish_id": 0,
        },
        "Matte": {
            "paint_finish_id": 0,
            "custom_finish_id": 0,
        },
        "Metallic Accent": {
            "paint_finish_id": 1978,
            "custom_finish_id": 1978,
        },
    }
    team_color_map = {
        "Blue Default": 27,
        "Blue Esports": 35,
        "Orange Default": 32,
        "Orange Esports": 33,
    }
    custom_color_map = {
        "No Accent": 0,
    }

    shared_defaults = {
        "car_id": 23,
        "decal_id": 307,
        "wheels_id": 1568,
        "boost_id": 32,
        "paint_finish_id": 1978,
        "custom_finish_id": 1978,
        "goal_explosion_id": 1901,
    }

    def _apply_team_loadout(section: str, team: str, default_body: str, default_color: str):
        preset_name = str(settings.get(f"{team}_car_preset", settings.get("car_preset", default_body)))
        preset = car_preset_map.get(preset_name, car_preset_map[default_body])
        decal_name = str(settings.get(f"{team}_decal_preset", settings.get("decal_preset", "Default Style")))
        wheel_name = str(settings.get(f"{team}_wheel_preset", settings.get("wheel_preset", "Bender")))
        boost_name = str(settings.get(f"{team}_boost_preset", settings.get("boost_preset", "Gold Rush")))
        goal_name = str(settings.get(f"{team}_goal_explosion_preset", settings.get("goal_explosion_preset", "Classic")))
        finish_name = str(settings.get(f"{team}_finish_preset", settings.get("finish_preset", "Brushed Metal")))

        selected_decal = decal_preset_map.get(decal_name, decal_preset_map["Default Style"])
        selected_wheels = wheel_preset_map.get(wheel_name, wheel_preset_map["Bender"])
        selected_boost = boost_preset_map.get(boost_name, boost_preset_map["Gold Rush"])
        selected_goal = goal_explosion_map.get(goal_name, goal_explosion_map["Classic"])
        selected_finish = finish_preset_map.get(finish_name, finish_preset_map["Brushed Metal"])

        for key, default in shared_defaults.items():
            resolved_default = preset.get(key, default)
            if key == "decal_id":
                resolved_default = selected_decal.get(preset_name, resolved_default)
            elif key == "wheels_id":
                resolved_default = selected_wheels
            elif key == "boost_id":
                resolved_default = selected_boost
            elif key == "goal_explosion_id":
                resolved_default = selected_goal
            elif key in ("paint_finish_id", "custom_finish_id"):
                resolved_default = selected_finish.get(key, resolved_default)
            value = _safe_int(settings.get(f"{team}_{key}", settings.get(key, resolved_default)), resolved_default)
            cfg.set(section, key, str(value))

        team_color_name = str(settings.get(f"team_color_{team}", default_color))
        accent_name = str(settings.get(f"custom_color_{team}", "No Accent"))
        cfg.set(section, "team_color_id", str(team_color_map.get(team_color_name, team_color_map[default_color])))
        cfg.set(section, "custom_color_id", str(custom_color_map.get(accent_name, 0)))

    _apply_team_loadout(blue, "blue", "Octane", "Blue Default")
    _apply_team_loadout(orange, "orange", "Octane", "Orange Default")

    for key in ("antenna_id", "hat_id", "engine_audio_id", "trails_id"):
        if not cfg.has_option(blue, key):
            cfg.set(blue, key, "0")
        if not cfg.has_option(orange, key):
            cfg.set(orange, key, "0")

    with open(looks_path, "w", encoding="utf-8") as handle:
        cfg.write(handle)


def _wait_for_user_match():
    """Block until the user starts a match (at least one car visible in packets)."""
    from rlbot.socket.socket_data_reporter import SocketDataReporter
    reporter = SocketDataReporter()
    try:
        waited = 0
        while True:
            packet = reporter.latest_packet
            if packet is not None:
                num_players = packet.PlayersLength()
                game_info = packet.GameInfo()
                is_ended = game_info.IsMatchEnded() if game_info else True
                if num_players > 0 and not is_ended:
                    print(f"[INFO] Match detected \u2014 {num_players} player(s) in game. Injecting bot...")
                    time.sleep(2.0)  # let the match fully settle
                    return
            time.sleep(1.0)
            waited += 1
            if waited % 30 == 0:
                print("[INFO] Still waiting \u2014 start any match from inside Rocket League...")
    finally:
        reporter.disconnect()


class _SettingsGUI:
    """Full match-settings window — mirrors RL's Exhibition Match + Mutators screens.

    All 16 RLBot mutators are exposed as drop-down selectors, plus game mode,
    arena, team size, AI difficulty, model choice, and a 'play as human' toggle.
    The window persists across matches; after a match ends it re-opens for the
    next game.
    """

    _BG      = "#0A1520"
    _FG      = "#DDEEFF"
    _SEC     = "#7FDBFF"
    _CB_BG   = "#162230"
    _HL      = "#2ECC40"

    _SOCCER_ARENAS = [
        ("DFH Stadium",       "DFHStadium"),
        ("DFH Stadium (Stormy)", "DFHStadium_Stormy"),
        ("Mannfield",         "Mannfield"),
        ("Mannfield (Night)", "Mannfield_Night"),
        ("Champions Field",   "ChampionsField"),
        ("Urban Central",     "UrbanCentral"),
        ("Beckwith Park",     "BeckwithPark"),
        ("Beckwith Park (Stormy)", "BeckwithPark_Stormy"),
        ("Utopia Coliseum",   "UtopiaColiseum"),
        ("Wasteland",         "wasteland"),
        ("Neo Tokyo",         "NeoTokyo"),
        ("Aqua Dome",         "AquaDome"),
        ("Farmstead",         "Farmstead"),
        ("Salty Shores",      "SaltyShores"),
        ("Forbidden Temple",  "ForbiddenTemple"),
        ("Deadeye Canyon",    "DeadeyeCanyon"),
        ("Estadio Vida",      "EstadioVida"),
        ("Sovereign Heights", "SovereignHeights"),
        ("Throwback Stadium", "ThrowbackStadium"),
        ("Starbase Arc",      "StarbaseArc"),
    ]

    _ARENAS_BY_MODE = {
        "Soccer":     _SOCCER_ARENAS,
        "Rumble":     _SOCCER_ARENAS,
        "Heatseeker": _SOCCER_ARENAS,
        "Gridiron":   _SOCCER_ARENAS,
        "Hoops":      [("Dunk House", "Park_P"),
                       ("Dunk House Rainy", "Park_Rainy_P")],
        "Hockey":     [("Urban Central Snow", "TrainStation_P"),
                       ("Ice Dome", "IceDome")],
        "Dropshot":   [("Core 707", "ShatterShot_P")],
    }

    _MUTATOR_OPTS = {
        "match_length":    ["5 Minutes", "10 Minutes", "20 Minutes", "Unlimited"],
        "max_score":       ["Unlimited", "1 Goal", "3 Goals", "5 Goals"],
        "overtime":        ["Unlimited", "+5 Max, First Score", "+5 Max, Random Team"],
        "series_length":   ["Unlimited", "3 Games", "5 Games", "7 Games"],
        "game_speed":      ["Default", "Slo-Mo", "Time Warp"],
        "boost_amount":    ["Default", "Unlimited", "Recharge (Slow)",
                            "Recharge (Fast)", "No Boost"],
        "boost_strength":  ["1x", "1.5x", "2x", "10x"],
        "rumble":          ["None", "Default", "Slow", "Civilized",
                            "Destruction Derby", "Spring Loaded",
                            "Spikes Only", "Spike Rush"],
        "gravity":         ["Default", "Low", "High", "Super High"],
        "demolish":        ["Default", "Disabled", "Friendly Fire",
                            "On Contact", "On Contact (FF)"],
        "respawn_time":    ["3 Seconds", "2 Seconds", "1 Second",
                            "Disable Goal Reset"],
        "ball_max_speed":  ["Default", "Slow", "Fast", "Super Fast"],
        "ball_type":       ["Default", "Cube", "Puck", "Basketball"],
        "ball_weight":     ["Default", "Light", "Heavy", "Super Light"],
        "ball_size":       ["Default", "Small", "Large", "Gigantic"],
        "ball_bounciness": ["Default", "Low", "High", "Super High"],
    }

    _CAR_PRESETS = {
        "Octane": {
            "car_id": 23, "decal_id": 307, "wheels_id": 1568,
            "boost_id": 32, "paint_finish_id": 1978,
            "custom_finish_id": 1978, "goal_explosion_id": 1901,
        },
        "Dominus": {
            "car_id": 403, "decal_id": 508, "wheels_id": 1568,
            "boost_id": 32, "paint_finish_id": 1978,
            "custom_finish_id": 1978, "goal_explosion_id": 1901,
        },
        "Breakout": {
            "car_id": 22, "decal_id": 300, "wheels_id": 1568,
            "boost_id": 32, "paint_finish_id": 1978,
            "custom_finish_id": 1978, "goal_explosion_id": 1901,
        },
        "Merc": {
            "car_id": 30, "decal_id": 350, "wheels_id": 1568,
            "boost_id": 32, "paint_finish_id": 1978,
            "custom_finish_id": 1978, "goal_explosion_id": 1901,
        },
        "Paladin": {
            "car_id": 24, "decal_id": 314, "wheels_id": 1568,
            "boost_id": 32, "paint_finish_id": 1978,
            "custom_finish_id": 1978, "goal_explosion_id": 1901,
        },
    }

    _DECAL_PRESETS = {
        "Default Style": {
            "Octane": 307,
            "Dominus": 508,
            "Breakout": 300,
            "Merc": 350,
            "Paladin": 314,
        },
        "Flames": {
            "Octane": 302,
            "Dominus": 504,
            "Breakout": 295,
            "Merc": 345,
            "Paladin": 309,
        },
        "Wings": {
            "Octane": 308,
            "Dominus": 510,
            "Breakout": 301,
            "Merc": 350,
            "Paladin": 315,
        },
    }

    _WHEEL_PRESETS = {
        "Bender": 360,
        "Almas": 364,
        "Mountaineer": 375,
        "Vortex": 381,
    }

    _BOOST_PRESETS = {
        "Gold Rush": 32,
        "Flamethrower Red": 41,
        "Thermal Yellow": 58,
        "Standard": 63,
    }

    _GOAL_EXPLOSION_PRESETS = {
        "Classic": 1903,
        "Fireworks": 1905,
        "Hellfire": 1908,
        "Dueling Dragons": 2044,
        "Ballistic": 2791,
        "Atomizer": 2817,
    }

    _FINISH_PRESETS = {
        "Brushed Metal": {
            "paint_finish_id": 1978,
            "custom_finish_id": 1978,
        },
        "Glossy": {
            "paint_finish_id": 1978,
            "custom_finish_id": 0,
        },
        "Matte": {
            "paint_finish_id": 0,
            "custom_finish_id": 0,
        },
        "Metallic Accent": {
            "paint_finish_id": 1978,
            "custom_finish_id": 1978,
        },
    }

    _TEAM_COLOR_PRESETS = {
        "Blue Default": 27,
        "Blue Esports": 35,
        "Orange Default": 32,
        "Orange Esports": 33,
    }

    _CUSTOM_COLOR_PRESETS = {
        "No Accent": 0,
    }

    _RL_TUNING_PRESETS = {
        "Custom": None,
        "Safe": {
            "movement_speed": 0.85,
            "steering_sensitivity": 0.9,
            "boost_aggression_scale": 0.75,
            "aerial_commitment": 0.8,
        },
        "Balanced": {
            "movement_speed": 1.0,
            "steering_sensitivity": 1.0,
            "boost_aggression_scale": 1.0,
            "aerial_commitment": 1.0,
        },
        "Aggressive": {
            "movement_speed": 1.15,
            "steering_sensitivity": 1.1,
            "boost_aggression_scale": 1.25,
            "aerial_commitment": 1.05,
        },
        "Aerial": {
            "movement_speed": 1.0,
            "steering_sensitivity": 1.05,
            "boost_aggression_scale": 1.1,
            "aerial_commitment": 1.3,
        },
    }

    # Search algorithm presets.
    # Each entry: (algorithm list, pros string, cons string)
    _SEARCH_ALGO_PRESETS = {
        "Custom": (
            ["A*", "BFS", "UCS", "DFS", "Greedy", "Decision Tree", "Beam Search", "IDA*"],
            "Choose your own combination of algorithms.",
            "",
        ),
        "Optimal (A* + UCS)": (
            ["A*", "UCS"],
            "Guaranteed shortest / lowest-cost paths. Great for precision positioning.",
            "Slower per tick. May feel less reactive at high ball speed.",
        ),
        "Fast (Greedy + Beam Search)": (
            ["Greedy", "Beam Search"],
            "Fastest decision time. Low CPU usage. Very responsive.",
            "May not find the globally optimal path. Weaker on defence.",
        ),
        "Balanced ⭐ (A* + BFS + Greedy)": (
            ["A*", "BFS", "Greedy"],
            "Best speed / quality trade-off. Recommended for most play styles.",
            "Skips UCS cost guarantees and IDA* intercept prediction.",
        ),
        "Thorough (A* + BFS + DFS + UCS)": (
            ["A*", "BFS", "DFS", "UCS"],
            "Very complete search coverage. Safer on complex or cramped fields.",
            "Higher CPU per tick. DFS can return long suboptimal paths occasionally.",
        ),
        "Full Ensemble (all 8)": (
            ["A*", "BFS", "UCS", "DFS", "Greedy", "Decision Tree", "Beam Search", "IDA*"],
            "Maximum information. Auto-adapts to any situation. A.I. picks the winner.",
            "Highest CPU usage. Car can feel slightly less snappy on low-end machines.",
        ),
    }

    _QUICK_THEMES = {
        "Custom": None,
        "Esports Blue": {
            "blue_car_preset": "Octane",
            "blue_decal_preset": "Default Style",
            "blue_wheel_preset": "Vortex",
            "blue_boost_preset": "Standard",
            "blue_goal_explosion_preset": "Ballistic",
            "blue_finish_preset": "Metallic Accent",
            "team_color_blue": "Blue Esports",
            "custom_color_blue": "No Accent",
            "orange_car_preset": "Dominus",
            "orange_decal_preset": "Default Style",
            "orange_wheel_preset": "Almas",
            "orange_boost_preset": "Flamethrower Red",
            "orange_goal_explosion_preset": "Fireworks",
            "orange_finish_preset": "Glossy",
            "team_color_orange": "Orange Esports",
            "custom_color_orange": "No Accent",
        },
        "Flame Attack": {
            "blue_car_preset": "Breakout",
            "blue_decal_preset": "Wings",
            "blue_wheel_preset": "Vortex",
            "blue_boost_preset": "Thermal Yellow",
            "blue_goal_explosion_preset": "Ballistic",
            "blue_finish_preset": "Brushed Metal",
            "team_color_blue": "Blue Default",
            "custom_color_blue": "No Accent",
            "orange_car_preset": "Dominus",
            "orange_decal_preset": "Flames",
            "orange_wheel_preset": "Almas",
            "orange_boost_preset": "Flamethrower Red",
            "orange_goal_explosion_preset": "Fireworks",
            "orange_finish_preset": "Glossy",
            "team_color_orange": "Orange Default",
            "custom_color_orange": "No Accent",
        },
        "Dark Merc": {
            "blue_car_preset": "Merc",
            "blue_decal_preset": "Wings",
            "blue_wheel_preset": "Mountaineer",
            "blue_boost_preset": "Gold Rush",
            "blue_goal_explosion_preset": "Hellfire",
            "blue_finish_preset": "Matte",
            "team_color_blue": "Blue Default",
            "custom_color_blue": "No Accent",
            "orange_car_preset": "Merc",
            "orange_decal_preset": "Default Style",
            "orange_wheel_preset": "Bender",
            "orange_boost_preset": "Standard",
            "orange_goal_explosion_preset": "Atomizer",
            "orange_finish_preset": "Matte",
            "team_color_orange": "Orange Esports",
            "custom_color_orange": "No Accent",
        },
        "Royal Duel": {
            "blue_car_preset": "Octane",
            "blue_decal_preset": "Wings",
            "blue_wheel_preset": "Vortex",
            "blue_boost_preset": "Standard",
            "blue_goal_explosion_preset": "Dueling Dragons",
            "blue_finish_preset": "Metallic Accent",
            "team_color_blue": "Blue Esports",
            "custom_color_blue": "No Accent",
            "orange_car_preset": "Paladin",
            "orange_decal_preset": "Flames",
            "orange_wheel_preset": "Almas",
            "orange_boost_preset": "Gold Rush",
            "orange_goal_explosion_preset": "Hellfire",
            "orange_finish_preset": "Glossy",
            "team_color_orange": "Orange Esports",
            "custom_color_orange": "No Accent",
        },
        "Neon Nights": {
            "blue_car_preset": "Breakout",
            "blue_decal_preset": "Default Style",
            "blue_wheel_preset": "Bender",
            "blue_boost_preset": "Standard",
            "blue_goal_explosion_preset": "Atomizer",
            "blue_finish_preset": "Glossy",
            "team_color_blue": "Blue Esports",
            "custom_color_blue": "No Accent",
            "orange_car_preset": "Dominus",
            "orange_decal_preset": "Wings",
            "orange_wheel_preset": "Mountaineer",
            "orange_boost_preset": "Thermal Yellow",
            "orange_goal_explosion_preset": "Fireworks",
            "orange_finish_preset": "Metallic Accent",
            "team_color_orange": "Orange Default",
            "custom_color_orange": "No Accent",
        },
        "Turbo Rivals": {
            "blue_car_preset": "Paladin",
            "blue_decal_preset": "Flames",
            "blue_wheel_preset": "Almas",
            "blue_boost_preset": "Thermal Yellow",
            "blue_goal_explosion_preset": "Ballistic",
            "blue_finish_preset": "Brushed Metal",
            "team_color_blue": "Blue Default",
            "custom_color_blue": "No Accent",
            "orange_car_preset": "Octane",
            "orange_decal_preset": "Flames",
            "orange_wheel_preset": "Vortex",
            "orange_boost_preset": "Flamethrower Red",
            "orange_goal_explosion_preset": "Dueling Dragons",
            "orange_finish_preset": "Metallic Accent",
            "team_color_orange": "Orange Esports",
            "custom_color_orange": "No Accent",
        },
    }

    def __init__(self):
        self._initial_values = _load_ui_preferences()
        self._result     = None
        self._root       = None
        self._signal_var = None
        self._vars: dict = {}
        self._comboboxes: dict = {}
        self._row_options: dict = {}
        self._car_preview_loadout_var = None
        self._car_preview_blue_var = None
        self._car_preview_orange_var = None
        self._car_preview_blue_label = None
        self._car_preview_orange_label = None
        self._car_preview_canvas = None
        self._car_search_var = None
        self._applying_theme = False
        self._applying_rl_preset = False
        self._apply_theme_blue_button = None
        self._apply_theme_orange_button = None

    # ── public ──────────────────────────────────────────────────────────────
    def ask(self) -> dict | None:
        """Show the window and block until Start or close. Returns settings dict."""
        self._result = None
        if self._root is None:
            self._build_window()
        self._signal_var.set("waiting")
        self._root.deiconify()
        self._root.lift()
        self._root.attributes("-topmost", True)
        self._root.after(300, lambda: self._root and self._root.attributes("-topmost", False))
        self._root.wait_variable(self._signal_var)
        return self._result

    def destroy(self):
        if self._root:
            try:
                self._root.destroy()
            except Exception:
                pass
        self._root = None
        self._signal_var = None

    # ── internal ─────────────────────────────────────────────────────────────
    def _build_window(self):
        import tkinter as tk
        from tkinter import ttk

        BG = self._BG
        root = tk.Tk()
        self._root = root
        self._signal_var = tk.StringVar(master=root, value="idle")

        root.title("medo dyaa  ·  Match Settings  ·  إعدادات الماتش")
        W, H = 760, 860
        root.geometry(f"{W}x{H}")
        root.resizable(False, False)
        root.configure(bg=BG)
        root.update_idletasks()
        sx = (root.winfo_screenwidth()  - W) // 2
        sy = (root.winfo_screenheight() - H) // 2
        root.geometry(f"{W}x{H}+{sx}+{sy}")

        # Dark dropdown lists
        root.option_add("*TCombobox*Listbox.background",      self._CB_BG)
        root.option_add("*TCombobox*Listbox.foreground",      self._FG)
        root.option_add("*TCombobox*Listbox.selectBackground", self._HL)
        root.option_add("*TCombobox*Listbox.selectForeground", "white")

        # ttk styles
        st = ttk.Style()
        st.theme_use("clam")
        st.configure("NB.TNotebook",
                     background=BG, borderwidth=0)
        st.configure("NB.TNotebook.Tab",
                     background="#162230", foreground=self._SEC,
                     font=("Segoe UI", 10, "bold"), padding=[14, 6])
        st.map("NB.TNotebook.Tab",
               background=[("selected", self._HL)],
               foreground=[("selected", "white")])
        st.configure("D.TFrame",    background=BG)
        st.configure("D.TCombobox",
                     fieldbackground=self._CB_BG, background=self._CB_BG,
                     foreground=self._FG, arrowcolor=self._SEC,
                     selectbackground=self._HL, selectforeground="white")
        st.map("D.TCombobox",
               fieldbackground=[("readonly", self._CB_BG)],
               foreground=[("readonly", self._FG)])

        # Title
        tk.Label(root, text="MATCH SETTINGS  ·  إعدادات الماتش",
                 fg=self._SEC, bg=BG,
                 font=("Segoe UI", 16, "bold")).pack(pady=(12, 4))

        # Notebook tabs
        nb = ttk.Notebook(root, style="NB.TNotebook")
        nb.pack(fill="both", expand=True, padx=12, pady=2)

        for label, builder in [
            ("MATCH  ⚽",  self._tab_match),
            ("GAME  🎮",  self._tab_game),
            ("BALL  🔵",  self._tab_ball),
            ("CAR  🚗",   self._tab_car),
            ("RL  🧠",    self._tab_rl),
            ("BOT  🤖",   self._tab_bot),
        ]:
            self._add_scrollable_tab(nb, label, builder)

        # Always-visible Start button
        tk.Button(
            root,
            text="▶   ابدأ الماتش   —   START MATCH",
            bg=self._HL, fg="white",
            font=("Segoe UI", 14, "bold"),
            relief="flat", cursor="hand2", height=2,
            command=self._on_start,
        ).pack(fill="x", padx=20, pady=(4, 14))

        root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── helpers ──────────────────────────────────────────────────────────────
    def _sec(self, parent, text):
        import tkinter as tk
        tk.Label(parent, text=text, bg=self._BG, fg=self._SEC,
                 font=("Segoe UI", 10, "bold"),
                 anchor="w").pack(fill="x", padx=20, pady=(10, 1))

    def _row(self, parent, label, key, opts, default=None, autocomplete=False):
        import tkinter as tk
        from tkinter import ttk
        fr = tk.Frame(parent, bg=self._BG)
        fr.pack(fill="x", padx=22, pady=3)
        tk.Label(fr, text=label, bg=self._BG, fg="#99BBCC",
                 font=("Segoe UI", 9), width=24, anchor="w").pack(side="left")
        initial = self._initial_values.get(key, default or (opts[0] if opts else ""))
        if opts and initial not in opts:
            initial = default or opts[0]
        var = tk.StringVar(value=initial)
        combo = ttk.Combobox(fr, textvariable=var, values=opts,
                             state="normal" if autocomplete else "readonly", width=28,
                             style="D.TCombobox")
        combo.pack(side="left")
        self._vars[key] = var
        self._comboboxes[key] = combo
        self._row_options[key] = list(opts)
        if autocomplete:
            self._bind_combobox_autocomplete(key)

    def _bind_combobox_autocomplete(self, key: str):
        combo = self._comboboxes.get(key)
        if combo is None:
            return
        combo.bind("<KeyRelease>", lambda _event, combo_key=key: self._on_combobox_key_release(combo_key))
        combo.bind("<FocusOut>", lambda _event, combo_key=key: self._normalize_combobox_value(combo_key))

    def _apply_combobox_autocomplete(self, key: str, typed_text: str, commit_first: bool = False):
        combo = self._comboboxes.get(key)
        all_options = self._row_options.get(key)
        combo_var = self._vars.get(key)
        if combo is None or all_options is None or combo_var is None:
            return

        query = typed_text.strip().lower()
        if not query:
            combo["values"] = all_options
            return

        matches = [option for option in all_options if query in option.lower()]
        combo["values"] = matches or all_options
        if matches and commit_first:
            combo_var.set(matches[0])
        try:
            combo.event_generate("<Down>")
        except Exception:
            pass

    def _on_combobox_key_release(self, key: str):
        combo = self._comboboxes.get(key)
        if combo is None:
            return
        self._apply_combobox_autocomplete(key, combo.get())

    def _normalize_combobox_value(self, key: str):
        combo_var = self._vars.get(key)
        all_options = self._row_options.get(key)
        if combo_var is None or not all_options:
            return
        typed = combo_var.get().strip()
        if not typed:
            combo_var.set(all_options[0])
            return
        exact = next((option for option in all_options if option.lower() == typed.lower()), None)
        if exact is not None:
            combo_var.set(exact)
            return
        partial = next((option for option in all_options if typed.lower() in option.lower()), None)
        if partial is not None:
            combo_var.set(partial)

    def _slider_row(self, parent, label, key, min_value, max_value, default, resolution=0.05):
        import tkinter as tk
        fr = tk.Frame(parent, bg=self._BG)
        fr.pack(fill="x", padx=22, pady=5)
        tk.Label(fr, text=label, bg=self._BG, fg="#99BBCC",
                 font=("Segoe UI", 9), width=24, anchor="w").pack(side="left")
        var = tk.DoubleVar(value=default)
        scale = tk.Scale(
            fr,
            from_=min_value,
            to=max_value,
            resolution=resolution,
            orient="horizontal",
            variable=var,
            length=240,
            bg=self._BG,
            fg=self._FG,
            highlightthickness=0,
            troughcolor=self._CB_BG,
            activebackground=self._HL,
            font=("Segoe UI", 8),
        )
        scale.pack(side="left", padx=(0, 8))
        tk.Label(fr, textvariable=var, bg=self._BG, fg="#DDEEFF",
                 font=("Segoe UI", 9, "bold"), width=5, anchor="w").pack(side="left")
        self._vars[key] = var

    def _add_scrollable_tab(self, notebook, label, builder):
        import tkinter as tk
        from tkinter import ttk

        outer = ttk.Frame(notebook, style="D.TFrame")
        notebook.add(outer, text=label)

        canvas = tk.Canvas(
            outer,
            bg=self._BG,
            bd=0,
            highlightthickness=0,
            yscrollincrement=18,
        )
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        content = tk.Frame(canvas, bg=self._BG)

        content_window = canvas.create_window((0, 0), window=content, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        def _sync_scroll_region(_event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _fit_content_width(event):
            canvas.itemconfigure(content_window, width=event.width)

        def _on_mousewheel(event):
            delta = event.delta
            if delta == 0:
                return
            canvas.yview_scroll(-int(delta / 120), "units")

        content.bind("<Configure>", _sync_scroll_region)
        canvas.bind("<Configure>", _fit_content_width)
        canvas.bind("<Enter>", lambda _e: canvas.bind_all("<MouseWheel>", _on_mousewheel))
        canvas.bind("<Leave>", lambda _e: canvas.unbind_all("<MouseWheel>"))

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        builder(content)
        _sync_scroll_region()

    def _install_preview_hooks(self, keys):
        for key in keys:
            var = self._vars.get(key)
            if var is not None:
                var.trace_add("write", lambda *_args, changed_key=key: self._on_car_setting_changed(changed_key))

    def _set_var_value(self, key: str, value: str):
        var = self._vars.get(key)
        if var is not None and var.get() != value:
            var.set(value)

    def _set_quick_theme_custom(self):
        quick_theme_var = self._vars.get("quick_theme")
        if quick_theme_var is not None and quick_theme_var.get() != "Custom":
            quick_theme_var.set("Custom")

    @staticmethod
    def _team_theme_keys(team: str):
        return (
            f"{team}_car_preset",
            f"{team}_decal_preset",
            f"{team}_wheel_preset",
            f"{team}_boost_preset",
            f"{team}_goal_explosion_preset",
            f"{team}_finish_preset",
            f"team_color_{team}",
            f"custom_color_{team}",
        )

    def _update_theme_action_buttons(self):
        enabled = bool(self._vars.get("quick_theme")) and self._vars["quick_theme"].get() != "Custom"
        state = "normal" if enabled else "disabled"
        if self._apply_theme_blue_button is not None:
            self._apply_theme_blue_button.configure(state=state)
        if self._apply_theme_orange_button is not None:
            self._apply_theme_orange_button.configure(state=state)

    def _refresh_car_option_filters(self, *_):
        query = self._car_search_var.get().strip().lower() if self._car_search_var is not None else ""
        filter_keys = [
            "quick_theme",
            "blue_car_preset", "blue_decal_preset", "blue_wheel_preset", "blue_boost_preset",
            "blue_goal_explosion_preset", "blue_finish_preset",
            "orange_car_preset", "orange_decal_preset", "orange_wheel_preset", "orange_boost_preset",
            "orange_goal_explosion_preset", "orange_finish_preset",
            "team_color_blue", "team_color_orange", "custom_color_blue", "custom_color_orange",
        ]
        for key in filter_keys:
            combo = self._comboboxes.get(key)
            all_options = self._row_options.get(key)
            if combo is None or all_options is None:
                continue
            if not query:
                combo["values"] = all_options
                continue
            filtered = [option for option in all_options if query in option.lower()]
            combo["values"] = filtered or all_options

    def _apply_arena_autocomplete(self, typed_text: str, commit_first: bool = False):
        self._apply_combobox_autocomplete("arena_display", typed_text, commit_first)

    def _on_arena_key_release(self, event=None):
        self._on_combobox_key_release("arena_display")

    def _normalize_arena_display_value(self):
        self._normalize_combobox_value("arena_display")

    def _set_rl_tuning_preset_custom(self):
        preset_var = self._vars.get("rl_tuning_preset")
        if preset_var is not None and preset_var.get() != "Custom":
            preset_var.set("Custom")

    def _on_rl_slider_changed(self, changed_key: str):
        if not self._applying_rl_preset and changed_key != "rl_tuning_preset":
            preset_var = self._vars.get("rl_tuning_preset")
            if preset_var is not None and preset_var.get() != "Custom":
                self._set_rl_tuning_preset_custom()

    def _apply_rl_tuning_preset(self, *_):
        if self._applying_rl_preset:
            return
        preset_var = self._vars.get("rl_tuning_preset")
        preset_name = preset_var.get() if preset_var is not None else "Balanced"
        preset = self._RL_TUNING_PRESETS.get(preset_name)
        if not preset:
            return
        self._applying_rl_preset = True
        try:
            for key, value in preset.items():
                self._set_var_value(key, value)
        finally:
            self._applying_rl_preset = False

    def _show_search_algo_selector(self):
        """Open the search algorithm selection dialog from the RL tab."""
        import tkinter as tk
        BG = "#101820"
        FG = "#DDEEFF"

        # Determine currently-selected preset / active algorithms
        preset_var = self._vars.get("search_algo_preset")
        current_preset = preset_var.get() if preset_var is not None else "Balanced ⭐ (A* + BFS + Greedy)"
        preset_entry = self._SEARCH_ALGO_PRESETS.get(
            current_preset, self._SEARCH_ALGO_PRESETS["Full Ensemble (all 8)"]
        )
        current_algos, _, _ = preset_entry

        all_algos = ["A*", "BFS", "UCS", "DFS", "Greedy", "Decision Tree", "Beam Search", "IDA*"]

        _ALGO_INFO = {
            "A*":            ("A* (A-Star)",      "Optimal & complete. Balances cost + heuristic. Best all-rounder."),
            "BFS":           ("BFS",              "Explores all neighbours level by level. Great for wide coverage."),
            "UCS":           ("UCS",              "Finds minimum-cost path. Safe, defensive positioning."),
            "DFS":           ("DFS",              "Aggressive deep search. Creative but sometimes suboptimal."),
            "Greedy":        ("Greedy",           "Fastest — always chases the best-looking move. Low CPU."),
            "Decision Tree": ("Decision Tree",    "Context-aware rule tree. Reads the situation explicitly."),
            "Beam Search":   ("Beam Search",      "Long look-ahead. Best for planning ahead during open plays."),
            "IDA*":          ("IDA*",             "Memory-efficient optimal. Excellent for intercept prediction."),
        }

        win = tk.Toplevel()
        win.title("Search Algorithm Selector / اختر خوارزميات البحث")
        win.geometry("520x620")
        win.configure(bg=BG)
        win.resizable(False, False)
        win.attributes("-topmost", True)
        win.grab_set()

        tk.Label(win, text="Search Algorithm Selection  خوارزميات البحث",
                 fg="#7FDBFF", bg=BG, font=("Segoe UI", 13, "bold")).pack(pady=(12, 2))
        tk.Label(
            win,
            text="This window controls SEARCH algorithms only. RL and advanced-ML systems stay enabled automatically.",
            fg="#AAAAAA", bg=BG, font=("Segoe UI", 8), wraplength=460,
        ).pack(pady=(0, 6))

        # ── Checkboxes ────────────────────────────────────────────────────────
        check_frame = tk.LabelFrame(win, text=" Algorithms ",
                                    fg="#7FDBFF", bg=BG, font=("Segoe UI", 9, "bold"))
        check_frame.pack(fill="x", padx=16, pady=4)

        algo_vars: dict = {}
        for algo in all_algos:
            name, desc = _ALGO_INFO[algo]
            var = tk.BooleanVar(value=(algo in current_algos))
            algo_vars[algo] = var
            row = tk.Frame(check_frame, bg=BG)
            row.pack(fill="x", padx=4, pady=1)
            tk.Checkbutton(row, variable=var, bg=BG, fg="white",
                           activebackground=BG, selectcolor="#2C3E50").pack(side="left")
            tk.Label(row, text=name, fg="#FFFFFF", bg=BG,
                     font=("Segoe UI", 9, "bold"), width=15, anchor="w").pack(side="left")
            tk.Label(row, text=desc, fg="#AAAAAA", bg=BG,
                     font=("Segoe UI", 8), anchor="w").pack(side="left")

        always_on = tk.LabelFrame(win, text=" Always Active Systems ",
                                  fg="#2ECC40", bg=BG, font=("Segoe UI", 9, "bold"))
        always_on.pack(fill="x", padx=16, pady=4)
        tk.Label(
            always_on,
            text=(
                "RL: Q-Learning, SARSA, DQN, PPO, A2C, Monte Carlo, Model-Based,\n"
                "Policy Gradient, Actor-Critic, Ensemble Voter, Online Learner\n\n"
                "Advanced ML: Anomaly Detector, Multi-Task Net, Passive-Aggressive,\n"
                "MAML Adapter, DeepRL Net, Causal Estimator\n\n"
                "Core modules: adaptive_learner.py, advanced_ml.py, all_algorithms.py,\n"
                "reward_calculator.py, rl_algorithms.py, rl_state.py"
            ),
            fg="#DDEEFF", bg=BG,
            font=("Segoe UI", 8, "bold"), justify="left", anchor="w",
            wraplength=460,
        ).pack(fill="x", padx=8, pady=8)

        # ── Pros / cons area ──────────────────────────────────────────────────
        notes_frame = tk.LabelFrame(win, text=" Preset Info ",
                                    fg="#FFDC00", bg=BG, font=("Segoe UI", 9, "bold"))
        notes_frame.pack(fill="x", padx=16, pady=4)
        pros_var = tk.StringVar(value="Select a preset, or tick boxes manually, then Apply.")
        cons_var = tk.StringVar(value="")
        tk.Label(notes_frame, textvariable=pros_var, fg="#2ECC40", bg=BG,
                 font=("Segoe UI", 8), justify="left", anchor="w",
                 wraplength=460).pack(fill="x", padx=8, pady=(4, 0))
        tk.Label(notes_frame, textvariable=cons_var, fg="#FF4136", bg=BG,
                 font=("Segoe UI", 8), justify="left", anchor="w",
                 wraplength=460).pack(fill="x", padx=8, pady=(0, 4))

        # ── Preset buttons ────────────────────────────────────────────────────
        preset_frame = tk.LabelFrame(win, text=" Recommended Presets ",
                                     fg="#7FDBFF", bg=BG, font=("Segoe UI", 9, "bold"))
        preset_frame.pack(fill="x", padx=16, pady=4)

        _PRESET_COLORS = {
            "Optimal (A* + UCS)":             "#0074D9",
            "Fast (Greedy + Beam Search)":    "#39CCCC",
            "Balanced ⭐ (A* + BFS + Greedy)": "#2ECC40",
            "Thorough (A* + BFS + DFS + UCS)": "#FF851B",
            "Full Ensemble (all 8)":           "#FF4136",
        }

        def apply_preset(pname: str):
            algos, pros, cons = self._SEARCH_ALGO_PRESETS[pname]
            for a, v in algo_vars.items():
                v.set(a in algos)
            pros_var.set(f"Pros: {pros}")
            cons_var.set(f"Cons: {cons}" if cons else "")
            if preset_var is not None:
                preset_var.set(pname)

        btn_row1 = tk.Frame(preset_frame, bg=BG)
        btn_row1.pack(pady=3)
        btn_row2 = tk.Frame(preset_frame, bg=BG)
        btn_row2.pack(pady=(0, 4))

        for i, (pname, color) in enumerate(_PRESET_COLORS.items()):
            row_frame = btn_row1 if i < 3 else btn_row2
            tk.Button(
                row_frame, text=pname, bg=color, fg="white",
                font=("Segoe UI", 8), cursor="hand2",
                command=lambda n=pname: apply_preset(n),
            ).pack(side="left", padx=3)

        # ── Apply / Cancel ────────────────────────────────────────────────────
        action_frame = tk.Frame(win, bg=BG)
        action_frame.pack(fill="x", padx=16, pady=8)

        def on_apply():
            chosen = [a for a, v in algo_vars.items() if v.get()]
            if not chosen:
                chosen = ["A*", "BFS", "Greedy"]   # safety fallback
            # Store as comma-separated string in a settings var
            algo_str = ",".join(chosen)
            if self._vars.get("active_search_algorithms") is None:
                import tkinter as _tk
                self._vars["active_search_algorithms"] = _tk.StringVar(value=algo_str)
            else:
                self._vars["active_search_algorithms"].set(algo_str)
            # Also set the dropdown to "Custom" if the selection doesn't match any preset
            if preset_var is not None:
                matched = next(
                    (name for name, (algos, _, _) in self._SEARCH_ALGO_PRESETS.items()
                     if set(algos) == set(chosen)),
                    "Custom",
                )
                preset_var.set(matched)
            win.destroy()

        def on_cancel():
            win.destroy()

        tk.Button(action_frame, text="Apply  تطبيق", width=12,
                  bg="#2ECC40", fg="white",
                  command=on_apply).pack(side="left", padx=4)
        tk.Button(action_frame, text="Cancel  إلغاء", width=12,
                  bg="#666666", fg="white",
                  command=on_cancel).pack(side="left", padx=4)

    def _on_car_setting_changed(self, changed_key: str):
        if not self._applying_theme and changed_key != "quick_theme":
            quick_theme_var = self._vars.get("quick_theme")
            if quick_theme_var is not None and quick_theme_var.get() != "Custom":
                self._set_quick_theme_custom()
                return
        self._update_car_preview()

    def _apply_quick_theme(self, *_):
        if self._applying_theme:
            return
        theme_name = self._vars.get("quick_theme").get() if self._vars.get("quick_theme") else "Custom"
        theme = self._QUICK_THEMES.get(theme_name)
        self._update_theme_action_buttons()
        if not theme:
            self._update_car_preview()
            return
        self._applying_theme = True
        try:
            for key, value in theme.items():
                self._set_var_value(key, value)
        finally:
            self._applying_theme = False
        self._update_car_preview()

    def _apply_theme_to_team(self, team: str):
        if self._applying_theme:
            return
        theme_name = self._vars.get("quick_theme").get() if self._vars.get("quick_theme") else "Custom"
        theme = self._QUICK_THEMES.get(theme_name)
        if not theme:
            self._update_car_preview()
            return

        if team not in ("blue", "orange"):
            return

        self._applying_theme = True
        try:
            for key in self._team_theme_keys(team):
                value = theme.get(key)
                if value is not None:
                    self._set_var_value(key, value)
        finally:
            self._applying_theme = False

        # A per-team apply produces a mixed configuration, so keep the selector honest.
        self._set_quick_theme_custom()
        self._update_car_preview()

    @staticmethod
    def _preview_color_for(name: str) -> str:
        color_map = {
            "Blue Default": "#4AA3FF",
            "Blue Esports": "#7FDBFF",
            "Orange Default": "#FF851B",
            "Orange Esports": "#FFB347",
        }
        return color_map.get(name, "#DDEEFF")

    def _update_car_preview(self):
        if self._car_preview_loadout_var is None:
            return
        theme_name = self._vars.get("quick_theme").get() if self._vars.get("quick_theme") else "Custom"
        blue_body = self._vars.get("blue_car_preset").get() if self._vars.get("blue_car_preset") else "Octane"
        blue_decal = self._vars.get("blue_decal_preset").get() if self._vars.get("blue_decal_preset") else "Default Style"
        blue_wheels = self._vars.get("blue_wheel_preset").get() if self._vars.get("blue_wheel_preset") else "Bender"
        blue_boost = self._vars.get("blue_boost_preset").get() if self._vars.get("blue_boost_preset") else "Gold Rush"
        blue_goal = self._vars.get("blue_goal_explosion_preset").get() if self._vars.get("blue_goal_explosion_preset") else "Classic"
        blue_finish = self._vars.get("blue_finish_preset").get() if self._vars.get("blue_finish_preset") else "Brushed Metal"
        orange_body = self._vars.get("orange_car_preset").get() if self._vars.get("orange_car_preset") else "Octane"
        orange_decal = self._vars.get("orange_decal_preset").get() if self._vars.get("orange_decal_preset") else "Default Style"
        orange_wheels = self._vars.get("orange_wheel_preset").get() if self._vars.get("orange_wheel_preset") else "Bender"
        orange_boost = self._vars.get("orange_boost_preset").get() if self._vars.get("orange_boost_preset") else "Gold Rush"
        orange_goal = self._vars.get("orange_goal_explosion_preset").get() if self._vars.get("orange_goal_explosion_preset") else "Classic"
        orange_finish = self._vars.get("orange_finish_preset").get() if self._vars.get("orange_finish_preset") else "Brushed Metal"
        blue = self._vars.get("team_color_blue").get() if self._vars.get("team_color_blue") else "Blue Default"
        orange = self._vars.get("team_color_orange").get() if self._vars.get("team_color_orange") else "Orange Default"
        accent = self._vars.get("custom_color_blue").get() if self._vars.get("custom_color_blue") else "No Accent"
        orange_accent = self._vars.get("custom_color_orange").get() if self._vars.get("custom_color_orange") else "No Accent"
        self._car_preview_loadout_var.set(
            f"Theme: {theme_name}  |  Blue and Orange loadouts are independent"
        )
        self._car_preview_blue_var.set(
            f"Blue: {blue_body} + {blue_decal} + {blue_wheels} + {blue_boost} + {blue_goal} + {blue_finish}  |  {blue} / {accent}"
        )
        self._car_preview_orange_var.set(
            f"Orange: {orange_body} + {orange_decal} + {orange_wheels} + {orange_boost} + {orange_goal} + {orange_finish}  |  {orange} / {orange_accent}"
        )
        if self._car_preview_blue_label is not None:
            self._car_preview_blue_label.configure(fg=self._preview_color_for(blue))
        if self._car_preview_orange_label is not None:
            self._car_preview_orange_label.configure(fg=self._preview_color_for(orange))
        self._draw_car_preview_canvas(blue_body, orange_body, blue, orange, accent, orange_accent)

    def _draw_car_preview_canvas(self, blue_body: str, orange_body: str, blue: str, orange: str,
                                 accent: str, orange_accent: str):
        if self._car_preview_canvas is None:
            return
        canvas = self._car_preview_canvas
        canvas.delete("all")

        blue_fill = self._preview_color_for(blue)
        orange_fill = self._preview_color_for(orange)
        accent_fill = "#EAEAEA" if accent == "No Accent" else "#A0A0A0"
        orange_accent_fill = "#EAEAEA" if orange_accent == "No Accent" else "#A0A0A0"

        canvas.create_rectangle(0, 0, 300, 110, fill="#0A1520", outline="#1F3A4D")
        canvas.create_text(150, 12, text="Blue vs Orange Preview", fill="#7FDBFF", font=("Segoe UI", 10, "bold"))

        canvas.create_rectangle(32, 34, 128, 78, fill=blue_fill, outline="#DDEEFF", width=2)
        canvas.create_polygon(54, 34, 110, 34, 96, 22, 66, 22, fill=blue_fill, outline="#DDEEFF", width=2)
        canvas.create_oval(44, 68, 68, 92, fill="#202020", outline="#CCCCCC")
        canvas.create_oval(92, 68, 116, 92, fill="#202020", outline="#CCCCCC")
        canvas.create_rectangle(70, 46, 104, 60, fill=accent_fill, outline="")
        canvas.create_text(80, 98, text=blue_body, fill=blue_fill, font=("Segoe UI", 9, "bold"))

        canvas.create_rectangle(172, 34, 268, 78, fill=orange_fill, outline="#DDEEFF", width=2)
        canvas.create_polygon(194, 34, 250, 34, 236, 22, 206, 22, fill=orange_fill, outline="#DDEEFF", width=2)
        canvas.create_oval(184, 68, 208, 92, fill="#202020", outline="#CCCCCC")
        canvas.create_oval(232, 68, 256, 92, fill="#202020", outline="#CCCCCC")
        canvas.create_rectangle(210, 46, 244, 60, fill=orange_accent_fill, outline="")
        canvas.create_text(220, 98, text=orange_body, fill=orange_fill, font=("Segoe UI", 9, "bold"))

    # ── tab builders ─────────────────────────────────────────────────────────
    def _tab_match(self, parent):
        import tkinter as tk
        from tkinter import ttk
        BG = self._BG

        self._sec(parent, "⚽  EXHIBITION MATCH  /  الدور")

        # Game Mode row (special — drives arena list)
        r = tk.Frame(parent, bg=BG); r.pack(fill="x", padx=22, pady=3)
        tk.Label(r, text="Game Mode  نوع اللعبه", bg=BG, fg="#99BBCC",
                 font=("Segoe UI", 9), width=24, anchor="w").pack(side="left")
        gm_var = tk.StringVar(value="Soccer")
        self._vars["game_mode"] = gm_var
        gm_cb = ttk.Combobox(
            r, textvariable=gm_var,
            values=["Soccer", "Hoops", "Hockey", "Dropshot",
                    "Rumble", "Heatseeker", "Gridiron"],
            state="readonly", width=28, style="D.TCombobox")
        gm_cb.pack(side="left")

        # Arena row (editable autocomplete)
        r2 = tk.Frame(parent, bg=BG); r2.pack(fill="x", padx=22, pady=3)
        tk.Label(r2, text="Arena  الملعب", bg=BG, fg="#99BBCC",
                 font=("Segoe UI", 9), width=24, anchor="w").pack(side="left")
        init_arenas = [n for n, _ in self._ARENAS_BY_MODE["Soccer"]]
        ar_var = tk.StringVar(value=init_arenas[0])
        self._vars["arena_display"] = ar_var
        ar_cb = ttk.Combobox(r2, textvariable=ar_var, values=init_arenas,
                             state="normal", width=28, style="D.TCombobox")
        ar_cb.pack(side="left")
        self._comboboxes["arena_display"] = ar_cb
        self._row_options["arena_display"] = list(init_arenas)
        self._bind_combobox_autocomplete("arena_display")

        def _on_mode(*_):
            options = [n for n, _ in
                       self._ARENAS_BY_MODE.get(gm_var.get(), self._SOCCER_ARENAS)]
            self._row_options["arena_display"] = list(options)
            ar_cb["values"] = options
            if options:
                ar_var.set(options[0])
        gm_cb.bind("<<ComboboxSelected>>", _on_mode)

        self._row(parent, "Team Size  حجم الفريق", "team_size",
                  ["1v1", "2v2", "3v3"], "1v1")
        self._row(parent, "Psyonix Bot Skill", "psyonix_difficulty",
              ["Rookie", "Pro", "All-Star"], "All-Star")

        self._sec(parent, "🏆  MATCH MUTATORS")
        self._row(parent, "Match Length  الوقت",  "match_length",
                  self._MUTATOR_OPTS["match_length"],  "5 Minutes", autocomplete=True)
        self._row(parent, "Max Score  أقصى جول",  "max_score",
                  self._MUTATOR_OPTS["max_score"],     "Unlimited", autocomplete=True)
        self._row(parent, "Overtime",              "overtime",
                  self._MUTATOR_OPTS["overtime"],      "Unlimited", autocomplete=True)
        self._row(parent, "Series Length",         "series_length",
                  self._MUTATOR_OPTS["series_length"], "Unlimited", autocomplete=True)

    def _tab_game(self, parent):
        self._sec(parent, "🎮  GAME MUTATORS")
        self._row(parent, "Game Speed",             "game_speed",
                  self._MUTATOR_OPTS["game_speed"],     "Default", autocomplete=True)
        self._row(parent, "Boost Amount  البوست",   "boost_amount",
                  self._MUTATOR_OPTS["boost_amount"],   "Default", autocomplete=True)
        self._row(parent, "Boost Strength",         "boost_strength",
                  self._MUTATOR_OPTS["boost_strength"], "1x", autocomplete=True)
        self._row(parent, "Rumble  رمبل",           "rumble",
                  self._MUTATOR_OPTS["rumble"],         "None", autocomplete=True)
        self._row(parent, "Gravity  الجاذبية",      "gravity",
                  self._MUTATOR_OPTS["gravity"],        "Default", autocomplete=True)
        self._row(parent, "Demolish",               "demolish",
                  self._MUTATOR_OPTS["demolish"],       "Default", autocomplete=True)
        self._row(parent, "Respawn Time",           "respawn_time",
                  self._MUTATOR_OPTS["respawn_time"],   "3 Seconds", autocomplete=True)

    def _tab_ball(self, parent):
        self._sec(parent, "  BALL MUTATORS")
        self._row(parent, "Ball Max Speed",         "ball_max_speed",
                  self._MUTATOR_OPTS["ball_max_speed"],  "Default", autocomplete=True)
        self._row(parent, "Ball Type  نوع الكرة",   "ball_type",
                  self._MUTATOR_OPTS["ball_type"],       "Default", autocomplete=True)
        self._row(parent, "Ball Weight  الوزن",     "ball_weight",
                  self._MUTATOR_OPTS["ball_weight"],     "Default", autocomplete=True)
        self._row(parent, "Ball Size  الحجم",       "ball_size",
                  self._MUTATOR_OPTS["ball_size"],       "Default", autocomplete=True)
        self._row(parent, "Ball Bounciness",        "ball_bounciness",
                  self._MUTATOR_OPTS["ball_bounciness"], "Default", autocomplete=True)

    def _tab_car(self, parent):
        import tkinter as tk
        self._sec(parent, "🚗  CAR PRESET / شكل العربية")
        search_row = tk.Frame(parent, bg=self._BG)
        search_row.pack(fill="x", padx=22, pady=(2, 6))
        tk.Label(search_row, text="Search / بحث", bg=self._BG, fg="#99BBCC",
                 font=("Segoe UI", 9), width=24, anchor="w").pack(side="left")
        self._car_search_var = tk.StringVar(value="")
        search_entry = tk.Entry(
            search_row,
            textvariable=self._car_search_var,
            bg=self._CB_BG,
            fg=self._FG,
            insertbackground=self._FG,
            relief="flat",
            width=31,
            font=("Segoe UI", 9),
        )
        search_entry.pack(side="left")
        self._car_search_var.trace_add("write", self._refresh_car_option_filters)
        self._row(parent, "Quick Theme", "quick_theme",
              list(self._QUICK_THEMES.keys()), "Custom", autocomplete=True)
        theme_actions = tk.Frame(parent, bg=self._BG)
        theme_actions.pack(fill="x", padx=22, pady=(2, 6))
        self._apply_theme_blue_button = tk.Button(
            theme_actions,
            text="Apply Theme To Blue Only",
            bg="#0E3A5D",
            fg="white",
            relief="flat",
            cursor="hand2",
            command=lambda: self._apply_theme_to_team("blue"),
        )
        self._apply_theme_blue_button.pack(side="left", padx=(0, 8))
        self._apply_theme_orange_button = tk.Button(
            theme_actions,
            text="Apply Theme To Orange Only",
            bg="#7A3412",
            fg="white",
            relief="flat",
            cursor="hand2",
            command=lambda: self._apply_theme_to_team("orange"),
        )
        self._apply_theme_orange_button.pack(side="left")

        self._sec(parent, "🔵  BLUE TEAM LOADOUT")
        self._row(parent, "Blue Body", "blue_car_preset",
              list(self._CAR_PRESETS.keys()), "Octane", autocomplete=True)
        self._row(parent, "Blue Decal", "blue_decal_preset",
              list(self._DECAL_PRESETS.keys()), "Default Style", autocomplete=True)
        self._row(parent, "Blue Wheels", "blue_wheel_preset",
              list(self._WHEEL_PRESETS.keys()), "Bender", autocomplete=True)
        self._row(parent, "Blue Boost", "blue_boost_preset",
              list(self._BOOST_PRESETS.keys()), "Gold Rush", autocomplete=True)
        self._row(parent, "Blue Goal Explosion", "blue_goal_explosion_preset",
              list(self._GOAL_EXPLOSION_PRESETS.keys()), "Classic", autocomplete=True)
        self._row(parent, "Blue Finish", "blue_finish_preset",
              list(self._FINISH_PRESETS.keys()), "Brushed Metal", autocomplete=True)

        self._sec(parent, "🟠  ORANGE TEAM LOADOUT")
        self._row(parent, "Orange Body", "orange_car_preset",
              list(self._CAR_PRESETS.keys()), "Octane", autocomplete=True)
        self._row(parent, "Orange Decal", "orange_decal_preset",
              list(self._DECAL_PRESETS.keys()), "Default Style", autocomplete=True)
        self._row(parent, "Orange Wheels", "orange_wheel_preset",
              list(self._WHEEL_PRESETS.keys()), "Bender", autocomplete=True)
        self._row(parent, "Orange Boost", "orange_boost_preset",
              list(self._BOOST_PRESETS.keys()), "Gold Rush", autocomplete=True)
        self._row(parent, "Orange Goal Explosion", "orange_goal_explosion_preset",
              list(self._GOAL_EXPLOSION_PRESETS.keys()), "Classic", autocomplete=True)
        self._row(parent, "Orange Finish", "orange_finish_preset",
              list(self._FINISH_PRESETS.keys()), "Brushed Metal", autocomplete=True)

        self._sec(parent, "🎨  COLORS / الألوان")
        self._row(parent, "Blue Team Color", "team_color_blue",
                  ["Blue Default", "Blue Esports"], "Blue Default")
        self._row(parent, "Orange Team Color", "team_color_orange",
                  ["Orange Default", "Orange Esports"], "Orange Default")
        self._row(parent, "Blue Accent", "custom_color_blue",
                  list(self._CUSTOM_COLOR_PRESETS.keys()), "No Accent")
        self._row(parent, "Orange Accent", "custom_color_orange",
                  list(self._CUSTOM_COLOR_PRESETS.keys()), "No Accent")

        self._sec(parent, "👀  PREVIEW / معاينة")
        self._car_preview_canvas = tk.Canvas(
            parent, width=300, height=110,
            bg=self._BG, highlightthickness=1, highlightbackground="#1F3A4D"
        )
        self._car_preview_canvas.pack(fill="x", padx=22, pady=(4, 8))
        self._car_preview_loadout_var = tk.StringVar(value="")
        self._car_preview_blue_var = tk.StringVar(value="")
        self._car_preview_orange_var = tk.StringVar(value="")
        tk.Label(parent, textvariable=self._car_preview_loadout_var,
                 bg=self._BG, fg="#DDEEFF",
                 font=("Segoe UI", 9, "bold"),
                 justify="left", anchor="w", wraplength=560).pack(fill="x", padx=22, pady=(4, 6))
        colors_frame = tk.Frame(parent, bg=self._BG)
        colors_frame.pack(fill="x", padx=22, pady=(0, 10))
        self._car_preview_blue_label = tk.Label(
            colors_frame, textvariable=self._car_preview_blue_var,
            bg=self._BG, fg="#4AA3FF", font=("Segoe UI", 9, "bold"),
            anchor="w", justify="left", wraplength=680
        )
        self._car_preview_blue_label.pack(fill="x", pady=(0, 4))
        self._car_preview_orange_label = tk.Label(
            colors_frame, textvariable=self._car_preview_orange_var,
            bg=self._BG, fg="#FF851B", font=("Segoe UI", 9, "bold"),
            anchor="w", justify="left", wraplength=680
        )
        self._car_preview_orange_label.pack(fill="x")
        quick_theme_var = self._vars.get("quick_theme")
        if quick_theme_var is not None:
            quick_theme_var.trace_add("write", self._apply_quick_theme)
        self._install_preview_hooks([
            "blue_car_preset", "blue_decal_preset", "blue_wheel_preset", "blue_boost_preset",
            "blue_goal_explosion_preset", "blue_finish_preset",
            "orange_car_preset", "orange_decal_preset", "orange_wheel_preset", "orange_boost_preset",
            "orange_goal_explosion_preset", "orange_finish_preset",
            "team_color_blue", "team_color_orange",
            "custom_color_blue", "custom_color_orange",
        ])
        self._update_theme_action_buttons()
        self._refresh_car_option_filters()
        self._update_car_preview()

    def _tab_rl(self, parent):
        import tkinter as tk
        BG = self._BG
        self._sec(parent, "🧠  RL SETTINGS / إعدادات التعلم والحركة")
        self._row(parent, "Bot Difficulty  الصعوبة", "difficulty",
                  ["Easy  سهل", "Medium  متوسط", "Hard  صعب"],
                  "Medium  متوسط")
        self._row(parent, "AI Model  الموديل", "model_choice",
                  ["Persistent  بيتطور", "Session  للماتش بس"],
                  "Persistent  بيتطور")
        self._row(parent, "Overlay Menu Position", "overlay_menu_position",
                  ["Top Right", "Bottom Right", "Hidden by Default"],
                  "Top Right")
        self._row(parent, "Overlay Menu Size", "overlay_menu_size",
                  ["Auto", "Small", "Medium"],
                  "Auto")
        self._row(parent, "Launcher Program  برنامج التشغيل", "launcher_platform",
                  ["Epic", "Steam"], "Epic")
        self._row(parent, "Game Launch  تشغيل اللعبة", "launcher_launch_mode",
                  ["Direct EXE (No Launcher)", "Auto via RLBot", "Manual Only (Do Not Launch Game)"],
                  "Direct EXE (No Launcher)")
        tk.Label(
            parent,
            text="Search selector below controls pathfinding only. RL algorithms and advanced ML stay active automatically.",
            bg=BG,
            fg="#99BBCC",
            font=("Segoe UI", 8),
            justify="left",
            anchor="w",
            wraplength=640,
        ).pack(fill="x", padx=22, pady=(0, 8))
        self._row(parent, "RL Preset", "rl_tuning_preset",
                  list(self._RL_TUNING_PRESETS.keys()), "Balanced")

        # ── Search algorithm preset ───────────────────────────────────────────
        self._row(parent, "Search Preset  خوارزميات البحث", "search_algo_preset",
                  list(self._SEARCH_ALGO_PRESETS.keys()),
                  "Balanced ⭐ (A* + BFS + Greedy)")

        r_search = tk.Frame(parent, bg=BG)
        r_search.pack(fill="x", padx=22, pady=(0, 6))
        tk.Button(
            r_search,
            text="Choose Algorithms / اختر  (A)",
            bg="#0074D9", fg="white",
            font=("Segoe UI", 9, "bold"),
            relief="flat", cursor="hand2",
            command=lambda: self._show_search_algo_selector(),
        ).pack(side="left")

        self._sec(parent, "🎚  MOVEMENT TUNING / شريط الحركة")
        self._slider_row(parent, "Movement Speed", "movement_speed", 0.5, 1.5, 1.0)
        self._slider_row(parent, "Steering Sensitivity", "steering_sensitivity", 0.5, 1.5, 1.0)
        self._slider_row(parent, "Boost Aggression", "boost_aggression_scale", 0.5, 1.5, 1.0)
        self._slider_row(parent, "Aerial Commitment", "aerial_commitment", 0.5, 1.5, 1.0)
        rl_preset_var = self._vars.get("rl_tuning_preset")
        if rl_preset_var is not None:
            rl_preset_var.trace_add("write", self._apply_rl_tuning_preset)
        for slider_key in (
            "movement_speed",
            "steering_sensitivity",
            "boost_aggression_scale",
            "aerial_commitment",
        ):
            slider_var = self._vars.get(slider_key)
            if slider_var is not None:
                slider_var.trace_add("write", lambda *_args, changed_key=slider_key: self._on_rl_slider_changed(changed_key))

        self._sec(parent, "👤  PLAYER / اللاعب")
        r = tk.Frame(parent, bg=BG)
        r.pack(fill="x", padx=22, pady=6)
        tk.Label(r, text="أنا بالعب  Play as Human",
                 bg=BG, fg="#99BBCC",
                 font=("Segoe UI", 9), width=24, anchor="w").pack(side="left")
        ph_var = tk.BooleanVar(value=False)
        self._vars["play_as_human"] = ph_var
        tk.Checkbutton(r, variable=ph_var, bg=BG, fg=self._FG,
                       activebackground=BG, selectcolor=self._CB_BG,
                       cursor="hand2").pack(side="left")
        tk.Label(r, text="تشغيل  →  لازم تكون جوه اللعبة",
                 bg=BG, fg="#556677",
                 font=("Segoe UI", 8)).pack(side="left", padx=8)

    def _tab_bot(self, parent):
        import tkinter as tk
        BG = self._BG
        self._sec(parent, "🤖  AI BOT STATUS / حالة الذكاء")

        self._sec(parent, "🧠  ALL ALGORITHMS ACTIVE  /  كل الخوارزميات شغالة")
        tk.Label(
            parent,
            text=(
                "Search ensemble: A*, BFS, UCS, DFS, Greedy, Decision Tree, Beam Search, IDA*\n"
                "Adaptive RL: Q-Learning, SARSA, DQN, PPO, A2C, Monte Carlo, Model-Based, Policy Gradient, Actor-Critic, Ensemble Voter, Online Learner\n"
                "Advanced ML: Anomaly Detector, Multi-Task Net, Passive-Aggressive, MAML Adapter, DeepRL Net, Causal Estimator"
            ),
            bg=BG,
            fg="#DDEEFF",
            font=("Segoe UI", 9, "bold"),
            justify="left",
            anchor="w",
            wraplength=640,
        ).pack(fill="x", padx=22, pady=(4, 8))

        self._sec(parent, "🧩  CORE STACK / دمج كل خوارزميات core")
        tk.Label(
            parent,
            text=(
                "Prediction + Search + RL are all wired into the live bot runtime.\n"
                "Core files in use: adaptive_learner.py, advanced_ml.py, all_algorithms.py,\n"
                "reward_calculator.py, rl_algorithms.py, rl_state.py.\n"
                "Live systems: reward shaping, state discretization, anomaly threat, counterfactual regret,\n"
                "role adaptation, mechanic adaptation, and ensemble control."
            ),
            bg=BG,
            fg="#99BBCC",
            font=("Segoe UI", 9),
            justify="left",
            anchor="w",
            wraplength=640,
        ).pack(fill="x", padx=22, pady=(4, 8))

        self._sec(parent, "📘  NOTES / ملاحظات")
        tk.Label(
            parent,
            text=(
                "The algorithm picker window shows the 8 search algorithms only.\n"
                "The remaining RL and advanced-ML algorithms are always active in the background and listed above.\n"
                "Use Manual mode in-game if you want to teach the policy from human driving."
            ),
            bg=BG,
            fg="#99BBCC",
            font=("Segoe UI", 9),
            justify="left",
            anchor="w",
            wraplength=640,
        ).pack(fill="x", padx=22, pady=(4, 8))

    # ── events ───────────────────────────────────────────────────────────────
    def _on_start(self):
        result = {k: (v.get() if not hasattr(v, "get") else v.get())
                  for k, v in self._vars.items()}
        # Resolve arena display → rlbot internal name
        gm   = result.get("game_mode", "Soccer")
        self._normalize_arena_display_value()
        disp = result.pop("arena_display", "DFH Stadium")
        arena_map = {n: k
                     for n, k in self._ARENAS_BY_MODE.get(gm, self._SOCCER_ARENAS)}
        if disp not in arena_map:
            fallback_display = next((name for name in arena_map if disp.lower() in name.lower()), "DFH Stadium")
            disp = fallback_display
        result["arena"] = arena_map.get(disp, "DFHStadium")
        # Strip Arabic label suffixes
        result["difficulty"]   = result["difficulty"].split("  ")[0]
        result["model_choice"] = result["model_choice"].split("  ")[0]
        _save_ui_preferences(result)
        self._result = result
        self._root.withdraw()
        self._signal_var.set("start")

    def _on_close(self):
        self._result = None
        self._signal_var.set("close")
        self._root.destroy()
        self._root = None
        self._signal_var = None


def _ask_settings_console() -> dict | None:
    """Console fallback when tkinter is unavailable."""
    print("\n=== Match Settings ===")
    modes = ["Soccer", "Hoops", "Hockey", "Dropshot", "Rumble", "Heatseeker", "Gridiron"]
    for i, m in enumerate(modes, 1):
        print(f"  {i} = {m}")
    while True:
        c = input("Game mode (1-7): ").strip()
        if c.isdigit() and 1 <= int(c) <= len(modes):
            game_mode = modes[int(c) - 1]
            break
        print("  Invalid, try again.")

    diff_opts = ["Easy", "Medium", "Hard"]
    while True:
        c = input("Difficulty (1=Easy, 2=Medium, 3=Hard): ").strip()
        if c in ("1", "2", "3"):
            difficulty = diff_opts[int(c) - 1]
            break
        print("  Invalid, try again.")

    time_opts = ["5 Minutes", "10 Minutes", "20 Minutes", "Unlimited"]
    while True:
        c = input("Match length (1=5min 2=10min 3=20min 4=Unlimited): ").strip()
        if c in ("1", "2", "3", "4"):
            match_length = time_opts[int(c) - 1]
            break
        print("  Invalid, try again.")

    while True:
        c = input("Gravity (1=Default 2=Low 3=High 4=Super High): ").strip()
        if c in ("1", "2", "3", "4"):
            gravity = ["Default", "Low", "High", "Super High"][int(c) - 1]
            break
        print("  Invalid, try again.")

    while True:
        c = input("AI Model (1=Persistent/evolving, 2=Session only): ").strip()
        if c == "1":
            model_choice = "Persistent"
            break
        elif c == "2":
            model_choice = "Session"
            break
        print("  Invalid, try again.")

    while True:
        c = input("Launcher platform (1=Epic, 2=Steam): ").strip()
        if c == "1":
            launcher_platform = "Epic"
            break
        elif c == "2":
            launcher_platform = "Steam"
            break
        print("  Invalid, try again.")

    while True:
        c = input("Game launch (1=Direct EXE, 2=Auto via RLBot, 3=Manual only): ").strip()
        if c == "1":
            launcher_launch_mode = "Direct EXE (No Launcher)"
            break
        elif c == "2":
            launcher_launch_mode = "Auto via RLBot"
            break
        elif c == "3":
            launcher_launch_mode = "Manual Only (Do Not Launch Game)"
            break
        print("  Invalid, try again.")

    return {
        "game_mode": game_mode, "arena": "DFHStadium", "team_size": "1v1",
        "psyonix_difficulty": "All-Star",
        "difficulty": difficulty, "model_choice": model_choice,
        "active_search_algorithms": "A*,BFS,UCS,DFS,Greedy,Decision Tree,Beam Search,IDA*",
        "active_rl_models": "q_learning,actor_critic,online_learner,dqn,ppo,a2c,monte_carlo,model_based",
        "active_advanced_models": "anomaly,multi_task,passive_aggressive,maml,deep_rl,causal",
        "overlay_menu_position": "Top Right",
        "launcher_platform": launcher_platform,
        "launcher_launch_mode": launcher_launch_mode,
        "match_length": match_length, "max_score": "Unlimited",
        "overtime": "Unlimited", "series_length": "Unlimited",
        "game_speed": "Default", "boost_amount": "Default",
        "boost_strength": "1x", "rumble": "None",
        "gravity": gravity, "demolish": "Default", "respawn_time": "3 Seconds",
        "ball_max_speed": "Default", "ball_type": "Default",
        "ball_weight": "Default", "ball_size": "Default",
        "ball_bounciness": "Default", "play_as_human": False,
        "rl_tuning_preset": "Balanced",
        "movement_speed": 1.0,
        "steering_sensitivity": 1.0,
        "boost_aggression_scale": 1.0,
        "aerial_commitment": 1.0,
        "quick_theme": "Custom",
        "team_color_blue": "Blue Default",
        "team_color_orange": "Orange Default",
        "custom_color_blue": "No Accent",
        "custom_color_orange": "No Accent",
        "blue_car_preset": "Octane",
        "blue_decal_preset": "Default Style",
        "blue_wheel_preset": "Bender",
        "blue_boost_preset": "Gold Rush",
        "blue_goal_explosion_preset": "Classic",
        "blue_finish_preset": "Brushed Metal",
        "orange_car_preset": "Octane",
        "orange_decal_preset": "Default Style",
        "orange_wheel_preset": "Bender",
        "orange_boost_preset": "Gold Rush",
        "orange_goal_explosion_preset": "Classic",
        "orange_finish_preset": "Brushed Metal",
    }

def _settings_from_dashboard_json(data: dict) -> dict:
    """Convert the JSON layout written by the dashboard to the flat settings dict
    expected by _apply_settings_to_engine and the match loop."""
    inner = data.get("settings", data)   # handle both wrapped and flat layouts
    size_raw = inner.get("team_size", "1v1")
    if isinstance(size_raw, str) and "v" in size_raw:
        size_str = size_raw
    else:
        try:
            size_str = {1: "1v1", 2: "2v2", 3: "3v3"}[int(size_raw)]
        except (KeyError, ValueError, TypeError):
            size_str = "1v1"
    temporary_reset = bool(data.get("temporary_reset", False))
    return {
        "game_mode":             inner.get("game_mode", "Soccer"),
        "arena":                 inner.get("arena", "DFHStadium"),
        "team_size":             size_str,
        "match_length":          inner.get("match_length", "5 Minutes"),
        "boost_amount":          inner.get("boost_amount", "Default"),
        "gravity":               inner.get("gravity", "Default"),
        "difficulty":            inner.get("difficulty", "Medium"),
        "psyonix_difficulty":    inner.get("psyonix_difficulty", "All-Star"),
        "model_choice":          "Session" if temporary_reset else "Persistent",
        # Dashboard sends lists; _apply_settings_to_engine handles them
        "search_algorithms":     inner.get("search_algorithms", ["A*"]),
        "rl_models":             inner.get("rl_models", []),
        "active_advanced_models": inner.get("active_advanced_models",
                                            "anomaly,multi_task,passive_aggressive,maml,deep_rl,causal"),
        "strategy":              inner.get("strategy", "Balanced"),
        "overlay_menu_position": inner.get("overlay_menu_position", "Top Right"),
        "launcher_platform":     inner.get("launcher_platform", "Epic"),
        "launcher_launch_mode":  inner.get("launcher_launch_mode", "Auto via RLBot"),
        "max_score":             inner.get("max_score", "Unlimited"),
        "overtime":              inner.get("overtime", "Unlimited"),
        "series_length":         inner.get("series_length", "Unlimited"),
        "game_speed":            inner.get("game_speed", "Default"),
        "boost_strength":        inner.get("boost_strength", "1x"),
        "rumble":                inner.get("rumble", "None"),
        "demolish":              inner.get("demolish", "Default"),
        "respawn_time":          inner.get("respawn_time", "3 Seconds"),
        "ball_max_speed":        inner.get("ball_max_speed", "Default"),
        "ball_type":             inner.get("ball_type", "Default"),
        "ball_weight":           inner.get("ball_weight", "Default"),
        "ball_size":             inner.get("ball_size", "Default"),
        "ball_bounciness":       inner.get("ball_bounciness", "Default"),
        "play_as_human":         inner.get("play_as_human", False),
        "rl_tuning_preset":      inner.get("rl_tuning_preset", "Balanced"),
        "movement_speed":        float(inner.get("movement_speed", 1.0)),
        "steering_sensitivity":  float(inner.get("steering_sensitivity", 1.0)),
        "boost_aggression_scale": float(inner.get("boost_aggression_scale", 1.0)),
        "aerial_commitment":     float(inner.get("aerial_commitment", 1.0)),
        "quick_theme":           inner.get("quick_theme", "Custom"),
        "blue_car_preset":       inner.get("blue_car_preset", "Octane"),
        "orange_car_preset":     inner.get("orange_car_preset", "Octane"),
    }


def run(mode: str = "1v1", opponent_skill: float = 0.5, no_gui: bool = False):
    global _pre_initialized_engine, _starting_temporary_reset, _pending_match_settings

    # ── When launched by the dashboard (--no-gui), skip the settings window ──
    if no_gui:
        settings_gui = None
        use_gui = False
    else:
        # ── Create settings GUI (persistent across matches) ──
        try:
            import tkinter as _tk_test  # noqa: F401
            settings_gui = _SettingsGUI()
            use_gui = True
        except Exception:
            settings_gui = None
            use_gui = False

    # ── Match loop — keeps running until user closes settings window ──
    while True:
        # ── Step 0.5: Collect all settings from the user ──
        if no_gui:
            # Dashboard mode: read directly from _match_settings.json
            loaded = _load_match_settings_from_file()
            if loaded:
                settings = _settings_from_dashboard_json(loaded)
                print("[INFO] Loaded match settings from dashboard JSON.")
            else:
                print("[INFO] No settings file found — using defaults.")
                settings = _settings_from_dashboard_json({})
        elif use_gui:
            print("\n[INFO] Waiting for user to configure match settings...")
            settings = settings_gui.ask()
        else:
            settings = _ask_settings_console()

        if settings is None:
            print("[INFO] User closed settings — exiting.")
            break

        gm = settings["game_mode"]
        print(f"[INFO] Settings: mode={gm}  arena={settings.get('arena')}  "
              f"difficulty={settings['difficulty']}  length={settings.get('match_length')}  "
              f"rumble={settings.get('rumble')}  gravity={settings.get('gravity')}  "
              f"model={settings['model_choice']}")

        print("[INIT] Pre-loading search algorithms & knowledge...")
        _pre_initialized_engine = DecisionEngine(PROJECT_ROOT)
        rl_status = "loaded" if _pre_initialized_engine.adaptive.q_role.q_table else "fresh"
        print(f"[INIT] Algorithms ready.  RL model: {rl_status}")

        _starting_temporary_reset = settings["model_choice"] == "Session"
        _pending_match_settings = dict(settings)
        if _starting_temporary_reset:
            print("[INFO] Model: session-only (will NOT save progress)")
        else:
            print("[INFO] Model: persistent (will save & evolve)")

        _apply_settings_to_engine(_pre_initialized_engine, settings)
        _apply_looks_settings(settings)

        # Save settings for the bot subprocess (globals don't cross process boundary).
        _save_match_settings_for_subprocess(settings, _starting_mode, _starting_temporary_reset)
        print(f"[INFO] Saved match settings sidecar for subprocess: mode={_starting_mode}, temp_reset={_starting_temporary_reset}")

        # ── Step 1: Check Rocket League state ──
        if _is_rl_running():
            print()
            print("[WARNING] Rocket League is already running!")
            print("          RLBot needs to start RL with special flags (-rlbot).")
            print("          If the bot fails to connect, close Rocket League")
            print("          and run this again — RLBot will launch it for you.")
            print()

        # ── Step 2: Build match configuration ────────────────────────────
        print("[INFO] Building match configuration...")
        try:
            match_config = build_match(settings)
        except Exception as _mc_exc:
            print(f"[ERROR] Could not build match config: {_mc_exc}")
            if no_gui:
                break
            continue

        # ── Step 3: Run match via RLBot ──────────────────────────────────
        print("[INFO] Starting RLBot match runner...")
        _active_manager = SetupManager()
        _active_manager.num_metadata_attempts = 10
        try:
            _active_manager.load_match_config(match_config)
            _active_manager.connect_to_game()
            _active_manager.start_match()
            _active_manager.launch_bot_processes()
            print("[INFO] Match running — bot processes launched.")
            _reopen_settings_requested.clear()
            _active_manager.infinite_loop()
        except KeyboardInterrupt:
            pass
        except Exception as _run_exc:
            print(f"[ERROR] RLBot match error: {_run_exc}")
        finally:
            try:
                ipc.write_bot_status({"alive": False})
            except Exception:
                pass
            try:
                _active_manager.shut_down()
            except Exception:
                pass

        # Dashboard (no_gui) mode: only one match per process invocation.
        # Restart is requested externally by the dashboard via _gui_commands.json.
        if no_gui:
            break

        def _show_search_algo_selector(self):
            """Open a compact control center for Search, RL, and Advanced ML."""
            import tkinter as tk
            from tkinter import ttk

            BG = "#101820"
            FG = "#DDEEFF"

            preset_var = self._vars.get("search_algo_preset")
            current_preset = preset_var.get() if preset_var is not None else "Balanced ⭐ (A* + BFS + Greedy)"
            preset_entry = self._SEARCH_ALGO_PRESETS.get(
                current_preset, self._SEARCH_ALGO_PRESETS["Full Ensemble (all 8)"]
            )
            current_algos, _, _ = preset_entry

            all_algos = ["A*", "BFS", "UCS", "DFS", "Greedy", "Decision Tree", "Beam Search", "IDA*"]
            rl_keys = ["q_learning", "actor_critic", "online_learner", "dqn", "ppo", "a2c", "monte_carlo", "model_based"]
            adv_keys = ["anomaly", "multi_task", "passive_aggressive", "maml", "deep_rl", "causal"]

            current_rl_var = self._vars.get("active_rl_models")
            current_rl = set(filter(None, [x.strip() for x in (current_rl_var.get() if current_rl_var is not None else "q_learning,actor_critic,online_learner,dqn,ppo,a2c,monte_carlo,model_based").split(",")]))
            current_adv_var = self._vars.get("active_advanced_models")
            current_adv = set(filter(None, [x.strip() for x in (current_adv_var.get() if current_adv_var is not None else "anomaly,multi_task,passive_aggressive,maml,deep_rl,causal").split(",")]))

            search_info = {
                "A*": ("A* (A-Star)", "Optimal & complete. Balances cost + heuristic."),
                "BFS": ("BFS", "Wide level-by-level exploration."),
                "UCS": ("UCS", "Minimum-cost path. Defensive safe choice."),
                "DFS": ("DFS", "Deep aggressive exploration."),
                "Greedy": ("Greedy", "Fastest short-term pursuit."),
                "Decision Tree": ("Decision Tree", "Context-aware tactical rules."),
                "Beam Search": ("Beam Search", "Longer look-ahead with bounded width."),
                "IDA*": ("IDA*", "Low-memory optimal intercept search."),
            }
            rl_info = {
                "q_learning": ("Q-Learning", "Classic fast table learner"),
                "actor_critic": ("Actor-Critic", "Value-guided control refinement"),
                "online_learner": ("Online Learner", "Live pattern adaptation"),
                "dqn": ("DQN", "Deep Q tactical learner"),
                "ppo": ("PPO", "Stable long-horizon policy optimization"),
                "a2c": ("A2C", "Fast actor-critic updates"),
                "monte_carlo": ("Monte Carlo", "Episode reward learner"),
                "model_based": ("Model-Based", "Dyna-Q imagination rollouts"),
            }
            adv_info = {
                "anomaly": ("Anomaly", "Threat detection from unusual play"),
                "multi_task": ("Multi-Task", "Shared attack/defense/boost learner"),
                "passive_aggressive": ("Passive-Aggressive", "Online corrective classifier"),
                "maml": ("MAML", "Fast opponent adaptation"),
                "deep_rl": ("DeepRL Net", "Deep mechanic and replay learner"),
                "causal": ("Causal", "Counterfactual regret analysis"),
            }

            win = tk.Toplevel()
            win.title("Algorithm Control Center / مركز التحكم في الخوارزميات")
            win.geometry("560x560")
            win.configure(bg=BG)
            win.resizable(False, False)
            win.attributes("-topmost", True)
            win.grab_set()

            tk.Label(win, text="Algorithm Control Center  مركز التحكم في الخوارزميات",
                     fg="#7FDBFF", bg=BG, font=("Segoe UI", 13, "bold")).pack(pady=(12, 2))
            tk.Label(win,
                     text="Compact controls for Search, RL, and Advanced ML. Tick what you want active this match.",
                     fg="#AAAAAA", bg=BG, font=("Segoe UI", 8), wraplength=500).pack(pady=(0, 6))

            st = ttk.Style(win)
            st.theme_use("clam")
            st.configure("Algo.TNotebook", background=BG, borderwidth=0)
            st.configure("Algo.TNotebook.Tab", background="#162230", foreground="#7FDBFF", padding=[10, 4])
            st.map("Algo.TNotebook.Tab", background=[("selected", "#2ECC40")], foreground=[("selected", "white")])

            notebook = ttk.Notebook(win, style="Algo.TNotebook")
            notebook.pack(fill="both", expand=True, padx=14, pady=4)

            search_tab = tk.Frame(notebook, bg=BG)
            rl_tab = tk.Frame(notebook, bg=BG)
            adv_tab = tk.Frame(notebook, bg=BG)
            notebook.add(search_tab, text="Search")
            notebook.add(rl_tab, text="RL")
            notebook.add(adv_tab, text="Advanced")

            search_frame = tk.LabelFrame(search_tab, text=" Search Algorithms ", fg="#7FDBFF", bg=BG, font=("Segoe UI", 9, "bold"))
            search_frame.pack(fill="x", padx=10, pady=6)
            algo_vars: dict[str, tk.BooleanVar] = {}
            for algo in all_algos:
                name, desc = search_info[algo]
                var = tk.BooleanVar(value=(algo in current_algos))
                algo_vars[algo] = var
                row = tk.Frame(search_frame, bg=BG)
                row.pack(fill="x", padx=4, pady=1)
                tk.Checkbutton(row, variable=var, bg=BG, fg="white", activebackground=BG, selectcolor="#2C3E50").pack(side="left")
                tk.Label(row, text=name, fg="#FFFFFF", bg=BG, font=("Segoe UI", 9, "bold"), width=14, anchor="w").pack(side="left")
                tk.Label(row, text=desc, fg="#AAAAAA", bg=BG, font=("Segoe UI", 8), anchor="w").pack(side="left")

            notes_frame = tk.LabelFrame(search_tab, text=" Search Preset Info ", fg="#FFDC00", bg=BG, font=("Segoe UI", 9, "bold"))
            notes_frame.pack(fill="x", padx=10, pady=6)
            pros_var = tk.StringVar(value="Select a preset or tick boxes manually.")
            cons_var = tk.StringVar(value="")
            tk.Label(notes_frame, textvariable=pros_var, fg="#2ECC40", bg=BG, font=("Segoe UI", 8), justify="left", anchor="w", wraplength=480).pack(fill="x", padx=8, pady=(4, 0))
            tk.Label(notes_frame, textvariable=cons_var, fg="#FF4136", bg=BG, font=("Segoe UI", 8), justify="left", anchor="w", wraplength=480).pack(fill="x", padx=8, pady=(0, 4))

            preset_frame = tk.LabelFrame(search_tab, text=" Search Presets ", fg="#7FDBFF", bg=BG, font=("Segoe UI", 9, "bold"))
            preset_frame.pack(fill="x", padx=10, pady=6)
            preset_row1 = tk.Frame(preset_frame, bg=BG)
            preset_row1.pack(pady=3)
            preset_row2 = tk.Frame(preset_frame, bg=BG)
            preset_row2.pack(pady=(0, 4))
            preset_colors = {
                "Optimal (A* + UCS)": "#0074D9",
                "Fast (Greedy + Beam Search)": "#39CCCC",
                "Balanced ⭐ (A* + BFS + Greedy)": "#2ECC40",
                "Thorough (A* + BFS + DFS + UCS)": "#FF851B",
                "Full Ensemble (all 8)": "#FF4136",
            }

            def apply_preset(pname: str):
                algos, pros, cons = self._SEARCH_ALGO_PRESETS[pname]
                for alg_name, var in algo_vars.items():
                    var.set(alg_name in algos)
                pros_var.set(f"Pros: {pros}")
                cons_var.set(f"Cons: {cons}" if cons else "")
                if preset_var is not None:
                    preset_var.set(pname)

            for idx, (pname, color) in enumerate(preset_colors.items()):
                row = preset_row1 if idx < 3 else preset_row2
                tk.Button(row, text=pname, bg=color, fg="white", font=("Segoe UI", 8), cursor="hand2",
                          command=lambda n=pname: apply_preset(n)).pack(side="left", padx=3)

            rl_frame = tk.LabelFrame(rl_tab, text=" RL Models ", fg="#FF851B", bg=BG, font=("Segoe UI", 9, "bold"))
            rl_frame.pack(fill="x", padx=10, pady=6)
            rl_vars: dict[str, tk.BooleanVar] = {}
            for key in rl_keys:
                name, desc = rl_info[key]
                var = tk.BooleanVar(value=(key in current_rl))
                rl_vars[key] = var
                row = tk.Frame(rl_frame, bg=BG)
                row.pack(fill="x", padx=4, pady=1)
                tk.Checkbutton(row, variable=var, bg=BG, fg="white", activebackground=BG, selectcolor="#2C3E50").pack(side="left")
                tk.Label(row, text=name, fg="#FFFFFF", bg=BG, font=("Segoe UI", 9, "bold"), width=16, anchor="w").pack(side="left")
                tk.Label(row, text=desc, fg="#AAAAAA", bg=BG, font=("Segoe UI", 8), anchor="w").pack(side="left")

            tk.Label(rl_tab,
                     text="SARSA, heuristic rules, and reward shaping remain active in the base bot flow.",
                     fg="#AAAAAA", bg=BG, font=("Segoe UI", 8), justify="left", anchor="w", wraplength=480).pack(fill="x", padx=12, pady=(0, 6))

            adv_frame = tk.LabelFrame(adv_tab, text=" Advanced ML ", fg="#2ECC40", bg=BG, font=("Segoe UI", 9, "bold"))
            adv_frame.pack(fill="x", padx=10, pady=6)
            adv_vars: dict[str, tk.BooleanVar] = {}
            for key in adv_keys:
                name, desc = adv_info[key]
                var = tk.BooleanVar(value=(key in current_adv))
                adv_vars[key] = var
                row = tk.Frame(adv_frame, bg=BG)
                row.pack(fill="x", padx=4, pady=1)
                tk.Checkbutton(row, variable=var, bg=BG, fg="white", activebackground=BG, selectcolor="#2C3E50").pack(side="left")
                tk.Label(row, text=name, fg="#FFFFFF", bg=BG, font=("Segoe UI", 9, "bold"), width=18, anchor="w").pack(side="left")
                tk.Label(row, text=desc, fg="#AAAAAA", bg=BG, font=("Segoe UI", 8), anchor="w").pack(side="left")

            info_frame = tk.LabelFrame(adv_tab, text=" Core Files ", fg="#7FDBFF", bg=BG, font=("Segoe UI", 9, "bold"))
            info_frame.pack(fill="x", padx=10, pady=6)
            tk.Label(info_frame,
                     text=("adaptive_learner.py, advanced_ml.py, all_algorithms.py,\n"
                           "reward_calculator.py, rl_algorithms.py, rl_state.py"),
                     fg=FG, bg=BG, font=("Segoe UI", 8, "bold"), justify="left", anchor="w", wraplength=480).pack(fill="x", padx=8, pady=8)

            action_frame = tk.Frame(win, bg=BG)
            action_frame.pack(fill="x", padx=16, pady=8)

            def on_apply():
                chosen = [name for name, var in algo_vars.items() if var.get()]
                if not chosen:
                    chosen = ["A*", "BFS", "Greedy"]
                chosen_rl = [name for name, var in rl_vars.items() if var.get()]
                if not chosen_rl:
                    chosen_rl = ["q_learning", "actor_critic", "dqn"]
                chosen_adv = [name for name, var in adv_vars.items() if var.get()]
                if not chosen_adv:
                    chosen_adv = ["anomaly", "maml", "causal"]

                algo_str = ",".join(chosen)
                rl_str = ",".join(chosen_rl)
                adv_str = ",".join(chosen_adv)
                if self._vars.get("active_search_algorithms") is None:
                    import tkinter as _tk
                    self._vars["active_search_algorithms"] = _tk.StringVar(value=algo_str)
                else:
                    self._vars["active_search_algorithms"].set(algo_str)
                if self._vars.get("active_rl_models") is None:
                    import tkinter as _tk
                    self._vars["active_rl_models"] = _tk.StringVar(value=rl_str)
                else:
                    self._vars["active_rl_models"].set(rl_str)
                if self._vars.get("active_advanced_models") is None:
                    import tkinter as _tk
                    self._vars["active_advanced_models"] = _tk.StringVar(value=adv_str)
                else:
                    self._vars["active_advanced_models"].set(adv_str)

                if preset_var is not None:
                    matched = next(
                        (name for name, (algos, _, _) in self._SEARCH_ALGO_PRESETS.items() if set(algos) == set(chosen)),
                        "Custom",
                    )
                    preset_var.set(matched)
                win.destroy()

            tk.Button(action_frame, text="Apply  تطبيق", width=12, bg="#2ECC40", fg="white", command=on_apply).pack(side="left", padx=4)
            tk.Button(action_frame, text="Cancel  إلغاء", width=12, bg="#666666", fg="white", command=win.destroy).pack(side="left", padx=4)
    if use_gui and settings_gui is not None:
        settings_gui.destroy()


def main():
    global _starting_mode
    cfg = _load_launcher_settings()

    default_mode = cfg.get('match', 'mode', fallback='1v1')
    default_skill = cfg.getfloat('match', 'opponent_skill', fallback=0.5)
    default_starting = cfg.get('bot', 'starting_mode', fallback='balanced')

    parser = argparse.ArgumentParser(description="Rocket League search-only AI launcher")
    parser.add_argument("--mode", type=str, default=default_mode, help="Match mode like 1v1, 2v2, 3v3")
    parser.add_argument("--opponent-skill", type=float, default=default_skill, help="Psyonix bot skill in [0.0, 1.0]")
    parser.add_argument("--starting-mode", type=str, default=default_starting,
                        choices=["balanced", "attack", "defense", "manual"],
                        help="Initial bot mode")
    parser.add_argument("--no-gui", action="store_true",
                        help="Run bot without GUI (used when launched from control panel)")
    args = parser.parse_args()

    _starting_mode = args.starting_mode
    run(args.mode, max(0.0, min(1.0, args.opponent_skill)), no_gui=args.no_gui)


if __name__ == "__main__":
    main()
