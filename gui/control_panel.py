from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from typing import Dict, List, Optional, Set

try:
    import tkinter as tk
    from tkinter import ttk, messagebox
except Exception:
    tk = None
    ttk = None
    messagebox = None

# Project root (gui/ → parent)
_GUI_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_GUI_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    import runtime.ipc as _ipc
except Exception:
    _ipc = None  # type: ignore[assignment]

try:
    from runtime.refresh_loop import RefreshLoop as _RefreshLoop
except Exception:
    _RefreshLoop = None  # type: ignore[assignment,misc]

try:
    from model.model_status import ModelStatus as _ModelStatus
except Exception:
    _ModelStatus = None  # type: ignore[assignment,misc]

_VENV_PYTHON = os.path.join(_PROJECT_ROOT, ".venv", "Scripts", "python.exe")
if not os.path.isfile(_VENV_PYTHON):
    _VENV_PYTHON = sys.executable

_MATCH_SETTINGS_FILE   = os.path.join(_PROJECT_ROOT, "runtime", "_match_settings.json")
_MODEL_SETTINGS_FILE   = os.path.join(_PROJECT_ROOT, "model",   "model_settings.json")
_ALGO_EXPLANATION_FILE = os.path.join(_PROJECT_ROOT, "documentation", "algorithm_explanation.txt")
_ALGO_STATS_FILE       = os.path.join(_PROJECT_ROOT, "model", "algorithm_stats.json")

# Ordered list used by the model selector dialog
_ALL_MODEL_KEYS = [
    "q_learning", "actor_critic", "online_learner",
    "dqn", "ppo", "a2c", "monte_carlo", "model_based",
]

# Human-readable names and one-line descriptions
_MODEL_INFO = {
    "q_learning":    ("Q-Learning",      "Classic table lookup — instant, low CPU"),
    "actor_critic":  ("Actor-Critic",     "Value-based control refinement"),
    "online_learner":("Online Learner",   "Real-time per-state adaptation"),
    "dqn":           ("DQN",              "Deep Q-Network — learns complex tactics"),
    "ppo":           ("PPO",              "Stable policy gradient, great long-term"),
    "a2c":           ("A2C",              "Fast advantage actor-critic"),
    "monte_carlo":   ("Monte Carlo",      "Episode-level goal/concede rewards"),
    "model_based":   ("Model-Based RL",   "Dyna-Q imagination rollouts"),
}

# Preset combinations: name -> (models set, pros, cons)
_PRESETS = {
    "Lightning": (
        {"q_learning", "actor_critic", "online_learner"},
        "Fastest CPU, instant response, low memory usage",
        "Limited to tabular patterns; slower at learning complex tactics",
    ),
    "Deep Core": (
        {"dqn", "ppo", "a2c"},
        "Neural-net power; learns intricate tactics over time",
        "Needs warmup matches; higher CPU/memory footprint",
    ),
    "Hybrid ⭐": (
        {"q_learning", "actor_critic", "dqn", "ppo", "online_learner"},
        "Quick-start via Q-Learning + deep DQN/PPO power; recommended balance",
        "Moderate CPU usage; PPO and Q-Learning can disagree during warmup",
    ),
    "Full Power": (
        set(_ALL_MODEL_KEYS),
        "All models vote — maximum intelligence and adaptability",
        "Highest CPU; A2C and PPO are partially redundant",
    ),
    "Imitation": (
        {"q_learning", "online_learner", "monte_carlo"},
        "Best when you play Manual mode first (learns from you); episode rewards",
        "Less effective when playing autonomously without human demos",
    ),
}


def _fmt_active(models: set) -> str:
    """Short label for the active models set shown in the main panel."""
    if not models or models >= set(_ALL_MODEL_KEYS):
        return "All models active (8)"
    for name, (keys, *_) in _PRESETS.items():
        if keys == models:
            return f"Preset: {name} ({len(models)} models)"
    return f"Custom: {len(models)} of {len(_ALL_MODEL_KEYS)} models active"


class ControlPanel:
    def __init__(self, initial_mode: str = "manual"):
        self.current_algorithm = "A*"
        self.current_planner_mode = "ensemble"
        self.path_cost = 0.0
        self.last_compute_ms = 0.0
        self.current_mode = initial_mode
        self.temporary_reset = False
        self.runtime_status = "Waiting for game..."
        self.surface_status = "Car: unknown | Ball: unknown"
        self.game_mode_display = "soccer"
        self.setup_hint = "Press L or click New Match after the round"
        self._usage: Dict[str, int] = {"A*": 0, "BFS": 0, "UCS": 0, "Greedy": 0, "DFS": 0, "Decision Tree": 0, "Beam Search": 0, "IDA*": 0}
        self._timing: Dict[str, float] = {"A*": 0.0, "BFS": 0.0, "UCS": 0.0, "Greedy": 0.0, "DFS": 0.0, "Decision Tree": 0.0, "Beam Search": 0.0, "IDA*": 0.0}
        self._timing_count: Dict[str, int] = {"A*": 0, "BFS": 0, "UCS": 0, "Greedy": 0, "DFS": 0, "Decision Tree": 0, "Beam Search": 0, "IDA*": 0}
        self._mode_override_pending: Optional[str] = None
        self._model_override_pending: Optional[bool] = None
        self._open_setup_pending = False
        self._gui_started = False
        self._gui_error: Optional[str] = None
        self._gui_ready = threading.Event()
        self._root = None
        # Model selector state
        self._pending_model_selection: Optional[Set[str]] = None
        self.active_models: Set[str] = set(_ALL_MODEL_KEYS)
        self.threat_level: float = 0.0

        if tk is not None:
            thread = threading.Thread(target=self._run_gui, daemon=True)
            thread.start()
        else:
            self._gui_error = "tkinter is unavailable in this Python environment"
            self._gui_ready.set()

    def _run_gui(self):
        try:
            root = tk.Tk()
            self._root = root
            root.title("Rocket League Search AI")
            root.geometry("440x345")
            root.resizable(False, False)
            root.configure(bg="#101820")
            root.attributes('-topmost', True)
            root.lift()
            root.focus_force()

            self._mode_var = tk.StringVar(value=f"Mode: {self.current_mode}")
            self._model_var = tk.StringVar(value="Model: PERSISTENT")
            self._algo_var = tk.StringVar(value=f"Algorithm: {self.current_algorithm}")
            self._gamemode_var = tk.StringVar(value=f"Game: {self.game_mode_display}")
            self._status_var = tk.StringVar(value=self.runtime_status)
            self._surface_var = tk.StringVar(value=self.surface_status)
            self._setup_var = tk.StringVar(value=self.setup_hint)
            self._active_models_var = tk.StringVar(value=_fmt_active(self.active_models))
            self._threat_var = tk.StringVar(value="")

            def make_label(var, fg, font):
                lbl = tk.Label(root, textvariable=var, fg=fg, bg="#101820", font=font, anchor="w", justify="left")
                lbl.pack(fill="x", padx=12, pady=4)
                return lbl

            make_label(self._mode_var,         "#7FDBFF", ("Segoe UI", 13, "bold"))
            make_label(self._model_var,         "#39CCCC", ("Segoe UI", 13, "bold"))
            make_label(self._active_models_var, "#FF851B", ("Segoe UI",  9))
            make_label(self._gamemode_var,      "#2ECC40", ("Segoe UI", 11, "bold"))
            make_label(self._algo_var,          "#FFFFFF", ("Segoe UI", 10, "bold"))
            make_label(self._status_var,        "#F5F5F5", ("Segoe UI",  9))
            make_label(self._surface_var,       "#FFDC00", ("Segoe UI",  9))
            make_label(self._threat_var,        "#FF4136", ("Segoe UI",  9, "bold"))
            make_label(self._setup_var,         "#7FDBFF", ("Segoe UI",  9, "bold"))

            btn_frame = tk.Frame(root, bg="#101820")
            btn_frame.pack(fill="x", padx=12, pady=6)

            def mode_button(text, value, bg):
                tk.Button(
                    btn_frame, text=text, width=8, bg=bg, fg="white",
                    command=lambda: self._queue_mode_override(value)
                ).pack(side="left", padx=4)

            mode_button("Manual",   "manual",   "#B10DC9")
            mode_button("Balanced", "balanced", "#2ECC40")
            mode_button("Attack",   "attack",   "#FF851B")
            mode_button("Defense",  "defense",  "#0074D9")

            model_frame = tk.Frame(root, bg="#101820")
            model_frame.pack(fill="x", padx=12, pady=4)
            tk.Button(
                model_frame, text="Session (P)", width=11, bg="#FF4136", fg="white",
                command=lambda: self._queue_model_override(True)
            ).pack(side="left", padx=3)
            tk.Button(
                model_frame, text="Persistent (O)", width=11, bg="#39CCCC", fg="black",
                command=lambda: self._queue_model_override(False)
            ).pack(side="left", padx=3)
            tk.Button(
                model_frame, text="New Match (L)", width=11, bg="#FFDC00", fg="black",
                command=self._queue_open_setup
            ).pack(side="left", padx=3)
            tk.Button(
                model_frame, text="Models (I)", width=9, bg="#FF851B", fg="white",
                command=lambda: root.after_idle(self._show_model_selector)
            ).pack(side="left", padx=3)

            self._gui_started = True
            self._gui_error = None
            self._gui_ready.set()

            def refresh():
                self._mode_var.set(f"Mode: {self.current_mode.upper()}")
                model_name = "SESSION MODEL" if self.temporary_reset else "PERSISTENT MODEL"
                self._model_var.set(f"Model: {model_name}")
                self._active_models_var.set(_fmt_active(self.active_models))
                self._gamemode_var.set(f"Game: {self.game_mode_display.upper()}")
                planner = self.current_planner_mode.upper()
                self._algo_var.set(f"Planner: {planner} | Winner: {self.current_algorithm} | Cost: {self.path_cost:.1f}")
                self._status_var.set(self.runtime_status)
                self._surface_var.set(self.surface_status)
                # Threat level from anomaly detection
                _thr = self.threat_level
                if _thr > 0.7:
                    self._threat_var.set(f"⚠ THREAT HIGH  {_thr:.2f}  (شبح خطر)")
                elif _thr > 0.4:
                    self._threat_var.set(f"⚠ Threat: {_thr:.2f}")
                else:
                    self._threat_var.set("")
                self._setup_var.set(self.setup_hint)
                root.after(150, refresh)

            refresh()
            root.mainloop()
            # Null out StringVar refs so GC doesn't call back into a dead Tk thread
            for _attr in (
                '_mode_var', '_model_var', '_algo_var', '_gamemode_var',
                '_status_var', '_surface_var', '_setup_var', '_active_models_var',
                '_threat_var',
            ):
                try:
                    setattr(self, _attr, None)
                except Exception:
                    pass
        except Exception as exc:
            self._gui_started = False
            self._gui_error = str(exc)
            self._gui_ready.set()

    # ── Model selector dialog ────────────────────────────────────────
    def _show_model_selector(self):
        """Open the RL model selection Toplevel dialog (called from GUI thread)."""
        if tk is None or self._root is None:
            return
        win = tk.Toplevel(self._root)
        win.title("Choose RL Models")
        win.geometry("480x540")
        win.configure(bg="#101820")
        win.resizable(False, False)
        win.attributes('-topmost', True)
        win.grab_set()

        tk.Label(win, text="RL Model Selection",
                 fg="#7FDBFF", bg="#101820", font=("Segoe UI", 13, "bold")).pack(pady=(10, 2))
        tk.Label(win, text="Pick which models vote on decisions each frame (SARSA + heuristic always active):",
                 fg="#AAAAAA", bg="#101820", font=("Segoe UI", 8), wraplength=440).pack(pady=(0, 6))

        # ── Checkboxes ───────────────────────────────────────────────
        check_outer = tk.LabelFrame(win, text=" Models ",
                                    fg="#7FDBFF", bg="#101820", font=("Segoe UI", 9, "bold"))
        check_outer.pack(fill="x", padx=16, pady=4)

        vars_: Dict[str, tk.BooleanVar] = {}
        for key in _ALL_MODEL_KEYS:
            name, desc = _MODEL_INFO[key]
            var = tk.BooleanVar(value=(key in self.active_models))
            vars_[key] = var
            row = tk.Frame(check_outer, bg="#101820")
            row.pack(fill="x", padx=4, pady=1)
            tk.Checkbutton(row, variable=var, bg="#101820", fg="white",
                           activebackground="#101820", selectcolor="#2C3E50").pack(side="left")
            tk.Label(row, text=name, fg="#FFFFFF", bg="#101820",
                     font=("Segoe UI", 9, "bold"), width=15, anchor="w").pack(side="left")
            tk.Label(row, text=desc, fg="#AAAAAA", bg="#101820",
                     font=("Segoe UI", 8), anchor="w").pack(side="left")

        # ── Pros / cons text area ────────────────────────────────────
        notes_frame = tk.LabelFrame(win, text=" Preset Notes ",
                                    fg="#FFDC00", bg="#101820", font=("Segoe UI", 9, "bold"))
        notes_frame.pack(fill="x", padx=16, pady=4)
        pros_var = tk.StringVar(value="Select a preset below, or tick boxes manually, then Apply.")
        cons_var = tk.StringVar(value="")
        tk.Label(notes_frame, textvariable=pros_var, fg="#2ECC40", bg="#101820",
                 font=("Segoe UI", 8), justify="left", anchor="w",
                 wraplength=430).pack(fill="x", padx=8, pady=(4, 0))
        tk.Label(notes_frame, textvariable=cons_var, fg="#FF4136", bg="#101820",
                 font=("Segoe UI", 8), justify="left", anchor="w",
                 wraplength=430).pack(fill="x", padx=8, pady=(0, 4))

        # ── Preset buttons ───────────────────────────────────────────
        preset_frame = tk.LabelFrame(win, text=" Recommended Presets ",
                                     fg="#7FDBFF", bg="#101820", font=("Segoe UI", 9, "bold"))
        preset_frame.pack(fill="x", padx=16, pady=4)
        btn_row = tk.Frame(preset_frame, bg="#101820")
        btn_row.pack(pady=5)

        preset_colors = {
            "Lightning":  "#39CCCC",
            "Deep Core":  "#0074D9",
            "Hybrid ⭐":  "#2ECC40",
            "Full Power": "#FF851B",
            "Imitation":  "#B10DC9",
        }

        def apply_preset(preset_name: str):
            keys, pros, cons = _PRESETS[preset_name]
            for k, v in vars_.items():
                v.set(k in keys)
            pros_var.set(f"Pros: {pros}")
            cons_var.set(f"Cons: {cons}")

        for pname, color in preset_colors.items():
            tk.Button(btn_row, text=pname, width=9, bg=color, fg="white",
                      font=("Segoe UI", 8),
                      command=lambda n=pname: apply_preset(n)).pack(side="left", padx=3)

        # ── Apply / Cancel ───────────────────────────────────────────
        action_frame = tk.Frame(win, bg="#101820")
        action_frame.pack(fill="x", padx=16, pady=8)

        def on_apply():
            selected = {k for k, v in vars_.items() if v.get()}
            if not selected:
                selected = {"q_learning", "actor_critic", "dqn"}  # safety fallback
            self.active_models = selected
            self._pending_model_selection = selected
            win.destroy()

        def on_cancel():
            win.destroy()

        tk.Button(action_frame, text="Apply",  width=10, bg="#2ECC40", fg="white",
                  command=on_apply).pack(side="left", padx=4)
        tk.Button(action_frame, text="Cancel", width=10, bg="#666666", fg="white",
                  command=on_cancel).pack(side="left", padx=4)
        tk.Label(action_frame, text="Keyboard: I to reopen this dialog",
                 fg="#666666", bg="#101820", font=("Segoe UI", 8)).pack(side="left", padx=10)

    def wait_for_startup(self, timeout: float = 2.0) -> bool:
        self._gui_ready.wait(timeout)
        return self._gui_started

    def startup_error(self) -> Optional[str]:
        return self._gui_error

    def close(self):
        root = self._root
        if root is None:
            return
        self._root = None
        try:
            root.after_idle(root.destroy)
        except Exception:
            pass

    def hide(self):
        """Minimize / withdraw the panel window during active gameplay."""
        root = self._root
        if root is None:
            return
        try:
            root.after_idle(root.withdraw)
        except Exception:
            pass

    def show(self):
        """Restore the panel window (e.g. when the match ends)."""
        root = self._root
        if root is None:
            return
        try:
            def _restore():
                root.deiconify()
                root.attributes('-topmost', True)
                root.lift()
                root.focus_force()
            root.after_idle(_restore)
        except Exception:
            pass

    def _queue_mode_override(self, mode: str):
        self._mode_override_pending = mode

    def _queue_model_override(self, temporary_reset: bool):
        self._model_override_pending = bool(temporary_reset)

    def _queue_open_setup(self):
        self._open_setup_pending = True

    def consume_mode_override(self) -> Optional[str]:
        override = self._mode_override_pending
        self._mode_override_pending = None
        return override

    def consume_model_override(self) -> Optional[bool]:
        override = self._model_override_pending
        self._model_override_pending = None
        return override

    def consume_model_selection(self) -> Optional[Set[str]]:
        """Return the newly chosen model set (or None if unchanged since last call)."""
        sel = self._pending_model_selection
        self._pending_model_selection = None
        return sel

    def consume_open_setup_request(self) -> bool:
        pending = self._open_setup_pending
        self._open_setup_pending = False
        return pending

    def set_mode(self, mode: str):
        self.current_mode = mode

    def set_temporary_reset(self, temporary_reset: bool):
        self.temporary_reset = bool(temporary_reset)

    def set_runtime_status(self, status: str):
        self.runtime_status = status

    def set_surface_info(self, info: str):
        self.surface_status = info

    def set_game_mode(self, mode: str):
        self.game_mode_display = mode

    def set_algorithm(self, algorithm: str):
        self.current_algorithm = algorithm
        if algorithm in self._usage:
            self._usage[algorithm] += 1

    def set_planner_mode(self, planner_mode: str):
        self.current_planner_mode = planner_mode

    def set_setup_hint(self, setup_hint: str):
        self.setup_hint = setup_hint

    def set_path_cost(self, cost: float):
        self.path_cost = cost

    def record_timing(self, algorithm: str, elapsed_ms: float):
        self.last_compute_ms = elapsed_ms
        if algorithm in self._timing:
            self._timing[algorithm] += elapsed_ms
            self._timing_count[algorithm] += 1

    def average_timing_ms(self) -> Dict[str, float]:
        result: Dict[str, float] = {}
        for name in self._timing:
            count = self._timing_count[name]
            result[name] = self._timing[name] / count if count > 0 else 0.0
        return result

    def usage_percentage(self) -> Dict[str, float]:
        total = sum(self._usage.values())
        if total == 0:
            return {name: 0.0 for name in self._usage}
        return {name: (count / total) * 100.0 for name, count in self._usage.items()}

    def usage_counts(self) -> Dict[str, int]:
        return dict(self._usage)

    def set_threat_level(self, level: float):
        """Update the anomaly-detection threat level shown on the overlay."""
        self.threat_level = max(0.0, min(1.0, float(level)))


# ═══════════════════════════════════════════════════════════════════════════════
#  NEW STANDALONE DASHBOARD  (main-thread, launched via main())
# ═══════════════════════════════════════════════════════════════════════════════

# ── Colours ─────────────────────────────────────────────────────────────────
_C = {
    "bg":         "#0D1117",
    "panel":      "#161B22",
    "sidebar":    "#0D1117",
    "border":     "#30363D",
    "text":       "#C9D1D9",
    "dim":        "#8B949E",
    "accent":     "#58A6FF",
    "green":      "#3FB950",
    "red":        "#F85149",
    "yellow":     "#D29922",
    "orange":     "#E3B341",
    "purple":     "#BC8CFF",
    "cyan":       "#39D0D8",
    "header":     "#21262D",
    "btn_active": "#238636",
    "btn_stop":   "#DA3633",
    "btn_nav":    "#1C2128",
    "btn_navhl":  "#21262D",
}

# ── Search algorithm info ─────────────────────────────────────────────────────
_SEARCH_ALGO_INFO: Dict[str, tuple] = {
    "A*":            ("A* (A-Star)",         "Optimal & complete. f(n)=g(n)+h(n). Best general pathing.", "O(b^d)", "High memory"),
    "BFS":           ("Breadth First Search", "Guarantees shortest path by steps. Unweighted graph.", "O(b^d)", "Very high memory"),
    "UCS":           ("Uniform Cost Search",  "Optimal under non-negative costs. Safe, defensive.", "O(b^(C*/ε))", "Slow if costs vary little"),
    "Greedy":        ("Greedy Best First",    "Fast heuristic pursuit. Aggressive, not optimal.", "O(b^m)", "Can miss optimal path"),
    "DFS":           ("Depth First Search",   "Low memory baseline. Not optimal or complete.", "O(b^m)", "Only O(b·m) space"),
    "IDA*":          ("Iterative Deepening A*","Memory-efficient A*. Iterative depth cutoffs.", "O(b^d)", "Revisits nodes"),
    "Beam Search":   ("Beam Search",          "Long-horizon planning. Keeps top-k candidates.", "O(k·b·d)", "May miss optimal"),
    "Decision Tree": ("Decision Tree",        "Rule-based tactic selector. Fast, interpretable.", "O(depth)", "Needs good rules"),
    "Ball Prediction":("Ball Prediction",     "Simulates future ball positions for intercepts.", "O(frames)", "Computational cost"),
}

# ── RL model info ─────────────────────────────────────────────────────────────
_RL_MODEL_INFO: Dict[str, tuple] = {
    "q_learning":     ("Q-Learning",       "Classic tabular Q(s,a). Instant, low CPU. Great baseline.", "Tabular"),
    "sarsa":          ("SARSA",             "On-policy Q-Learning variant. Conservative updates.", "Tabular"),
    "dqn":            ("DQN",               "Neural Q-Network. Learns complex function approximations.", "Neural"),
    "ppo":            ("PPO",               "Proximal Policy Optimisation. Stable policy gradient.", "Neural"),
    "a2c":            ("A2C",               "Synchronous advantage actor-critic. Fast stable updates.", "Neural"),
    "monte_carlo":    ("Monte Carlo",       "Episode-return based learning. Excellent for sparse rewards.", "Episode"),
    "model_based":    ("Model-Based RL",    "Dyna-Q with imagination rollouts. Sample efficient.", "Model"),
    "actor_critic":   ("Actor-Critic",      "Value + policy heads. Balances bias-variance.", "Neural"),
    "online_learner": ("Online Learner",    "Real-time state-to-action table. Fast online adaptation.", "Tabular"),
    "policy_gradient":("Policy Gradient",  "REINFORCE. Direct policy optimisation via returns.", "Neural"),
    "ensemble":       ("Ensemble Voting",   "All active models vote on the final action.", "Meta"),
}

# ── Smart algorithm combos ─────────────────────────────────────────────────────
_COMBOS: List[tuple] = [
    ("Aggressive Rush",   ["A*", "Greedy", "Ball Prediction"],      ["q_learning", "dqn"],         "Fast attack — high pressure, maximum offense"),
    ("Rock-Solid Defense",["UCS", "A*"],                             ["actor_critic", "ppo"],       "Cost-safe repositioning, block shot angles"),
    ("Counter Attack",    ["Beam Search", "A*"],                     ["ppo", "a2c"],                "Plan ahead, strike on opponent mistake"),
    ("Ball Possession",   ["A*", "BFS"],                             ["q_learning", "online_learner"],"Control tempo, reward small advances"),
    ("Demo Play",         ["Greedy", "A*"],                          ["dqn", "monte_carlo"],        "Demo-focused — reward 3× for demolitions"),
    ("Academic Baseline", ["A*", "BFS", "UCS", "DFS"],              ["q_learning", "online_learner"],"All classical algos — great for presentations"),
    ("Full AUTO",         list(_SEARCH_ALGO_INFO.keys()),            list(_RL_MODEL_INFO.keys()),   "All algorithms — maximum intelligence"),
]

# ── Arena list ────────────────────────────────────────────────────────────────
_ARENAS = [
    "DFHStadium", "Mannfield", "ChampionsField", "UrbanCentral",
    "BeckwithPark", "UtopiaColiseum", "wasteland", "NeoTokyo",
    "AquaDome", "Farmstead", "SaltyShores", "ForbiddenTemple",
    "Throwback Stadium", "StarbaseArc", "Park_P",
]
_GAME_MODES = ["Soccer", "Hoops", "Dropshot", "Rumble", "Heatseeker", "Gridiron", "Hockey"]
_MATCH_LENGTHS = ["5 Minutes", "10 Minutes", "20 Minutes", "Unlimited"]
_BOOST_OPTS = ["Default", "Unlimited", "Recharge (Slow)", "Recharge (Fast)", "No Boost"]
_GRAVITY_OPTS = ["Default", "Low", "High", "Super High"]
_TEAM_SIZES = ["1v1", "2v2", "3v3"]
_MAX_SCORE_OPTS = ["Unlimited", "1 Goal", "3 Goals", "5 Goals"]


def _load_algo_explanations() -> Dict[str, str]:
    """Parse algorithm_explanation.txt into a {algorithm_name: text} dict."""
    result: Dict[str, str] = {}
    try:
        with open(_ALGO_EXPLANATION_FILE, "r", encoding="utf-8") as f:
            text = f.read()
        # Split by numbered sections: try to extract A*, BFS, UCS, Greedy, DFS
        import re
        sections = re.split(r'\n(?=\w+[\w\s*()]+\n\s+)', text)
        for sec in sections:
            for key in _SEARCH_ALGO_INFO:
                if sec.strip().startswith(key):
                    result[key] = sec.strip()
                    break
    except Exception:
        pass
    return result


class DashboardApp:
    """Standalone Tkinter dashboard that runs on the main thread.
    Launches the bot as a subprocess with --no-gui and communicates via IPC."""

    def __init__(self, root: "tk.Tk"):
        self._root = root
        self._bot_proc: Optional[subprocess.Popen] = None  # type: ignore[type-arg]
        self._status_data: dict = {}
        self._reward_history: List[float] = []
        self._algo_explanations = _load_algo_explanations()

        # Settings state
        self._game_mode_var = tk.StringVar(value="Soccer")
        self._arena_var = tk.StringVar(value="DFHStadium")
        self._match_len_var = tk.StringVar(value="5 Minutes")
        self._boost_var = tk.StringVar(value="Default")
        self._gravity_var = tk.StringVar(value="Default")
        self._team_size_var = tk.StringVar(value="1v1")
        self._start_mode_var = tk.StringVar(value="balanced")
        self._strategy_preset_var = tk.StringVar(value="Balanced")
        self._temp_model_var = tk.StringVar(value="Persistent")
        self._max_score_var = tk.StringVar(value="Unlimited")
        self._algo_stats_data: dict = {}
        self._algo_stats_poll = 0

        # Search algo checkbox vars
        self._search_vars: Dict[str, tk.BooleanVar] = {
            k: tk.BooleanVar(value=True) for k in _SEARCH_ALGO_INFO
        }
        # RL model checkbox vars
        self._rl_vars: Dict[str, tk.BooleanVar] = {
            k: tk.BooleanVar(value=True) for k in _RL_MODEL_INFO
        }

        self._active_panel: str = "Dashboard"
        self._panel_frames: Dict[str, tk.Frame] = {}
        self._nav_buttons: Dict[str, tk.Button] = {}
        self._model_status_data: dict = {}

        self._build_ui()
        # Use RefreshLoop so the poll tick is always rescheduled, even on exception
        if _RefreshLoop is not None:
            self._refresh_loop = _RefreshLoop(self._poll_bot_status, interval_ms=500)
            self._refresh_loop.start(self._root)
        else:
            self._poll_bot_status()

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self):
        root = self._root
        root.title("Rocket League AI Dashboard — medo dyaa")
        root.geometry("1100x700")
        root.minsize(800, 550)
        root.configure(bg=_C["bg"])
        root.rowconfigure(0, weight=1)
        root.columnconfigure(1, weight=1)

        # ── Sidebar ──────────────────────────────────────────────────────────
        sidebar = tk.Frame(root, bg=_C["sidebar"], width=160)
        sidebar.grid(row=0, column=0, sticky="ns")
        sidebar.grid_propagate(False)
        sidebar.rowconfigure(20, weight=1)  # push launch buttons to bottom

        logo = tk.Label(sidebar, text="Rocket League AI Control Panel\nby medo dyaa",
                        bg=_C["sidebar"], fg=_C["accent"],
                        font=("Segoe UI", 10, "bold"), pady=8, wraplength=150, justify="center")
        logo.grid(row=0, column=0, sticky="ew", padx=8)
        tk.Frame(sidebar, bg=_C["border"], height=1).grid(row=1, column=0, sticky="ew", padx=8, pady=2)

        nav_items = [
            ("Dashboard",        "DASHBOARD"),
            ("Algorithms",       "ALGORITHMS"),
            ("Match Settings",   "MATCH SETTINGS"),
            ("Model Settings",   "MODEL SETTINGS"),
            ("AI Architecture",  "AI ARCHITECTURE"),
            ("Decision Flow",    "DECISION FLOW"),
            ("Analytics",        "ANALYTICS"),
            ("Stats",            "STATS"),
            ("Ball Prediction",  "BALL PREDICTION"),
            ("Match Control",    "MATCH CONTROL"),
            ("Guidance",         "GUIDANCE"),
            ("Presentation",     "PRESENTATION"),
            ("Strategy Book",    "STRATEGY BOOK"),
            ("Ability Discovery", "ABILITY DISCOVERY"),
            ("Team Config",       "TEAM CONFIG"),
        ]
        for row_idx, (label, panel_key) in enumerate(nav_items, start=2):
            btn = tk.Button(
                sidebar, text=label, anchor="w", padx=12,
                bg=_C["btn_nav"], fg=_C["text"],
                activebackground=_C["btn_navhl"], activeforeground=_C["accent"],
                relief="flat", font=("Segoe UI", 9),
                command=lambda k=panel_key: self._show_panel(k),
            )
            btn.grid(row=row_idx, column=0, sticky="ew", padx=4, pady=1)
            self._nav_buttons[panel_key] = btn

        tk.Frame(sidebar, bg=_C["border"], height=1).grid(row=20, column=0, sticky="ew", padx=8, pady=4)

        self._launch_btn = tk.Button(
            sidebar, text="▶  Launch Bot", bg=_C["btn_active"], fg="white",
            activebackground="#2EA043", font=("Segoe UI", 9, "bold"),
            relief="flat", padx=8, pady=4,
            command=self._launch_bot,
        )
        self._launch_btn.grid(row=21, column=0, sticky="ew", padx=6, pady=2)

        self._stop_btn = tk.Button(
            sidebar, text="■  Stop Bot", bg=_C["btn_stop"], fg="white",
            activebackground="#B91C1C", font=("Segoe UI", 9, "bold"),
            relief="flat", padx=8, pady=4,
            command=self._stop_bot, state="disabled",
        )
        self._stop_btn.grid(row=22, column=0, sticky="ew", padx=6, pady=2)

        self._restart_match_btn = tk.Button(
            sidebar, text="↺  Restart Match", bg=_C["btn_nav"], fg=_C["yellow"],
            font=("Segoe UI", 9), relief="flat", padx=8, pady=4,
            command=lambda: self._send_match_command("restart_match"),
        )
        self._restart_match_btn.grid(row=23, column=0, sticky="ew", padx=6, pady=1)

        self._next_match_btn = tk.Button(
            sidebar, text="⏭  Next Match", bg=_C["btn_nav"], fg=_C["cyan"],
            font=("Segoe UI", 9), relief="flat", padx=8, pady=4,
            command=lambda: self._send_match_command("next_match"),
        )
        self._next_match_btn.grid(row=24, column=0, sticky="ew", padx=6, pady=1)

        self._bot_status_lbl = tk.Label(
            sidebar, text="Bot: stopped", bg=_C["sidebar"],
            fg=_C["dim"], font=("Segoe UI", 8),
        )
        self._bot_status_lbl.grid(row=25, column=0, padx=4, pady=(2, 8))

        # ── Main content area ─────────────────────────────────────────────────
        self._main = tk.Frame(root, bg=_C["bg"])
        self._main.grid(row=0, column=1, sticky="nsew")
        self._main.rowconfigure(0, weight=1)
        self._main.columnconfigure(0, weight=1)

        self._build_dashboard_panel()
        self._build_algorithms_panel()
        self._build_match_settings_panel()
        self._build_model_settings_panel()
        self._build_architecture_panel()
        self._build_flowchart_panel()
        self._build_analytics_panel()
        self._build_stats_panel()
        self._build_ball_prediction_panel()
        self._build_match_control_panel()
        self._build_guidance_panel()
        self._build_presentation_panel()
        self._build_strategy_book_panel()
        self._build_ability_discovery_panel()
        self._build_team_config_panel()

        self._show_panel("DASHBOARD")

    # ── Panel management ─────────────────────────────────────────────────────

    def _show_panel(self, key: str):
        for pkey, frame in self._panel_frames.items():
            frame.grid_remove()
        if key in self._panel_frames:
            self._panel_frames[key].grid(row=0, column=0, sticky="nsew")
        self._active_panel = key
        for bkey, btn in self._nav_buttons.items():
            btn.configure(
                bg=_C["btn_navhl"] if bkey == key else _C["btn_nav"],
                fg=_C["accent"] if bkey == key else _C["text"],
            )

    def _new_panel(self, key: str) -> "tk.Frame":
        frame = tk.Frame(self._main, bg=_C["bg"])
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        self._panel_frames[key] = frame
        return frame

    def _section_header(self, parent, title: str, color=None) -> "tk.Label":
        lbl = tk.Label(parent, text=title,
                       bg=_C["header"], fg=color or _C["accent"],
                       font=("Segoe UI", 11, "bold"), anchor="w",
                       padx=12, pady=6,
                       relief="flat")
        lbl.pack(fill="x", padx=0, pady=(0, 1))
        return lbl

    # ── Dashboard panel ───────────────────────────────────────────────────────

    def _build_dashboard_panel(self):
        p = self._new_panel("DASHBOARD")
        self._section_header(p, "  Live Match Dashboard")

        # Stats grid
        stats_frame = tk.Frame(p, bg=_C["bg"])
        stats_frame.pack(fill="both", expand=True, padx=12, pady=6)
        stats_frame.columnconfigure((0, 1, 2), weight=1)
        stats_frame.rowconfigure((0, 1, 2, 3), weight=1)

        def stat_card(parent, row, col, label, var, color):
            card = tk.Frame(parent, bg=_C["panel"], relief="flat")
            card.grid(row=row, column=col, sticky="nsew", padx=4, pady=4)
            tk.Label(card, text=label, bg=_C["panel"], fg=_C["dim"],
                     font=("Segoe UI", 8)).pack(anchor="w", padx=8, pady=(6, 0))
            lbl = tk.Label(card, textvariable=var, bg=_C["panel"], fg=color,
                           font=("Segoe UI", 14, "bold"))
            lbl.pack(anchor="w", padx=8, pady=(0, 6))
            return lbl

        self._dash_score_var     = tk.StringVar(value="0 — 0")
        self._dash_mode_var      = tk.StringVar(value="balanced")
        self._dash_algo_var      = tk.StringVar(value="A*")
        self._dash_strategy_var  = tk.StringVar(value="BALANCED")
        self._dash_state_var     = tk.StringVar(value="—")
        self._dash_reward_var    = tk.StringVar(value="0")
        self._dash_threat_var    = tk.StringVar(value="0.00")
        self._dash_tick_var      = tk.StringVar(value="0")
        self._dash_rl_role_var   = tk.StringVar(value="—")

        stat_card(stats_frame, 0, 0, "Score (us — them)", self._dash_score_var, _C["green"])
        stat_card(stats_frame, 0, 1, "Mode",              self._dash_mode_var,  _C["accent"])
        stat_card(stats_frame, 0, 2, "Algorithm",         self._dash_algo_var,  _C["cyan"])
        stat_card(stats_frame, 1, 0, "Strategy",          self._dash_strategy_var, _C["orange"])
        stat_card(stats_frame, 1, 1, "Game State",        self._dash_state_var,  _C["yellow"])
        stat_card(stats_frame, 1, 2, "Episode Reward",    self._dash_reward_var, _C["green"])
        stat_card(stats_frame, 2, 0, "Threat Level",      self._dash_threat_var, _C["red"])
        stat_card(stats_frame, 2, 1, "Tick Counter",      self._dash_tick_var,   _C["dim"])
        stat_card(stats_frame, 2, 2, "RL Role",           self._dash_rl_role_var, _C["purple"])

        # Reward breakdown
        rb_frame = tk.LabelFrame(p, text=" Reward Breakdown ", fg=_C["accent"],
                                 bg=_C["panel"], font=("Segoe UI", 9, "bold"))
        rb_frame.pack(fill="x", padx=12, pady=4)
        self._dash_reward_text = tk.Text(rb_frame, height=3, bg=_C["panel"],
                                          fg=_C["text"], font=("Consolas", 8),
                                          relief="flat", state="disabled")
        self._dash_reward_text.pack(fill="x", padx=8, pady=4)

        # Search + RL advice
        advice_frame = tk.LabelFrame(p, text=" Strategy Advice ", fg=_C["yellow"],
                                     bg=_C["panel"], font=("Segoe UI", 9, "bold"))
        advice_frame.pack(fill="x", padx=12, pady=4)
        self._dash_advice_var = tk.StringVar(value="Waiting for first match tick...")
        tk.Label(advice_frame, textvariable=self._dash_advice_var,
                 bg=_C["panel"], fg=_C["text"], font=("Segoe UI", 9),
                 wraplength=780, justify="left").pack(fill="x", padx=8, pady=4)

        # Model status bar
        model_frame = tk.LabelFrame(p, text=" Model Status ", fg=_C["cyan"],
                                    bg=_C["panel"], font=("Segoe UI", 9, "bold"))
        model_frame.pack(fill="x", padx=12, pady=4)
        self._dash_model_var = tk.StringVar(value="Model: — | State: — | LR: — | Goals: —")
        tk.Label(model_frame, textvariable=self._dash_model_var,
                 bg=_C["panel"], fg=_C["text"], font=("Consolas", 8),
                 anchor="w").pack(fill="x", padx=8, pady=4)

        # Match Progress section
        prog_frame = tk.LabelFrame(p, text=" Match Progress ", fg=_C["orange"],
                                   bg=_C["panel"], font=("Segoe UI", 9, "bold"))
        prog_frame.pack(fill="x", padx=12, pady=4)
        prog_inner = tk.Frame(prog_frame, bg=_C["panel"])
        prog_inner.pack(fill="x", padx=8, pady=6)
        self._dash_team_goals_var = tk.StringVar(value="0")
        self._dash_opp_goals_var  = tk.StringVar(value="0")
        self._dash_goal_diff_var  = tk.StringVar(value="±0")
        self._dash_goal_limit_var = tk.StringVar(value="7")
        self._dash_goals_left_var = tk.StringVar(value="7")
        for col, (lbl, var, color) in enumerate([
            ("Team Goals",  self._dash_team_goals_var, _C["green"]),
            ("Opp Goals",   self._dash_opp_goals_var,  _C["red"]),
            ("Difference",  self._dash_goal_diff_var,  _C["yellow"]),
            ("Goal Limit",  self._dash_goal_limit_var, _C["cyan"]),
            ("Need to Win", self._dash_goals_left_var, _C["orange"]),
        ]):
            prog_inner.columnconfigure(col, weight=1)
            cf = tk.Frame(prog_inner, bg=_C["panel"])
            cf.grid(row=0, column=col, sticky="ew", padx=4)
            tk.Label(cf, text=lbl, bg=_C["panel"], fg=_C["dim"],
                     font=("Segoe UI", 7)).pack()
            tk.Label(cf, textvariable=var, bg=_C["panel"], fg=color,
                     font=("Segoe UI", 13, "bold")).pack()

    # ── Algorithms panel ─────────────────────────────────────────────────────

    def _build_algorithms_panel(self):
        p = self._new_panel("ALGORITHMS")
        self._section_header(p, "  Algorithm Selection")

        outer = tk.Frame(p, bg=_C["bg"])
        outer.pack(fill="both", expand=True, padx=8, pady=4)
        outer.columnconfigure(0, weight=1)
        outer.columnconfigure(1, weight=1)
        outer.columnconfigure(2, weight=2)
        outer.rowconfigure(0, weight=1)

        # ── Search algorithms (left) ──────────────────────────────────────
        sf = tk.LabelFrame(outer, text=" Search Algorithms (9) ", fg=_C["accent"],
                           bg=_C["panel"], font=("Segoe UI", 9, "bold"))
        sf.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)

        for key, (name, desc, tc, sc) in _SEARCH_ALGO_INFO.items():
            row = tk.Frame(sf, bg=_C["panel"])
            row.pack(fill="x", padx=4, pady=1)
            cb = tk.Checkbutton(row, variable=self._search_vars[key],
                                bg=_C["panel"], fg=_C["text"],
                                selectcolor=_C["bg"], activebackground=_C["panel"],
                                command=lambda k=key: self._on_algo_click("search", k))
            cb.pack(side="left")
            lbl = tk.Label(row, text=name, bg=_C["panel"], fg=_C["text"],
                           font=("Segoe UI", 8, "bold"), cursor="hand2",
                           anchor="w")
            lbl.pack(side="left", fill="x")
            lbl.bind("<Button-1>", lambda e, k=key: self._show_algo_detail("search", k))

        # ── RL models (middle) ─────────────────────────────────────────────
        mf = tk.LabelFrame(outer, text=" RL Models (11) ", fg=_C["purple"],
                           bg=_C["panel"], font=("Segoe UI", 9, "bold"))
        mf.grid(row=0, column=1, sticky="nsew", padx=4, pady=4)

        for key, (name, desc, kind) in _RL_MODEL_INFO.items():
            row = tk.Frame(mf, bg=_C["panel"])
            row.pack(fill="x", padx=4, pady=1)
            cb = tk.Checkbutton(row, variable=self._rl_vars[key],
                                bg=_C["panel"], fg=_C["text"],
                                selectcolor=_C["bg"], activebackground=_C["panel"],
                                command=lambda k=key: self._on_algo_click("rl", k))
            cb.pack(side="left")
            lbl = tk.Label(row, text=name, bg=_C["panel"], fg=_C["text"],
                           font=("Segoe UI", 8, "bold"), cursor="hand2",
                           anchor="w")
            lbl.pack(side="left", fill="x")
            lbl.bind("<Button-1>", lambda e, k=key: self._show_algo_detail("rl", k))

        # ── Detail pane + combos (right) ──────────────────────────────────
        df = tk.Frame(outer, bg=_C["bg"])
        df.grid(row=0, column=2, sticky="nsew", padx=4, pady=4)
        df.rowconfigure(0, weight=1)
        df.rowconfigure(1, weight=0)
        df.columnconfigure(0, weight=1)

        detail_container = tk.LabelFrame(df, text=" Algorithm Details ", fg=_C["cyan"],
                                          bg=_C["panel"], font=("Segoe UI", 9, "bold"))
        detail_container.grid(row=0, column=0, sticky="nsew", pady=(0, 4))

        self._algo_detail_text = tk.Text(detail_container, bg=_C["panel"], fg=_C["text"],
                                          font=("Consolas", 8), relief="flat",
                                          state="disabled", wrap="word")
        self._algo_detail_text.pack(fill="both", expand=True, padx=6, pady=6)

        # Smart combos
        combo_frame = tk.LabelFrame(df, text=" Smart Combos ", fg=_C["yellow"],
                                     bg=_C["panel"], font=("Segoe UI", 9, "bold"))
        combo_frame.grid(row=1, column=0, sticky="ew", pady=(4, 0))

        for name, search, rl, desc in _COMBOS:
            row = tk.Frame(combo_frame, bg=_C["panel"])
            row.pack(fill="x", padx=4, pady=1)
            tk.Button(row, text=name, width=18, bg=_C["btn_nav"], fg=_C["accent"],
                      font=("Segoe UI", 8), relief="flat",
                      command=lambda s=search, r=rl: self._apply_combo(s, r)
                      ).pack(side="left", padx=2)
            tk.Label(row, text=desc, bg=_C["panel"], fg=_C["dim"],
                     font=("Segoe UI", 7)).pack(side="left", padx=4, fill="x")

        # Apply button
        apply_frame = tk.Frame(p, bg=_C["bg"])
        apply_frame.pack(fill="x", padx=8, pady=4)
        tk.Button(apply_frame, text="Apply Selection to Bot",
                  bg=_C["btn_active"], fg="white",
                  font=("Segoe UI", 9, "bold"), relief="flat",
                  padx=12, pady=4,
                  command=self._apply_algo_selection).pack(side="left", padx=4)
        tk.Button(apply_frame, text="Select All", bg=_C["btn_nav"], fg=_C["text"],
                  font=("Segoe UI", 8), relief="flat",
                  command=self._select_all_algos).pack(side="left", padx=4)
        tk.Button(apply_frame, text="Clear All", bg=_C["btn_nav"], fg=_C["text"],
                  font=("Segoe UI", 8), relief="flat",
                  command=self._clear_all_algos).pack(side="left", padx=4)

    # ── Match Settings panel ─────────────────────────────────────────────────

    def _build_match_settings_panel(self):
        p = self._new_panel("MATCH SETTINGS")
        self._section_header(p, "  Match Settings")

        content = tk.Frame(p, bg=_C["bg"])
        content.pack(fill="both", expand=True, padx=20, pady=10)
        content.columnconfigure(1, weight=1)

        def row_widget(parent, r, label, widget_factory):
            tk.Label(parent, text=label, bg=_C["bg"], fg=_C["dim"],
                     font=("Segoe UI", 9), anchor="e", width=18
                     ).grid(row=r, column=0, sticky="e", padx=(0, 8), pady=5)
            w = widget_factory(parent)
            w.grid(row=r, column=1, sticky="w", pady=5)

        def make_combo(parent, var, values, width=22):
            cb = ttk.Combobox(parent, textvariable=var, values=values,
                              state="readonly", width=width)
            cb.configure(style="Dark.TCombobox")
            return cb

        style = ttk.Style()
        style.configure("Dark.TCombobox",
                        fieldbackground=_C["panel"],
                        background=_C["panel"],
                        foreground=_C["text"],
                        selectbackground=_C["accent"],
                        selectforeground="white")

        row_widget(content, 0, "Game Mode:", lambda pr: make_combo(pr, self._game_mode_var, _GAME_MODES))
        row_widget(content, 1, "Arena:", lambda pr: make_combo(pr, self._arena_var, _ARENAS, width=28))
        row_widget(content, 2, "Team Size:", lambda pr: make_combo(pr, self._team_size_var, _TEAM_SIZES))
        row_widget(content, 3, "Match Length:", lambda pr: make_combo(pr, self._match_len_var, _MATCH_LENGTHS))
        row_widget(content, 4, "Boost Amount:", lambda pr: make_combo(pr, self._boost_var, _BOOST_OPTS))
        row_widget(content, 5, "Gravity:", lambda pr: make_combo(pr, self._gravity_var, _GRAVITY_OPTS))
        row_widget(content, 6, "Starting Mode:", lambda pr: make_combo(
            pr, self._start_mode_var, ["balanced", "attack", "defense", "manual"]))
        row_widget(content, 7, "Strategy Preset:", lambda pr: make_combo(
            pr, self._strategy_preset_var,
            ["Balanced", "Aggressive", "Defensive", "Counter", "Goalkeeper", "Possession", "Demo"]))
        row_widget(content, 8, "Temp Model:", lambda pr: make_combo(
            pr, self._temp_model_var,
            ["Persistent", "Experimental", "Training", "Testing"]))
        row_widget(content, 9, "Goal Limit:", lambda pr: make_combo(
            pr, self._max_score_var, _MAX_SCORE_OPTS))

        btn_frame = tk.Frame(p, bg=_C["bg"])
        btn_frame.pack(fill="x", padx=20, pady=6)
        tk.Button(btn_frame, text="Save & Launch Bot",
                  bg=_C["btn_active"], fg="white",
                  font=("Segoe UI", 9, "bold"), relief="flat",
                  padx=12, pady=6,
                  command=self._save_and_launch).pack(side="left", padx=4)
        tk.Button(btn_frame, text="Save Settings Only",
                  bg=_C["btn_nav"], fg=_C["text"],
                  font=("Segoe UI", 9), relief="flat",
                  padx=12, pady=6,
                  command=self._save_match_settings).pack(side="left", padx=4)

    # ── Model Settings panel ─────────────────────────────────────────────────

    def _build_model_settings_panel(self):
        p = self._new_panel("MODEL SETTINGS")
        self._section_header(p, "  RL Model Settings")

        content = tk.Frame(p, bg=_C["bg"])
        content.pack(fill="both", expand=True, padx=12, pady=8)
        content.columnconfigure((0, 1), weight=1)
        content.rowconfigure(0, weight=1)

        # Model checkboxes
        check_frame = tk.LabelFrame(content, text=" Active RL Models ",
                                     fg=_C["purple"], bg=_C["panel"],
                                     font=("Segoe UI", 9, "bold"))
        check_frame.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)

        for key, (name, desc, kind) in _RL_MODEL_INFO.items():
            if key not in self._rl_vars:
                self._rl_vars[key] = tk.BooleanVar(value=True)
            row = tk.Frame(check_frame, bg=_C["panel"])
            row.pack(fill="x", padx=6, pady=2)
            tk.Checkbutton(row, variable=self._rl_vars[key],
                           bg=_C["panel"], fg=_C["text"],
                           selectcolor=_C["bg"],
                           activebackground=_C["panel"]).pack(side="left")
            tk.Label(row, text=f"{name}", bg=_C["panel"], fg=_C["text"],
                     font=("Segoe UI", 8, "bold"), width=16, anchor="w").pack(side="left")
            tk.Label(row, text=f"[{kind}]  {desc}", bg=_C["panel"], fg=_C["dim"],
                     font=("Segoe UI", 7), anchor="w").pack(side="left", fill="x")

        # Preset buttons
        preset_frame = tk.LabelFrame(content, text=" Presets ",
                                      fg=_C["yellow"], bg=_C["panel"],
                                      font=("Segoe UI", 9, "bold"))
        preset_frame.grid(row=0, column=1, sticky="nsew", padx=4, pady=4)

        self._preset_notes_var = tk.StringVar(value="Click a preset to see details.")
        tk.Label(preset_frame, textvariable=self._preset_notes_var,
                 bg=_C["panel"], fg=_C["text"], font=("Segoe UI", 8),
                 wraplength=300, justify="left").pack(fill="x", padx=8, pady=(8, 4))

        preset_colors_map = {
            "Lightning": _C["cyan"], "Deep Core": _C["accent"],
            "Hybrid ⭐": _C["green"], "Full Power": _C["orange"],
            "Imitation": _C["purple"],
        }
        for pname, pcolor in preset_colors_map.items():
            keys, pros, cons = _PRESETS[pname]
            tk.Button(preset_frame, text=pname, width=18,
                      bg=_C["btn_nav"], fg=pcolor,
                      font=("Segoe UI", 9, "bold"), relief="flat",
                      command=lambda k=keys, pr=pros, co=cons: self._apply_model_preset(k, pr, co)
                      ).pack(fill="x", padx=12, pady=3)

        apply_frame = tk.Frame(p, bg=_C["bg"])
        apply_frame.pack(fill="x", padx=12, pady=4)
        tk.Button(apply_frame, text="Apply to Running Bot",
                  bg=_C["btn_active"], fg="white",
                  font=("Segoe UI", 9, "bold"), relief="flat",
                  padx=12, pady=4,
                  command=self._apply_model_settings).pack(side="left", padx=4)

    # ── AI Architecture panel ─────────────────────────────────────────────────

    def _build_architecture_panel(self):
        p = self._new_panel("AI ARCHITECTURE")
        self._section_header(p, "  AI System Architecture — 7 Layers")
        canvas = tk.Canvas(p, bg=_C["bg"], highlightthickness=0)
        canvas.pack(fill="both", expand=True, padx=8, pady=8)
        canvas.bind("<Configure>", lambda e: self._draw_architecture(canvas))
        self._arch_canvas = canvas

    def _draw_architecture(self, canvas: "tk.Canvas"):
        canvas.delete("all")
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        if cw < 10 or ch < 10:
            return

        layers = [
            ("Game Input",        "Sensor: pos/vel/boost/score  ➜  GameTickPacket", _C["cyan"]),
            ("State Detection",   "30 Game States: KICKOFF, ATTACK, BALL_CHASE, DEMO_OPPORTUNITY…", _C["accent"]),
            ("Strategy Layer",    "7 Strategies: AGGRESSIVE, DEFENSIVE, COUNTER, POSSESSION, DEMO, BOOST, BALANCED", _C["orange"]),
            ("Decision Engine",   "9 Search Algos: A*, BFS, UCS, DFS, Greedy, IDA*, Beam, DTree, Ball", _C["yellow"]),
            ("RL Ensemble",       "11 Models: Q-Learn, DQN, PPO, A2C, SARSA, Monte Carlo, Model-Based…", _C["purple"]),
            ("Path Planner",      "8-dir Grid  |  allowed_algorithms enforcement  |  PathResult(cost, path)", _C["green"]),
            ("Controller Output", "SimpleControllerState  ➜  throttle/steer/boost/jump/pitch/yaw/roll", _C["red"]),
        ]
        n = len(layers)
        margin_x = max(40, cw * 0.08)
        margin_y = 20
        box_w = cw - 2 * margin_x
        total_h = ch - 2 * margin_y
        box_h = min(56, (total_h - (n - 1) * 10) / n)
        step_y = (total_h - box_h) / max(n - 1, 1)
        cx = cw / 2

        for i, (title, desc, color) in enumerate(layers):
            y_top = margin_y + i * step_y
            y_bot = y_top + box_h
            x_left = margin_x
            x_right = margin_x + box_w

            # Shadow
            canvas.create_rectangle(x_left + 3, y_top + 3, x_right + 3, y_bot + 3,
                                     fill="#000000", outline="", tags="shadow")
            # Box
            canvas.create_rectangle(x_left, y_top, x_right, y_bot,
                                     fill=_C["panel"], outline=color, width=2)
            # Layer index
            canvas.create_text(x_left + 20, (y_top + y_bot) / 2,
                                text=str(i + 1), fill=color,
                                font=("Segoe UI", 11, "bold"))
            # Title
            canvas.create_text(x_left + 60, y_top + box_h * 0.32,
                                text=title, fill=color, anchor="w",
                                font=("Segoe UI", 9, "bold"))
            # Description
            canvas.create_text(x_left + 60, y_top + box_h * 0.70,
                                text=desc, fill=_C["dim"], anchor="w",
                                font=("Segoe UI", 7))

            # Arrow to next layer
            if i < n - 1:
                arr_y = y_bot + 2
                arr_y2 = y_bot + step_y - box_h + y_bot - 2
                canvas.create_line(cx, arr_y, cx, y_bot + step_y - box_h + y_bot,
                                   fill=color, width=2, arrow="last",
                                   arrowshape=(8, 10, 3))

    # ── Decision Flowchart panel ──────────────────────────────────────────────

    def _build_flowchart_panel(self):
        p = self._new_panel("DECISION FLOW")
        self._section_header(p, "  Decision Pipeline — 8 Steps")
        canvas = tk.Canvas(p, bg=_C["bg"], highlightthickness=0)
        canvas.pack(fill="both", expand=True, padx=8, pady=8)
        canvas.bind("<Configure>", lambda e: self._draw_flowchart(canvas))
        self._flow_canvas = canvas

    def _draw_flowchart(self, canvas: "tk.Canvas"):
        canvas.delete("all")
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        if cw < 10 or ch < 10:
            return

        steps = [
            ("Receive Packet",      "GameTickPacket arrives every ~16 ms",      _C["cyan"]),
            ("Poll IPC Commands",   "Read mode/algo overrides from GUI",          _C["accent"]),
            ("Detect Game State",   "30-state classifier: pos, vel, boost, score",_C["orange"]),
            ("Strategy Adapt",      "StrategyLayer selects 1 of 7 strategies",   _C["yellow"]),
            ("Filter Algorithms",   "active_search_algorithms whitelist applied", _C["purple"]),
            ("Parallel Ensemble",   "ThreadPool runs all allowed algos simultaneously",_C["cyan"]),
            ("Score & Select Best", "score_candidate() weights by mode/history", _C["green"]),
            ("Output + Learn",      "Apply controller state; update RL + reward",_C["red"]),
        ]
        n = len(steps)
        cols = 2
        rows = (n + 1) // 2
        margin_x = 30
        margin_y = 20
        gap_x = 20
        gap_y = 12
        area_w = cw - 2 * margin_x - gap_x
        area_h = ch - 2 * margin_y
        box_w = area_w / 2
        box_h = (area_h - (rows - 1) * gap_y) / rows

        positions = []
        for i, (title, desc, color) in enumerate(steps):
            col = i % 2
            row = i // 2
            x = margin_x + col * (box_w + gap_x)
            y = margin_y + row * (box_h + gap_y)
            positions.append((x, y, title, desc, color))

        for idx, (x, y, title, desc, color) in enumerate(positions):
            xr = x + box_w
            yr = y + box_h
            canvas.create_rectangle(x + 2, y + 2, xr + 2, yr + 2,
                                     fill="#000", outline="", tags="sh")
            canvas.create_rectangle(x, y, xr, yr,
                                     fill=_C["panel"], outline=color, width=2)
            label = f"Step {idx + 1}: {title}"
            canvas.create_text(x + box_w * 0.5, y + box_h * 0.33,
                                text=label, fill=color, anchor="center",
                                font=("Segoe UI", 8, "bold"))
            canvas.create_text(x + box_w * 0.5, y + box_h * 0.72,
                                text=desc, fill=_C["dim"], anchor="center",
                                font=("Segoe UI", 7))

        # Arrows: 1→2 (same row), 2→3 (down), 3→4 (same row), etc.
        for i in range(len(positions) - 1):
            x0, y0 = positions[i][0] + box_w, positions[i][1] + box_h / 2
            x1, y1 = positions[i + 1][0], positions[i + 1][1] + box_h / 2
            # Same row? horizontal arrow
            if i % 2 == 0:
                canvas.create_line(x0, y0, x1, y1,
                                   fill=positions[i][4], width=2,
                                   arrow="last", arrowshape=(8, 10, 3))
            else:
                # Different row: go down from right-col to next left-col
                mid_x = x0 + 10
                x1b = positions[i + 1][0] + box_w
                y1b = positions[i + 1][1] + box_h / 2
                canvas.create_line(x0, y0,  # right side of box i
                                   mid_x, y0,
                                   mid_x, y1b,
                                   x1b, y1b,
                                   fill=positions[i][4], width=2,
                                   arrow="last", arrowshape=(8, 10, 3))

    # ── Analytics panel ───────────────────────────────────────────────────────

    def _build_analytics_panel(self):
        p = self._new_panel("ANALYTICS")
        self._section_header(p, "  Algorithm Usage Analytics")

        self._analytics_canvas = tk.Canvas(p, bg=_C["bg"], highlightthickness=0, height=200)
        self._analytics_canvas.pack(fill="x", padx=8, pady=4)

        self._reward_canvas = tk.Canvas(p, bg=_C["bg"], highlightthickness=0)
        self._reward_canvas.pack(fill="both", expand=True, padx=8, pady=4)

        tk.Button(p, text="Refresh Charts", bg=_C["btn_nav"], fg=_C["text"],
                  font=("Segoe UI", 8), relief="flat",
                  command=self._refresh_analytics).pack(padx=8, pady=4, anchor="w")

    # ── Stats panel ───────────────────────────────────────────────────────────

    def _build_stats_panel(self):
        p = self._new_panel("STATS")
        self._section_header(p, "  Algorithm Success Rate")

        top_bar = tk.Frame(p, bg=_C["bg"])
        top_bar.pack(fill="x", padx=8, pady=4)
        tk.Button(top_bar, text="Refresh Stats", bg=_C["btn_nav"], fg=_C["text"],
                  font=("Segoe UI", 8), relief="flat",
                  command=self._refresh_algo_stats).pack(side="left", padx=4)
        self._stats_last_updated_var = tk.StringVar(value="Click Refresh to load")
        tk.Label(top_bar, textvariable=self._stats_last_updated_var,
                 bg=_C["bg"], fg=_C["dim"], font=("Segoe UI", 8)).pack(side="left", padx=8)

        self._stats_canvas = tk.Canvas(p, bg=_C["bg"], highlightthickness=0)
        self._stats_canvas.pack(fill="both", expand=True, padx=8, pady=4)
        self._stats_canvas.bind("<Configure>", lambda e: self._draw_stats_chart())

    def _refresh_algo_stats(self):
        import time
        try:
            with open(_ALGO_STATS_FILE, "r", encoding="utf-8") as f:
                self._algo_stats_data = json.load(f)
            self._stats_last_updated_var.set(f"Updated {time.strftime('%H:%M:%S')}")
        except Exception:
            self._stats_last_updated_var.set("No stats file found")
        self._draw_stats_chart()

    def _draw_stats_chart(self):
        if not hasattr(self, "_stats_canvas"):
            return
        canvas = self._stats_canvas
        canvas.delete("all")
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        if cw < 10 or ch < 10:
            return

        data = self._algo_stats_data
        if not data:
            canvas.create_text(cw // 2, ch // 2,
                                text="No statistics loaded \u2014 click Refresh Stats",
                                fill=_C["dim"], font=("Segoe UI", 10))
            return

        algos = list(data.keys())
        # Compute success rate: (goals+interceptions+saves) / max(attempts,1)
        def success_rate(d: dict) -> float:
            successes = d.get("goals_scored", 0) + d.get("interceptions", 0) + d.get("defensive_saves", 0)
            attempts  = max(1, d.get("attempts", 1))
            return min(100.0, successes / attempts * 100)

        rates = [success_rate(data[a]) for a in algos]
        n = len(algos)
        margin_l, margin_r, margin_t, margin_b = 70, 20, 40, 50
        bar_area_w = cw - margin_l - margin_r
        bar_w = bar_area_w / n * 0.65
        gap   = bar_area_w / n * 0.35
        max_h = ch - margin_t - margin_b
        colors = [_C["cyan"], _C["accent"], _C["orange"], _C["yellow"],
                  _C["purple"], _C["green"], _C["red"], _C["dim"]]

        canvas.create_text(cw // 2, 18, text="Algorithm Success Rate (%)",
                           fill=_C["accent"], font=("Segoe UI", 10, "bold"))

        for i, (algo, rate) in enumerate(zip(algos, rates)):
            x     = margin_l + i * (bar_w + gap)
            bar_h = rate / 100 * max_h
            y_bot = ch - margin_b
            y_top = y_bot - bar_h
            col   = colors[i % len(colors)]
            canvas.create_rectangle(x, y_top, x + bar_w, y_bot, fill=col, outline="")
            canvas.create_text(x + bar_w / 2, y_bot + 5,
                                text=algo[:8], fill=_C["dim"],
                                font=("Segoe UI", 7), anchor="n")
            canvas.create_text(x + bar_w / 2, max(y_top - 3, margin_t),
                                text=f"{rate:.0f}%", fill=col,
                                font=("Segoe UI", 8), anchor="s")
            # Raw numbers below bar label
            d = data[algo]
            detail = f"G:{d.get('goals_scored',0)} I:{d.get('interceptions',0)} S:{d.get('defensive_saves',0)}"
            canvas.create_text(x + bar_w / 2, y_bot + 18,
                                text=detail, fill=_C["dim"],
                                font=("Segoe UI", 6), anchor="n")

    # ── Ball Prediction panel ─────────────────────────────────────────────────

    def _build_ball_prediction_panel(self):
        p = self._new_panel("BALL PREDICTION")
        self._section_header(p, "  Ball Trajectory Prediction Viewer")

        ctrl = tk.Frame(p, bg=_C["bg"])
        ctrl.pack(fill="x", padx=8, pady=4)
        tk.Button(ctrl, text="Simulate Prediction", bg=_C["btn_nav"], fg=_C["cyan"],
                  font=("Segoe UI", 8), relief="flat",
                  command=self._run_ball_prediction).pack(side="left", padx=4)
        self._bp_status_var = tk.StringVar(value="Press Simulate or wait for live match data.")
        tk.Label(ctrl, textvariable=self._bp_status_var,
                 bg=_C["bg"], fg=_C["dim"], font=("Segoe UI", 8)).pack(side="left", padx=8)

        self._bp_canvas = tk.Canvas(p, bg="#0A1020", highlightthickness=1,
                                     highlightbackground=_C["border"])
        self._bp_canvas.pack(fill="both", expand=True, padx=8, pady=4)
        self._bp_canvas.bind("<Configure>", lambda e: self._draw_bp_field())
        self._bp_trajectory: list = []
        self._bp_intercept: Optional[tuple] = None
        self._bp_shoot_aim: Optional[tuple] = None

    def _run_ball_prediction(self):
        """Pull ball/car state from IPC status and run prediction."""
        try:
            from core.ball_prediction import predict_ball_path, find_best_intercept
            d = self._status_data
            # Use last known position from IPC; fall back to centre-field
            bpos = d.get("ball_pos", (0.0, 0.0, 93.0))
            bvel = d.get("ball_vel", (300.0, -600.0, 0.0))
            cpos = d.get("car_pos",  (0.0, -2000.0, 17.0))
            if isinstance(bpos, list): bpos = tuple(bpos)
            if isinstance(bvel, list): bvel = tuple(bvel)
            if isinstance(cpos, list): cpos = tuple(cpos)
            self._bp_trajectory = predict_ball_path(bpos, bvel, steps=90)
            self._bp_intercept  = find_best_intercept(self._bp_trajectory, cpos)
            self._bp_shoot_aim  = (0.0, 5120.0)
            self._bp_status_var.set(f"{len(self._bp_trajectory)} pts predicted")
        except Exception as e:
            self._bp_status_var.set(f"Error: {e}")
        self._draw_bp_field()

    def _draw_bp_field(self):
        if not hasattr(self, "_bp_canvas"):
            return
        canvas = self._bp_canvas
        canvas.delete("all")
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        if cw < 10 or ch < 10:
            return

        # Map field coords to canvas coords
        # Field: X in [-4096,4096], Y in [-5120, 5120]
        def to_c(fx, fy):
            px = (fx + 4096) / 8192 * cw
            py = (1.0 - (fy + 5120) / 10240) * ch
            return px, py

        # Field outline
        x0, y0 = to_c(-4096, -5120)
        x1, y1 = to_c( 4096,  5120)
        canvas.create_rectangle(x0, y0, x1, y1, outline="#1A3A5C", width=1)
        # Halfway line
        mx0, my0 = to_c(-4096, 0)
        mx1, my1 = to_c( 4096, 0)
        canvas.create_line(mx0, my0, mx1, my1, fill="#1A3A5C", dash=(4, 4))
        # Goals
        gx0, gy0 = to_c(-893, 5020)
        gx1, gy1 = to_c( 893, 5120)
        canvas.create_rectangle(gx0, gy0, gx1, gy1, outline="#3FB950", width=2)
        ogx0, ogy0 = to_c(-893, -5120)
        ogx1, ogy1 = to_c( 893, -5020)
        canvas.create_rectangle(ogx0, ogy0, ogx1, ogy1, outline="#F85149", width=2)

        # Trajectory (cyan path)
        traj = self._bp_trajectory
        if len(traj) >= 2:
            for i in range(1, len(traj), 3):
                ax, ay = to_c(traj[i-1][0], traj[i-1][1])
                bx, by = to_c(traj[i][0],   traj[i][1])
                canvas.create_line(ax, ay, bx, by, fill="#39D0D8", width=2)

        # Intercept (green dot)
        if self._bp_intercept:
            ix, iy = to_c(self._bp_intercept[0], self._bp_intercept[1])
            canvas.create_oval(ix-6, iy-6, ix+6, iy+6, fill="#3FB950", outline="")
            canvas.create_text(ix+10, iy, text="intercept", fill="#3FB950",
                                font=("Segoe UI", 7), anchor="w")

        # Shoot aim (red dot)
        if self._bp_shoot_aim:
            sx, sy = to_c(self._bp_shoot_aim[0], self._bp_shoot_aim[1])
            canvas.create_oval(sx-5, sy-5, sx+5, sy+5, fill="#F85149", outline="")

        # Ball start (white dot)
        if traj:
            bsx, bsy = to_c(traj[0][0], traj[0][1])
            canvas.create_oval(bsx-5, bsy-5, bsx+5, bsy+5, fill="white", outline="")
            canvas.create_text(bsx+8, bsy, text="ball", fill="white",
                                font=("Segoe UI", 7), anchor="w")

        canvas.create_text(8, 8, text="OPP GOAL ▲", fill="#3FB950",
                           font=("Segoe UI", 7), anchor="nw")
        canvas.create_text(8, ch-8, text="OWN GOAL ▼", fill="#F85149",
                           font=("Segoe UI", 7), anchor="sw")

    # ── Match Control panel ────────────────────────────────────────────────────

    def _build_match_control_panel(self):
        p = self._new_panel("MATCH CONTROL")
        self._section_header(p, "  Match Control System")

        info = tk.Label(p,
            text="Commands are sent to the running bot via IPC.  "
                 "Rocket League stays open between matches.",
            bg=_C["bg"], fg=_C["dim"], font=("Segoe UI", 9),
            wraplength=700, justify="left")
        info.pack(fill="x", padx=12, pady=(4, 0))

        btn_frame = tk.Frame(p, bg=_C["bg"])
        btn_frame.pack(fill="x", padx=12, pady=10)

        cmds = [
            ("▶  Start Match",   "start_match",   _C["btn_active"]),
            ("↺  Restart Match", "restart_match",  _C["yellow"]),
            ("⏭  Next Match",    "next_match",      _C["cyan"]),
            ("■  Stop Bot",      "stop_bot",         _C["btn_stop"]),
        ]
        for label, cmd, color in cmds:
            tk.Button(
                btn_frame, text=label, width=18,
                bg=color, fg="white" if color not in (_C["yellow"], _C["cyan"]) else _C["bg"],
                font=("Segoe UI", 10, "bold"), relief="flat", padx=10, pady=8,
                command=lambda c=cmd: self._send_match_command(c),
            ).pack(side="left", padx=6)

        sep = tk.Frame(p, bg=_C["border"], height=1)
        sep.pack(fill="x", padx=12, pady=8)

        # Status
        status_frame = tk.LabelFrame(p, text=" Bot Status ", fg=_C["accent"],
                                      bg=_C["panel"], font=("Segoe UI", 9, "bold"))
        status_frame.pack(fill="x", padx=12, pady=4)
        self._mc_status_var = tk.StringVar(value="No bot running.")
        tk.Label(status_frame, textvariable=self._mc_status_var,
                 bg=_C["panel"], fg=_C["text"],
                 font=("Segoe UI", 9)).pack(anchor="w", padx=10, pady=6)

        # Game mode / restart notes
        notes = tk.LabelFrame(p, text=" Notes ", fg=_C["dim"],
                               bg=_C["panel"], font=("Segoe UI", 8, "bold"))
        notes.pack(fill="x", padx=12, pady=4)
        note_text = (
            "• start_match — tells the bot to begin a new match sequence.\n"
            "• restart_match — resets score/timer; keeps AI model and learning data.\n"
            "• next_match — moves to the next match without closing Rocket League.\n"
            "• stop_bot — gracefully shuts down the AI process."
        )
        tk.Label(notes, text=note_text, bg=_C["panel"], fg=_C["dim"],
                 font=("Segoe UI", 8), justify="left", anchor="w").pack(
            fill="x", padx=10, pady=6)

    def _send_match_command(self, command: str) -> None:
        """Write a match control command to the IPC command file.

        If the command is 'start_match' and no bot is running, launch it instead
        of silently dropping the IPC write.
        """
        if not _ipc:
            return
        # Auto-launch: if start_match is requested but bot is not alive, launch it
        if command == "start_match":
            bot_alive = (self._bot_proc is not None and self._bot_proc.poll() is None)
            if not bot_alive:
                self._save_and_launch()
                return
        try:
            _ipc.write_gui_commands({"command": command})
            if hasattr(self, "_mc_status_var"):
                self._mc_status_var.set(f"Sent: {command}")
        except Exception:
            pass

    def _refresh_analytics(self):
        self._draw_algo_bar_chart()
        self._draw_reward_line_chart()

    def _draw_algo_bar_chart(self):
        canvas = self._analytics_canvas
        canvas.delete("all")
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        if cw < 10 or ch < 10:
            return

        # Try to read algo usage from IPC status
        algos = list(_SEARCH_ALGO_INFO.keys())
        usage = self._status_data.get("algo_usage", {})
        vals = [float(usage.get(a, 0)) for a in algos]
        total = sum(vals) or 1.0
        pcts = [v / total * 100 for v in vals]

        n = len(algos)
        margin_l = 70
        margin_r = 20
        margin_t = 30
        margin_b = 40
        bar_area_w = cw - margin_l - margin_r
        bar_w = bar_area_w / n * 0.7
        gap = bar_area_w / n * 0.3
        max_h = ch - margin_t - margin_b
        max_pct = max(pcts) if max(pcts) > 0 else 100

        canvas.create_text(cw / 2, 14, text="Algorithm Usage %",
                           fill=_C["accent"], font=("Segoe UI", 9, "bold"))

        colors = [_C["cyan"], _C["accent"], _C["orange"], _C["yellow"],
                  _C["purple"], _C["green"], _C["red"], _C["dim"], _C["text"]]
        for i, (algo, pct) in enumerate(zip(algos, pcts)):
            x = margin_l + i * (bar_w + gap)
            bar_h = (pct / max_pct) * max_h if max_pct > 0 else 0
            y_bot = ch - margin_b
            y_top = y_bot - bar_h
            col = colors[i % len(colors)]
            canvas.create_rectangle(x, y_top, x + bar_w, y_bot,
                                     fill=col, outline="")
            canvas.create_text(x + bar_w / 2, y_bot + 5,
                                text=algo[:6], fill=_C["dim"],
                                font=("Segoe UI", 6), anchor="n")
            if pct > 0:
                canvas.create_text(x + bar_w / 2, y_top - 3,
                                    text=f"{pct:.0f}%", fill=col,
                                    font=("Segoe UI", 7), anchor="s")

    def _draw_reward_line_chart(self):
        canvas = self._reward_canvas
        canvas.delete("all")
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        if cw < 10 or ch < 10 or len(self._reward_history) < 2:
            canvas.create_text(cw // 2, ch // 2,
                                text="No reward history yet — launch a match",
                                fill=_C["dim"], font=("Segoe UI", 9))
            return

        hist = self._reward_history[-80:]
        mn, mx = min(hist), max(hist)
        if mx == mn:
            mx = mn + 1

        margin = 30
        w = cw - 2 * margin
        h = ch - 2 * margin
        pts = []
        for i, v in enumerate(hist):
            x = margin + i * w / max(len(hist) - 1, 1)
            y = margin + h - (v - mn) / (mx - mn) * h
            pts.append((x, y))

        canvas.create_text(cw / 2, 14, text="Episode Reward Trend",
                           fill=_C["green"], font=("Segoe UI", 9, "bold"))
        for i in range(len(pts) - 1):
            canvas.create_line(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1],
                                fill=_C["green"], width=2)
        # Zero line
        if mn < 0 < mx:
            y0 = margin + h - (0 - mn) / (mx - mn) * h
            canvas.create_line(margin, y0, cw - margin, y0,
                                fill=_C["dim"], dash=(4, 4))

    # ── Guidance panel ────────────────────────────────────────────────────────

    def _build_guidance_panel(self):
        p = self._new_panel("GUIDANCE")
        self._section_header(p, "  Usage Guidance")

        scroll_frame = tk.Frame(p, bg=_C["bg"])
        scroll_frame.pack(fill="both", expand=True, padx=8, pady=8)

        text_widget = tk.Text(scroll_frame, bg=_C["panel"], fg=_C["text"],
                              font=("Segoe UI", 9), relief="flat",
                              wrap="word", state="normal", padx=12, pady=8)
        scrollbar = tk.Scrollbar(scroll_frame, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        text_widget.pack(fill="both", expand=True)

        text_widget.tag_configure("heading", foreground=_C["accent"],
                                   font=("Segoe UI", 11, "bold"))
        text_widget.tag_configure("subhead", foreground=_C["yellow"],
                                   font=("Segoe UI", 9, "bold"))
        text_widget.tag_configure("body", foreground=_C["text"],
                                   font=("Segoe UI", 9))
        text_widget.tag_configure("tip", foreground=_C["green"],
                                   font=("Segoe UI", 9))

        guidance = [
            ("BEGINNER GUIDE", "heading"),
            ("1. Click Launch Bot — the AI will start playing automatically.\n", "body"),
            ("2. Use Match Settings to choose your arena and game mode.\n", "body"),
            ("3. Watch the Dashboard for live score, mode, and reward info.\n", "body"),
            ("4. The bot starts in Balanced mode — it attacks and defends.\n\n", "body"),
            ("ALGORITHM SELECTION", "subhead"),
            ("When the bot is running you can apply algorithm changes live:\n", "body"),
            ("  • A* — best general path planning (recommended default)\n", "tip"),
            ("  • Greedy — fastest, most aggressive offense\n", "tip"),
            ("  • UCS — safest defense positioning\n", "tip"),
            ("  • Beam Search — long-horizon attack planning\n", "tip"),
            ("  • Ball Prediction — intercept-based positioning\n\n", "tip"),
            ("REWARD SYSTEM", "subhead"),
            ("The bot learns from integer rewards each match:\n", "body"),
            ("  +100  Goal scored\n", "tip"),
            ("  +50   Assist\n", "tip"),
            ("  +40   Save\n", "tip"),
            ("  +25   Clear ball\n", "tip"),
            ("  +20   Shot on target\n", "tip"),
            ("  +30   Demo (defender near goal)\n", "tip"),
            ("  -30   Goal conceded\n", "body"),
            ("  -5    Idle (no movement)\n\n", "body"),
            ("ADVANCED TIPS", "subhead"),
            ("• Use 'Session Model' (P key) to start fresh each match.\n", "body"),
            ("• The 'Persistent Model' (O key) saves learning across sessions.\n", "body"),
            ("• In Manual mode, the bot learns from how you drive.\n", "body"),
            ("• Strategy Layer auto-switches tactics based on score and time.\n", "body"),
            ("• The AI Architecture panel shows the 7-layer decision pipeline.\n", "body"),
            ("• Use Presentation Mode for a clean academic display screen.\n\n", "body"),
            ("ACADEMIC USE", "heading"),
            ("This project demonstrates classical AI search algorithms in a real-time\n", "body"),
            ("game environment. 9 search algorithms and 11 RL models operate in\n", "body"),
            ("parallel, with the best result selected by the ensemble scorer.\n", "body"),
        ]
        for content_text, tag in guidance:
            text_widget.insert("end", content_text, tag)
        text_widget.configure(state="disabled")

    # ── Presentation panel ────────────────────────────────────────────────────

    def _build_presentation_panel(self):
        p = self._new_panel("PRESENTATION")
        self._section_header(p, "  University Presentation Mode")

        info_frame = tk.Frame(p, bg=_C["bg"])
        info_frame.pack(fill="both", expand=True, padx=20, pady=20)

        tk.Label(info_frame, text="Presentation Mode",
                 bg=_C["bg"], fg=_C["accent"],
                 font=("Segoe UI", 18, "bold")).pack(pady=(30, 10))
        tk.Label(info_frame,
                 text="Opens a clean 900×600 window showing:\n"
                      "• Real-time match stats  •  AI Architecture diagram\n"
                      "• Active algorithm  •  Reward breakdown  •  Strategy",
                 bg=_C["bg"], fg=_C["dim"],
                 font=("Segoe UI", 10), justify="center").pack(pady=10)

        tk.Button(info_frame, text="Open Presentation Window",
                  bg=_C["btn_active"], fg="white",
                  font=("Segoe UI", 12, "bold"), relief="flat",
                  padx=20, pady=10,
                  command=self._open_presentation).pack(pady=20)

        tk.Label(info_frame, text="Tip: Keep this dashboard open alongside.",
                 bg=_C["bg"], fg=_C["dim"], font=("Segoe UI", 8)).pack()

    # ── Presentation window ───────────────────────────────────────────────────

    def _open_presentation(self):
        win = tk.Toplevel(self._root)
        win.title("Rocket League AI — Presentation")
        win.geometry("1000x650")
        win.configure(bg=_C["bg"])
        win.resizable(True, True)
        win.rowconfigure(0, weight=1)
        win.columnconfigure(0, weight=1)
        win.columnconfigure(1, weight=1)

        # Left: live stats
        left = tk.Frame(win, bg=_C["panel"])
        left.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=10)
        left.columnconfigure(0, weight=1)

        tk.Label(left, text="Rocket League AI", bg=_C["panel"],
                 fg=_C["accent"], font=("Segoe UI", 18, "bold"),
                 pady=10).pack(fill="x")
        tk.Label(left, text="Autonomous AI Bot — 9 Search Algorithms  |  11 RL Models",
                 bg=_C["panel"], fg=_C["dim"],
                 font=("Segoe UI", 9)).pack(fill="x")

        tk.Frame(left, bg=_C["border"], height=1).pack(fill="x", pady=6)

        self._pres_vars: Dict[str, tk.StringVar] = {}
        stats_info = [
            ("Score",        "0 — 0",        _C["green"]),
            ("Mode",         "balanced",      _C["accent"]),
            ("Algorithm",    "A*",            _C["cyan"]),
            ("Strategy",     "BALANCED",      _C["orange"]),
            ("Game State",   "—",             _C["yellow"]),
            ("Ep. Reward",   "0",             _C["green"]),
            ("Tick",         "0",             _C["dim"]),
        ]
        for label, default, color in stats_info:
            var = tk.StringVar(value=default)
            self._pres_vars[label] = var
            row = tk.Frame(left, bg=_C["panel"])
            row.pack(fill="x", padx=12, pady=4)
            tk.Label(row, text=f"{label}:", bg=_C["panel"], fg=_C["dim"],
                     font=("Segoe UI", 9), width=12, anchor="e").pack(side="left")
            tk.Label(row, textvariable=var, bg=_C["panel"], fg=color,
                     font=("Segoe UI", 12, "bold"), anchor="w").pack(side="left", padx=8)

        tk.Frame(left, bg=_C["border"], height=1).pack(fill="x", pady=6)

        self._pres_reward_lbl = tk.Label(left, text="", bg=_C["panel"],
                                          fg=_C["text"], font=("Consolas", 8),
                                          justify="left", anchor="w")
        self._pres_reward_lbl.pack(fill="x", padx=12, pady=4)

        # Right: architecture canvas
        right = tk.Frame(win, bg=_C["bg"])
        right.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=10)
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        pres_canvas = tk.Canvas(right, bg=_C["bg"], highlightthickness=0)
        pres_canvas.grid(row=0, column=0, sticky="nsew")
        pres_canvas.bind("<Configure>", lambda e: self._draw_architecture(pres_canvas))

        def pres_refresh():
            if not win.winfo_exists():
                return
            d = self._status_data
            def gv(k, fb=""): return str(d.get(k, fb))
            ms = int(d.get("my_score", 0))
            os_ = int(d.get("opp_score", 0))
            self._pres_vars["Score"].set(f"{ms} — {os_}")
            self._pres_vars["Mode"].set(gv("current_mode", "balanced").upper())
            self._pres_vars["Algorithm"].set(gv("current_algo", "A*"))
            self._pres_vars["Strategy"].set(gv("strategy", "BALANCED").upper())
            self._pres_vars["Game State"].set(gv("current_game_state", "—"))
            self._pres_vars["Ep. Reward"].set(str(int(float(gv("episode_reward", "0")))))
            self._pres_vars["Tick"].set(gv("tick", "0"))
            rb = d.get("reward_breakdown", {})
            rb_text = "  ".join(f"{k}: {v:.0f}" for k, v in list(rb.items())[:6])
            self._pres_reward_lbl.configure(text=rb_text or "No reward data yet")
            win.after(600, pres_refresh)
        pres_refresh()

    # ── Bot lifecycle ─────────────────────────────────────────────────────────

    # ── Strategy Book panel ───────────────────────────────────────────────────

    def _build_strategy_book_panel(self):
        p = self._new_panel("STRATEGY BOOK")
        self._section_header(p, "  Strategy Book — Arena Tactical Knowledge")

        # ── Controls bar ─────────────────────────────────────────────────
        ctrl = tk.Frame(p, bg=_C["bg"])
        ctrl.pack(fill="x", padx=8, pady=6)

        # Arena selector
        tk.Label(ctrl, text="Arena:", bg=_C["bg"], fg=_C["dim"],
                 font=("Segoe UI", 9)).pack(side="left", padx=(4, 2))
        self._sb_arena_var = tk.StringVar(value="DFHStadium")
        _arena_list = [
            "DFHStadium", "Mannfield", "ChampionsField", "UrbanCentral",
            "BeckwithPark", "NeoTokyo", "AquaDome", "Farmstead",
            "StarbaseArc", "WastelandHS", "Utopia_Retro", "TheSanctuary",
        ]
        arena_cb = ttk.Combobox(ctrl, textvariable=self._sb_arena_var,
                                 values=_arena_list, width=18,
                                 state="readonly", font=("Segoe UI", 9))
        arena_cb.pack(side="left", padx=4)
        arena_cb.bind("<<ComboboxSelected>>", lambda e: self._refresh_strategy_book())

        tk.Button(ctrl, text="Refresh", bg=_C["btn_nav"], fg=_C["text"],
                  font=("Segoe UI", 8), relief="flat",
                  command=self._refresh_strategy_book).pack(side="left", padx=6)

        self._sb_adaptive_var = tk.BooleanVar(value=True)
        tk.Checkbutton(ctrl, text="Adaptive Strategies",
                       variable=self._sb_adaptive_var,
                       bg=_C["bg"], fg=_C["text"],
                       selectcolor=_C["panel"],
                       activebackground=_C["bg"],
                       command=self._toggle_adaptive_strategies).pack(side="left", padx=6)

        tk.Button(ctrl, text="Reset Strategy Book",
                  bg=_C["btn_stop"], fg="white",
                  font=("Segoe UI", 8), relief="flat",
                  command=self._reset_strategy_book).pack(side="right", padx=6)

        self._sb_status_var = tk.StringVar(value="Click Refresh to load strategies.")
        tk.Label(p, textvariable=self._sb_status_var,
                 bg=_C["bg"], fg=_C["dim"], font=("Segoe UI", 8)).pack(anchor="w", padx=12)

        # ── Active strategies info bar ────────────────────────────────────
        info_frame = tk.LabelFrame(p, text=" Current Match  (live from bot IPC) ",
                                    fg=_C["accent"], bg=_C["panel"],
                                    font=("Segoe UI", 9, "bold"))
        info_frame.pack(fill="x", padx=12, pady=(4, 0))

        self._sb_live_vars: Dict[str, tk.StringVar] = {}
        live_items = [
            ("Arena",          "sb_live_arena",   _C["orange"]),
            ("Active Strategy","sb_live_strategy", _C["cyan"]),
            ("Switch Reason",  "sb_live_reason",   _C["yellow"]),
        ]
        for label, key, color in live_items:
            row = tk.Frame(info_frame, bg=_C["panel"])
            row.pack(fill="x", padx=10, pady=2)
            tk.Label(row, text=f"{label}:", bg=_C["panel"], fg=_C["dim"],
                     font=("Segoe UI", 8), width=16, anchor="e").pack(side="left")
            var = tk.StringVar(value="—")
            self._sb_live_vars[key] = var
            tk.Label(row, textvariable=var, bg=_C["panel"], fg=color,
                     font=("Segoe UI", 9, "bold"), anchor="w").pack(side="left", padx=6)

        # ── Strategy table ────────────────────────────────────────────────
        tbl_outer = tk.Frame(p, bg=_C["bg"])
        tbl_outer.pack(fill="both", expand=True, padx=12, pady=8)

        # Canvas + scrollbar for the table
        sb_canvas = tk.Canvas(tbl_outer, bg=_C["bg"], highlightthickness=0)
        vsb = tk.Scrollbar(tbl_outer, orient="vertical", command=sb_canvas.yview)
        sb_canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        sb_canvas.pack(side="left", fill="both", expand=True)

        self._sb_table_frame = tk.Frame(sb_canvas, bg=_C["bg"])
        self._sb_table_window = sb_canvas.create_window(
            (0, 0), window=self._sb_table_frame, anchor="nw"
        )

        def _on_frame_configure(event):
            sb_canvas.configure(scrollregion=sb_canvas.bbox("all"))

        def _on_canvas_configure(event):
            sb_canvas.itemconfig(self._sb_table_window, width=event.width)

        self._sb_table_frame.bind("<Configure>", _on_frame_configure)
        sb_canvas.bind("<Configure>", _on_canvas_configure)

        # Column headers
        _cols = ["Strategy Type", "Active Strategy", "Best (learned)", "Success Rate", "Attempts", "Description"]
        _col_widths = [130, 150, 150, 100, 70, 280]
        header_row = tk.Frame(self._sb_table_frame, bg=_C["header"])
        header_row.pack(fill="x")
        for col, w in zip(_cols, _col_widths):
            tk.Label(header_row, text=col, bg=_C["header"], fg=_C["accent"],
                     font=("Segoe UI", 8, "bold"), width=w // 8, anchor="w",
                     padx=6).pack(side="left")

        # Data rows – populated by _refresh_strategy_book
        self._sb_data_rows: list = []

    def _refresh_strategy_book(self):
        """Load strategy_book.json and repopulate the table."""
        import time
        try:
            from strategy.strategy_book import StrategyBook
            sb = StrategyBook()
            arena = self._sb_arena_var.get()
            summary = sb.get_arena_summary(arena)

            # Wipe old data rows
            for widget in self._sb_data_rows:
                widget.destroy()
            self._sb_data_rows.clear()

            _row_colors = [_C["panel"], _C["bg"]]
            for idx, (stype, info) in enumerate(summary.items()):
                row_bg = _row_colors[idx % 2]
                row = tk.Frame(self._sb_table_frame, bg=row_bg)
                row.pack(fill="x")
                self._sb_data_rows.append(row)

                rate_pct = f"{info['success_rate'] * 100:.1f}%"
                rate_color = (
                    _C["green"]  if info["success_rate"] >= 0.60 else
                    _C["yellow"] if info["success_rate"] >= 0.45 else
                    _C["red"]
                )
                active_color = (
                    _C["cyan"] if info["active"] == info["best"] else _C["orange"]
                )

                cells = [
                    (stype.replace("_", " ").title(), _C["text"],    130),
                    (info["active"],                  active_color,  150),
                    (info["best"],                    _C["green"],   150),
                    (rate_pct,                        rate_color,    100),
                    (str(info["attempts"]),           _C["dim"],      70),
                    (info["description"][:50],        _C["dim"],     280),
                ]
                for text, color, w in cells:
                    tk.Label(row, text=text, bg=row_bg, fg=color,
                             font=("Segoe UI", 8), width=w // 8,
                             anchor="w", padx=6).pack(side="left")

            self._sb_adaptive_var.set(sb.adaptive_enabled)
            self._sb_status_var.set(
                f"Loaded {len(summary)} strategy types for {arena}  "
                f"({time.strftime('%H:%M:%S')})"
            )
        except Exception as exc:
            self._sb_status_var.set(f"Error loading strategy book: {exc}")

    def _toggle_adaptive_strategies(self):
        """Toggle adaptive_enabled flag in the strategy book."""
        try:
            from strategy.strategy_book import StrategyBook
            sb = StrategyBook()
            sb.adaptive_enabled = self._sb_adaptive_var.get()
            sb.save()
            state = "enabled" if sb.adaptive_enabled else "disabled"
            self._sb_status_var.set(f"Adaptive strategies {state}.")
        except Exception as exc:
            self._sb_status_var.set(f"Error: {exc}")

    def _reset_strategy_book(self):
        """Wipe all learned strategy data and restore defaults."""
        if not messagebox:
            return
        if not messagebox.askyesno(
            "Reset Strategy Book",
            "This will delete all learned strategies and restore defaults.\nContinue?",
        ):
            return
        try:
            from strategy.strategy_book import StrategyBook
            sb = StrategyBook()
            sb.reset()
            self._sb_status_var.set("Strategy book reset to defaults.")
            self._refresh_strategy_book()
        except Exception as exc:
            self._sb_status_var.set(f"Error resetting: {exc}")

    def _update_strategy_book_panel(self, d: dict):
        """Called from _poll_bot_status to refresh live bot strategy data."""
        if not hasattr(self, "_sb_live_vars"):
            return
        self._sb_live_vars["sb_live_arena"].set(
            d.get("arena", "—")
        )
        active_strats = d.get("active_strategies", {})
        atk = active_strats.get("attack", "—")
        dfn = active_strats.get("defense", "—")
        kof = active_strats.get("kickoff", "—")
        self._sb_live_vars["sb_live_strategy"].set(
            f"attack:{atk}  defense:{dfn}  kickoff:{kof}"
        )
        self._sb_live_vars["sb_live_reason"].set(
            d.get("strategy_switch_reason", "—")
        )
        # Auto-sync arena selector to current game arena
        bot_arena = d.get("arena", "")
        if bot_arena and bot_arena != self._sb_arena_var.get():
            self._sb_arena_var.set(bot_arena)

    # ── Ability Discovery panel ──────────────────────────────────────────────

    def _build_ability_discovery_panel(self):
        p = self._new_panel("ABILITY DISCOVERY")
        self._section_header(p, "  Ability Discovery — Rumble & Special Power-ups")

        # ── Live status bar ──────────────────────────────────────────────
        live_frame = tk.LabelFrame(
            p, text=" Live Bot Status (from IPC) ",
            fg=_C["accent"], bg=_C["panel"], font=("Segoe UI", 9, "bold"))
        live_frame.pack(fill="x", padx=12, pady=(6, 0))

        self._ad_live_vars: Dict[str, "tk.StringVar"] = {}
        live_items = [
            ("Held Ability",      "ad_ability",      _C["orange"]),
            ("Ability Type",      "ad_effect",       _C["cyan"]),
            ("Best Use",          "ad_use_case",     _C["yellow"]),
            ("Success Rate",      "ad_success_rate", _C["green"]),
            ("Experiment Mode",   "ad_experiment",   _C["dim"]),
            ("Team Directive",    "ad_team_dir",     _C["accent"]),
            ("My Role",           "ad_role",         _C["cyan"]),
        ]
        for label, key, color in live_items:
            row = tk.Frame(live_frame, bg=_C["panel"])
            row.pack(fill="x", padx=10, pady=2)
            tk.Label(row, text=f"{label}:", bg=_C["panel"], fg=_C["dim"],
                     font=("Segoe UI", 8), width=18, anchor="e").pack(side="left")
            var = tk.StringVar(value="—")
            self._ad_live_vars[key] = var
            tk.Label(row, textvariable=var, bg=_C["panel"], fg=color,
                     font=("Segoe UI", 9, "bold"), anchor="w").pack(side="left", padx=6)

        # ── Known abilities table ────────────────────────────────────────
        ctrl = tk.Frame(p, bg=_C["bg"])
        ctrl.pack(fill="x", padx=8, pady=6)
        tk.Button(ctrl, text="Refresh Ability Knowledge",
                  bg=_C["btn_nav"], fg=_C["text"],
                  font=("Segoe UI", 8), relief="flat",
                  command=self._refresh_ability_table).pack(side="left", padx=4)
        self._ad_status_var = tk.StringVar(value="Click Refresh to load.")
        tk.Label(p, textvariable=self._ad_status_var,
                 bg=_C["bg"], fg=_C["dim"], font=("Segoe UI", 8)
                 ).pack(anchor="w", padx=12)

        tbl_outer = tk.Frame(p, bg=_C["bg"])
        tbl_outer.pack(fill="both", expand=True, padx=12, pady=4)
        sb_canvas = tk.Canvas(tbl_outer, bg=_C["bg"], highlightthickness=0)
        vsb = tk.Scrollbar(tbl_outer, orient="vertical", command=sb_canvas.yview)
        sb_canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        sb_canvas.pack(side="left", fill="both", expand=True)
        self._ad_table_frame = tk.Frame(sb_canvas, bg=_C["bg"])
        sb_canvas.create_window((0, 0), window=self._ad_table_frame, anchor="nw")

        headers = ["Ability", "Type", "Best Use", "Success Rate", "Attempts"]
        col_w = [120, 120, 180, 100, 80]
        for ci, (h, w) in enumerate(zip(headers, col_w)):
            tk.Label(self._ad_table_frame, text=h, bg=_C["header"],
                     fg=_C["accent"], font=("Segoe UI", 8, "bold"),
                     width=w // 8, anchor="w").grid(
                         row=0, column=ci, sticky="ew", padx=1, pady=1)

        self._ad_table_rows: List = []

        def _on_frame_configure(e):
            sb_canvas.configure(scrollregion=sb_canvas.bbox("all"))
        self._ad_table_frame.bind("<Configure>", _on_frame_configure)

    def _refresh_ability_table(self):
        try:
            from game_logic.ability_discovery import AbilityDiscovery
            ad = AbilityDiscovery()
            # Clear existing rows
            for w in self._ad_table_rows:
                w.destroy()
            self._ad_table_rows.clear()

            for ri, name in enumerate(sorted(ad.list_known_abilities()), start=1):
                info = ad.get_ability_info(name)
                sr = float(info.get("success_rate", 0.5))
                color = _C["green"] if sr >= 0.60 else (_C["yellow"] if sr >= 0.45 else _C["btn_stop"])
                vals = [
                    name,
                    info.get("type", "?"),
                    info.get("best_use_case", "?"),
                    f"{sr:.0%}",
                    str(info.get("attempts", 0)),
                ]
                for ci, v in enumerate(vals):
                    fg = color if ci == 3 else _C["text"]
                    lbl = tk.Label(
                        self._ad_table_frame, text=v,
                        bg=_C["panel"] if ri % 2 == 0 else _C["bg"],
                        fg=fg, font=("Segoe UI", 8), anchor="w")
                    lbl.grid(row=ri, column=ci, sticky="ew", padx=1, pady=1)
                    self._ad_table_rows.append(lbl)
            self._ad_status_var.set(f"Loaded {len(ad.list_known_abilities())} abilities.")
        except Exception as exc:
            self._ad_status_var.set(f"Error: {exc}")

    def _update_ability_panel(self, d: dict):
        """Refresh live ability data from IPC status dict."""
        if not hasattr(self, "_ad_live_vars"):
            return
        self._ad_live_vars["ad_ability"].set(d.get("ability_name", "—"))
        self._ad_live_vars["ad_effect"].set(d.get("ability_type", "—"))
        self._ad_live_vars["ad_use_case"].set(d.get("ability_use_case", "—"))
        self._ad_live_vars["ad_success_rate"].set(d.get("ability_success_rate", "—"))
        self._ad_live_vars["ad_experiment"].set(
            "YES — experimenting" if d.get("ability_experimenting") else "No"
        )
        self._ad_live_vars["ad_team_dir"].set(d.get("team_directive", "—"))
        self._ad_live_vars["ad_role"].set(d.get("bot_role", "—"))

    # ── Team Configuration panel ──────────────────────────────────────────────

    def _build_team_config_panel(self):
        p = self._new_panel("TEAM CONFIG")
        self._section_header(p, "  Team Configuration — Bots & Roles")

        # ── Team mode selection ──────────────────────────────────────────
        mode_frame = tk.LabelFrame(
            p, text=" Team Mode ",
            fg=_C["accent"], bg=_C["panel"], font=("Segoe UI", 9, "bold"))
        mode_frame.pack(fill="x", padx=12, pady=(8, 4))

        self._team_mode_var = tk.StringVar(value="default_rl")
        modes = [
            ("Use Rocket League Default Bots", "default_rl"),
            ("Use Custom AI Bots (this project)",  "custom_ai"),
            ("Select Specific AI Models Manually", "specific"),
        ]
        for text, val in modes:
            tk.Radiobutton(
                mode_frame, text=text, variable=self._team_mode_var,
                value=val, bg=_C["panel"], fg=_C["text"],
                selectcolor=_C["bg"], activebackground=_C["panel"],
                font=("Segoe UI", 9),
                command=self._on_team_mode_changed,
            ).pack(anchor="w", padx=12, pady=3)

        # ── Specific model selectors (shown when mode == "specific") ───
        self._tc_specific_frame = tk.LabelFrame(
            p, text=" Role Model Assignment ",
            fg=_C["cyan"], bg=_C["panel"], font=("Segoe UI", 9, "bold"))
        self._tc_specific_frame.pack(fill="x", padx=12, pady=4)

        _model_options = [
            "model_attack_v1", "model_attack_v2", "model_attack_v3",
            "model_support_v1", "model_support_v2",
            "model_defense_v1", "model_defense_v2", "model_defense_v5",
            "custom_ai", "default_rl",
        ]
        self._tc_model_vars: Dict[str, "tk.StringVar"] = {}
        for role in ("attacker", "support", "defender"):
            row = tk.Frame(self._tc_specific_frame, bg=_C["panel"])
            row.pack(fill="x", padx=12, pady=4)
            tk.Label(row, text=f"{role.capitalize()} Model:",
                     bg=_C["panel"], fg=_C["text"],
                     font=("Segoe UI", 9), width=18, anchor="e"
                     ).pack(side="left")
            var = tk.StringVar(value=f"model_{role[:3]}_v1")
            self._tc_model_vars[role] = var
            ttk.Combobox(
                row, textvariable=var,
                values=_model_options, width=22,
                state="readonly", font=("Segoe UI", 9)
            ).pack(side="left", padx=6)

        # ── Role assignment live display ──────────────────────────────────
        roles_frame = tk.LabelFrame(
            p, text=" Live Role Assignments (from IPC) ",
            fg=_C["accent"], bg=_C["panel"], font=("Segoe UI", 9, "bold"))
        roles_frame.pack(fill="x", padx=12, pady=(4, 0))

        self._tc_live_vars: Dict[str, "tk.StringVar"] = {}
        for label, key in [
            ("My Role",           "tc_my_role"),
            ("Team Mode Active",  "tc_team_mode"),
            ("Last Rotation",     "tc_rotation"),
            ("Teammate Abilities","tc_tm_abilities"),
            ("Opp. Abilities",    "tc_opp_abilities"),
        ]:
            row = tk.Frame(roles_frame, bg=_C["panel"])
            row.pack(fill="x", padx=10, pady=2)
            tk.Label(row, text=f"{label}:", bg=_C["panel"], fg=_C["dim"],
                     font=("Segoe UI", 8), width=20, anchor="e").pack(side="left")
            var = tk.StringVar(value="—")
            self._tc_live_vars[key] = var
            tk.Label(row, textvariable=var, bg=_C["panel"], fg=_C["cyan"],
                     font=("Segoe UI", 9, "bold"), anchor="w").pack(side="left", padx=6)

        # ── Apply button ─────────────────────────────────────────────────
        tk.Button(
            p, text="Apply Team Configuration",
            bg=_C["btn_active"], fg="white",
            font=("Segoe UI", 9, "bold"), relief="flat",
            command=self._apply_team_config,
        ).pack(anchor="w", padx=12, pady=8)

        # ── Multi-bot team launch ─────────────────────────────────────────
        launch_frame = tk.LabelFrame(
            p, text=" Multi-Bot Team Control ",
            fg=_C["accent"], bg=_C["panel"], font=("Segoe UI", 9, "bold"))
        launch_frame.pack(fill="x", padx=12, pady=(0, 8))

        btn_row = tk.Frame(launch_frame, bg=_C["panel"])
        btn_row.pack(fill="x", padx=10, pady=6)

        self._launch_team_btn = tk.Button(
            btn_row, text="▶▶  Launch Full Team (3 bots)",
            bg="#1a7a40", fg="white",
            font=("Segoe UI", 9, "bold"), relief="flat",
            command=self._launch_full_team,
        )
        self._launch_team_btn.pack(side="left", padx=(0, 6))

        self._stop_team_btn = tk.Button(
            btn_row, text="■  Stop All Team Bots",
            bg="#7a1a1a", fg="white",
            font=("Segoe UI", 9, "bold"), relief="flat",
            command=self._stop_team_bots,
        )
        self._stop_team_btn.pack(side="left")

        # Per-role status labels
        self._team_pid_vars: Dict[str, "tk.StringVar"] = {}
        for role in ("attacker", "support", "defender"):
            row = tk.Frame(launch_frame, bg=_C["panel"])
            row.pack(fill="x", padx=10, pady=2)
            tk.Label(row, text=f"{role.capitalize()}:", bg=_C["panel"],
                     fg=_C["dim"], font=("Segoe UI", 8), width=10,
                     anchor="e").pack(side="left")
            var = tk.StringVar(value="not running")
            self._team_pid_vars[role] = var
            tk.Label(row, textvariable=var, bg=_C["panel"],
                     fg=_C["cyan"], font=("Segoe UI", 9, "bold"),
                     anchor="w").pack(side="left", padx=6)

        # State init
        self._on_team_mode_changed()

    def _on_team_mode_changed(self):
        mode = self._team_mode_var.get()
        if hasattr(self, "_tc_specific_frame"):
            if mode == "specific":
                self._tc_specific_frame.pack(fill="x", padx=12, pady=4)
            else:
                self._tc_specific_frame.pack_forget()

    def _apply_team_config(self):
        """Write team config into match settings for subprocess to pick up."""
        mode = self._team_mode_var.get()
        models = {
            role: var.get()
            for role, var in self._tc_model_vars.items()
        }
        import json as _json
        try:
            cfg_path = os.path.join(_PROJECT_ROOT, "runtime", "_match_settings.json")
            if os.path.isfile(cfg_path):
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = _json.load(f)
            else:
                cfg = {"settings": {}}
            cfg.setdefault("settings", {})["team_mode"] = mode
            cfg["settings"]["team_models"] = models
            with open(cfg_path, "w", encoding="utf-8") as f:
                _json.dump(cfg, f, indent=2)
            if hasattr(self, "_tc_live_vars"):
                self._tc_live_vars["tc_team_mode"].set(mode)
        except Exception:
            pass

    def _launch_full_team(self):
        """Spawn one headless bot process per role (attacker / support / defender)."""
        try:
            from game_logic.team_controller import TeamController
            if not hasattr(self, "_team_controller_gui"):
                self._team_controller_gui = TeamController()

            venv_py = os.path.join(_PROJECT_ROOT, ".venv", "Scripts", "python.exe")
            if not os.path.isfile(venv_py):
                venv_py = None   # fall back to sys.executable inside launch_team

            self._team_controller_gui.launch_team(_PROJECT_ROOT, venv_py)
            self._refresh_team_pid_labels()
            if hasattr(self, "_launch_team_btn"):
                self._launch_team_btn.configure(state="disabled")
        except Exception as exc:
            from tkinter import messagebox
            messagebox.showerror("Team Launch Error", str(exc))

    def _stop_team_bots(self):
        """Terminate all team-bot subprocesses."""
        if hasattr(self, "_team_controller_gui"):
            self._team_controller_gui.stop_team_bots()
        if hasattr(self, "_team_pid_vars"):
            for var in self._team_pid_vars.values():
                var.set("not running")
        if hasattr(self, "_launch_team_btn"):
            self._launch_team_btn.configure(state="normal")

    def _refresh_team_pid_labels(self):
        """Update per-role PID status labels after a team launch."""
        if not hasattr(self, "_team_pid_vars") or not hasattr(self, "_team_controller_gui"):
            return
        pids = self._team_controller_gui.get_launched_pids()
        for role, var in self._team_pid_vars.items():
            pid = pids.get(role)
            var.set(f"PID {pid}" if pid else "failed to start")

    def _update_team_config_panel(self, d: dict):
        """Refresh live team data from IPC status dict."""
        if not hasattr(self, "_tc_live_vars"):
            return
        self._tc_live_vars["tc_my_role"].set(d.get("bot_role", "—"))
        self._tc_live_vars["tc_team_mode"].set(d.get("team_mode", "—"))
        self._tc_live_vars["tc_rotation"].set(d.get("team_rotation_reason", "—"))
        tm = d.get("teammate_abilities", {})
        op = d.get("opponent_abilities", {})
        self._tc_live_vars["tc_tm_abilities"].set(
            ", ".join(f"{k}:{v}" for k, v in tm.items()) if tm else "none"
        )
        self._tc_live_vars["tc_opp_abilities"].set(
            ", ".join(f"{k}:{v}" for k, v in op.items()) if op else "none"
        )

    # ── Bot lifecycle ─────────────────────────────────────────────────────────

    def _launch_bot(self):
        if self._bot_proc and self._bot_proc.poll() is None:
            return  # already running
        self._save_match_settings()
        cmd = [_VENV_PYTHON,
               os.path.join(_PROJECT_ROOT, "runtime", "run_match.py"),
               "--no-gui"]
        try:
            self._bot_proc = subprocess.Popen(
                cmd, cwd=_PROJECT_ROOT,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
            )
            self._launch_btn.configure(state="disabled")
            self._stop_btn.configure(state="normal")
            self._bot_status_lbl.configure(text=f"Bot: running (pid {self._bot_proc.pid})",
                                            fg=_C["green"])
        except Exception as exc:
            if messagebox:
                messagebox.showerror("Launch Error", str(exc))

    def _stop_bot(self):
        if self._bot_proc:
            try:
                self._bot_proc.terminate()
            except Exception:
                pass
            self._bot_proc = None
        self._launch_btn.configure(state="normal")
        self._stop_btn.configure(state="disabled")
        self._bot_status_lbl.configure(text="Bot: stopped", fg=_C["dim"])

    def _save_and_launch(self):
        self._save_match_settings()
        self._launch_bot()
        self._show_panel("DASHBOARD")

    def _save_match_settings(self):
        size_map = {"1v1": 1, "2v2": 2, "3v3": 3}
        active_search = [k for k, v in self._search_vars.items() if v.get()] or ["A*"]
        active_rl     = [k for k, v in self._rl_vars.items() if v.get()]
        data = {
            "starting_mode":    self._start_mode_var.get(),
            "temporary_reset":  False,
            "settings": {
                "game_mode":          self._game_mode_var.get(),
                "arena":              self._arena_var.get(),
                "team_size":          size_map.get(self._team_size_var.get(), 1),
                "match_length":       self._match_len_var.get(),
                "boost_amount":       self._boost_var.get(),
                "gravity":            self._gravity_var.get(),
                "max_score":          self._max_score_var.get(),
                "search_algorithms":  active_search,
                "rl_models":          active_rl,
                "strategy":           self._strategy_preset_var.get(),
                "temporary_model":    self._temp_model_var.get(),
            },
        }
        try:
            with open(_MATCH_SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass
        self._sync_model_settings(data)

    def _sync_model_settings(self, match_data: dict) -> None:
        """Write a live snapshot of model configuration to model/model_settings.json."""
        s = match_data.get("settings", {})
        model_data = {
            "active_model":        "temporary" if s.get("temporary_model") else "persistent",
            "learning_enabled":    True,
            "adaptive_difficulty": True,
            "team_strategy":       s.get("strategy", "balanced"),
            "team_type":           "default_rl",
            "attacker_model":      "custom_ai",
            "support_model":       "custom_ai",
            "defender_model":      "custom_ai",
            "active_search":       s.get("search_algorithms", ["A*"]),
            "active_rl_models":    s.get("rl_models", []),
        }
        try:
            with open(_MODEL_SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(model_data, f, indent=2)
        except Exception:
            pass

    # ── IPC polling ───────────────────────────────────────────────────────────

    def _poll_bot_status(self):
        """Read IPC status files and update the dashboard.
        Called by RefreshLoop which guarantees rescheduling even on exceptions.
        """
        if _ipc:
            try:
                d = _ipc.read_bot_status()
                if d:
                    self._status_data = d
                    self._update_dashboard(d)
                    self._update_strategy_book_panel(d)
                    self._update_ability_panel(d)
                    self._update_team_config_panel(d)
                    ep_reward = float(d.get("episode_reward", 0))
                    if self._reward_history[-1:] != [ep_reward]:
                        self._reward_history.append(ep_reward)
                        if len(self._reward_history) > 200:
                            self._reward_history = self._reward_history[-200:]
                    # Update match control status
                    if hasattr(self, "_mc_status_var"):
                        gs   = d.get("current_game_state", "")
                        mode = d.get("current_mode", "balanced")
                        tick = d.get("tick", 0)
                        self._mc_status_var.set(
                            f"Running — mode: {mode}  state: {gs or '—'}  tick: {tick}"
                        )
            except Exception:
                pass

        # Read model status
        if _ModelStatus is not None:
            try:
                ms = _ModelStatus.read()
                if ms:
                    self._model_status_data = ms
                    if hasattr(self, "_dash_model_var"):
                        mdl   = ms.get("current_model",  "—")
                        state = ms.get("training_state", "—")
                        lr    = ms.get("learning_rate",  0.0)
                        goals = ms.get("goals_total",    0)
                        conc  = ms.get("concedes_total", 0)
                        ep    = ms.get("episodes",       0)
                        wr    = ms.get("win_rate",       0.0)
                        self._dash_model_var.set(
                            f"Model: {mdl}  |  State: {state}  |  LR: {lr:.4f}"
                            f"  |  Goals: {goals}  Concedes: {conc}"
                            f"  |  Episodes: {ep}  WR: {wr:.0%}"
                        )
            except Exception:
                pass

        # Refresh algorithm stats every ~6 s (12 × 500 ms)
        self._algo_stats_poll = getattr(self, "_algo_stats_poll", 0) + 1
        if self._algo_stats_poll >= 12:
            self._algo_stats_poll = 0
            self._refresh_algo_stats()

        # Check if bot process died
        try:
            if self._bot_proc and self._bot_proc.poll() is not None:
                self._bot_proc = None
                if hasattr(self, "_launch_btn"):
                    self._launch_btn.configure(state="normal")
                if hasattr(self, "_stop_btn"):
                    self._stop_btn.configure(state="disabled")
                if hasattr(self, "_bot_status_lbl"):
                    self._bot_status_lbl.configure(text="Bot: exited", fg=_C["yellow"])
                if hasattr(self, "_mc_status_var"):
                    self._mc_status_var.set("Bot process exited.")
        except Exception:
            pass
        # NOTE: no self._root.after() here — RefreshLoop handles rescheduling

    def _update_dashboard(self, d: dict):
        ms = int(d.get("my_score", 0))
        os_ = int(d.get("opp_score", 0))
        self._dash_score_var.set(f"{ms} — {os_}")
        self._dash_mode_var.set(str(d.get("current_mode", "—")).upper())
        self._dash_algo_var.set(str(d.get("current_algo", "A*")))
        self._dash_strategy_var.set(str(d.get("strategy", "BALANCED")).upper())
        self._dash_state_var.set(str(d.get("current_game_state", "—")))
        self._dash_reward_var.set(str(int(float(d.get("episode_reward", 0)))))
        self._dash_threat_var.set(f"{float(d.get('threat_level', 0)):.2f}")
        self._dash_tick_var.set(str(d.get("tick", 0)))
        self._dash_rl_role_var.set(str(d.get("rl_role", "—")))

        rb = d.get("reward_breakdown", {})
        rb_text = "  ".join(f"{k}: {v:.0f}" for k, v in list(rb.items())[:8])
        self._dash_reward_text.configure(state="normal")
        self._dash_reward_text.delete("1.0", "end")
        self._dash_reward_text.insert("end", rb_text or "No reward data yet.")
        self._dash_reward_text.configure(state="disabled")

        s_adv = d.get("search_advice", "")
        r_adv = d.get("rl_advice", "")
        strat = d.get("strategy", "")
        advice = f"Strategy: {strat}  |  Search: {s_adv}  |  RL: {r_adv}"
        self._dash_advice_var.set(advice or "Waiting for match…")

        # Match Progress
        if hasattr(self, "_dash_team_goals_var"):
            gl = int(d.get("goal_limit", 7))
            diff = ms - os_
            left = max(0, gl - ms)
            self._dash_team_goals_var.set(str(ms))
            self._dash_opp_goals_var.set(str(os_))
            self._dash_goal_diff_var.set(f"+{diff}" if diff > 0 else str(diff))
            self._dash_goal_limit_var.set(str(gl))
            self._dash_goals_left_var.set(str(left))

    # ── Algorithm helpers ─────────────────────────────────────────────────────

    def _on_algo_click(self, kind: str, key: str):
        self._show_algo_detail(kind, key)

    def _show_algo_detail(self, kind: str, key: str):
        if kind == "search":
            info = _SEARCH_ALGO_INFO.get(key)
            if info:
                name, desc, tc, sc = info
                txt = (f"{name}\n{'─'*40}\n"
                       f"Description:\n  {desc}\n\n"
                       f"Time Complexity: {tc}\n"
                       f"Space Note:      {sc}\n\n")
                # Add text from file if available
                file_text = self._algo_explanations.get(key, "")
                if file_text:
                    txt += "From documentation:\n" + file_text[:600]
            else:
                txt = key
        else:
            info = _RL_MODEL_INFO.get(key)
            if info:
                name, desc, kind_lbl = info
                txt = (f"{name}  [{kind_lbl}]\n{'─'*40}\n"
                       f"{desc}\n")
            else:
                txt = key

        self._algo_detail_text.configure(state="normal")
        self._algo_detail_text.delete("1.0", "end")
        self._algo_detail_text.insert("end", txt)
        self._algo_detail_text.configure(state="disabled")

    def _apply_combo(self, search_list: List[str], rl_list: List[str]):
        for k in self._search_vars:
            self._search_vars[k].set(k in search_list)
        for k in self._rl_vars:
            self._rl_vars[k].set(k in rl_list)
        self._apply_algo_selection()

    def _select_all_algos(self):
        for v in self._search_vars.values():
            v.set(True)
        for v in self._rl_vars.values():
            v.set(True)

    def _clear_all_algos(self):
        for v in self._search_vars.values():
            v.set(False)
        for v in self._rl_vars.values():
            v.set(False)

    def _apply_algo_selection(self):
        if not _ipc:
            return
        active_search = [k for k, v in self._search_vars.items() if v.get()]
        active_rl = [k for k, v in self._rl_vars.items() if v.get()]
        if not active_search:
            active_search = ["A*"]  # safety fallback
        try:
            _ipc.write_gui_commands({
                "active_search_algorithms": active_search,
                "active_rl_models": active_rl if active_rl else None,
            })
        except Exception:
            pass

    # ── Model settings helpers ────────────────────────────────────────────────

    def _apply_model_preset(self, keys: set, pros: str, cons: str):
        for k, v in self._rl_vars.items():
            v.set(k in keys)
        self._preset_notes_var.set(f"Pros: {pros}\n\nCons: {cons}")

    def _apply_model_settings(self):
        if not _ipc:
            return
        active_rl = [k for k, v in self._rl_vars.items() if v.get()]
        try:
            _ipc.write_gui_commands({"active_rl_models": active_rl or None})
        except Exception:
            pass


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    """Run the standalone dashboard on the main thread."""
    if tk is None:
        print("[ERROR] tkinter is not available. Cannot start dashboard.")
        sys.exit(1)

    root = tk.Tk()
    app = DashboardApp(root)  # noqa: F841
    root.mainloop()


if __name__ == "__main__":
    main()

