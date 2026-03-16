"""
Rocket League Search AI - Automatic Launcher

Startup sequence:
1. Initialize the search algorithms engine
2. Start the GUI and overlay system
3. Let the user pick mode and match type
4. Launch Rocket League via RLBot
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import time

RUNTIME_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(RUNTIME_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

_VENV_PYTHON = os.path.join(PROJECT_ROOT, ".venv", "Scripts", "python.exe")


def _relaunch_in_venv():
    """If running outside the .venv, re-launch this script with the venv Python."""
    if os.path.isfile(_VENV_PYTHON) and os.path.normcase(sys.executable) != os.path.normcase(_VENV_PYTHON):
        print(f"[INFO] Switching to .venv Python: {_VENV_PYTHON}")
        result = subprocess.run([_VENV_PYTHON] + sys.argv)
        sys.exit(result.returncode)


def _check_rl_running() -> bool:
    """Return True if Rocket League is already running."""
    try:
        output = subprocess.check_output(
            ["tasklist", "/FI", "IMAGENAME eq RocketLeague.exe"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return "RocketLeague.exe" in output
    except Exception:
        return False


def _check_rlbot_running() -> bool:
    """Return True if an RLBot process is already running."""
    try:
        output = subprocess.check_output(
            ["tasklist", "/FI", "IMAGENAME eq RLBot.exe"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return "RLBot.exe" in output
    except Exception:
        return False


def _step1_init_search_engine() -> bool:
    """Step 1: Load and validate all algorithm modules."""
    print("\n[Step 1/2] Initializing algorithms engine...")

    # Each entry: (module, label, required_attrs)
    modules_to_check = [
        ("core.all_algorithms",
         "Search algorithms (A* / BFS / UCS / DFS / Greedy / Beam / IDA* / DecisionTree)",
         ["astar_search", "bfs_search", "run_search"]),
        ("core.rl_algorithms",
         "RL algorithms (Q-Learning / SARSA / DQN / PPO / A2C / MC / ModelBased / AC / PG / Ensemble / Online)",
         ["QLearningRoleSelector", "DQNRoleSelector", "PPORoleSelector"]),
        ("core.rl_state",      "RL State Discretization",     []),
        ("core.reward_calculator", "Reward Calculator",        []),
        ("core.adaptive_learner",  "Adaptive RL Coordinator",  ["AdaptiveLearner"]),
    ]

    all_ok = True
    for mod_name, label, attrs in modules_to_check:
        try:
            mod = importlib.import_module(mod_name)
            missing = [a for a in attrs if not hasattr(mod, a)]
            if missing:
                print(f"  [WARN] {label} ({mod_name}) — missing: {missing}")
            else:
                print(f"  [OK] {label} ({mod_name})")
        except Exception as exc:
            print(f"  [FAIL] {label} ({mod_name}): {exc}")
            all_ok = False

    if all_ok:
        print("  All algorithm modules ready.\n")
    else:
        print("  WARNING: Some modules failed to load.\n")
    return all_ok


def _step2_init_gui_overlay() -> bool:
    """Step 2: Validate GUI and overlay modules can be loaded."""
    print("[Step 2/2] Initializing GUI and overlay system...")
    modules_to_check = [
        ("visualization.overlay_renderer", "Overlay renderer"),
        ("gui.control_panel", "Control panel"),
        ("game_logic.decision_engine", "Decision engine"),
        ("game_logic.mode_manager", "Mode manager"),
        ("navigation.path_planner", "Path planner"),
    ]
    all_ok = True
    for mod_name, label in modules_to_check:
        try:
            importlib.import_module(mod_name)
            print(f"  [OK] {label} loaded ({mod_name})")
        except Exception as exc:
            print(f"  [FAIL] {label} ({mod_name}): {exc}")
            all_ok = False

    panel = None
    try:
        from gui.control_panel import ControlPanel
        panel = ControlPanel(initial_mode="manual")
        if panel.wait_for_startup(timeout=1.5):
            print("  [OK] Control panel GUI opened successfully")
        else:
            gui_error = panel.startup_error() or "unknown startup failure"
            print(f"  [FAIL] Control panel GUI could not open: {gui_error}")
            all_ok = False
    except Exception as exc:
        print(f"  [FAIL] Control panel GUI smoke test failed: {exc}")
        all_ok = False
    finally:
        if panel is not None:
            panel.close()

    try:
        from game_logic.decision_engine import DecisionEngine
        engine = DecisionEngine(PROJECT_ROOT)
        q_states = len(engine.adaptive.q_role.q_table)
        opp_states = len(engine.adaptive.sarsa_opp.q_table)
        model_state = "loaded" if (q_states or opp_states) else "fresh"
        print(
            f"  [OK] Adaptive model ready ({model_state}: role_states={q_states}, "
            f"opp_states={opp_states})"
        )
    except Exception as exc:
        print(f"  [FAIL] Adaptive model smoke test failed: {exc}")
        all_ok = False

    knowledge_path = os.path.join(PROJECT_ROOT, "model", "search_knowledge.json")
    if os.path.exists(knowledge_path):
        print(f"  [OK] Knowledge base found ({knowledge_path})")
    else:
        print(f"  [INFO] No existing knowledge base; will create on first run.")

    if all_ok:
        print("  GUI and overlay systems ready.\n")
    else:
        print("  WARNING: Some GUI/overlay modules failed to load.\n")
    return all_ok


def _get_venv_python() -> str:
    """Return the .venv Python executable, falling back to sys.executable."""
    venv_python = os.path.join(PROJECT_ROOT, ".venv", "Scripts", "python.exe")
    if os.path.isfile(venv_python):
        return venv_python
    return sys.executable


def _launch_game() -> None:
    """Launch the bot and connect to whatever match is running."""
    print()
    print("=" * 60)
    print("  Launching bot...")
    print("=" * 60)

    if _check_rl_running():
        print("  [WARNING] Rocket League is already running.")
        print("  If the bot can't connect, close RL and run this again.")
        print("  RLBot needs to start RL with the -rlbot flag.")
    else:
        print("  [INFO] RLBot will launch Rocket League automatically.")

    run_match_path = os.path.join(RUNTIME_DIR, "run_match.py")
    if not os.path.exists(run_match_path):
        print(f"  [FAIL] run_match.py not found at {run_match_path}")
        return

    python_exe = _get_venv_python()
    print()
    print("=" * 60)
    print("  IN-GAME KEYBOARD CONTROLS")
    print("  ----------------------------------------")
    print("    M = Manual   (you play, bot learns)")
    print("    N = Balanced (attack + defense)")
    print("    B = Attack   (aggressive, push to goal)")
    print("    V = Defense  (protect own goal)")
    print("    P = Session-only model (fresh learning branch)")
    print("    O = Persistent model (return to saved evolving model)")
    print("  ----------------------------------------")
    print("  Manual mode:  WASD/Arrows  Space=Jump  LShift=Boost")
    print("  Match setup:  Start the match yourself from inside Rocket League")
    print("=" * 60)
    print()

    # Simple launch: bot will wait for Rocket League, then inject into
    # whatever match the user starts from inside the game.
    cmd = [python_exe, run_match_path]
    subprocess.run(cmd)


def launch() -> None:
    """Run the full startup sequence."""
    print()
    print("=" * 60)
    print(r"                                                          ")
    print(r"  ##   ##  #####  ####    ####     ####   ##  ##    #####     #####    ")
    print(r"  ### ###  ##     ## ##  ##  ##    ## ##  ##   ## ##     ## ##     ##   ")
    print(r"  ## # ##  ####   ##  ## ##  ##    ##  ##  ###### ######### ######### ")
    print(r"  ##   ##  ##     ## ##  ##  ##    ## ##    ####  ##     ## ##     ##   ")
    print(r"  ##   ##  #####  ####    ####     ####     ##    ##     ## ##     ##   ")
    print(r"                                                          ")
    print("       Rocket League AI - Classical Search + Deep RL")
    print("  A* | BFS | UCS | Greedy | DFS | Decision Tree | Beam Search | IDA*")
    print("  DQN | PPO | A2C | Monte Carlo | Model-Based RL | Q-Learning | SARSA")
    print("=" * 60)

    if not _step1_init_search_engine():
        print("ERROR: Search engine initialization failed. Aborting.")
        return

    if not _step2_init_gui_overlay():
        print("ERROR: GUI/Overlay initialization failed. Aborting.")
        return

    _launch_game()


def main():
    """Thin entry point: ensure we are in the .venv, then open the GUI dashboard."""
    _relaunch_in_venv()
    print("="*50)
    print("  Rocket League AI System")
    print("  Author: medo dyaa")
    print("="*50)

    # Add project root to path and launch the standalone dashboard
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    from gui.control_panel import main as dashboard_main
    dashboard_main()


if __name__ == "__main__":
    main()
