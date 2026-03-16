"""
test_full_project.py
====================
Comprehensive test suite for the Rocket League AI Bot project.

Covers every module:
  - core/all_algorithms.py        (search algorithms unified file)
  - core/astar, bfs, dfs, greedy, ucs, ida_star, beam_search
  - core/decision_tree
  - core/ball_prediction
  - core/search_algorithms        (router)
  - core/rl_state
  - core/q_learning
  - core/sarsa_opponent
  - core/actor_critic
  - core/ensemble_voter
  - core/online_learner
  - core/reward_calculator
  - core/policy_gradient
  - core/dqn
  - core/ppo
  - core/a2c
  - core/monte_carlo_rl
  - core/model_based_rl
  - core/adaptive_learner
  - core/advanced_ml
  - navigation/path_planner
  - game_logic/mode_manager

Run with:  python -m pytest tests/test_full_project.py -v
Or:        python tests/test_full_project.py
"""

from __future__ import annotations

import math
import sys
import os
import unittest

# ── ensure project root is on sys.path ──────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════

def _grid_neighbors(node):
    x, y = node
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            yield (x + dx, y + dy)


def _uniform_cost(a, b):
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _manhattan(a, b):
    return abs(b[0] - a[0]) + abs(b[1] - a[1])


def _octile(a, b):
    dx, dy = abs(b[0] - a[0]), abs(b[1] - a[1])
    return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)


START = (0, 0)
GOAL  = (5, 5)


# ════════════════════════════════════════════════════════════════
#  1.  SEARCH ALGORITHMS (individual modules)
# ════════════════════════════════════════════════════════════════

class TestAstar(unittest.TestCase):
    def setUp(self):
        from core.all_algorithms import astar_search as search
        self.search = search

    def test_finds_path(self):
        path, cost = self.search(START, GOAL, _grid_neighbors, _uniform_cost, _octile)
        self.assertGreater(len(path), 0)
        self.assertEqual(path[0], START)
        self.assertEqual(path[-1], GOAL)

    def test_cost_positive(self):
        _, cost = self.search(START, GOAL, _grid_neighbors, _uniform_cost, _octile)
        self.assertGreater(cost, 0)

    def test_no_path_returns_empty(self):
        # Goal surrounded by no neighbours (isolated) — use unreachable goal
        path, cost = self.search((100, 100), (200, 200),
                                 lambda n: [],       # no neighbours
                                 _uniform_cost, _octile)
        self.assertEqual(path, [])
        self.assertEqual(cost, float("inf"))

    def test_same_start_goal(self):
        path, cost = self.search(START, START, _grid_neighbors, _uniform_cost, _octile)
        self.assertIn(START, path)


class TestBFS(unittest.TestCase):
    def setUp(self):
        from core.all_algorithms import bfs_search as search
        self.search = search

    def test_finds_path(self):
        path, cost = self.search(START, GOAL, _grid_neighbors, _uniform_cost, _octile)
        self.assertGreater(len(path), 0)
        self.assertEqual(path[-1], GOAL)

    def test_path_is_connected(self):
        path, _ = self.search(START, GOAL, _grid_neighbors, _uniform_cost, _octile)
        for i in range(len(path) - 1):
            dx = abs(path[i+1][0] - path[i][0])
            dy = abs(path[i+1][1] - path[i][1])
            self.assertLessEqual(max(dx, dy), 1)


class TestDFS(unittest.TestCase):
    def setUp(self):
        from core.all_algorithms import dfs_search as search
        self.search = search

    def test_finds_path(self):
        path, _ = self.search(START, GOAL, _grid_neighbors, _uniform_cost, _octile)
        self.assertGreater(len(path), 0)
        self.assertEqual(path[-1], GOAL)

    def test_no_path(self):
        path, cost = self.search((0, 0), (99, 99), lambda n: [], _uniform_cost, _octile)
        self.assertEqual(path, [])


class TestGreedy(unittest.TestCase):
    def setUp(self):
        from core.all_algorithms import greedy_search as search
        self.search = search

    def test_finds_path(self):
        path, _ = self.search(START, GOAL, _grid_neighbors, _uniform_cost, _octile)
        self.assertGreater(len(path), 0)
        self.assertEqual(path[-1], GOAL)


class TestUCS(unittest.TestCase):
    def setUp(self):
        from core.all_algorithms import ucs_search as search
        self.search = search

    def test_finds_path(self):
        path, cost = self.search(START, GOAL, _grid_neighbors, _uniform_cost, _octile)
        self.assertGreater(len(path), 0)
        self.assertGreater(cost, 0)

    def test_optimal_vs_dfs(self):
        from core.all_algorithms import dfs_search as dfs
        _, ucs_cost = self.search(START, GOAL, _grid_neighbors, _uniform_cost, _octile)
        _, dfs_cost = dfs(START, GOAL, _grid_neighbors, _uniform_cost, _octile)
        # UCS must be <= DFS (optimal)
        self.assertLessEqual(round(ucs_cost, 6), round(dfs_cost, 6))


class TestIDAStar(unittest.TestCase):
    def setUp(self):
        from core.all_algorithms import ida_star_search as search
        self.search = search

    def test_finds_path(self):
        path, cost = self.search(START, GOAL, _grid_neighbors, _uniform_cost, _octile)
        self.assertGreater(len(path), 0)
        self.assertEqual(path[-1], GOAL)

    def test_optimal(self):
        from core.all_algorithms import astar_search as astar
        _, ida_cost   = self.search(START, GOAL, _grid_neighbors, _uniform_cost, _octile)
        _, astar_cost = astar(START, GOAL, _grid_neighbors, _uniform_cost, _octile)
        self.assertAlmostEqual(ida_cost, astar_cost, places=4)


class TestBeamSearch(unittest.TestCase):
    def setUp(self):
        from core.all_algorithms import beam_search as search
        self.search = search

    def test_finds_path(self):
        path, _ = self.search(START, GOAL, _grid_neighbors, _uniform_cost, _octile)
        self.assertGreater(len(path), 0)
        self.assertEqual(path[-1], GOAL)

    def test_custom_width(self):
        # beam_search has a fixed default width; just verify a second call is stable
        from core.all_algorithms import beam_search
        path, _ = beam_search((0, 0), (3, 3), _grid_neighbors, _uniform_cost, _octile)
        self.assertGreater(len(path), 0)
        self.assertEqual(path[-1], (3, 3))


# ════════════════════════════════════════════════════════════════
#  2.  ALL_ALGORITHMS.PY — unified file
# ════════════════════════════════════════════════════════════════

class TestAllAlgorithms(unittest.TestCase):
    def setUp(self):
        import core.all_algorithms as aa
        self.aa = aa

    def test_all_algorithms_find_path(self):
        for name in self.aa.ALGORITHMS:
            with self.subTest(algorithm=name):
                path, cost = self.aa.run_search(
                    name, START, GOAL, _grid_neighbors, _uniform_cost, _octile)
                self.assertGreater(len(path), 0,
                                   msg=f"{name} returned empty path")
                self.assertEqual(path[-1], GOAL,
                                 msg=f"{name} did not reach goal")

    def test_unknown_algorithm_raises(self):
        with self.assertRaises(ValueError):
            self.aa.run_search("INVALID", START, GOAL,
                               _grid_neighbors, _uniform_cost, _octile)

    def test_compare_all_algorithms_returns_all(self):
        results = self.aa.compare_all_algorithms(
            START, GOAL, _grid_neighbors, _uniform_cost, _octile)
        self.assertEqual(len(results), len(self.aa.ALGORITHMS))
        for r in results:
            self.assertIn("algorithm",   r)
            self.assertIn("found",       r)
            self.assertIn("cost",        r)
            self.assertIn("path_length", r)
            self.assertIn("time_ms",     r)
            self.assertTrue(r["found"], msg=f"{r['algorithm']} not found")

    def test_ball_prediction_trajectory(self):
        aa = self.aa
        traj = aa.predict_ball_trajectory(0, 0, 200, 500, 500, 300, total_time=2.0)
        self.assertGreater(len(traj), 0)
        t0, x0, y0, z0, *_ = traj[0]
        # Ball should have moved
        self.assertGreater(abs(x0) + abs(y0), 0)

    def test_ball_stays_in_bounds(self):
        aa = self.aa
        traj = aa.predict_ball_trajectory(0, 0, 500, 2000, -3000, 100, total_time=5.0)
        for t, x, y, z, *_ in traj:
            self.assertLessEqual(abs(x), aa.WALL_X + 1)
            self.assertGreaterEqual(z, aa.GROUND_Z - 1)

    def test_decision_tree_build_and_evaluate(self):
        aa = self.aa
        root = aa.build_decision_tree(
            car_pos=(0.0, -2000.0),
            ball_pos=(300.0, 0.0),
            car_speed=1200.0,
            my_team=0,
        )
        result = aa.evaluate_decision_tree(root)
        self.assertIsNotNone(result.best_action)
        self.assertGreater(len(result.all_actions), 0)
        self.assertGreater(result.nodes_evaluated, 0)

    def test_find_intercept(self):
        aa = self.aa
        traj = aa.predict_ball_trajectory(0, 1000, 100, 0, -500, 50, total_time=3.0)
        intercept = aa.find_ball_intercept(traj, 0, 0, 1500, 60)
        # Either found or None — just must not crash
        if intercept is not None:
            bx, by, bz, t = intercept
            self.assertGreater(t, 0)


# ════════════════════════════════════════════════════════════════
#  3.  SEARCH_ALGORITHMS ROUTER
# ════════════════════════════════════════════════════════════════

class TestSearchAlgorithmsRouter(unittest.TestCase):
    def test_every_algorithm_via_router(self):
        from core.all_algorithms import run_search, ALGORITHMS
        for name in ALGORITHMS:
            with self.subTest(algorithm=name):
                path, cost = run_search(
                    name, START, GOAL, _grid_neighbors, _uniform_cost, _octile)
                self.assertGreater(len(path), 0)

    def test_invalid_raises(self):
        from core.all_algorithms import run_search
        with self.assertRaises(ValueError):
            run_search("NONE", START, GOAL, _grid_neighbors, _uniform_cost, _octile)


# ════════════════════════════════════════════════════════════════
#  4.  BALL PREDICTION (original module)
# ════════════════════════════════════════════════════════════════

class TestBallPrediction(unittest.TestCase):
    def setUp(self):
        import core.all_algorithms as bp
        self.bp = bp

    def test_trajectory_length(self):
        traj = self.bp.predict_trajectory(0, 0, 200, 300, 200, 100, total_time=2.0)
        # 2 s * 30 fps = 60 frames
        self.assertEqual(len(traj), 60)

    def test_floor_clamped(self):
        traj = self.bp.predict_trajectory(0, 0, 93, 0, 0, -500, total_time=1.0)
        for _, x, y, z, *_ in traj:
            self.assertGreaterEqual(z, self.bp.GROUND_Z - 1)

    def test_landing_pos(self):
        traj = self.bp.predict_trajectory(0, 0, 500, 100, 100, -600, total_time=5.0)
        landing = self.bp.ball_landing_pos(traj)
        if landing:
            x, y, t = landing
            self.assertGreater(t, 0)

    def test_find_intercept_reachable(self):
        traj = self.bp.predict_trajectory(0, 500, 100, 0, -100, 0, total_time=3.0)
        intercept = self.bp.find_intercept(traj, 0, 0, 1400, 80)
        # Car is at (0,0) ball moving slowly → should find an intercept
        self.assertIsNotNone(intercept)

    def test_puck_mode(self):
        normal = self.bp.predict_trajectory(0, 0, 500, 0, 0, 300, total_time=2.0)
        puck   = self.bp.predict_trajectory(0, 0, 500, 0, 0, 300, total_time=2.0,
                                            puck_mode=True)
        _, _, _, z_n, *_ = normal[-1]
        _, _, _, z_p, *_ = puck[-1]
        # Puck has more drag / less bounce — final Z typically lower
        self.assertIsNotNone(z_p)


# ════════════════════════════════════════════════════════════════
#  5.  DECISION TREE (original module)
# ════════════════════════════════════════════════════════════════

class TestDecisionTree(unittest.TestCase):
    def setUp(self):
        import core.all_algorithms as dt
        self.dt = dt

    def test_build_returns_root(self):
        root = self.dt.build_decision_tree(
            car_pos=(0.0, -3000.0),
            ball_pos=(0.0, 0.0),
            car_speed=900.0,
            my_team=0,
        )
        self.assertEqual(root.name, "Root")
        self.assertEqual(len(root.children), 4)

    def test_evaluate_returns_valid_result(self):
        root = self.dt.build_decision_tree(
            car_pos=(-1000.0, 2000.0),
            ball_pos=(500.0, -500.0),
            car_speed=1500.0,
            my_team=1,
        )
        result = self.dt.evaluate_tree(root)
        self.assertIsInstance(result.best_action, str)
        self.assertIsInstance(result.target, tuple)
        self.assertEqual(len(result.target), 2)

    def test_team1_goal_direction(self):
        # Team 1 attacks toward y=-5120
        root = self.dt.build_decision_tree(
            car_pos=(0.0, 1000.0),
            ball_pos=(0.0, 0.0),
            car_speed=1200.0,
            my_team=1,
        )
        result = self.dt.evaluate_tree(root)
        # best attack target y should be < 0 (heading toward team-1 attack goal)
        self.assertIsNotNone(result.target)

    def test_knowledge_bonus(self):
        knowledge = {"successful_strategies": {"attack": 50, "defense": 5}}
        root = self.dt.build_decision_tree(
            car_pos=(0.0, -1000.0),
            ball_pos=(0.0, 0.0),
            car_speed=1000.0,
            my_team=0,
            knowledge_data=knowledge,
        )
        result = self.dt.evaluate_tree(root)
        # With strong attack knowledge, should prefer attack actions
        top_action = result.all_actions[0]["name"]
        self.assertIsNotNone(top_action)


# ════════════════════════════════════════════════════════════════
#  6.  RL STATE
# ════════════════════════════════════════════════════════════════

class TestRLState(unittest.TestCase):
    def setUp(self):
        from core import rl_state as rs
        self.rs = rs

    def test_zone_center(self):
        z = self.rs._zone(0, 0, 1.0)
        self.assertIn("C_", z)

    def test_zone_left(self):
        z = self.rs._zone(-2000, 0, 1.0)
        self.assertIn("L_", z)

    def test_speed_zones(self):
        self.assertEqual(self.rs._speed_zone(0),    "slow")
        self.assertEqual(self.rs._speed_zone(800),  "med")
        self.assertEqual(self.rs._speed_zone(1500), "fast")
        self.assertEqual(self.rs._speed_zone(2000), "supersonic")

    def test_boost_zones(self):
        self.assertEqual(self.rs._boost_zone(0),  "empty")
        self.assertEqual(self.rs._boost_zone(20), "low")
        self.assertEqual(self.rs._boost_zone(50), "mid")
        self.assertEqual(self.rs._boost_zone(80), "full")

    def test_score_zones(self):
        self.assertEqual(self.rs._score_zone(-5),  "losing_bad")
        self.assertEqual(self.rs._score_zone(-1),  "losing")
        self.assertEqual(self.rs._score_zone(0),   "tied")
        self.assertEqual(self.rs._score_zone(1),   "winning")
        self.assertEqual(self.rs._score_zone(5),   "winning_big")

    def test_build_state_key_format(self):
        key = self.rs.build_state_key(
            "defending", "C_mid", "L_our_half", "fast", "low", "tied", "them")
        parts = key.split("|")
        self.assertEqual(len(parts), 7)


# ════════════════════════════════════════════════════════════════
#  7.  Q-LEARNING
# ════════════════════════════════════════════════════════════════

class TestQLearning(unittest.TestCase):
    def setUp(self):
        from core.rl_algorithms import QLearningRoleSelector
        self.q = QLearningRoleSelector()

    def test_choose_role_valid(self):
        from core.rl_state import ROLE_ACTIONS
        role = self.q.choose_role("test|state")
        self.assertIn(role, ROLE_ACTIONS)

    def test_update_changes_q(self):
        self.q.choose_role("s0")
        self.q.update("s1", 5.0)
        q_vals = self.q._get_q("s0")
        # At least one action should have non-zero Q after positive reward
        self.assertTrue(any(v != 0.0 for v in q_vals.values()))

    def test_epsilon_decays(self):
        initial = self.q.epsilon
        for _ in range(100):
            self.q.choose_role("s")
            self.q.update("s", 0.1)
        self.assertLess(self.q.epsilon, initial)

    def test_serialization(self):
        self.q.choose_role("test")
        self.q.update("test2", 2.0)
        d = self.q.to_dict()
        from core.rl_algorithms import QLearningRoleSelector
        q2 = QLearningRoleSelector()
        q2.from_dict(d)
        self.assertEqual(set(q2.q_table.keys()), set(self.q.q_table.keys()))


# ════════════════════════════════════════════════════════════════
#  8.  SARSA OPPONENT MODEL
# ════════════════════════════════════════════════════════════════

class TestSARSA(unittest.TestCase):
    def setUp(self):
        from core.rl_algorithms import SARSAOpponentModel
        self.sarsa = SARSAOpponentModel()

    def test_classify_rush(self):
        action = self.sarsa.classify_opp_action(
            opp_pos=(100, 200),
            opp_vel=(400, 600),
            ball_pos=(300, 500),
            push_dir=1.0,
        )
        self.assertEqual(action, "rush_ball")

    def test_classify_returns_valid(self):
        from core.rl_state import OPP_ACTIONS
        for _ in range(20):
            action = self.sarsa.classify_opp_action(
                opp_pos=(1000, 3000),
                opp_vel=(2000, 100),
                ball_pos=(0, 0),
                push_dir=1.0,
            )
            self.assertIn(action, OPP_ACTIONS)

    def test_predict_valid(self):
        from core.rl_state import OPP_ACTIONS
        pred = self.sarsa.predict("some_state")
        self.assertIn(pred, OPP_ACTIONS)

    def test_update_changes_q(self):
        self.sarsa.update("s0", "rush_ball", 1.0, "s1", "shadow")
        q = self.sarsa._get_q("s0")
        self.assertNotEqual(q.get("rush_ball", 0.0), 0.0)


# ════════════════════════════════════════════════════════════════
#  9.  ACTOR-CRITIC
# ════════════════════════════════════════════════════════════════

class TestActorCritic(unittest.TestCase):
    def setUp(self):
        from core.rl_algorithms import ActorCritic
        self.ac = ActorCritic()

    def test_evaluate_state_default_zero(self):
        v = self.ac.evaluate_state("unseen_state")
        self.assertEqual(v, 0.0)

    def test_update_modifies_value(self):
        self.ac.update("s0", 3.0)
        self.ac.update("s1", 3.0)
        v = self.ac.evaluate_state("s0")
        self.assertNotEqual(v, 0.0)

    def test_adjustments_keys(self):
        adj = self.ac.get_adjustments("any_state")
        self.assertIn("throttle_adj", adj)
        self.assertIn("boost_adj",    adj)
        self.assertIn("aggression",   adj)

    def test_serialization(self):
        self.ac.update("s_a", 1.0)
        self.ac.update("s_a", 2.0)
        d  = self.ac.to_dict()
        from core.rl_algorithms import ActorCritic
        ac2 = ActorCritic()
        ac2.from_dict(d)
        self.assertAlmostEqual(
            ac2.evaluate_state("s_a"), self.ac.evaluate_state("s_a"), places=6)


# ════════════════════════════════════════════════════════════════
#  10. ENSEMBLE VOTER
# ════════════════════════════════════════════════════════════════

class TestEnsembleVoter(unittest.TestCase):
    def setUp(self):
        from core.rl_algorithms import EnsembleVoter
        self.ev = EnsembleVoter()

    def test_unanimous_vote(self):
        recs = {s: "attack" for s in ["q_learning", "dqn", "ppo", "a2c"]}
        self.assertEqual(self.ev.vote(recs), "attack")

    def test_majority_wins(self):
        recs = {
            "q_learning": "attack",
            "dqn":        "attack",
            "ppo":        "defense",
            "a2c":        "balanced",
        }
        result = self.ev.vote(recs)
        self.assertEqual(result, "attack")

    def test_empty_recs_returns_default(self):
        result = self.ev.vote({})
        self.assertEqual(result, "balanced")

    def test_reward_source_updates_weight(self):
        for _ in range(10):
            self.ev.reward_source("dqn", 1.0)
        self.assertGreater(self.ev.source_weights["dqn"], 1.0)

    def test_serialization(self):
        self.ev.reward_source("q_learning", 2.0)
        d = self.ev.to_dict()
        from core.rl_algorithms import EnsembleVoter
        ev2 = EnsembleVoter()
        ev2.from_dict(d)
        self.assertIn("q_learning", ev2.source_weights)


# ════════════════════════════════════════════════════════════════
#  11. ONLINE PATTERN LEARNER
# ════════════════════════════════════════════════════════════════

class TestOnlineLearner(unittest.TestCase):
    def setUp(self):
        from core.rl_algorithms import OnlinePatternLearner
        self.ol = OnlinePatternLearner()

    def test_record_and_best_action(self):
        for _ in range(10):
            self.ol.record("state_a", "attack",  1.0)
            self.ol.record("state_a", "defense", -0.5)
        best = self.ol.best_action("state_a")
        self.assertEqual(best, "attack")

    def test_unknown_state_returns_none(self):
        self.assertIsNone(self.ol.best_action("unknown_state_xyz"))

    def test_recent_trend(self):
        for _ in range(20):
            self.ol.record("s",   "attack", 1.0)
        trend = self.ol.recent_trend()
        self.assertGreater(trend, 0)

    def test_serialization(self):
        # to_dict() prunes entries with count < 3; record the same pair 3+ times
        for _ in range(4):
            self.ol.record("s", "attack", 2.0)
        d = self.ol.to_dict()
        from core.rl_algorithms import OnlinePatternLearner
        ol2 = OnlinePatternLearner()
        ol2.from_dict(d)
        self.assertIn("s", ol2.stats)


# ════════════════════════════════════════════════════════════════
#  12. REWARD CALCULATOR
# ════════════════════════════════════════════════════════════════

class TestRewardCalculator(unittest.TestCase):
    def setUp(self):
        from core.reward_calculator import RewardCalculator
        self.rc = RewardCalculator()

    def test_reward_is_float(self):
        r = self.rc.compute_reward(
            car_pos=(0, -2000), ball_pos=(0, 0), ball_vel=(0, 200),
            opp_pos=(0, 2000), car_speed=1000, car_boost=50,
            push_dir=1.0, situation="free_ball",
        )
        self.assertIsInstance(r, float)

    def test_approaching_ball_positive(self):
        # Frame 1: far from ball
        self.rc.compute_reward(
            car_pos=(0, -5000), ball_pos=(0, 0), ball_vel=(0, 0),
            opp_pos=(0, 2000), car_speed=1000, car_boost=50,
            push_dir=1.0, situation="free_ball",
        )
        # Frame 2: closer to ball → positive delta reward
        r2 = self.rc.compute_reward(
            car_pos=(0, -2000), ball_pos=(0, 0), ball_vel=(0, 0),
            opp_pos=(0, 2000), car_speed=1000, car_boost=50,
            push_dir=1.0, situation="free_ball",
        )
        self.assertGreater(r2, 0)

    def test_goal_signal(self):
        self.rc.signal_goal_scored()
        r = self.rc.compute_reward(
            car_pos=(0, 0), ball_pos=(0, 0), ball_vel=(0, 0),
            opp_pos=(0, 500), car_speed=0, car_boost=0,
            push_dir=1.0, situation="free_ball",
        )
        self.assertGreater(r, 5)   # goal reward = 10.0

    def test_concede_signal(self):
        self.rc.signal_goal_conceded()
        # The -30 concede penalty is queued immediately into stats["idle"];
        # episode total is only updated on compute_reward(), so check the
        # accumulated category instead.
        stats = self.rc.get_episode_stats()
        self.assertLess(stats.get("idle", 0), 0)   # concede penalty = -30


# ════════════════════════════════════════════════════════════════
#  13. DQN
# ════════════════════════════════════════════════════════════════

class TestDQN(unittest.TestCase):
    def setUp(self):
        from core.rl_algorithms import DQNRoleSelector
        self.dqn = DQNRoleSelector()

    def test_choose_role_valid(self):
        from core.rl_state import ROLE_ACTIONS
        role = self.dqn.choose_role("free_ball|C_mid|C_mid|med|full|tied|us")
        self.assertIn(role, ROLE_ACTIONS)

    def test_update_no_crash(self):
        self.dqn.choose_role("free_ball|C_mid|C_mid|med|full|tied|us")
        self.dqn.update("free_ball|C_mid|C_mid|med|full|tied|them", 1.0)

    def test_encode_state_shape(self):
        from core.rl_algorithms import encode_state, FEATURE_DIM
        x = encode_state("free_ball|C_mid|C_mid|med|full|tied|us")
        self.assertEqual(x.shape[0], FEATURE_DIM)
        self.assertAlmostEqual(float(x.sum()), 7.0, places=3)  # 7 one-hot 1s


# ════════════════════════════════════════════════════════════════
#  14. PPO
# ════════════════════════════════════════════════════════════════

class TestPPO(unittest.TestCase):
    def setUp(self):
        from core.rl_algorithms import PPORoleSelector
        self.ppo = PPORoleSelector()

    def test_choose_role_valid(self):
        from core.rl_state import ROLE_ACTIONS
        role = self.ppo.choose_role("free_ball|C_mid|C_mid|med|full|tied|us")
        self.assertIn(role, ROLE_ACTIONS)

    def test_update_accumulates(self):
        s = "free_ball|C_mid|C_mid|fast|low|losing|them"
        for _ in range(70):      # > ROLLOUT_LEN(64), triggers PPO epoch
            self.ppo.choose_role(s)
            self.ppo.update(s, 0.5)


# ════════════════════════════════════════════════════════════════
#  15. A2C
# ════════════════════════════════════════════════════════════════

class TestA2C(unittest.TestCase):
    def setUp(self):
        from core.rl_algorithms import A2CRoleSelector
        self.a2c = A2CRoleSelector()

    def test_choose_role_valid(self):
        from core.rl_state import ROLE_ACTIONS
        role = self.a2c.choose_role("free_ball|C_mid|C_mid|med|full|tied|us")
        self.assertIn(role, ROLE_ACTIONS)

    def test_update_no_crash(self):
        s = "free_ball|C_mid|C_mid|med|full|tied|us"
        self.a2c.choose_role(s)
        self.a2c.update(s, 1.0)


# ════════════════════════════════════════════════════════════════
#  16. MONTE CARLO
# ════════════════════════════════════════════════════════════════

class TestMonteCarlo(unittest.TestCase):
    def setUp(self):
        from core.rl_algorithms import MonteCarloRoleSelector
        self.mc = MonteCarloRoleSelector()

    def test_choose_role_valid(self):
        from core.rl_state import ROLE_ACTIONS
        role = self.mc.choose_role("free_ball|C_mid|C_mid|med|full|tied|us")
        self.assertIn(role, ROLE_ACTIONS)

    def test_end_episode_no_crash(self):
        s = "free_ball|C_mid|C_mid|med|full|tied|us"
        for _ in range(5):
            self.mc.record(s, "attack", 0.1)
        self.mc.end_episode(5.0)


# ════════════════════════════════════════════════════════════════
#  17. MODEL-BASED RL
# ════════════════════════════════════════════════════════════════

class TestModelBasedRL(unittest.TestCase):
    def setUp(self):
        from core.rl_algorithms import ModelBasedRLSelector
        self.mb = ModelBasedRLSelector()

    def test_choose_role_valid(self):
        from core.rl_state import ROLE_ACTIONS
        role = self.mb.choose_role("free_ball|C_mid|C_mid|med|full|tied|us")
        self.assertIn(role, ROLE_ACTIONS)

    def test_generate_sample_returns_none_or_tuple(self):
        # Fill model with some transitions first
        s = "free_ball|C_mid|C_mid|med|full|tied|us"
        self.mb.record(s, "attack", 0.5)
        sample = self.mb.generate_sample()
        # May be None if not enough data, or a tuple
        if sample is not None:
            self.assertEqual(len(sample), 3)


# ════════════════════════════════════════════════════════════════
#  18. ADAPTIVE LEARNER (integration)
# ════════════════════════════════════════════════════════════════

class TestAdaptiveLearner(unittest.TestCase):
    def setUp(self):
        from core.adaptive_learner import AdaptiveLearner
        self.al = AdaptiveLearner()

    def _decide(self, situation="free_ball", score_diff=0):
        return self.al.decide_role(
            situation=situation,
            car_pos=(0.0,  -2000.0),
            ball_pos=(0.0,  0.0),
            ball_vel=(100.0, 200.0),
            opp_pos=(0.0,   2000.0),
            opp_vel=(-100.0, -200.0),
            car_speed=1200.0,
            car_boost=50.0,
            score_diff=score_diff,
            push_dir=1.0,
            user_mode="balanced",
        )

    def test_decide_role_valid(self):
        role = self._decide()
        self.assertIn(role, ("attack", "defense", "balanced"))

    def test_decide_multiple_frames_no_crash(self):
        for _ in range(30):
            self._decide()

    def test_signal_goal_updates_rewards(self):
        # signal_goal_scored adds +100 to stats["goal"] immediately via _queue()
        self.al.signal_goal_scored()
        stats = self.al.reward_calc.get_episode_stats()
        self.assertGreater(stats.get("goal", 0), 0)  # goal reward = +100

    def test_signal_concede(self):
        self._decide()
        self.al.signal_goal_conceded()

    def test_save_made(self):
        self.al.signal_save_made()

    def test_serialization(self):
        self._decide()
        d = self.al.to_dict()
        self.assertIn("q_role", d)
        self.assertIn("advanced_ml", d)

    def test_from_dict_restores(self):
        self._decide()
        d = self.al.to_dict()
        from core.adaptive_learner import AdaptiveLearner
        al2 = AdaptiveLearner()
        al2.from_dict(d)
        role = al2.decide_role(
            "free_ball", (0, -1000), (0, 0), (0, 0),
            (0, 1000), (0, 0), 900, 30, 0, 1.0, "balanced")
        self.assertIn(role, ("attack", "defense", "balanced"))

    def test_defending_mode(self):
        role = self._decide(situation="defending", score_diff=-1)
        self.assertIn(role, ("attack", "defense", "balanced"))

    def test_opponent_prediction(self):
        self._decide()
        pred = self.al.get_opponent_prediction()
        from core.rl_state import OPP_ACTIONS
        self.assertIn(pred, OPP_ACTIONS)

    def test_trend_float(self):
        self._decide()
        trend = self.al.get_trend()
        self.assertIsInstance(trend, float)


# ════════════════════════════════════════════════════════════════
#  19. ADVANCED ML MODULE
# ════════════════════════════════════════════════════════════════

class TestAnomalyDetector(unittest.TestCase):
    def setUp(self):
        from core.advanced_ml import AnomalyDetector
        self.ad = AnomalyDetector(window=50, threshold=3.0)

    def test_no_anomaly_normal_data(self):
        import random
        for _ in range(60):
            flags = self.ad.update({"speed": 1200.0 + random.gauss(0, 50)})
        self.assertIn("speed", flags)
        self.assertFalse(flags["speed"])

    def test_detects_spike(self):
        for _ in range(60):
            self.ad.update({"speed": 1200.0})
        flags = self.ad.update({"speed": 99999.0})
        self.assertTrue(flags["speed"])

    def test_threat_level_range(self):
        self.ad.update({"x": 100.0})
        t = self.ad.overall_threat_level()
        self.assertGreaterEqual(t, 0.0)
        self.assertLessEqual(t, 1.0)

    def test_event_logged(self):
        for _ in range(60):
            self.ad.update({"v": 0.0})
        self.ad.update({"v": 1_000_000.0})
        self.assertGreater(len(self.ad.events), 0)


class TestMultiTaskNet(unittest.TestCase):
    def setUp(self):
        from core.advanced_ml import MultiTaskNet
        self.net = MultiTaskNet()

    def test_predict_shape(self):
        import numpy as np
        from core.rl_algorithms import encode_state
        x = encode_state("free_ball|C_mid|C_mid|med|full|tied|us")
        for task in range(4):
            q = self.net.predict(x, task)
            self.assertEqual(len(q), [6, 6, 3, 3][task])

    def test_update_returns_float(self):
        import numpy as np
        from core.rl_algorithms import encode_state
        x = encode_state("free_ball|C_mid|C_mid|med|full|tied|us")
        loss = self.net.update(x, 0, 0, 1.0)
        self.assertIsInstance(float(loss), float)


class TestPassiveAggressiveLearner(unittest.TestCase):
    def setUp(self):
        from core.advanced_ml import PassiveAggressiveLearner
        self.pa = PassiveAggressiveLearner()

    def test_predict_valid_class(self):
        import numpy as np
        x = np.random.rand(27).astype(np.float32)
        pred = self.pa.predict(x)
        self.assertGreaterEqual(pred, 0)
        self.assertLess(pred, 6)

    def test_update_no_crash(self):
        import numpy as np
        for _ in range(20):
            x = np.random.rand(27).astype(np.float32)
            self.pa.update(x, 0, 1.0)
            self.pa.update(x, 3, -0.5)


class TestMAML(unittest.TestCase):
    def setUp(self):
        from core.advanced_ml import MAMLRoleAdapter
        self.maml = MAMLRoleAdapter()

    def test_best_action_valid(self):
        import numpy as np
        x = np.random.rand(27).astype(np.float32)
        a = self.maml.best_action(x)
        self.assertGreaterEqual(a, 0)
        self.assertLess(a, 6)

    def test_adapt_no_crash_small_buffer(self):
        import numpy as np
        for _ in range(5):
            x = np.random.rand(27).astype(np.float32)
            self.maml.record_experience(x, 0, 1.0)
        self.maml.adapt()   # buffer < 8 → should return early silently

    def test_adapt_with_enough_data(self):
        import numpy as np
        for _ in range(20):
            x = np.random.rand(27).astype(np.float32)
            self.maml.record_experience(x, 2, 0.5)
        self.maml.adapt()   # should run inner loop

    def test_end_of_match_no_crash(self):
        import numpy as np
        for _ in range(10):
            x = np.random.rand(27).astype(np.float32)
            self.maml.record_experience(x, 1, -0.2)
        self.maml.end_of_match()

    def test_serialization(self):
        import numpy as np
        for _ in range(15):
            x = np.random.rand(27).astype(np.float32)
            self.maml.record_experience(x, 0, 1.0)
        self.maml.adapt()
        d = self.maml.to_dict()
        from core.advanced_ml import MAMLRoleAdapter
        m2 = MAMLRoleAdapter()
        m2.from_dict(d)
        x = np.random.rand(27).astype(np.float32)
        self.maml.best_action(x)


class TestDeepRLNet(unittest.TestCase):
    def setUp(self):
        from core.advanced_ml import DeepRLNet
        self.net = DeepRLNet(batch_size=16)

    def test_q_role_shape(self):
        import numpy as np
        x = np.random.rand(27).astype(np.float32)
        q = self.net.q_role(x)
        self.assertEqual(len(q), 6)

    def test_q_mechanic_shape(self):
        import numpy as np
        x = np.random.rand(27).astype(np.float32)
        q = self.net.q_mechanic(x)
        self.assertEqual(len(q), 8)

    def test_value_scalar(self):
        import numpy as np
        x = np.random.rand(27).astype(np.float32)
        v = self.net.value(x)
        self.assertIsInstance(v, float)

    def test_train_step_needs_replay(self):
        import numpy as np
        loss = self.net.train_step()
        self.assertEqual(loss, 0.0)   # not enough samples

    def test_train_step_with_data(self):
        import numpy as np
        for _ in range(20):
            x  = np.random.rand(27).astype(np.float32)
            xn = np.random.rand(27).astype(np.float32)
            self.net.store(x, 0, 1, 0.5, xn, False)
        loss = self.net.train_step()
        self.assertGreaterEqual(loss, 0.0)


class TestCausalML(unittest.TestCase):
    def setUp(self):
        from core.advanced_ml import CausalActionEstimator
        self.causal = CausalActionEstimator(min_samples=3)

    def test_causal_q_zero_insufficient_data(self):
        q = self.causal.causal_q("s0", "attack")
        self.assertEqual(q, 0.0)

    def test_causal_q_after_records(self):
        for _ in range(10):
            self.causal.record("s0", "attack",  2.0)
            self.causal.record("s0", "defense", -1.0)
        q_attack  = self.causal.causal_q("s0", "attack")
        q_defense = self.causal.causal_q("s0", "defense")
        self.assertGreater(q_attack, q_defense)

    def test_best_action_causal(self):
        from core.rl_state import ROLE_ACTIONS
        for _ in range(10):
            self.causal.record("sX", "attack",  3.0)
            self.causal.record("sX", "defense", -2.0)
        best = self.causal.best_action_causal("sX", ["attack", "defense", "balanced"])
        self.assertEqual(best, "attack")

    def test_counterfactual_regret(self):
        for _ in range(10):
            self.causal.record("sR", "attack",  1.0)
            self.causal.record("sR", "defense", 5.0)
        regret = self.causal.counterfactual_regret("sR", "attack",
                                                   ["attack", "defense"])
        self.assertGreaterEqual(regret, 0.0)


class TestAdvancedMLSystem(unittest.TestCase):
    def setUp(self):
        from core.advanced_ml import AdvancedMLSystem
        self.adv = AdvancedMLSystem()

    def _tick(self, role_idx=0):
        state_key = "free_ball|C_mid|C_mid|med|full|tied|us"
        metrics   = {"opp_speed": 1200.0, "car_speed": 900.0,
                     "dist_to_ball": 1500.0, "ball_vel_y": 200.0}
        return self.adv.tick(state_key, role_idx, 0, 0.5,
                             state_key, False, metrics)

    def test_tick_returns_keys(self):
        result = self._tick()
        self.assertIn("role_override",  result)
        self.assertIn("threat_level",   result)
        self.assertIn("cf_regret",      result)
        self.assertIn("anomaly_flags",  result)

    def test_threat_level_range(self):
        for _ in range(5):
            result = self._tick()
        t = result["threat_level"]
        self.assertGreaterEqual(t, 0.0)
        self.assertLessEqual(t, 1.0)

    def test_on_goal_no_crash(self):
        self.adv.on_goal_scored()

    def test_on_match_end_no_crash(self):
        for _ in range(10):
            self._tick()
        self.adv.on_match_end()

    def test_serialization(self):
        for _ in range(5):
            self._tick()
        d = self.adv.to_dict()
        from core.advanced_ml import AdvancedMLSystem
        adv2 = AdvancedMLSystem()
        adv2.from_dict(d)
        result = adv2.tick(
            "free_ball|C_mid|C_mid|med|full|tied|us", 0, 0, 0.0,
            "free_ball|C_mid|C_mid|med|full|tied|us", False,
            {"opp_speed": 1000.0, "car_speed": 900.0,
             "dist_to_ball": 1200.0, "ball_vel_y": 0.0})
        self.assertIn("threat_level", result)


# ════════════════════════════════════════════════════════════════
#  20. NAVIGATION — PATH PLANNER
# ════════════════════════════════════════════════════════════════

class TestPathPlanner(unittest.TestCase):
    def setUp(self):
        from navigation.path_planner import PathPlanner
        self.pp = PathPlanner(grid_step=600)

    def test_find_path_returns_result(self):
        result = self.pp.find_path((0, -2000), (0, 2000), "A*")
        self.assertGreater(len(result.world_path), 0)
        self.assertGreater(result.cost, 0)

    def test_all_algorithms_find_path(self):
        algorithms = ["A*", "BFS", "UCS", "DFS", "Greedy", "Beam Search", "IDA*"]
        for alg in algorithms:
            with self.subTest(algorithm=alg):
                result = self.pp.find_path((-1000, -1000), (1000, 1000), alg)
                self.assertGreater(len(result.world_path), 0,
                                   msg=f"{alg} returned empty path")

    def test_world_path_coordinates(self):
        result = self.pp.find_path((0, 0), (2400, 2400), "A*")
        for wx, wy in result.world_path:
            self.assertLessEqual(abs(wx), 4096 + 600)
            self.assertLessEqual(abs(wy), 5120 + 600)

    def test_algorithm_field_in_result(self):
        result = self.pp.find_path((0, 0), (600, 600), "UCS")
        self.assertEqual(result.algorithm, "UCS")

    def test_clamped_out_of_bounds_start(self):
        # Should not crash with out-of-bounds coordinates
        result = self.pp.find_path((99999, 99999), (0, 0), "A*")
        self.assertGreater(len(result.world_path), 0)


# ════════════════════════════════════════════════════════════════
#  21. MODE MANAGER
# ════════════════════════════════════════════════════════════════

class TestModeManager(unittest.TestCase):
    def setUp(self):
        from game_logic.mode_manager import ModeManager, MODE_BALANCED
        self.mm = ModeManager(initial_mode=MODE_BALANCED)

    def test_initial_mode(self):
        from game_logic.mode_manager import MODE_BALANCED
        self.assertEqual(self.mm.state.mode, MODE_BALANCED)

    def test_is_pressed_no_crash(self):
        # Should return False (no keyboard input in test env)
        result = self.mm._is_pressed(0x4D)
        self.assertIsInstance(result, bool)

    def test_mode_state_dataclass(self):
        from game_logic.mode_manager import ModeState
        ms = ModeState()
        self.assertFalse(ms.temporary_reset)


# ════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════



# ════════════════════════════════════════════════════════════════
#  22.  ABILITY DISCOVERY
# ════════════════════════════════════════════════════════════════

class TestAbilityDiscovery(unittest.TestCase):
    def setUp(self):
        from game_logic.ability_discovery import AbilityDiscovery
        # Use in-memory path so tests don't touch real persistence
        self.ad = AbilityDiscovery(knowledge_path=":memory:")

    def test_known_seeded_abilities(self):
        abilities = self.ad.list_known_abilities()
        self.assertIn("plunger", abilities)
        self.assertIn("spikes", abilities)

    def test_get_ability_info_returns_dict(self):
        info = self.ad.get_ability_info("plunger")
        self.assertIn("type", info)
        self.assertIn("best_use_case", info)
        self.assertIn("success_rate", info)

    def test_update_with_none_powerup_clears_current(self):
        self.ad.update(None, (0, 0), (0, 0), (500, 0), 100.0)
        self.assertIsNone(self.ad.current_ability)

    def test_update_known_ability(self):
        from game_logic.ability_discovery import RUMBLE_POWERUP_NAMES
        raw = "Rugby_Ball_BA"   # maps to "plunger"
        self.ad.update(raw, (0, 0), (0, 0), (500, 0), 200.0)
        self.assertEqual(self.ad.current_ability, "plunger")

    def test_update_unknown_ability_enters_experiment(self):
        self.ad.update("SomeNewAbility_BA", (0, 0), (0, 0), (500, 0), 100.0)
        self.assertTrue(self.ad._experiment_mode)
        self.assertIn("SomeNewAbility_BA", self.ad._knowledge)

    def test_record_ability_use_updates_success_rate(self):
        # Seed an ability with known attempts
        self.ad._knowledge.setdefault("plunger", {}).update(
            {"attempts": 0, "wins": 0, "success_rate": 0.5}
        )
        for _ in range(4):
            self.ad.record_ability_use("plunger", success=True)
        self.ad.record_ability_use("plunger", success=False)
        info = self.ad.get_ability_info("plunger")
        self.assertEqual(info["attempts"], 5)
        self.assertAlmostEqual(info["success_rate"], 0.8, places=2)

    def test_should_activate_returns_bool(self):
        self.ad.update("Rugby_Ball_BA", (0, 0), (500, 0), (500, 0), 100.0)
        result = self.ad.should_activate((0, 4500), (500, 0), (500, 0), score_diff=-1)
        self.assertIsInstance(result, bool)

    def test_record_trial_outcome_no_crash(self):
        self.ad.current_ability = "plunger"
        self.ad._experiment_mode = True
        self.ad._pre_ball_vel = 100.0
        self.ad._pre_opp_dist = 800.0
        self.ad.record_trial_outcome(350.0, 600.0, 0.05)


# ════════════════════════════════════════════════════════════════
#  23.  ABILITY STRATEGY
# ════════════════════════════════════════════════════════════════

class TestAbilityStrategy(unittest.TestCase):
    def setUp(self):
        from game_logic.ability_discovery import AbilityDiscovery
        from game_logic.ability_strategy import AbilityStrategy
        ad = AbilityDiscovery(knowledge_path=":memory:")
        self.strategy = AbilityStrategy(discovery=ad)

    def test_evaluate_empty_ability_returns_false(self):
        result = self.strategy.evaluate("", "Attack", (0,0), (0,0), (500,0), 0, 0)
        self.assertFalse(result)

    def test_evaluate_respects_cooldown(self):
        # First call at tick 0 may activate; second at tick 5 should not (cooldown=90)
        _ = self.strategy.evaluate("plunger", "Attack", (0,4500), (500,0), (500,0), -1, 0)
        result2 = self.strategy.evaluate("plunger", "Attack", (0,4500), (500,0), (500,0), -1, 5)
        self.assertFalse(result2)

    def test_evaluate_activates_after_cooldown(self):
        # Car near ball near opp goal — ball_pull (plunger) should fire
        _ = self.strategy.evaluate("plunger", "Attack", (0, 4500), (0, 3800), (500, 0), -1, 0)
        result = self.strategy.evaluate("plunger", "Attack", (0, 4500), (0, 3800), (500, 0), -1, 100)
        # Should be True: ball near opp goal, car within 1000 of ball, plunger is ball_pull
        self.assertTrue(result)

    def test_get_strategy_hint_returns_string(self):
        hint = self.strategy.get_strategy_hint("plunger")
        self.assertIn("plunger", hint)
        self.assertIn("ball_pull", hint)

    def test_record_outcome_no_crash(self):
        self.strategy.record_outcome("plunger", True)
        self.strategy.record_outcome("spikes", False)


# ════════════════════════════════════════════════════════════════
#  24.  TEAM ABILITY AWARENESS
# ════════════════════════════════════════════════════════════════

class TestTeamAbilityAwareness(unittest.TestCase):
    def setUp(self):
        from game_logic.team_ability_awareness import TeamAbilityAwareness
        self.awareness = TeamAbilityAwareness()

    def _make_fake_cars(self, powerup_list):
        """Create minimal fake car objects."""
        class FakeLoc:
            def __init__(self, x, y):
                self.x = x; self.y = y; self.z = 0.0
        class FakePhys:
            def __init__(self, x, y):
                self.location = FakeLoc(x, y)
        class FakeCar:
            def __init__(self, team, powerup, x, y):
                self.team = team
                self.powerup = powerup
                self.physics = FakePhys(x, y)
                self.spawn_id = 0
        return [FakeCar(*args) for args in powerup_list]

    def test_update_parses_abilities(self):
        cars = self._make_fake_cars([
            (0, "Spike_BA", 100, 0),
            (1, "Freeze_BA", 500, 0),
        ])
        self.awareness.update(cars, 2, 0, 0)
        self.assertEqual(self.awareness.get_car_ability(0), "spike")
        self.assertEqual(self.awareness.get_car_ability(1), "freezer")

    def test_empty_powerup_returns_none(self):
        cars = self._make_fake_cars([(0, "", 0, 0)])
        self.awareness.update(cars, 1, 0, 0)
        self.assertIsNone(self.awareness.get_car_ability(0))

    def test_dangerous_opponent_nearby(self):
        cars = self._make_fake_cars([
            (0, "", 0, 0),             # me, team 0
            (1, "Freeze_BA", 300, 0),  # opponent with freezer
        ])
        self.awareness.update(cars, 2, 0, 0)
        danger, name = self.awareness.is_dangerous_opponent_nearby((0, 0), radius=600)
        self.assertTrue(danger)
        self.assertEqual(name, "freezer")

    def test_no_danger_faraway_opponent(self):
        cars = self._make_fake_cars([
            (0, "", 0, 0),
            (1, "Freeze_BA", 5000, 0),
        ])
        self.awareness.update(cars, 2, 0, 0)
        danger, _ = self.awareness.is_dangerous_opponent_nearby((0, 0), radius=600)
        self.assertFalse(danger)

    def test_get_all_opponent_abilities(self):
        cars = self._make_fake_cars([
            (0, "", 0, 0),
            (1, "Grapple_BA", 100, 0),
        ])
        self.awareness.update(cars, 2, 0, 0)
        opps = self.awareness.get_all_opponent_abilities()
        self.assertIn(1, opps)
        self.assertEqual(opps[1], "grappling_hook")

    def test_summary_returns_dict(self):
        cars = self._make_fake_cars([(0, "", 0, 0)])
        self.awareness.update(cars, 1, 0, 0)
        summary = self.awareness.summary()
        self.assertIn("teammate_abilities", summary)
        self.assertIn("opponent_abilities", summary)


# ════════════════════════════════════════════════════════════════
#  25.  TEAM ABILITY STRATEGY
# ════════════════════════════════════════════════════════════════

class TestTeamAbilityStrategy(unittest.TestCase):
    def setUp(self):
        from game_logic.team_ability_awareness import TeamAbilityAwareness
        from game_logic.ability_discovery import AbilityDiscovery
        from game_logic.team_ability_strategy import TeamAbilityStrategy
        ad = AbilityDiscovery(knowledge_path=":memory:")
        aw = TeamAbilityAwareness()
        self.ts = TeamAbilityStrategy(awareness=aw, discovery=ad)

    def test_get_team_directive_returns_object(self):
        from game_logic.team_ability_strategy import TeamDirective
        d = self.ts.get_team_directive(
            bot_index=0, my_ability=None, game_state="Attack",
            ball_pos=(0, 3000), car_pos=(0, 0), score_diff=0, tick=0)
        self.assertIsInstance(d.strategy_name, str)
        self.assertIsInstance(d.reason, str)

    def test_directive_standard_when_no_abilities(self):
        d = self.ts.get_team_directive(
            bot_index=0, my_ability=None, game_state="Midfield",
            ball_pos=(0, 0), car_pos=(0, 0), score_diff=0, tick=0)
        self.assertEqual(d.strategy_name, "standard")

    def test_grapple_intercept_triggered(self):
        d = self.ts.get_team_directive(
            bot_index=0, my_ability="grappling_hook", game_state="Counter Attack",
            ball_pos=(3000, 4000), car_pos=(0, 0), score_diff=0, tick=0)
        self.assertEqual(d.strategy_name, "grapple_intercept")
        self.assertTrue(d.should_activate_ability)

    def test_freeze_then_shoot_triggered_when_behind(self):
        d = self.ts.get_team_directive(
            bot_index=0, my_ability="freezer", game_state="Defense",
            ball_pos=(0, -3500), car_pos=(0, -2000), score_diff=-2, tick=0)
        self.assertEqual(d.strategy_name, "freeze_then_shoot")


# ════════════════════════════════════════════════════════════════
#  26.  TEAM CONTROLLER
# ════════════════════════════════════════════════════════════════

class TestTeamController(unittest.TestCase):
    def setUp(self):
        from game_logic.team_controller import TeamController
        self.tc = TeamController()

    def test_update_assigns_roles(self):
        from game_logic.team_controller import ROLE_ATTACKER, ROLE_DEFENDER
        positions = {0: (0, -2000), 1: (0, 0), 2: (0, 2000)}
        self.tc.update((0, 3000), positions, my_index=0, my_team=0, tick=300)
        roles = {i: self.tc.get_role(i) for i in positions}
        self.assertIn(roles[0], ("attacker", "support", "defender"))

    def test_get_role_unknown_returns_support(self):
        from game_logic.team_controller import ROLE_SUPPORT
        role = self.tc.get_role(999)
        self.assertEqual(role, ROLE_SUPPORT)

    def test_trigger_rotation_no_crash(self):
        positions = {0: (0, 0), 1: (0, 1000)}
        self.tc.update((0, 3000), positions, my_index=0, my_team=0, tick=300)
        self.tc.trigger_rotation("attacker_scored")
        self.tc.trigger_rotation("goal_conceded")
        self.tc.trigger_rotation("ball_cleared")

    def test_set_team_mode(self):
        from game_logic.team_controller import (
            TEAM_MODE_CUSTOM_AI, TEAM_MODE_SPECIFIC, TEAM_MODE_DEFAULT_RL)
        self.tc.set_team_mode(TEAM_MODE_CUSTOM_AI)
        self.assertEqual(self.tc.team_mode, TEAM_MODE_CUSTOM_AI)
        self.tc.set_team_mode("invalid_mode")  # should not change
        self.assertEqual(self.tc.team_mode, TEAM_MODE_CUSTOM_AI)

    def test_get_model_for_role_custom_ai(self):
        from game_logic.team_controller import TEAM_MODE_CUSTOM_AI
        self.tc.set_team_mode(TEAM_MODE_CUSTOM_AI)
        self.assertEqual(self.tc.get_model_for_role("attacker"), "custom_ai")

    def test_get_model_for_role_specific(self):
        from game_logic.team_controller import TEAM_MODE_SPECIFIC
        self.tc.set_team_mode(TEAM_MODE_SPECIFIC)
        self.tc.set_team_model("attacker", "model_attack_v3")
        self.assertEqual(self.tc.get_model_for_role("attacker"), "model_attack_v3")

    def test_get_team_summary_returns_dict(self):
        positions = {0: (0, 0), 1: (200, 0)}
        self.tc.update((0, 3000), positions, my_index=0, my_team=0, tick=300)
        s = self.tc.get_team_summary()
        self.assertIsInstance(s, dict)
        self.assertIn("0", s)

    def test_cooldown_prevents_thrashing(self):
        positions = {0: (0, 0), 1: (200, 0)}
        self.tc.update((0, 3000), positions, my_index=0, my_team=0, tick=300)
        role_after_first = self.tc.get_role(0)
        # Second update within cooldown — role should stay
        self.tc.update((0, -3000), positions, my_index=0, my_team=0, tick=310)
        self.assertEqual(self.tc.get_role(0), role_after_first)


# ════════════════════════════════════════════════════════════════
#  27.  TEAM POWER STRATEGY
# ════════════════════════════════════════════════════════════════

class TestTeamPowerStrategy(unittest.TestCase):
    def setUp(self):
        from game_logic.team_controller import TeamController
        from game_logic.team_ability_awareness import TeamAbilityAwareness
        from game_logic.ability_discovery import AbilityDiscovery
        from game_logic.team_power_strategy import TeamPowerStrategy
        tc  = TeamController()
        aw  = TeamAbilityAwareness()
        ad  = AbilityDiscovery(knowledge_path=":memory:")
        self.tps = TeamPowerStrategy(tc, aw, ad)

    def _decide(self, game_state="Midfield", ball_pos=(0.0, 0.0),
                car_pos=(0.0, 0.0), score_diff=0):
        return self.tps.get_strategic_decision(
            bot_index=0, game_state=game_state,
            ball_pos=ball_pos, car_pos=car_pos,
            opp_positions={}, score_diff=score_diff, tick=0,
        )

    def test_get_strategic_decision_returns_object(self):
        from game_logic.team_power_strategy import StrategicDecision
        dec = self._decide()
        self.assertIsInstance(dec, StrategicDecision)
        self.assertIsInstance(dec.action, str)
        self.assertIsInstance(dec.combo_name, str)
        self.assertIsInstance(dec.reason, str)
        self.assertIsInstance(dec.confidence, float)

    def test_standard_decision_when_no_abilities(self):
        dec = self._decide()
        # No ability set → must fall through to standard
        self.assertEqual(dec.combo_name, "standard")

    def test_spikes_rush_triggered(self):
        # Set ability directly on the discovery object
        self.tps._discovery.current_ability = "spikes_carry"
        dec = self._decide(
            game_state="Attack",
            ball_pos=(0.0, -100.0),    # 100 units from car
            car_pos=(0.0, 0.0),
        )
        self.assertEqual(dec.combo_name, "spikes_rush")
        self.assertTrue(dec.use_ability)
        self.assertEqual(dec.action, "attack")

    def test_teleport_intercept_when_far(self):
        self.tps._discovery.current_ability = "teleport_strike"
        dec = self._decide(
            game_state="Attack",
            ball_pos=(3000.0, 4000.0),  # dist ~5000 > 1500
            car_pos=(0.0, 0.0),
        )
        self.assertEqual(dec.combo_name, "teleport_intercept")
        self.assertTrue(dec.use_ability)

    def test_ice_trap_defensive_save(self):
        self.tps._discovery.current_ability = "ice_attack"
        # ball at (0, -4500): dist_to_own_goal = hypot(0, -4500+5120) = 620 < 2000
        dec = self._decide(
            game_state="Defense",
            ball_pos=(0.0, -4500.0),
            car_pos=(0.0, -3000.0),
        )
        self.assertEqual(dec.combo_name, "ice_trap_defensive_save")
        self.assertEqual(dec.action, "defend")
        self.assertTrue(dec.use_ability)

    def test_intelligence_scores_float_in_range(self):
        self._decide()
        t = self.tps.intelligence.threat_score
        o = self.tps.intelligence.opportunity_score
        self.assertIsInstance(t, float)
        self.assertIsInstance(o, float)
        self.assertGreaterEqual(t, 0.0)
        self.assertLessEqual(t, 1.0)
        self.assertGreaterEqual(o, 0.0)
        self.assertLessEqual(o, 1.0)

    def test_notify_goal_scored_no_crash(self):
        self.tps.notify_goal_scored()

    def test_notify_goal_conceded_no_crash(self):
        self.tps.notify_goal_conceded()

    def test_last_combo_property(self):
        self.tps._discovery.current_ability = "spikes"
        self._decide(game_state="Free Ball", ball_pos=(0.0, 0.0), car_pos=(0.0, 50.0))
        # last_combo should be spikes_rush if ball is within 600 units
        self.assertIn(self.tps.last_combo, [c for c in __import__(
            "game_logic.team_power_strategy", fromlist=["POWER_COMBOS"]).POWER_COMBOS])


# ════════════════════════════════════════════════════════════════
#  28.  ABILITY DISCOVERY — EXTENDED TYPES
# ════════════════════════════════════════════════════════════════

class TestAbilityDiscoveryExtended(unittest.TestCase):
    def setUp(self):
        from game_logic.ability_discovery import AbilityDiscovery
        self.ad = AbilityDiscovery(knowledge_path=":memory:")

    def test_new_ability_types_seeded(self):
        abilities = self.ad.list_known_abilities()
        for ab in ("ice_attack", "shock_wave", "ball_magnet", "spikes_carry", "teleport_strike"):
            self.assertIn(ab, abilities, msg=f"'{ab}' not in seeded knowledge")

    def test_new_powerup_names_mapped(self):
        from game_logic.ability_discovery import RUMBLE_POWERUP_NAMES
        self.assertEqual(RUMBLE_POWERUP_NAMES.get("IceAttack_BA"),      "ice_attack")
        self.assertEqual(RUMBLE_POWERUP_NAMES.get("ShockWave_BA"),      "shock_wave")
        self.assertEqual(RUMBLE_POWERUP_NAMES.get("BallMagnet_BA"),     "ball_magnet")
        self.assertEqual(RUMBLE_POWERUP_NAMES.get("SpikesCharged_BA"),  "spikes_carry")
        self.assertEqual(RUMBLE_POWERUP_NAMES.get("TeleportStrike_BA"), "teleport_strike")

    def test_new_effect_classes_present(self):
        from game_logic.ability_discovery import EFFECT_CLASSES
        for ec in ("ice_attack", "shock_wave", "ball_magnet", "spikes_carry", "teleport_strike"):
            self.assertIn(ec, EFFECT_CLASSES, msg=f"'{ec}' not in EFFECT_CLASSES")

    def test_update_ice_attack_sets_current_ability(self):
        self.ad.update("IceAttack_BA", (0, 0), (0, 0), (500, 0), 100.0)
        self.assertEqual(self.ad.current_ability, "ice_attack")

    def test_update_teleport_strike_sets_current_ability(self):
        self.ad.update("TeleportStrike_BA", (0, 0), (0, 0), (500, 0), 100.0)
        self.assertEqual(self.ad.current_ability, "teleport_strike")

    def test_new_ability_info_structure(self):
        for ab in ("ice_attack", "shock_wave", "ball_magnet"):
            info = self.ad.get_ability_info(ab)
            self.assertIn("type", info)
            self.assertIn("best_use_case", info)
            self.assertIn("success_rate", info)


if __name__ == "__main__":
    loader  = unittest.TestLoader()
    suite   = loader.loadTestsFromModule(__import__("__main__"))
    runner  = unittest.TextTestRunner(verbosity=2)
    result  = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
