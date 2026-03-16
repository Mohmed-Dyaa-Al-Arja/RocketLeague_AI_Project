"""
all_algorithms.py
=================
Unified collection of ALL search & AI algorithms used in the Rocket League AI Bot.
Required by university submission — all algorithms in one file.

Algorithms included:
  1.  A* (A-Star)               — Optimal heuristic-guided pathfinding
  2.  BFS                        — Breadth-First Search (shortest hops)
  3.  DFS                        — Depth-First Search
  4.  Greedy Best-First Search   — Heuristic-only, fast but suboptimal
  5.  UCS                        — Uniform-Cost Search (Dijkstra variant)
  6.  IDA*                       — Iterative-Deepening A* (low memory)
  7.  Beam Search                — Bounded-width best-first search
  8.  Decision Tree              — Tactical tree scoring & best-leaf selection
  9.  Ball Prediction            — Physics-based trajectory simulation
  10. Search Algorithms Router   — Unified run_search() dispatcher

Author: RocketLeague AI Project
Date  : 2026
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────
#  Standard imports
# ─────────────────────────────────────────────────────────────────
import heapq
import math
from collections import deque
from dataclasses import dataclass
from typing import (
    Callable, Dict, Iterable, List, Optional, Tuple
)

# ═════════════════════════════════════════════════════════════════
#  SHARED TYPES
# ═════════════════════════════════════════════════════════════════

Node = Tuple[int, int]   # (grid_x, grid_y) — used by all path algorithms


# ═════════════════════════════════════════════════════════════════
#  1. A* ALGORITHM
# ═════════════════════════════════════════════════════════════════

def astar_search(
    start: Node,
    goal: Node,
    neighbors_fn: Callable[[Node], Iterable[Node]],
    cost_fn: Callable[[Node, Node], float],
    heuristic_fn: Callable[[Node, Node], float],
) -> Tuple[List[Node], float]:
    """
    A* Search — finds the optimal path from *start* to *goal*.

    Uses a priority queue ordered by f = g + h where:
      g = cost from start to current node
      h = admissible heuristic estimate to goal

    Time  : O(b^d)  where b = branching factor, d = depth
    Space : O(b^d)
    Optimal: Yes (if heuristic is admissible)

    Returns (path, total_cost).  path is empty if no path exists.
    """
    frontier: List[Tuple[float, Node]] = []
    heapq.heappush(frontier, (0.0, start))

    came_from: Dict[Node, Optional[Node]] = {start: None}
    g_score: Dict[Node, float] = {start: 0.0}

    while frontier:
        f_val, current = heapq.heappop(frontier)
        if current == goal:
            break

        # Skip stale entries
        if f_val > g_score.get(current, 0.0) + heuristic_fn(current, goal) + 1e-9:
            continue

        for nxt in neighbors_fn(current):
            tentative_g = g_score[current] + cost_fn(current, nxt)
            if nxt not in g_score or tentative_g < g_score[nxt]:
                g_score[nxt] = tentative_g
                f_score = tentative_g + heuristic_fn(nxt, goal)
                heapq.heappush(frontier, (f_score, nxt))
                came_from[nxt] = current

    if goal not in came_from:
        return [], float("inf")

    path: List[Node] = []
    cur: Optional[Node] = goal
    while cur is not None:
        path.append(cur)
        cur = came_from[cur]
    path.reverse()
    return path, g_score.get(goal, float("inf"))


# ═════════════════════════════════════════════════════════════════
#  2. BFS — Breadth-First Search
# ═════════════════════════════════════════════════════════════════

def bfs_search(
    start: Node,
    goal: Node,
    neighbors_fn: Callable[[Node], Iterable[Node]],
    cost_fn: Callable[[Node, Node], float],
    heuristic_fn: Callable[[Node, Node], float],   # unused — kept for uniform interface
) -> Tuple[List[Node], float]:
    """
    BFS — explores nodes level by level (FIFO queue).

    Guarantees shortest path in terms of *hop count* (unweighted graph).
    Does NOT minimise total cost.

    Time  : O(V + E)
    Space : O(V)
    Optimal: Yes (for unweighted graphs; not for weighted)
    """
    queue: deque[Node] = deque([start])
    came_from: Dict[Node, Optional[Node]] = {start: None}

    while queue:
        current = queue.popleft()
        if current == goal:
            break

        for nxt in neighbors_fn(current):
            if nxt not in came_from:
                came_from[nxt] = current
                queue.append(nxt)

    if goal not in came_from:
        return [], float("inf")

    path: List[Node] = []
    cur: Optional[Node] = goal
    while cur is not None:
        path.append(cur)
        cur = came_from[cur]
    path.reverse()

    cost = sum(cost_fn(path[i - 1], path[i]) for i in range(1, len(path)))
    return path, cost


# ═════════════════════════════════════════════════════════════════
#  3. DFS — Depth-First Search
# ═════════════════════════════════════════════════════════════════

def dfs_search(
    start: Node,
    goal: Node,
    neighbors_fn: Callable[[Node], Iterable[Node]],
    cost_fn: Callable[[Node, Node], float],
    heuristic_fn: Callable[[Node, Node], float],   # unused
) -> Tuple[List[Node], float]:
    """
    DFS — explores one branch as deep as possible before backtracking (LIFO stack).

    Fast exploration but found path is NOT guaranteed optimal.
    Useful for existence checks, not cost minimisation.

    Time  : O(V + E)
    Space : O(V)
    Optimal: No
    """
    stack: List[Node] = [start]
    came_from: Dict[Node, Optional[Node]] = {start: None}

    while stack:
        current = stack.pop()
        if current == goal:
            break

        for nxt in neighbors_fn(current):
            if nxt not in came_from:
                came_from[nxt] = current
                stack.append(nxt)

    if goal not in came_from:
        return [], float("inf")

    path: List[Node] = []
    cur: Optional[Node] = goal
    while cur is not None:
        path.append(cur)
        cur = came_from[cur]
    path.reverse()

    cost = sum(cost_fn(path[i - 1], path[i]) for i in range(1, len(path)))
    return path, cost


# ═════════════════════════════════════════════════════════════════
#  4. GREEDY BEST-FIRST SEARCH
# ═════════════════════════════════════════════════════════════════

def greedy_search(
    start: Node,
    goal: Node,
    neighbors_fn: Callable[[Node], Iterable[Node]],
    cost_fn: Callable[[Node, Node], float],
    heuristic_fn: Callable[[Node, Node], float],
) -> Tuple[List[Node], float]:
    """
    Greedy Best-First Search — always expands the node closest to the goal
    according to the heuristic h(n), ignoring path cost so far.

    Fast but NOT optimal — may find a longer path than A*.

    Time  : O(b^m)  in the worst case
    Space : O(b^m)
    Optimal: No
    """
    frontier: List[Tuple[float, Node]] = [(heuristic_fn(start, goal), start)]
    came_from: Dict[Node, Optional[Node]] = {start: None}
    visited = {start}

    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal:
            break

        for nxt in neighbors_fn(current):
            if nxt in visited:
                continue
            visited.add(nxt)
            came_from[nxt] = current
            heapq.heappush(frontier, (heuristic_fn(nxt, goal), nxt))

    if goal not in came_from:
        return [], float("inf")

    path: List[Node] = []
    cur: Optional[Node] = goal
    while cur is not None:
        path.append(cur)
        cur = came_from[cur]
    path.reverse()

    cost = sum(cost_fn(path[i - 1], path[i]) for i in range(1, len(path)))
    return path, cost


# ═════════════════════════════════════════════════════════════════
#  5. UCS — Uniform-Cost Search
# ═════════════════════════════════════════════════════════════════

def ucs_search(
    start: Node,
    goal: Node,
    neighbors_fn: Callable[[Node], Iterable[Node]],
    cost_fn: Callable[[Node, Node], float],
    heuristic_fn: Callable[[Node, Node], float],   # unused
) -> Tuple[List[Node], float]:
    """
    UCS (Uniform-Cost Search / Dijkstra) — expands the cheapest-cost node first.

    Equivalent to A* with h(n)=0.  Guarantees optimal cost path.

    Time  : O((V + E) log V)
    Space : O(V)
    Optimal: Yes (for non-negative edge weights)
    """
    frontier: List[Tuple[float, Node]] = [(0.0, start)]
    came_from: Dict[Node, Optional[Node]] = {start: None}
    cost_so_far: Dict[Node, float] = {start: 0.0}

    while frontier:
        current_cost, current = heapq.heappop(frontier)
        if current == goal:
            break

        if current_cost > cost_so_far.get(current, float("inf")):
            continue

        for nxt in neighbors_fn(current):
            new_cost = cost_so_far[current] + cost_fn(current, nxt)
            if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                cost_so_far[nxt] = new_cost
                came_from[nxt] = current
                heapq.heappush(frontier, (new_cost, nxt))

    if goal not in came_from:
        return [], float("inf")

    path: List[Node] = []
    cur: Optional[Node] = goal
    while cur is not None:
        path.append(cur)
        cur = came_from[cur]
    path.reverse()
    return path, cost_so_far.get(goal, float("inf"))


# ═════════════════════════════════════════════════════════════════
#  6. IDA* — Iterative Deepening A*
# ═════════════════════════════════════════════════════════════════

_IDA_INF = float("inf")
_IDA_MAX_ITER = 80   # safeguard against infinite f-cost growth


def _ida_dfs(
    path_stack: List[Node],
    g_stack: List[float],
    goal: Node,
    threshold: float,
    neighbors_fn: Callable[[Node], Iterable[Node]],
    cost_fn: Callable[[Node, Node], float],
    heuristic_fn: Callable[[Node, Node], float],
) -> Tuple[Optional[List[Node]], float]:
    """Recursive DFS with f-cost threshold for IDA*."""
    node = path_stack[-1]
    g = g_stack[-1]
    f = g + heuristic_fn(node, goal)

    if f > threshold:
        return None, f

    if node == goal:
        return list(path_stack), f

    min_next = _IDA_INF
    visited = set(path_stack)

    for nxt in neighbors_fn(node):
        if nxt in visited:
            continue
        step = cost_fn(node, nxt)
        path_stack.append(nxt)
        g_stack.append(g + step)
        result, t = _ida_dfs(path_stack, g_stack, goal, threshold,
                              neighbors_fn, cost_fn, heuristic_fn)
        if result is not None:
            return result, t
        if t < min_next:
            min_next = t
        path_stack.pop()
        g_stack.pop()

    return None, min_next


def ida_star_search(
    start: Node,
    goal: Node,
    neighbors_fn: Callable[[Node], Iterable[Node]],
    cost_fn: Callable[[Node, Node], float],
    heuristic_fn: Callable[[Node, Node], float],
) -> Tuple[List[Node], float]:
    """
    IDA* — Iterative Deepening A*.

    Combines DFS memory efficiency (O(depth)) with A*'s optimal cost guidance.
    Each iteration increases the f-cost threshold to the smallest over-threshold
    value found in the previous pass.

    Time  : O(b^d)  (same as A*, but repeated expansions in practice)
    Space : O(d)    — only current path kept in memory
    Optimal: Yes (admissible heuristic required)
    """
    threshold = heuristic_fn(start, goal)
    path_stack: List[Node] = [start]
    g_stack: List[float] = [0.0]

    for _ in range(_IDA_MAX_ITER):
        result, new_threshold = _ida_dfs(path_stack, g_stack, goal, threshold,
                                         neighbors_fn, cost_fn, heuristic_fn)
        if result is not None:
            return result, g_stack[-1] if g_stack else 0.0
        if new_threshold == _IDA_INF:
            break
        threshold = new_threshold

    return [], _IDA_INF


# ═════════════════════════════════════════════════════════════════
#  7. BEAM SEARCH
# ═════════════════════════════════════════════════════════════════

BEAM_WIDTH = 32   # keep top-32 candidates per level


def beam_search(
    start: Node,
    goal: Node,
    neighbors_fn: Callable[[Node], Iterable[Node]],
    cost_fn: Callable[[Node, Node], float],
    heuristic_fn: Callable[[Node, Node], float],
    beam_width: int = BEAM_WIDTH,
) -> Tuple[List[Node], float]:
    """
    Beam Search — bounded-width best-first search.

    At each depth level only the top *beam_width* nodes (lowest f-score)
    are kept. Trades optimality for speed when the state space is large.

    Time  : O(beam_width * d)   much faster than full A* in practice
    Space : O(beam_width)
    Optimal: No — may miss optimal path if it falls outside the beam
    """
    came_from: Dict[Node, Optional[Node]] = {start: None}
    g_score: Dict[Node, float] = {start: 0.0}
    beam: List[Tuple[float, Node]] = [(heuristic_fn(start, goal), start)]

    for _ in range(500):
        if not beam:
            break

        for _, node in beam:
            if node == goal:
                path: List[Node] = []
                cur: Optional[Node] = goal
                while cur is not None:
                    path.append(cur)
                    cur = came_from[cur]
                path.reverse()
                return path, g_score.get(goal, float("inf"))

        candidates: List[Tuple[float, Node]] = []
        for _, current in beam:
            for nxt in neighbors_fn(current):
                tent_g = g_score.get(current, 0.0) + cost_fn(current, nxt)
                if nxt not in g_score or tent_g < g_score[nxt]:
                    g_score[nxt] = tent_g
                    came_from[nxt] = current
                    f = tent_g + heuristic_fn(nxt, goal)
                    candidates.append((f, nxt))

        if not candidates:
            break

        candidates.sort(key=lambda x: x[0])
        beam = candidates[:beam_width]

    if goal in came_from:
        path = []
        cur = goal
        while cur is not None:
            path.append(cur)
            cur = came_from[cur]
        path.reverse()
        return path, g_score.get(goal, float("inf"))

    return [], float("inf")


# ═════════════════════════════════════════════════════════════════
#  8. DECISION TREE
#     Tactical decision-making for Rocket League
# ═════════════════════════════════════════════════════════════════

@dataclass
class DTNode:
    """A node in the tactical decision tree."""
    name: str
    target: Tuple[float, float]
    score: float = 0.0
    children: Optional[List["DTNode"]] = None
    depth: int = 0


@dataclass
class DTResult:
    """Result returned by the decision tree."""
    best_action: str
    target: Tuple[float, float]
    score: float
    tree_depth: int
    nodes_evaluated: int
    all_actions: List[Dict]


# Major boost pad XY positions (Rocket League standard field)
_BOOST_PADS = [
    (-3072, -4096), (3072, -4096),
    (-3584, 0),     (3584, 0),
    (-3072,  4096), (3072,  4096),
]


def _dt_dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _nearest_boost(car_pos: Tuple[float, float]) -> Tuple[float, float]:
    return min(_BOOST_PADS, key=lambda p: _dt_dist(car_pos, p))


def build_decision_tree(
    car_pos: Tuple[float, float],
    ball_pos: Tuple[float, float],
    car_speed: float,
    my_team: int,
    knowledge_data: Optional[Dict] = None,
) -> DTNode:
    """
    Build a 2-level tactical decision tree for Rocket League.

    Branches:
      1. Push through ball → goal            (offensive)
      2. Get behind ball   → then push/center (positioning)
      3. Defend own goal   → intercept/shadow (defensive)
      4. Grab boost        → rush/rotate      (resource)

    Each leaf is scored; higher score = better action right now.
    Returns root DTNode with all children populated and scored.
    """
    opp_goal_y  = 5120.0  if my_team == 0 else -5120.0
    own_goal_y  = -5120.0 if my_team == 0 else  5120.0
    opp_goal    = (0.0, opp_goal_y)
    own_goal    = (0.0, own_goal_y)
    push_dir    = 1.0     if my_team == 0 else -1.0

    dist_to_ball       = _dt_dist(car_pos, ball_pos)
    dist_ball_own_goal = _dt_dist(ball_pos, own_goal)
    ball_on_our_side   = (ball_pos[1] * push_dir) < 0

    # Optional knowledge bonuses from RL learning history
    attack_bonus = defense_bonus = 0.0
    if knowledge_data:
        strats = knowledge_data.get("successful_strategies", {})
        attack_bonus  = min(10.0, float(strats.get("attack",  0)) * 0.2)
        defense_bonus = min(10.0, float(strats.get("defense", 0)) * 0.2)

    def _score(target: Tuple[float, float], role: str, urgency: float = 0.0) -> float:
        d_car  = _dt_dist(car_pos,  target)
        d_ball = _dt_dist(ball_pos, target)
        tbd    = _dt_dist(target,   opp_goal) - _dt_dist(ball_pos, opp_goal)
        s = 0.0
        s -= d_car  / 12000.0
        s -= max(0.0, d_ball - 900.0) / 9000.0
        s += urgency
        s += 0.05 * min(1.0, car_speed / 2300.0)
        if role == "attack":
            s -= max(0.0, tbd) / 6000.0
            if tbd < -250.0:
                s += 0.08
            s += 0.15 + attack_bonus / 100.0
        elif role == "defense":
            s += 0.10 + defense_bonus / 100.0
        return s

    # ── Branch 1: Drive through ball ──
    dx_g = opp_goal[0] - ball_pos[0]
    dy_g = opp_goal[1] - ball_pos[1]
    d_g  = max(1.0, math.hypot(dx_g, dy_g))
    push_target = (
        max(-4000.0, min(4000.0, ball_pos[0] + dx_g / d_g * 950.0)),
        max(-5100.0, min(5100.0, ball_pos[1] + dy_g / d_g * 950.0)),
    )
    clear_x      = (3000.0 if ball_pos[0] < 0 else -3000.0)
    clear_target = (clear_x, ball_pos[1] * 0.5)
    chase_urgency = max(0.0, 0.5 - dist_to_ball / 6000.0)

    chase_push  = DTNode("Push to goal",   push_target,  _score(push_target,  "attack",  chase_urgency + 0.1), depth=2)
    chase_clear = DTNode("Clear to side",  clear_target, _score(clear_target, "defense", chase_urgency),       depth=2)
    chase_ball  = DTNode("Push through ball", push_target, children=[chase_push, chase_clear], depth=1)
    chase_ball.score = max(chase_push.score, chase_clear.score)

    # ── Branch 2: Get behind ball ──
    behind_y    = ball_pos[1] - 500.0 * push_dir
    behind_pos  = (ball_pos[0], behind_y)
    then_push   = (ball_pos[0] * 0.25,
                   max(-5100.0, min(5100.0, ball_pos[1] + push_dir * 1400.0)))
    center_tgt  = (0.0, ball_pos[1] * 0.5 + opp_goal_y * 0.5)

    behind_push   = DTNode("Push to goal", then_push,  _score(then_push,  "attack", 0.10), depth=2)
    behind_center = DTNode("Center ball",  center_tgt, _score(center_tgt, "attack", 0.05), depth=2)
    get_behind    = DTNode("Get behind ball", behind_pos, children=[behind_push, behind_center], depth=1)
    get_behind.score = max(behind_push.score, behind_center.score)

    # ── Branch 3: Defend ──
    intercept_pos = (ball_pos[0] * 0.5, own_goal_y * 0.65)
    shadow_pos    = (0.0, own_goal_y * 0.8)
    defend_urgency = 0.0
    if ball_on_our_side:
        defend_urgency = max(0.0, 0.6 - dist_ball_own_goal / 5120.0)

    defend_int    = DTNode("Intercept ball",   intercept_pos, _score(intercept_pos, "defense", defend_urgency),       depth=2)
    defend_shad   = DTNode("Shadow position",  shadow_pos,    _score(shadow_pos,    "defense", defend_urgency * 0.7), depth=2)
    defend_goal   = DTNode("Defend goal", (0.0, own_goal_y * 0.7), children=[defend_int, defend_shad], depth=1)
    defend_goal.score = max(defend_int.score, defend_shad.score)

    # ── Branch 4: Boost ──
    boost_pos    = _nearest_boost(car_pos)
    boost_rush   = DTNode("Rush ball",    ball_pos,               _score(ball_pos,               "attack",  -0.05), depth=2)
    boost_rotate = DTNode("Rotate back",  (0.0, own_goal_y * 0.5), _score((0.0, own_goal_y * 0.5), "defense", -0.10), depth=2)
    grab_boost   = DTNode("Grab boost", boost_pos, children=[boost_rush, boost_rotate], depth=1)
    grab_boost.score = max(boost_rush.score, boost_rotate.score)

    root = DTNode("Root", car_pos,
                  children=[chase_ball, get_behind, defend_goal, grab_boost], depth=0)
    root.score = max(c.score for c in root.children)
    return root


def evaluate_decision_tree(root: DTNode) -> DTResult:
    """
    Walk the tree depth-first and return the best leaf node decision.
    """
    best_leaf: Optional[DTNode] = None
    best_score = float("-inf")
    nodes_evaluated = 0
    max_depth = 0
    all_actions: List[Dict] = []

    stack: List[DTNode] = [root]
    while stack:
        node = stack.pop()
        nodes_evaluated += 1
        max_depth = max(max_depth, node.depth)

        if not node.children:
            all_actions.append({"name": node.name, "score": node.score})
            if node.score > best_score:
                best_score = node.score
                best_leaf = node
        else:
            for child in node.children:
                stack.append(child)

    if best_leaf is None:
        best_leaf = root

    all_actions.sort(key=lambda x: x["score"], reverse=True)
    return DTResult(
        best_action=best_leaf.name,
        target=best_leaf.target,
        score=best_score,
        tree_depth=max_depth,
        nodes_evaluated=nodes_evaluated,
        all_actions=all_actions,
    )


# ═════════════════════════════════════════════════════════════════
#  9. BALL PREDICTION (Physics Simulation)
# ═════════════════════════════════════════════════════════════════

# ── Rocket League field constants ──
BALL_RADIUS     = 92.75
WALL_X          = 4096.0
WALL_Y          = 5120.0
CEILING_Z       = 2044.0
GROUND_Z        = BALL_RADIUS
GOAL_HALF_WIDTH = 893.0
GRAVITY         = -650.0
RESTITUTION     = 0.6
DRAG_PER_SEC    = 0.0305

BallState = Tuple[float, float, float, float, float, float, float]  # (t, x, y, z, vx, vy, vz)


def predict_ball_trajectory(
    ball_x: float, ball_y: float, ball_z: float,
    vel_x: float,  vel_y: float,  vel_z: float,
    total_time: float = 4.0,
    dt: float = 1 / 30,
    gravity: float = GRAVITY,
    restitution: float = RESTITUTION,
    puck_mode: bool = False,
) -> List[BallState]:
    """
    Simulate the ball's physics trajectory over *total_time* seconds.

    Models:
      - Gravity (downward acceleration)
      - Air drag (velocity multiplied by (1 - drag * dt) each step)
      - Wall bounces (x walls, y end-walls, floor, ceiling)
      - Goal opening logic (no y-wall bounce inside goal mouth)
      - Puck mode: heavier, less bouncy (Snow Day variant)

    Returns list of (t, x, y, z, vx, vy, vz) states at 30 Hz.
    """
    trajectory: List[BallState] = []
    x, y, z = ball_x, ball_y, max(ball_z, GROUND_Z)
    vx, vy, vz = vel_x, vel_y, vel_z

    eff_restitution = restitution * (0.4 if puck_mode else 1.0)
    eff_drag        = DRAG_PER_SEC * (1.3 if puck_mode else 1.0)

    for i in range(1, int(total_time / dt) + 1):
        drag = 1.0 - eff_drag * dt
        vx  *= drag;  vy *= drag;  vz *= drag
        vz  += gravity * dt
        x   += vx * dt;  y += vy * dt;  z += vz * dt

        # X-wall bounces
        if x < -WALL_X + BALL_RADIUS:
            x = -WALL_X + BALL_RADIUS;  vx = abs(vx) * eff_restitution
        elif x > WALL_X - BALL_RADIUS:
            x = WALL_X - BALL_RADIUS;   vx = -abs(vx) * eff_restitution

        # Y end-wall bounces (skip if inside goal opening)
        in_goal = abs(x) < GOAL_HALF_WIDTH and z < 643.0
        if y < -WALL_Y + BALL_RADIUS and not in_goal:
            y = -WALL_Y + BALL_RADIUS;  vy = abs(vy) * eff_restitution
        elif y > WALL_Y - BALL_RADIUS and not in_goal:
            y = WALL_Y - BALL_RADIUS;   vy = -abs(vy) * eff_restitution

        # Floor bounce
        if z < GROUND_Z:
            z = GROUND_Z;  vz = abs(vz) * eff_restitution
            if abs(vz) < 20.0:
                vz = 0.0

        # Ceiling bounce
        if z > CEILING_Z - BALL_RADIUS:
            z = CEILING_Z - BALL_RADIUS;  vz = -abs(vz) * RESTITUTION

        trajectory.append((i * dt, x, y, z, vx, vy, vz))

    return trajectory


def find_ball_intercept(
    trajectory: List[BallState],
    car_x: float,
    car_y: float,
    car_speed: float,
    car_boost: float,
    max_z: float = 400.0,
) -> Optional[Tuple[float, float, float, float]]:
    """
    Return the earliest (x, y, z, time) point in *trajectory* the car can reach.

    Compares drive time (distance / estimated speed) to ball arrival time.
    Returns None if ball is always too fast or too high.
    """
    avg_speed = max(800.0, car_speed)
    if car_boost > 40:
        avg_speed = min(2300.0, avg_speed + 500.0)
    elif car_boost > 15:
        avg_speed = min(2300.0, avg_speed + 250.0)

    for t, bx, by, bz, *_ in trajectory:
        if bz > max_z:
            continue
        drive_dist = math.hypot(bx - car_x, by - car_y)
        if drive_dist / avg_speed <= t + 0.15:
            return (bx, by, bz, t)

    return None


def ball_landing_position(
    trajectory: List[BallState],
) -> Optional[Tuple[float, float, float]]:
    """Return (x, y, time) of the first frame the ball settles on the ground."""
    for t, x, y, z, _vx, _vy, vz in trajectory:
        if z <= GROUND_Z + 5.0 and vz <= 0:
            return (x, y, t)
    return None


# ═════════════════════════════════════════════════════════════════
#  10. SEARCH ALGORITHMS ROUTER
#      Unified dispatcher — pass an algorithm name and get a result
# ═════════════════════════════════════════════════════════════════

SearchFn = Callable[
    [Node, Node,
     Callable[[Node], Iterable[Node]],
     Callable[[Node, Node], float],
     Callable[[Node, Node], float]],
    Tuple[List[Node], float]
]

ALGORITHMS: Dict[str, SearchFn] = {
    "A*":            astar_search,
    "BFS":           bfs_search,
    "DFS":           dfs_search,
    "Greedy":        greedy_search,
    "UCS":           ucs_search,
    "IDA*":          ida_star_search,
    "Beam Search":   beam_search,
    "Decision Tree": astar_search,   # DT uses A* for grid pathfinding; see build_decision_tree() for tactics
}


def run_search(
    algorithm: str,
    start: Node,
    goal: Node,
    neighbors_fn: Callable[[Node], Iterable[Node]],
    cost_fn: Callable[[Node, Node], float],
    heuristic_fn: Callable[[Node, Node], float],
) -> Tuple[List[Node], float]:
    """
    Unified search dispatcher.

    Parameters
    ----------
    algorithm   : one of ALGORITHMS keys (case-sensitive)
    start/goal  : (grid_x, grid_y) tuples
    neighbors_fn: given a node, returns iterable of adjacent nodes
    cost_fn     : edge cost between two adjacent nodes
    heuristic_fn: admissible heuristic estimate to goal

    Returns (path, cost) — path is [] and cost is inf if no path found.
    """
    if algorithm not in ALGORITHMS:
        raise ValueError(
            f"Unknown algorithm '{algorithm}'. "
            f"Available: {list(ALGORITHMS.keys())}"
        )
    return ALGORITHMS[algorithm](start, goal, neighbors_fn, cost_fn, heuristic_fn)


# ─────────────────────────────────────────────────────────────────
#  Algorithm comparison utility (university demo helper)
# ─────────────────────────────────────────────────────────────────

def compare_all_algorithms(
    start: Node,
    goal: Node,
    neighbors_fn: Callable[[Node], Iterable[Node]],
    cost_fn: Callable[[Node, Node], float],
    heuristic_fn: Callable[[Node, Node], float],
) -> List[Dict]:
    """
    Run every search algorithm on the same problem and return a
    list of result dicts for comparison/benchmarking.

    Each dict: { "algorithm", "path_length", "cost", "found" }
    """
    import time
    results = []
    for name in ALGORITHMS:
        t0 = time.perf_counter()
        path, cost = run_search(name, start, goal, neighbors_fn, cost_fn, heuristic_fn)
        elapsed = time.perf_counter() - t0
        results.append({
            "algorithm":   name,
            "path_length": len(path),
            "cost":        round(cost, 4),
            "found":       len(path) > 0,
            "time_ms":     round(elapsed * 1000, 3),
        })
    return results


# ─────────────────────────────────────────────────────────────────
#  Backward-compatibility aliases
#  (allow code that imported from the old individual modules to keep
#   working with just an import path change)
# ─────────────────────────────────────────────────────────────────

# search_algorithms.py aliases — each old module exposed a bare `search`
astar_fn      = astar_search
bfs_fn        = bfs_search
dfs_fn        = dfs_search
greedy_fn     = greedy_search
ucs_fn        = ucs_search
ida_star_fn   = ida_star_search
beam_search_fn = beam_search

# ball_prediction.py aliases
predict_trajectory  = predict_ball_trajectory
find_intercept      = find_ball_intercept
ball_landing_pos    = ball_landing_position

# decision_tree.py alias
evaluate_tree = evaluate_decision_tree
