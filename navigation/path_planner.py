from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Set, Tuple

from core.all_algorithms import run_search

Node = Tuple[int, int]

# All algorithm keys accepted by run_search()
_ALL_ALGORITHMS: List[str] = [
    "A*", "BFS", "UCS", "DFS", "Greedy",
    "Decision Tree", "Beam Search", "IDA*",
]


@dataclass
class PathResult:
    algorithm: str
    world_path: List[Tuple[float, float]]
    node_path: List[Node]
    cost: float


class PathPlanner:
    def __init__(self, grid_step: int = 600):
        self.grid_step = max(200, grid_step)
        self.field_half_width = 4096
        self.field_half_length = 5120

    def _clamp_world(self, x: float, y: float) -> Tuple[float, float]:
        cx = max(-self.field_half_width, min(self.field_half_width, x))
        cy = max(-self.field_half_length, min(self.field_half_length, y))
        return cx, cy

    def _to_node(self, x: float, y: float) -> Node:
        x, y = self._clamp_world(x, y)
        return (int(round(x / self.grid_step)), int(round(y / self.grid_step)))

    def _to_world(self, node: Node) -> Tuple[float, float]:
        return node[0] * self.grid_step, node[1] * self.grid_step

    def _neighbors(self, node: Node) -> Iterable[Node]:
        x, y = node
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                wx, wy = self._to_world((nx, ny))
                if abs(wx) <= self.field_half_width and abs(wy) <= self.field_half_length:
                    yield nx, ny

    @staticmethod
    def _cost(a: Node, b: Node) -> float:
        return math.hypot(b[0] - a[0], b[1] - a[1])

    @staticmethod
    def _heuristic(a: Node, b: Node) -> float:
        # Octile distance: tighter admissible heuristic for 8-directional grids
        dx = abs(b[0] - a[0])
        dy = abs(b[1] - a[1])
        return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)

    def find_path(
        self,
        start_xy: Tuple[float, float],
        goal_xy: Tuple[float, float],
        algorithm: str,
        allowed_algorithms: Optional[Set[str]] = None,
    ) -> PathResult:
        """Find a path from start_xy to goal_xy using the given algorithm.

        Parameters
        ----------
        allowed_algorithms : optional set of algorithm names the planner is
            allowed to use.  If *algorithm* is not in the set, fall back to
            the first allowed algorithm (or A* if the set is empty / None).
        """
        # Resolve the effective algorithm respecting the allow-list
        effective_algo = self._resolve_algorithm(algorithm, allowed_algorithms)

        start = self._to_node(*start_xy)
        goal = self._to_node(*goal_xy)

        node_path, cost = run_search(
            algorithm=effective_algo,
            start=start,
            goal=goal,
            neighbors_fn=self._neighbors,
            cost_fn=self._cost,
            heuristic_fn=self._heuristic,
        )

        if not node_path:
            node_path = [start, goal]
            cost = self._cost(start, goal)

        world_path = [self._to_world(n) for n in node_path]
        return PathResult(
            algorithm=effective_algo,
            world_path=world_path,
            node_path=node_path,
            cost=cost,
        )

    @staticmethod
    def _resolve_algorithm(
        requested: str,
        allowed: Optional[Set[str]],
    ) -> str:
        """Return the algorithm to actually use given optional constraints.

        - If *allowed* is None or empty → use *requested* (no restriction).
        - If *requested* is in *allowed* → use *requested*.
        - Otherwise → use the first entry in *allowed* (preserving insertion
          order) or fall back to "A*" if resolution fails.
        """
        if not allowed:
            return requested
        if requested in allowed:
            return requested
        # Pick first allowed algorithm that run_search knows about
        for alg in _ALL_ALGORITHMS:
            if alg in allowed:
                return alg
        return "A*"
