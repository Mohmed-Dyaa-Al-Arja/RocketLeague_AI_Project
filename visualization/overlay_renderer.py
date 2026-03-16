from __future__ import annotations

import ctypes
import math
from typing import Dict, List, Optional, Tuple


def _wire_box(renderer, cx: float, cy: float, cz: float, hx: float, hy: float, hz: float, color):
    pts = [
        (cx - hx, cy - hy, cz - hz), (cx + hx, cy - hy, cz - hz),
        (cx + hx, cy + hy, cz - hz), (cx - hx, cy + hy, cz - hz),
        (cx - hx, cy - hy, cz + hz), (cx + hx, cy - hy, cz + hz),
        (cx + hx, cy + hy, cz + hz), (cx - hx, cy + hy, cz + hz),
    ]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    for i, j in edges:
        renderer.draw_line_3d(pts[i], pts[j], color)


def _draw_text_2d_safe(renderer, x: int, y: int, scale_x: int, scale_y: int, text: str, color):
    try:
        renderer.draw_string_2d(int(x), int(y), int(scale_x), int(scale_y), text, color)
        return
    except Exception:
        pass

    try:
        vec = renderer._RenderingManager__create_vector([int(x), int(y), 0])
        renderer.native_draw_string_2d(
            renderer.builder,
            text.encode("utf-8"),
            color,
            vec,
            ctypes.c_int(int(scale_x)),
            ctypes.c_int(int(scale_y)),
        )
    except Exception:
        return


class OverlayRenderer:
    def __init__(self, agent_renderer):
        self.r = agent_renderer
        self._colors_cached = False
        self._c_green = None
        self._c_blue = None
        self._c_yellow = None
        self._c_red = None
        self._c_purple = None
        self._c_white = None

    # ── Debug visualisations ──────────────────────────────────────────────────

    def draw_predicted_trajectory(
        self,
        trajectory: List[Tuple[float, float, float]],
        step_stride: int = 5,
    ) -> None:
        """
        Draw the predicted ball trajectory as a cyan polyline.

        Parameters
        ----------
        trajectory  : list of (x, y, z) positions from ball_prediction.py.
        step_stride : only draw every Nth point for performance.
        """
        self._ensure_colors()
        if not trajectory:
            return
        prev = trajectory[0]
        for i, pt in enumerate(trajectory):
            if i % step_stride != 0:
                continue
            self.r.draw_line_3d(prev, pt, self._c_cyan)
            prev = pt

    def draw_intercept_point(
        self,
        intercept: Tuple[float, float, float],
        radius: float = 80.0,
    ) -> None:
        """
        Draw a green circle (cross + ring approximation) at the intercept point.
        """
        self._ensure_colors()
        x, y, z = intercept
        # Cross marker
        self.r.draw_line_3d((x - radius, y, z), (x + radius, y, z), self._c_green)
        self.r.draw_line_3d((x, y - radius, z), (x, y + radius, z), self._c_green)
        # Vertical spike
        self.r.draw_line_3d((x, y, z), (x, y, z + radius * 2), self._c_green)

    def draw_shoot_direction(
        self,
        ball_pos: Tuple[float, float],
        goal_aim: Tuple[float, float],
        length: float = 500.0,
    ) -> None:
        """
        Draw a red arrow from ball_pos toward goal_aim indicating shoot direction.
        """
        self._ensure_colors()
        bx, by = ball_pos
        gx, gy = goal_aim
        dx = gx - bx
        dy = gy - by
        d  = max(1.0, math.hypot(dx, dy))
        # Arrow tip
        tip_x = bx + dx / d * length
        tip_y = by + dy / d * length
        bz = 93.0   # ball height
        self.r.draw_line_3d((bx, by, bz), (tip_x, tip_y, bz), self._c_red)
        # Arrowhead (two short diagonal lines)
        perp_x = -dy / d * 80.0
        perp_y =  dx / d * 80.0
        back_x = tip_x - dx / d * 120.0
        back_y = tip_y - dy / d * 120.0
        self.r.draw_line_3d((tip_x, tip_y, bz),
                             (back_x + perp_x, back_y + perp_y, bz), self._c_red)
        self.r.draw_line_3d((tip_x, tip_y, bz),
                             (back_x - perp_x, back_y - perp_y, bz), self._c_red)

    @staticmethod
    def _detect_screen_size() -> Tuple[int, int]:
        try:
            user32 = ctypes.windll.user32
            return max(1280, int(user32.GetSystemMetrics(0))), max(720, int(user32.GetSystemMetrics(1)))
        except Exception:
            return 1920, 1080

    def _menu_layout(self, menu_position: str = "auto", menu_size: str = "auto") -> Dict[str, int | bool]:
        screen_w, screen_h = self._detect_screen_size()
        if menu_size == "small":
            compact = True
        elif menu_size == "medium":
            compact = False
        else:
            compact = screen_w < 1600 or screen_h < 900
        # Always top-right to avoid blocking gameplay view
        panel_width = 320 if compact else 400
        px = max(20, screen_w - panel_width - 30)
        py = 20
        return {
            "x": px,
            "y": py,
            "title_scale": 1 if compact else 2,
            "line_scale": 1,
            "line_gap": 14 if compact else 16,
            "section_gap": 18 if compact else 24,
            "compact": compact,
        }

    def _ensure_colors(self):
        if self._colors_cached:
            return
        r = self.r
        self._c_green = r.create_color(180, 40, 255, 40)
        self._c_blue = r.create_color(180, 40, 120, 255)
        self._c_yellow = r.create_color(255, 255, 230, 50)
        self._c_red = r.create_color(255, 255, 50, 50)
        self._c_purple = r.create_color(255, 160, 50, 255)
        self._c_white = r.create_color(255, 255, 255, 255)
        self._c_cyan = r.create_color(255, 80, 230, 255)
        self._c_orange = r.create_color(255, 255, 160, 40)
        self._colors_cached = True

    def _field_guidance(self, color_goal, color_own,
                        opponent_goal_y: float,
                        game_mode: str,
                        goal_width: float = 893.0):
        """Draw the relevant scoring area for the current mode."""
        own_goal_y = -opponent_goal_y
        half_goal_w = max(500.0, goal_width)

        if game_mode == "hoops":
            rim_y = opponent_goal_y * 0.78
            self.r.draw_line_3d((-700, rim_y, 290), (700, rim_y, 290), color_goal)
            _wire_box(self.r, 0, rim_y, 390, 520, 120, 120, color_goal)
            _wire_box(self.r, 0, own_goal_y * 0.78, 390, 520, 120, 120, color_own)
        elif game_mode == "dropshot":
            zone_y = opponent_goal_y * 0.45
            self.r.draw_line_3d((-3800, zone_y, 12), (3800, zone_y, 12), color_goal)
            self.r.draw_line_3d((-3800, own_goal_y * 0.45, 12), (3800, own_goal_y * 0.45, 12), color_own)
            _wire_box(self.r, 0, zone_y, 60, 3600, 900, 40, color_goal)
        else:
            self.r.draw_line_3d((-half_goal_w, opponent_goal_y, 17), (half_goal_w, opponent_goal_y, 17), color_goal)
            self.r.draw_line_3d((-half_goal_w, own_goal_y, 17), (half_goal_w, own_goal_y, 17), color_own)

        self.r.draw_line_3d((-4000, 0, 10), (4000, 0, 10), color_own)

    def _draw_path(self, path_points: List[Tuple[float, float]], color):
        if len(path_points) < 2:
            return
        for i in range(1, len(path_points)):
            a = path_points[i - 1]
            b = path_points[i]
            self.r.draw_line_3d((a[0], a[1], 30), (b[0], b[1], 30), color)

    def _draw_ball_to_goal(self, ball_pos: Tuple[float, float, float],
                           goal_x: float, goal_y: float, color):
        """Draw dashed projection line from ball to the target goal."""
        bx, by, bz = ball_pos
        segments = 10
        for i in range(segments):
            if i % 2 == 1:  # skip every other segment for dashed look
                continue
            t0 = i / segments
            t1 = (i + 1) / segments
            x0 = bx + (goal_x - bx) * t0
            y0 = by + (goal_y - by) * t0
            x1 = bx + (goal_x - bx) * t1
            y1 = by + (goal_y - by) * t1
            z = max(bz, 30.0)
            # Gently descend toward ground at goal end
            z0 = z * (1 - t0) + 30.0 * t0
            z1 = z * (1 - t1) + 30.0 * t1
            self.r.draw_line_3d((x0, y0, z0), (x1, y1, z1), color)
        # Arrow-head: small V at goal end
        dx = goal_x - bx
        dy = goal_y - by
        d = max(1.0, math.hypot(dx, dy))
        ux, uy = dx / d, dy / d
        px, py = -uy, ux  # perpendicular
        tip = (goal_x, goal_y, 30.0)
        wing = 180.0
        self.r.draw_line_3d(tip, (goal_x - ux * wing + px * wing * 0.4,
                                   goal_y - uy * wing + py * wing * 0.4, 30.0), color)
        self.r.draw_line_3d(tip, (goal_x - ux * wing - px * wing * 0.4,
                                   goal_y - uy * wing - py * wing * 0.4, 30.0), color)

    def _draw_aerial_arc(self, car_pos: Tuple[float, float, float],
                         ball_pos: Tuple[float, float, float], color):
        """Draw a curved arc from car up to an aerial ball."""
        cx, cy, cz = car_pos
        bx, by, bz = ball_pos
        arc_segments = 14
        prev = (cx, cy, cz)
        for i in range(1, arc_segments + 1):
            t = i / arc_segments
            x = cx + (bx - cx) * t
            y = cy + (by - cy) * t
            # Parabolic arc: rises to ball height with overshoot
            peak = bz * 1.15
            z = cz + (peak - cz) * (2 * t - t * t)  # quadratic ease
            # At t=1, z ≈ peak*(2-1) = peak, close to ball height
            z = max(z, 20.0)
            self.r.draw_line_3d(prev, (x, y, z), color)
            prev = (x, y, z)

    def render(
        self,
        my_team: int,
        car_pos: Tuple[float, float, float],
        ball_pos: Tuple[float, float, float],
        opponent_goal_y: float,
        mode: str,
        algorithm: str,
        usage_pct: Dict[str, float],
        path_cost: float,
        path_points: List[Tuple[float, float]],
        temporary_reset: bool,
        target: Tuple[float, float] = (0.0, 0.0),
        situation: str = "free_ball",
        rl_role: str = "",
        opp_prediction: str = "",
        rl_trend: float = 0.0,
        human_policy_active: bool = False,
        show_menu: bool = True,
        menu_position: str = "auto",
        menu_size: str = "auto",
        game_mode: str = "soccer",
        goal_width: float = 893.0,
        # Visual debug overlays (items 8, 9 in requirements)
        predicted_trajectory: Optional[List[Tuple[float, float, float]]] = None,
        intercept_point: Optional[Tuple[float, float, float]] = None,
        shoot_aim: Optional[Tuple[float, float]] = None,
        strategy: str = "",
        match_ended: bool = False,
        my_score: int = 0,
        opp_score: int = 0,
        goal_limit: int = 0,
    ):
        self._ensure_colors()
        r = self.r
        c_green = self._c_green
        c_blue = self._c_blue
        c_yellow = self._c_yellow
        c_red = self._c_red
        c_purple = self._c_purple
        c_white = self._c_white
        c_cyan = self._c_cyan
        c_orange = self._c_orange
        menu_layout = self._menu_layout(menu_position, menu_size)

        # ── Auto-close: when match has ended, skip all overlay content ──
        if match_ended:
            return

        # ── 3D World Elements ──
        self._field_guidance(c_red, c_purple, opponent_goal_y, game_mode, goal_width)

        _wire_box(r, car_pos[0], car_pos[1], car_pos[2] + 20, 70, 45, 25, c_green)
        _wire_box(r, ball_pos[0], ball_pos[1], ball_pos[2], 95, 95, 95, c_blue)

        target_goal_y = opponent_goal_y
        target_goal_z = 320
        if game_mode == "hoops":
            target_goal_y = opponent_goal_y * 0.78
            target_goal_z = 390
        elif game_mode == "dropshot":
            target_goal_y = opponent_goal_y * 0.45
            target_goal_z = 60
        _wire_box(r, 0, target_goal_y, target_goal_z, 900, 220, 320, c_red)
        own_goal_y = -opponent_goal_y
        _wire_box(r, 0, own_goal_y * 0.88, 250, 1200, 500, 250, c_purple)

        # Direct line: car → ball
        r.draw_line_3d((car_pos[0], car_pos[1], car_pos[2] + 20),
                       (ball_pos[0], ball_pos[1], ball_pos[2]), c_green)

        # Car → Target driving path
        self._draw_path(path_points, c_yellow)

        # Ball → Goal projection
        self._draw_ball_to_goal(ball_pos, 0.0, target_goal_y, c_orange)

        # ── Visual debug: predicted trajectory (cyan), intercept (green), shoot dir (red) ──
        if predicted_trajectory:
            self.draw_predicted_trajectory(predicted_trajectory)
        if intercept_point is not None:
            self.draw_intercept_point(intercept_point)
        if shoot_aim is not None:
            self.draw_shoot_direction((ball_pos[0], ball_pos[1]), shoot_aim)

        # Target marker (small cross)
        tx, ty = target
        r.draw_line_3d((tx - 80, ty, 25), (tx + 80, ty, 25), c_red)
        r.draw_line_3d((tx, ty - 80, 25), (tx, ty + 80, 25), c_red)

        # ── Resolve mode display ──
        mode_color = c_green
        mode_label = mode.upper()
        if mode == "attack":
            mode_color = c_orange
        elif mode == "defense":
            mode_color = c_blue
        elif mode == "manual":
            mode_color = c_red
        elif mode == "attack/defense":
            mode_color = c_yellow
            mode_label = "BALANCED"

        sit_map = {
            "we_have_ball": ("WE HAVE BALL", c_green),
            "opp_has_ball": ("OPP HAS BALL", c_red),
            "free_ball": ("FREE BALL", c_yellow),
            "defending": ("DEFENDING", c_blue),
        }
        sit_text, sit_color = sit_map.get(situation, ("---", c_white))

        model_text = "SESSION" if temporary_reset else "PERSISTENT"
        model_color = c_red if temporary_reset else c_cyan

        # ── Single consolidated HUD panel ──
        if show_menu:
            mx = int(menu_layout["x"])
            my = int(menu_layout["y"])
            lg = int(menu_layout["line_gap"])
            sg = int(menu_layout["section_gap"])
            ts = int(menu_layout["title_scale"])
            ls = int(menu_layout["line_scale"])
            compact = bool(menu_layout["compact"])

            _draw_text_2d_safe(r, mx, my, ts, ts, "medo dyaa", c_cyan)
            my += 20 if compact else 30

            _draw_text_2d_safe(r, mx, my, ls, ls, f"Mode: {mode_label}  |  {sit_text}", mode_color)
            my += lg
            _draw_text_2d_safe(r, mx, my, ls, ls, f"Algo: {algorithm}  |  Model: {model_text}", c_white)
            my += lg
            if strategy:
                _draw_text_2d_safe(r, mx, my, ls, ls, f"Strategy: {strategy}", c_orange)
                my += lg

            # RL role + opponent prediction on one line
            rl_info_parts = []
            if rl_role:
                rl_info_parts.append(f"Role: {rl_role.upper()}")
            if opp_prediction:
                rl_info_parts.append(f"Opp: {opp_prediction}")
            if rl_info_parts:
                _draw_text_2d_safe(r, mx, my, ls, ls, "  |  ".join(rl_info_parts), c_yellow)
                my += lg

            # Trend indicator
            trend_color = c_green if rl_trend >= 0 else c_red
            trend_bar = "+" * int(max(0, rl_trend * 100)) if rl_trend >= 0 else "-" * int(min(20, abs(rl_trend * 100)))
            _draw_text_2d_safe(r, mx, my, ls, ls, f"Trend: {rl_trend:+.3f} {trend_bar}", trend_color)
            my += sg

            # Top 3 algorithms by usage (compact)
            sorted_algs = sorted(usage_pct.items(), key=lambda kv: kv[1], reverse=True)
            top3 = sorted_algs[:3]
            if top3:
                parts = [f"{name} {pct:.0f}%" for name, pct in top3]
                _draw_text_2d_safe(r, mx, my, ls, ls, "Top: " + "  ".join(parts), c_white)
                my += lg

            if human_policy_active:
                _draw_text_2d_safe(r, mx, my, ls, ls, "Human Policy: ACTIVE", c_yellow)
                my += lg

            my += 4
            if compact:
                _draw_text_2d_safe(r, mx, my, 1, 1, "M/N/B/V=Mode P/O=Model L=Match H=Hide", c_green)
            else:
                _draw_text_2d_safe(r, mx, my, 1, 1, "M=Manual N=Balanced B=Attack V=Defense", c_green)
                my += lg
                _draw_text_2d_safe(r, mx, my, 1, 1, "P=Session O=Persistent L=Match H=Hide", c_green)

            if mode == "manual":
                my += lg
                _draw_text_2d_safe(r, mx, my, 1, 1, "WASD=Drive Space=Jump LShift=Boost", c_yellow)

            # ── Match Progress (score / target) ──────────────────────────────
            my += sg
            if goal_limit > 0:
                remaining = max(0, goal_limit - my_score)
                score_clr = c_green if my_score > opp_score else (c_red if my_score < opp_score else c_white)
                _draw_text_2d_safe(r, mx, my, ls, ls, f"Score: {my_score} — {opp_score}", score_clr)
                my += lg
                rem_clr = c_green if remaining <= 1 else (c_yellow if remaining <= 2 else c_white)
                _draw_text_2d_safe(r, mx, my, ls, ls, f"Target: {goal_limit} Goals  |  Need: {remaining}", rem_clr)
        else:
            _draw_text_2d_safe(r, int(menu_layout["x"]), int(menu_layout["y"]), 1, 1, "H = Show Menu", c_cyan)
