"""
rl_algorithms.py
================
Unified Reinforcement Learning algorithm library for the Rocket League AI Bot.

Algorithms organised by category (matching university categorisation):

  ── Category 1: State Encoding ──────────────────────────────────────────────
    encode_state()              — 27-dim one-hot feature vector
    FEATURE_DIM, N_ACTIONS      — shared dimension constants

  ── Category 2: Classical Reinforcement Learning ────────────────────────────
    QLearningRoleSelector       — Tabular Q-Learning (ε-greedy, TD(0))
    SARSAOpponentModel          — On-policy SARSA for opponent prediction
    ActorCritic                 — Tabular Actor-Critic (TD-error baseline)
    MonteCarloRoleSelector      — First-Visit Monte Carlo control

  ── Category 3: Deep Reinforcement Learning ─────────────────────────────────
    DQNRoleSelector             — Deep Q-Network (3-layer numpy MLP, replay buffer)
    PPORoleSelector             — Proximal Policy Optimization (GAE, clip=0.2)
    A2CRoleSelector             — Advantage Actor-Critic (shared backbone)

  ── Category 4: Model-Based RL ──────────────────────────────────────────────
    ModelBasedRLSelector        — Dyna-Q style: empirical reward model + planning

  ── Category 5: Policy Gradient / Imitation Learning ───────────────────────
    PolicyGradientHumanLearner  — REINFORCE from human demonstrations

  ── Category 6: Ensemble / Meta-voting ──────────────────────────────────────
    EnsembleVoter               — Weighted majority vote across all selectors

  ── Category 7: Online / Incremental Learning ───────────────────────────────
    OnlinePatternLearner        — Per-state running stats, best-action recall

Usage
-----
from core.rl_algorithms import (
    encode_state, FEATURE_DIM, N_ACTIONS,
    QLearningRoleSelector, SARSAOpponentModel, ActorCritic,
    MonteCarloRoleSelector, DQNRoleSelector, PPORoleSelector,
    A2CRoleSelector, ModelBasedRLSelector, PolicyGradientHumanLearner,
    EnsembleVoter, OnlinePatternLearner,
)
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.rl_state import ROLE_ACTIONS, OPP_ACTIONS


# ═════════════════════════════════════════════════════════════════════════════
#  0.  STATE ENCODING
#      Shared vocabulary and 27-dim one-hot encoder used by all neural-net RLs
# ═════════════════════════════════════════════════════════════════════════════

_SITUATIONS = ["we_have_ball", "free_ball", "defending", "opp_has_ball"]
_BALL_X     = ["L", "C", "R"]
_BALL_Y     = ["opp_deep", "opp_half", "mid", "our_half", "our_deep"]
_SPEEDS     = ["slow", "med", "fast", "supersonic"]
_BOOSTS     = ["empty", "low", "mid", "full"]
_SCORES     = ["losing_bad", "losing", "tied", "winning", "winning_big"]
_POSS       = ["us", "them"]

FEATURE_DIM: int = (
    len(_SITUATIONS) + len(_BALL_X) + len(_BALL_Y) +
    len(_SPEEDS) + len(_BOOSTS) + len(_SCORES) + len(_POSS)
)   # = 4+3+5+4+4+5+2 = 27
N_ACTIONS: int = len(ROLE_ACTIONS)   # = 6


def _one_hot(value: str, vocab: List[str]) -> np.ndarray:
    arr = np.zeros(len(vocab), dtype=np.float32)
    try:
        arr[vocab.index(value)] = 1.0
    except ValueError:
        pass
    return arr


def encode_state(state_key: str) -> np.ndarray:
    """
    Encode a state string key into a float32 one-hot feature vector.

    State key format: "situation|ball_zone|car_zone|speed|boost|score|poss"
    Optional context prefix is stripped: "mode_tag::actual_state_key"
    """
    if "::" in state_key:
        state_key = state_key.split("::", 1)[1]
    parts = state_key.split("|")
    sit       = parts[0] if len(parts) > 0 else "free_ball"
    ball_zone = parts[1] if len(parts) > 1 else "C_mid"
    bz_parts  = ball_zone.split("_", 1)
    bx        = bz_parts[0] if len(bz_parts) > 0 else "C"
    by        = bz_parts[1] if len(bz_parts) > 1 else "mid"
    speed     = parts[3] if len(parts) > 3 else "med"
    boost     = parts[4] if len(parts) > 4 else "mid"
    score     = parts[5] if len(parts) > 5 else "tied"
    poss      = parts[6] if len(parts) > 6 else "them"
    return np.concatenate([
        _one_hot(sit,   _SITUATIONS),
        _one_hot(bx,    _BALL_X),
        _one_hot(by,    _BALL_Y),
        _one_hot(speed, _SPEEDS),
        _one_hot(boost, _BOOSTS),
        _one_hot(score, _SCORES),
        _one_hot(poss,  _POSS),
    ])


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / (e.sum() + 1e-8)


# ═════════════════════════════════════════════════════════════════════════════
#  2.  CLASSICAL REINFORCEMENT LEARNING
# ═════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
#  2a. Q-Learning  (Tabular, ε-greedy, TD(0))
# ─────────────────────────────────────────────────────────────────────────────

class QLearningRoleSelector:
    """
    Tabular Q-Learning for deciding which role/behaviour to adopt.

    Q(s,a) ← Q(s,a) + α [ r + γ·max_a' Q(s',a') − Q(s,a) ]

    Exploration: ε-greedy with multiplicative decay.
    """

    def __init__(self, alpha: float = 0.1, gamma: float = 0.95,
                 epsilon: float = 0.15, epsilon_decay: float = 0.9999,
                 epsilon_min: float = 0.03):
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min   = epsilon_min
        self.q_table: Dict[str, Dict[str, float]] = {}
        self._prev_state:  Optional[str] = None
        self._prev_action: Optional[str] = None
        self._episode_rewards: List[float] = []

    def _get_q(self, state: str) -> Dict[str, float]:
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in ROLE_ACTIONS}
        return self.q_table[state]

    def choose_role(self, state_key: str) -> str:
        """Epsilon-greedy role selection."""
        if random.random() < self.epsilon:
            action = random.choice(ROLE_ACTIONS)
        else:
            q_vals = self._get_q(state_key)
            action = max(q_vals, key=q_vals.get)
        self._prev_state  = state_key
        self._prev_action = action
        return action

    def update(self, new_state_key: str, reward: float):
        """Q-Learning TD update."""
        if self._prev_state is None or self._prev_action is None:
            return
        old_q    = self._get_q(self._prev_state)
        new_q    = self._get_q(new_state_key)
        old_val  = old_q[self._prev_action]
        best_next = max(new_q.values())
        td_target = reward + self.gamma * best_next
        old_q[self._prev_action] = old_val + self.alpha * (td_target - old_val)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self._episode_rewards.append(reward)

    def get_episode_return(self) -> float:
        return sum(self._episode_rewards)

    def reset_episode(self):
        self._episode_rewards.clear()

    def to_dict(self) -> Dict:
        return {
            "q_table": {k: dict(v) for k, v in self.q_table.items()},
            "epsilon": self.epsilon,
        }

    def from_dict(self, data: Dict):
        q = data.get("q_table", {})
        for state, actions in q.items():
            self.q_table[state] = {a: float(v) for a, v in actions.items()}
            for a in ROLE_ACTIONS:
                self.q_table[state].setdefault(a, 0.0)
        self.epsilon = data.get("epsilon", self.epsilon)


# ─────────────────────────────────────────────────────────────────────────────
#  2b. SARSA  (On-policy opponent prediction)
# ─────────────────────────────────────────────────────────────────────────────

class SARSAOpponentModel:
    """
    On-policy SARSA for predicting opponent behaviour.

    Uses observable features (position, velocity, distance to ball) to
    classify what the opponent is currently doing, then SARSA-updates a
    Q-table to improve future predictions.

    Q(s,a) ← Q(s,a) + α [ r + γ·Q(s',a') − Q(s,a) ]   (on-policy)
    """

    def __init__(self, alpha: float = 0.08, gamma: float = 0.9,
                 epsilon: float = 0.1):
        self.alpha   = alpha
        self.gamma   = gamma
        self.epsilon = epsilon
        self.q_table: Dict[str, Dict[str, float]] = {}
        self._prev_state:  Optional[str] = None
        self._prev_action: Optional[str] = None

    def _get_q(self, state: str) -> Dict[str, float]:
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in OPP_ACTIONS}
        return self.q_table[state]

    def classify_opp_action(self, opp_pos: Tuple[float, float],
                            opp_vel: Tuple[float, float],
                            ball_pos: Tuple[float, float],
                            push_dir: float) -> str:
        """Rule-based classification of the current opponent action."""
        dist_to_ball = math.hypot(opp_pos[0] - ball_pos[0],
                                  opp_pos[1] - ball_pos[1])
        opp_speed    = math.hypot(opp_vel[0], opp_vel[1])
        opp_vel_toward_ball = 0.0
        if dist_to_ball > 1:
            dx = ball_pos[0] - opp_pos[0]
            dy = ball_pos[1] - opp_pos[1]
            d  = math.hypot(dx, dy)
            opp_vel_toward_ball = (opp_vel[0] * dx + opp_vel[1] * dy) / d

        opp_y_rel = opp_pos[1] * push_dir

        if dist_to_ball < 800 and opp_vel_toward_ball > 300:
            return "rush_ball"
        if opp_y_rel > 2500 and opp_vel_toward_ball < 100:
            return "rotate_back"
        if opp_speed > 1500 and dist_to_ball > 2000:
            return "boost_grab"
        if opp_y_rel > 0 and abs(opp_vel_toward_ball) < 200 and dist_to_ball > 1200:
            return "shadow"
        if dist_to_ball > 3000 and opp_speed > 1800:
            return "demolish"
        if opp_pos[1] > 300:
            return "aerial"
        return "rush_ball"

    def predict(self, state_key: str) -> str:
        """Predict the opponent's most likely next action."""
        q_vals = self._get_q(state_key)
        return max(q_vals, key=q_vals.get)

    def update(self, state_key: str, action: str, reward: float,
               next_state: str, next_action: str):
        """SARSA on-policy update."""
        old_q    = self._get_q(state_key)
        next_q   = self._get_q(next_state)
        old_val  = old_q.get(action, 0.0)
        next_val = next_q.get(next_action, 0.0)
        td_target = reward + self.gamma * next_val
        old_q[action] = old_val + self.alpha * (td_target - old_val)
        self._prev_state  = state_key
        self._prev_action = action

    def to_dict(self) -> Dict:
        return {"q_table": {k: dict(v) for k, v in self.q_table.items()}}

    def from_dict(self, data: Dict):
        q = data.get("q_table", {})
        for state, actions in q.items():
            self.q_table[state] = {a: float(v) for a, v in actions.items()}
            for a in OPP_ACTIONS:
                self.q_table[state].setdefault(a, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
#  2c. Actor-Critic  (Tabular, TD-error baseline)
# ─────────────────────────────────────────────────────────────────────────────

class ActorCritic:
    """
    Tabular Actor-Critic for real-time control blending.

    Critic  → estimates V(s) (how good the current game state is)
    Actor   → adjusts control weights using the TD-error signal from the Critic

    TD-error  δ = r + γ·V(s') − V(s)
    V(s) ← V(s) + α_c·δ
    Actor adjustments ← adjustments + α_a·δ
    """

    def __init__(self, alpha_actor: float = 0.003, alpha_critic: float = 0.01,
                 gamma: float = 0.95):
        self.alpha_actor  = alpha_actor
        self.alpha_critic = alpha_critic
        self.gamma        = gamma
        self.value_table: Dict[str, float] = {}
        self.actor_adjustments: Dict[str, Dict[str, float]] = {}
        self._prev_state: Optional[str]  = None
        self._prev_value: float          = 0.0

    def _get_value(self, state: str) -> float:
        return self.value_table.get(state, 0.0)

    def _get_adjustments(self, state: str) -> Dict[str, float]:
        if state not in self.actor_adjustments:
            self.actor_adjustments[state] = {
                "throttle_adj": 0.0, "steer_adj": 0.0,
                "boost_adj": 0.0,    "aggression": 0.0,
            }
        return self.actor_adjustments[state]

    def evaluate_state(self, state_key: str) -> float:
        return self._get_value(state_key)

    def get_adjustments(self, state_key: str) -> Dict[str, float]:
        return dict(self._get_adjustments(state_key))

    def update(self, state_key: str, reward: float):
        if self._prev_state is None:
            self._prev_state = state_key
            self._prev_value = self._get_value(state_key)
            return
        current_value = self._get_value(state_key)
        td_error = reward + self.gamma * current_value - self._prev_value
        old_v = self.value_table.get(self._prev_state, 0.0)
        self.value_table[self._prev_state] = old_v + self.alpha_critic * td_error
        adj = self._get_adjustments(self._prev_state)
        adj["aggression"] += self.alpha_actor * td_error * 0.1
        adj["aggression"]  = max(-0.5, min(0.5, adj["aggression"]))
        self._prev_state = state_key
        self._prev_value = current_value

    def to_dict(self) -> Dict:
        sorted_values = sorted(self.value_table.items(),
                               key=lambda x: abs(x[1]), reverse=True)[:500]
        sorted_adj = sorted(self.actor_adjustments.items(),
                            key=lambda x: abs(x[1].get("aggression", 0)),
                            reverse=True)[:500]
        return {
            "value_table": dict(sorted_values),
            "actor_adjustments": {k: dict(v) for k, v in sorted_adj},
        }

    def from_dict(self, data: Dict):
        self.value_table = {k: float(v)
                            for k, v in data.get("value_table", {}).items()}
        for k, v in data.get("actor_adjustments", {}).items():
            self.actor_adjustments[k] = {
                "throttle_adj": v.get("throttle_adj", 0.0),
                "steer_adj":    v.get("steer_adj",    0.0),
                "boost_adj":    v.get("boost_adj",    0.0),
                "aggression":   v.get("aggression",   0.0),
            }


# ─────────────────────────────────────────────────────────────────────────────
#  2d. Monte Carlo RL  (First-Visit control)
# ─────────────────────────────────────────────────────────────────────────────

class MonteCarloRoleSelector:
    """
    First-Visit Monte Carlo control.

    Unlike TD methods, Monte Carlo waits until the end of an episode
    (goal/concede event) and uses the ACTUAL discounted return G_t.

    G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + …
    Q(s,a) ← mean of all returns observed for (s,a)

    Especially useful for evaluating long-horizon defensive positioning
    and attack sequences.
    """

    def __init__(self):
        self.q_table: Dict[str, Dict[str, float]] = {}
        self._returns: Dict[str, Dict[str, List[float]]] = {}
        self._episode:    List[Tuple[str, str]] = []
        self._ep_rewards: List[float]           = []
        self.gamma        = 0.95
        self._explore_eps = 0.35

    def _get_q(self, state_key: str) -> Dict[str, float]:
        if state_key not in self.q_table:
            self.q_table[state_key] = {a: 0.0 for a in ROLE_ACTIONS}
        return self.q_table[state_key]

    def choose_role(self, state_key: str) -> str:
        if random.random() < self._explore_eps:
            return random.choice(ROLE_ACTIONS)
        q = self._get_q(state_key)
        return max(q, key=q.get)

    def record(self, state_key: str, action: str, reward: float):
        self._episode.append((state_key, action))
        self._ep_rewards.append(reward)
        if len(self._episode) > 400:
            self._episode.pop(0)
            self._ep_rewards.pop(0)

    def end_episode(self, terminal_reward: float = 0.0):
        if not self._episode:
            return
        self._ep_rewards[-1] += terminal_reward
        G = 0.0
        visited: set = set()
        for t in reversed(range(len(self._episode))):
            s, a = self._episode[t]
            G    = self.gamma * G + self._ep_rewards[t]
            sa   = (s, a)
            if sa not in visited:
                visited.add(sa)
                if s not in self._returns:
                    self._returns[s] = {}
                if a not in self._returns[s]:
                    self._returns[s][a] = []
                history = self._returns[s][a]
                history.append(G)
                if len(history) > 120:
                    history.pop(0)
                self._get_q(s)[a] = sum(history) / len(history)
        self._episode.clear()
        self._ep_rewards.clear()
        self._explore_eps = max(0.07, self._explore_eps * 0.993)

    def to_dict(self) -> Dict:
        return {"q_table": self.q_table, "explore_eps": self._explore_eps}

    def from_dict(self, data: Dict):
        if "q_table" in data:
            self.q_table = data["q_table"]
        if "explore_eps" in data:
            self._explore_eps = float(data["explore_eps"])


# ═════════════════════════════════════════════════════════════════════════════
#  3.  DEEP REINFORCEMENT LEARNING
# ═════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
#  3a. Deep Q-Network (DQN)
# ─────────────────────────────────────────────────────────────────────────────

class DQNRoleSelector:
    """
    Deep Q-Network using a 3-layer numpy MLP as a Q-function approximator.

    Architecture:  state(27) → 64 → 32 → Q-values(6)
    Training:      TD(0) with experience replay and ε-greedy exploration.
    Generalises:   Handles unseen states gracefully via weight sharing.
    """

    def __init__(self):
        rng = np.random.default_rng(42)
        self.W1 = (rng.standard_normal((FEATURE_DIM, 64)) * 0.1).astype(np.float32)
        self.b1 = np.zeros(64, dtype=np.float32)
        self.W2 = (rng.standard_normal((64, 32)) * 0.1).astype(np.float32)
        self.b2 = np.zeros(32, dtype=np.float32)
        self.W3 = (rng.standard_normal((32, N_ACTIONS)) * 0.05).astype(np.float32)
        self.b3 = np.zeros(N_ACTIONS, dtype=np.float32)

        self._buffer: List[Tuple[np.ndarray, int, float, np.ndarray]] = []
        self._buf_max     = 2000
        self._batch       = 32
        self._train_every = 50
        self._step        = 0

        self.epsilon       = 0.6
        self.epsilon_min   = 0.08
        self.epsilon_decay = 0.9996
        self.lr            = 0.001
        self.gamma         = 0.90

        self._prev_feat:   Optional[np.ndarray] = None
        self._prev_action: Optional[int]        = None

    def _forward(self, feat: np.ndarray) -> np.ndarray:
        h1 = _relu(feat @ self.W1 + self.b1)
        h2 = _relu(h1  @ self.W2 + self.b2)
        return h2 @ self.W3 + self.b3

    def choose_role(self, state_key: str) -> str:
        feat = encode_state(state_key)
        if random.random() < self.epsilon:
            idx = random.randrange(N_ACTIONS)
        else:
            idx = int(np.argmax(self._forward(feat)))
        self._prev_feat   = feat
        self._prev_action = idx
        return ROLE_ACTIONS[idx]

    def update(self, state_key: str, reward: float):
        self._step += 1
        feat = encode_state(state_key)
        if self._prev_feat is not None and self._prev_action is not None:
            self._buffer.append((self._prev_feat.copy(), self._prev_action,
                                 reward, feat.copy()))
            if len(self._buffer) > self._buf_max:
                self._buffer.pop(0)
        if self._step % self._train_every == 0 and len(self._buffer) >= self._batch:
            self._train()
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def _train(self):
        batch = random.sample(self._buffer, self._batch)
        for s, a, r, s2 in batch:
            q_curr = self._forward(s)
            q_next = self._forward(s2)
            td_target = r + self.gamma * float(np.max(q_next))
            err       = td_target - q_curr[a]

            h1 = _relu(s @ self.W1 + self.b1)
            h2 = _relu(h1 @ self.W2 + self.b2)

            dq    = np.zeros(N_ACTIONS, dtype=np.float32)
            dq[a] = err
            dW3   = np.outer(h2, dq)
            dh2   = dq @ self.W3.T * (h2 > 0)
            dW2   = np.outer(h1, dh2)
            dh1   = dh2 @ self.W2.T * (h1 > 0)
            dW1   = np.outer(s, dh1)

            self.W3 += self.lr * dW3;  self.b3 += self.lr * dq
            self.W2 += self.lr * dW2;  self.b2 += self.lr * dh2
            self.W1 += self.lr * dW1;  self.b1 += self.lr * dh1

    def add_imaginary(self, state_key: str, action_name: str, reward: float):
        """Accept a model-generated sample for Dyna-Q replay augmentation."""
        feat = encode_state(state_key)
        try:
            a = ROLE_ACTIONS.index(action_name)
        except ValueError:
            return
        self._buffer.append((feat.copy(), a, reward, feat.copy()))
        if len(self._buffer) > self._buf_max:
            self._buffer.pop(0)

    def to_dict(self) -> Dict:
        return {
            "W1": self.W1.tolist(), "b1": self.b1.tolist(),
            "W2": self.W2.tolist(), "b2": self.b2.tolist(),
            "W3": self.W3.tolist(), "b3": self.b3.tolist(),
            "epsilon": self.epsilon,
        }

    def from_dict(self, data: Dict):
        for key in ("W1", "b1", "W2", "b2", "W3", "b3"):
            if key in data:
                setattr(self, key, np.array(data[key], dtype=np.float32))
        if "epsilon" in data:
            self.epsilon = float(data["epsilon"])


# ─────────────────────────────────────────────────────────────────────────────
#  3b. PPO (Proximal Policy Optimization, GAE, clip=0.2)
# ─────────────────────────────────────────────────────────────────────────────

_PPO_CLIP    = 0.2
_PPO_GAMMA   = 0.95
_PPO_LAMBDA  = 0.95
_PPO_LR      = 0.0005
_PPO_EPOCHS  = 3
_PPO_ROLLOUT = 64


class PPORoleSelector:
    """
    Proximal Policy Optimization (PPO-Clip) with Generalised Advantage
    Estimation (GAE).

    Architecture:  state(27) → 64 → 32 → [policy(6) || value(1)]
    Objective:     L_CLIP = E[ min(ratio·Â, clip(ratio, 1±ε)·Â) ]
    Update:        Mini-rollout of 64 frames → 3 PPO epochs.

    The clipped surrogate prevents large destructive policy updates.
    Stochastic policy (softmax) enables natural exploration.
    """

    def __init__(self):
        rng = np.random.default_rng(99)
        self.W1 = (rng.standard_normal((FEATURE_DIM, 64)) * 0.1).astype(np.float32)
        self.b1 = np.zeros(64, dtype=np.float32)
        self.W2 = (rng.standard_normal((64, 32)) * 0.1).astype(np.float32)
        self.b2 = np.zeros(32, dtype=np.float32)
        self.Wp = (rng.standard_normal((32, N_ACTIONS)) * 0.05).astype(np.float32)
        self.bp = np.zeros(N_ACTIONS, dtype=np.float32)
        self.Wv = (rng.standard_normal((32, 1)) * 0.05).astype(np.float32)
        self.bv = np.zeros(1, dtype=np.float32)

        self._states:    List[np.ndarray] = []
        self._actions:   List[int]        = []
        self._rewards:   List[float]      = []
        self._log_probs: List[float]      = []
        self._values:    List[float]      = []

        self._prev_feat:     Optional[np.ndarray] = None
        self._prev_action:   Optional[int]        = None
        self._prev_log_prob: float                = 0.0
        self._prev_value:    float                = 0.0

    def _forward(self, feat: np.ndarray) -> Tuple[np.ndarray, float]:
        h1    = _relu(feat @ self.W1 + self.b1)
        h2    = _relu(h1   @ self.W2 + self.b2)
        probs = _softmax(h2 @ self.Wp + self.bp)
        value = float((h2  @ self.Wv + self.bv)[0])
        return probs, value

    def choose_role(self, state_key: str) -> str:
        feat = encode_state(state_key)
        probs, value = self._forward(feat)
        idx = int(np.random.choice(N_ACTIONS, p=probs))
        self._prev_feat     = feat
        self._prev_action   = idx
        self._prev_log_prob = float(np.log(probs[idx] + 1e-8))
        self._prev_value    = value
        return ROLE_ACTIONS[idx]

    def update(self, state_key: str, reward: float):
        if self._prev_feat is not None:
            self._states.append(self._prev_feat.copy())
            self._actions.append(self._prev_action or 0)
            self._rewards.append(reward)
            self._log_probs.append(self._prev_log_prob)
            self._values.append(self._prev_value)
        if len(self._states) >= _PPO_ROLLOUT:
            self._ppo_update()
            self._states.clear();    self._actions.clear()
            self._rewards.clear();   self._log_probs.clear()
            self._values.clear()

    def _ppo_update(self):
        n        = len(self._rewards)
        rewards  = self._rewards
        values   = self._values

        advantages = np.zeros(n, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(n)):
            nv    = values[t + 1] if t + 1 < n else 0.0
            delta = rewards[t] + _PPO_GAMMA * nv - values[t]
            gae   = delta + _PPO_GAMMA * _PPO_LAMBDA * gae
            advantages[t] = gae
        returns    = advantages + np.array(values, dtype=np.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        old_lps    = np.array(self._log_probs, dtype=np.float32)
        states     = np.array(self._states,    dtype=np.float32)
        actions    = np.array(self._actions,   dtype=np.int32)

        for _epoch in range(_PPO_EPOCHS):
            order = list(range(n));  random.shuffle(order)
            for i in order:
                feat = states[i];  a = actions[i]
                adv  = float(advantages[i]);  ret = float(returns[i])
                probs, val = self._forward(feat)
                new_lp = float(np.log(probs[a] + 1e-8))
                ratio  = math.exp(new_lp - float(old_lps[i]))
                if adv >= 0:
                    clipped = min(ratio, 1 + _PPO_CLIP)
                else:
                    clipped = max(ratio, 1 - _PPO_CLIP)
                policy_loss = -min(ratio * adv, clipped * adv)
                val_loss    = (val - ret) ** 2

                h1 = _relu(feat @ self.W1 + self.b1)
                h2 = _relu(h1   @ self.W2 + self.b2)
                dv   = np.array([2.0 * (val - ret)], dtype=np.float32)
                dWv  = np.outer(h2, dv)
                dp   = probs.copy()
                dp[a] -= 1.0
                scale = policy_loss / (abs(adv) + 1e-8)
                dp   *= scale
                dWp  = np.outer(h2, dp)
                dh2  = (dp @ self.Wp.T + dv @ self.Wv.T) * (h2 > 0)
                dW2  = np.outer(h1, dh2)
                dh1  = (dh2 @ self.W2.T) * (h1 > 0)
                dW1  = np.outer(feat, dh1)

                self.Wp -= _PPO_LR * dWp;  self.bp -= _PPO_LR * dp
                self.Wv -= _PPO_LR * dWv;  self.bv -= _PPO_LR * dv
                self.W2 -= _PPO_LR * dW2;  self.b2 -= _PPO_LR * dh2
                self.W1 -= _PPO_LR * dW1;  self.b1 -= _PPO_LR * dh1

    def to_dict(self) -> Dict:
        return {k: getattr(self, k).tolist()
                for k in ("W1", "b1", "W2", "b2", "Wp", "bp", "Wv", "bv")}

    def from_dict(self, data: Dict):
        for k in ("W1", "b1", "W2", "b2", "Wp", "bp", "Wv", "bv"):
            if k in data:
                setattr(self, k, np.array(data[k], dtype=np.float32))


# ─────────────────────────────────────────────────────────────────────────────
#  3c. A2C (Advantage Actor-Critic)
# ─────────────────────────────────────────────────────────────────────────────

_A2C_GAMMA = 0.95
_A2C_LR    = 0.001


class A2CRoleSelector:
    """
    Advantage Actor-Critic (A2C) with a shared 2-layer backbone.

    Architecture:  state(27) → 64 → 32 → [actor(6) || critic(1)]
    Advantage:     A(s,a) = R_t − V(s)   (reduces policy gradient variance)
    The critic provides a value baseline that substitutes for a Monte Carlo
    baseline, giving lower variance while remaining unbiased.
    """

    def __init__(self):
        rng = np.random.default_rng(7)
        self.W1 = (rng.standard_normal((FEATURE_DIM, 64)) * 0.1).astype(np.float32)
        self.b1 = np.zeros(64, dtype=np.float32)
        self.W2 = (rng.standard_normal((64, 32)) * 0.1).astype(np.float32)
        self.b2 = np.zeros(32, dtype=np.float32)
        self.Wa = (rng.standard_normal((32, N_ACTIONS)) * 0.05).astype(np.float32)
        self.ba = np.zeros(N_ACTIONS, dtype=np.float32)
        self.Wc = (rng.standard_normal((32, 1)) * 0.05).astype(np.float32)
        self.bc = np.zeros(1, dtype=np.float32)

        self._prev_feat:   Optional[np.ndarray] = None
        self._prev_action: Optional[int]        = None
        self._prev_value:  float                = 0.0

    def _forward(self, feat: np.ndarray):
        h1    = _relu(feat @ self.W1 + self.b1)
        h2    = _relu(h1   @ self.W2 + self.b2)
        probs = _softmax(h2 @ self.Wa + self.ba)
        value = float((h2  @ self.Wc + self.bc)[0])
        return probs, value, h1, h2

    def choose_role(self, state_key: str) -> str:
        feat = encode_state(state_key)
        probs, value, _, _ = self._forward(feat)
        idx = int(np.random.choice(N_ACTIONS, p=probs))
        self._prev_feat   = feat
        self._prev_action = idx
        self._prev_value  = value
        return ROLE_ACTIONS[idx]

    def update(self, state_key: str, reward: float):
        if self._prev_feat is None or self._prev_action is None:
            return
        feat = encode_state(state_key)
        _, next_value, _, _ = self._forward(feat)
        td_target = reward + _A2C_GAMMA * next_value
        advantage = td_target - self._prev_value

        _, _, h1, h2 = self._forward(self._prev_feat)
        probs = _softmax(h2 @ self.Wa + self.ba)

        dp    = probs.copy()
        dp[self._prev_action] -= 1.0
        dp   *= (-advantage * 0.1)
        dWa   = np.outer(h2, dp)
        dvc   = np.array([2.0 * (self._prev_value - td_target)], dtype=np.float32)
        dWc   = np.outer(h2, dvc)
        dh2   = (dp @ self.Wa.T + dvc @ self.Wc.T) * (h2 > 0)
        dW2   = np.outer(h1, dh2)
        dh1   = (dh2 @ self.W2.T) * (h1 > 0)
        dW1   = np.outer(self._prev_feat, dh1)

        self.Wa -= _A2C_LR * dWa;  self.ba -= _A2C_LR * dp
        self.Wc -= _A2C_LR * dWc;  self.bc -= _A2C_LR * dvc
        self.W2 -= _A2C_LR * dW2;  self.b2 -= _A2C_LR * dh2
        self.W1 -= _A2C_LR * dW1;  self.b1 -= _A2C_LR * dh1

    def evaluate_state(self, state_key: str) -> float:
        feat = encode_state(state_key)
        _, value, _, _ = self._forward(feat)
        return value

    def to_dict(self) -> Dict:
        return {k: getattr(self, k).tolist()
                for k in ("W1", "b1", "W2", "b2", "Wa", "ba", "Wc", "bc")}

    def from_dict(self, data: Dict):
        for k in ("W1", "b1", "W2", "b2", "Wa", "ba", "Wc", "bc"):
            if k in data:
                setattr(self, k, np.array(data[k], dtype=np.float32))


# ═════════════════════════════════════════════════════════════════════════════
#  4.  MODEL-BASED RL (Dyna-Q)
# ═════════════════════════════════════════════════════════════════════════════

class ModelBasedRLSelector:
    """
    Dyna-Q style Model-Based RL.

    Learns an empirical reward model:  p(reward | state, action)

    Uses this model to:
      1. Select roles based on expected reward (model exploitation).
      2. Generate imaginary (state, action, reward) samples that are
         injected into the DQN replay buffer — the Dyna-Q "planning" step.

    This is a Rocket League adaptation of Dyna-Q: we model the reward
    distribution per (s,a) pair rather than a full transition model of
    the environment, since ball physics are already captured by ball_prediction.
    """

    def __init__(self):
        self._reward_model: Dict[str, Dict[str, List[float]]] = \
            defaultdict(lambda: defaultdict(list))
        self._max_per_sa   = 60
        self._explore_eps  = 0.40
        self._prev_state:  Optional[str] = None
        self._prev_action: Optional[str] = None

    def choose_role(self, state_key: str) -> str:
        if random.random() < self._explore_eps:
            return random.choice(ROLE_ACTIONS)
        model  = self._reward_model.get(state_key, {})
        best_a = "balanced"
        best_v = float("-inf")
        for action in ROLE_ACTIONS:
            hist = model.get(action, [])
            val  = (sum(hist) / len(hist)) if hist else 0.05
            if val > best_v:
                best_v = val
                best_a = action
        return best_a

    def record(self, state_key: str, action: str, reward: float):
        self._prev_state  = state_key
        self._prev_action = action
        hist = self._reward_model[state_key][action]
        hist.append(reward)
        if len(hist) > self._max_per_sa:
            hist.pop(0)
        self._explore_eps = max(0.07, self._explore_eps * 0.99985)

    def generate_sample(self) -> Optional[Tuple[str, str, float]]:
        """Sample a random (state, action, reward) from the learned model."""
        if not self._reward_model:
            return None
        state  = random.choice(list(self._reward_model.keys()))
        ad     = self._reward_model[state]
        if not ad:
            return None
        action = random.choice(list(ad.keys()))
        hist   = ad[action]
        if not hist:
            return None
        return state, action, random.choice(hist)

    def to_dict(self) -> Dict:
        return {
            "reward_model": {s: dict(ad)
                             for s, ad in self._reward_model.items()},
            "explore_eps":  self._explore_eps,
        }

    def from_dict(self, data: Dict):
        if "reward_model" in data:
            for s, ad in data["reward_model"].items():
                for a, rlist in ad.items():
                    self._reward_model[s][a] = list(rlist)
        if "explore_eps" in data:
            self._explore_eps = float(data["explore_eps"])


# ═════════════════════════════════════════════════════════════════════════════
#  5.  POLICY GRADIENT / IMITATION LEARNING
# ═════════════════════════════════════════════════════════════════════════════

class PolicyGradientHumanLearner:
    """
    REINFORCE-style policy gradient that learns from human demonstrations.

    When the human drives (M-key mode), the bot records (state, action)
    pairs and trains a linear function approximator to map game features
    to continuous controls (throttle, steer, boost, jump).

    This is an Imitation Learning baseline that bootstraps the policy
    before RL takes over.
    """

    def __init__(self, learning_rate: float = 0.005):
        self.lr = learning_rate
        self._num_features = 14
        self.weights: Dict[str, List[float]] = {
            "throttle": [0.0] * self._num_features,
            "steer":    [0.0] * self._num_features,
            "boost":    [0.0] * self._num_features,
            "jump":     [0.0] * self._num_features,
        }
        self._trajectory:    List[Dict] = []
        self._total_updates: int        = 0

    @staticmethod
    def _context_scalar(context_tag: str) -> float:
        if not context_tag:
            return 0.0
        total = sum(ord(ch) for ch in context_tag)
        return ((total % 2000) / 1000.0) - 1.0

    def _extract_features(self, car_pos: Tuple[float, float],
                          ball_pos: Tuple[float, float],
                          car_speed: float, car_boost: float,
                          opp_pos: Tuple[float, float],
                          push_dir: float,
                          context_tag: str = "") -> List[float]:
        dx_ball    = (ball_pos[0] - car_pos[0]) / 5000.0
        dy_ball    = (ball_pos[1] - car_pos[1]) / 5000.0
        dist_ball  = math.hypot(dx_ball, dy_ball)
        dx_opp     = (opp_pos[0] - car_pos[0]) / 5000.0
        dy_opp     = (opp_pos[1] - car_pos[1]) / 5000.0
        speed_norm = car_speed / 2300.0
        boost_norm = car_boost / 100.0
        car_y_rel  = car_pos[1]  * push_dir / 5120.0
        ball_y_rel = ball_pos[1] * push_dir / 5120.0
        car_x_norm = car_pos[0]  / 4096.0
        ball_x_norm= ball_pos[0] / 4096.0
        ctx_scalar = self._context_scalar(context_tag)
        return [dx_ball, dy_ball, dist_ball,
                dx_opp, dy_opp,
                speed_norm, boost_norm,
                car_y_rel, ball_y_rel,
                car_x_norm, ball_x_norm,
                ctx_scalar, ctx_scalar * ctx_scalar,
                1.0]   # bias

    def _predict_control(self, features: List[float], control: str) -> float:
        w = self.weights[control]
        return sum(f * wi for f, wi in zip(features, w))

    def record_human_frame(self, car_pos: Tuple[float, float],
                           ball_pos: Tuple[float, float],
                           car_speed: float, car_boost: float,
                           opp_pos: Tuple[float, float],
                           push_dir: float,
                           context_tag: str,
                           throttle: float, steer: float,
                           boost_in: float, jump_in: float):
        features = self._extract_features(
            car_pos, ball_pos, car_speed, car_boost, opp_pos, push_dir, context_tag)
        self._trajectory.append({
            "features": features,
            "actions": {"throttle": throttle, "steer": steer,
                        "boost": boost_in, "jump": jump_in},
        })
        if len(self._trajectory) >= 5:
            self._update_from_trajectory()

    def _update_from_trajectory(self):
        if not self._trajectory:
            return
        for frame in self._trajectory:
            features = frame["features"]
            actions  = frame["actions"]
            for control, target_val in actions.items():
                predicted = self._predict_control(features, control)
                error     = target_val - predicted
                w = self.weights[control]
                for i in range(len(w)):
                    w[i] += self.lr * error * features[i]
                    w[i]  = max(-5.0, min(5.0, w[i]))
        self._total_updates += len(self._trajectory)
        self._trajectory.clear()

    def suggest_controls(self, car_pos: Tuple[float, float],
                         ball_pos: Tuple[float, float],
                         car_speed: float, car_boost: float,
                         opp_pos: Tuple[float, float],
                         push_dir: float,
                         context_tag: str = "") -> Optional[Dict[str, float]]:
        if self._total_updates < 20:
            return None
        features = self._extract_features(
            car_pos, ball_pos, car_speed, car_boost, opp_pos, push_dir, context_tag)
        return {
            "throttle": max(-1.0, min(1.0,
                            self._predict_control(features, "throttle"))),
            "steer":    max(-1.0, min(1.0,
                            self._predict_control(features, "steer"))),
            "boost":    max(0.0,  min(1.0,
                            self._predict_control(features, "boost"))),
            "jump":     max(0.0,  min(1.0,
                            self._predict_control(features, "jump"))),
        }

    def has_learned(self) -> bool:
        return self._total_updates >= 20

    def to_dict(self) -> Dict:
        return {
            "weights":       {k: list(v) for k, v in self.weights.items()},
            "total_updates": self._total_updates,
        }

    def from_dict(self, data: Dict):
        w = data.get("weights", {})
        for control in ("throttle", "steer", "boost", "jump"):
            if control in w and len(w[control]) == self._num_features:
                self.weights[control] = list(w[control])
        self._total_updates = data.get("total_updates", 0)


# ═════════════════════════════════════════════════════════════════════════════
#  6.  ENSEMBLE / META-VOTING
# ═════════════════════════════════════════════════════════════════════════════

class EnsembleVoter:
    """
    Weighted Majority Vote over all strategy recommendations.

    Weights are updated online based on which sources lead to good
    outcomes (positive reward → weight ↑, negative → weight ↓).

    This is a lightweight meta-learning aggregator that combines
    Classical RL, Deep RL, Model-Based RL, and heuristic outputs.
    """

    def __init__(self):
        self.source_weights: Dict[str, float] = {
            "q_learning":    1.0,
            "sarsa_counter": 1.0,
            "actor_critic":  1.0,
            "online_learner":1.0,
            "dqn":           1.0,
            "ppo":           1.0,
            "a2c":           1.0,
            "monte_carlo":   1.0,
            "model_based":   1.0,
            "heuristic":     1.0,
        }
        self._source_rewards: Dict[str, List[float]] = defaultdict(list)

    def vote(self, recommendations: Dict[str, str]) -> str:
        action_scores: Dict[str, float] = defaultdict(float)
        for source, action in recommendations.items():
            weight = self.source_weights.get(source, 1.0)
            action_scores[action] += weight
        if not action_scores:
            return "balanced"
        return max(action_scores, key=action_scores.get)

    def reward_source(self, source: str, reward: float):
        self._source_rewards[source].append(reward)
        self._source_rewards[source] = self._source_rewards[source][-50:]
        for src in self.source_weights:
            rewards = self._source_rewards.get(src, [])
            if rewards:
                avg = sum(rewards) / len(rewards)
                self.source_weights[src] = max(0.3, min(2.0, 1.0 + avg))

    def to_dict(self) -> Dict:
        return {
            "source_weights": dict(self.source_weights),
            "source_rewards": {k: list(v)
                               for k, v in self._source_rewards.items()},
        }

    def from_dict(self, data: Dict):
        self.source_weights.update(data.get("source_weights", {}))
        for k, v in data.get("source_rewards", {}).items():
            self._source_rewards[k] = list(v)


# ═════════════════════════════════════════════════════════════════════════════
#  7.  ONLINE / INCREMENTAL LEARNING
# ═════════════════════════════════════════════════════════════════════════════

class OnlinePatternLearner:
    """
    Incremental Online Learner with running per-state statistics.

    Records every (state, action, reward) triple and provides:
      best_action(state) — the action with the highest mean reward
      recent_trend()     — average reward over a sliding window (≈5 s)

    Never requires batch training — updates every frame.
    Complements the Passive-Aggressive learner in advanced_ml.py.
    """

    def __init__(self):
        self.stats: Dict[str, Dict[str, Tuple[int, float]]] = {}
        self._window: List[Dict] = []
        self._window_size = 300   # ~5 s at 60 fps

    def record(self, state_key: str, action: str, reward: float):
        if state_key not in self.stats:
            self.stats[state_key] = {}
        entry = self.stats[state_key]
        if action not in entry:
            entry[action] = (0, 0.0)
        count, total = entry[action]
        entry[action] = (count + 1, total + reward)
        self._window.append({"state": state_key, "action": action,
                              "reward": reward})
        if len(self._window) > self._window_size:
            self._window.pop(0)

    def best_action(self, state_key: str) -> Optional[str]:
        entry = self.stats.get(state_key)
        if not entry:
            return None
        best_action = None
        best_avg    = float("-inf")
        for action, (count, total) in entry.items():
            if count >= 3:
                avg = total / count
                if avg > best_avg:
                    best_avg    = avg
                    best_action = action
        return best_action

    def recent_trend(self) -> float:
        if not self._window:
            return 0.0
        return sum(f["reward"] for f in self._window) / len(self._window)

    def to_dict(self) -> Dict:
        pruned = {}
        for state, actions in self.stats.items():
            filtered = {a: list(v) for a, v in actions.items() if v[0] >= 3}
            if filtered:
                pruned[state] = filtered
        if len(pruned) > 1000:
            sorted_states = sorted(
                pruned.items(),
                key=lambda x: max(v[0] for v in x[1].values()),
                reverse=True,
            )[:1000]
            pruned = dict(sorted_states)
        return {"stats": pruned}

    def from_dict(self, data: Dict):
        for state, actions in data.get("stats", {}).items():
            self.stats[state] = {a: tuple(v) for a, v in actions.items()}
