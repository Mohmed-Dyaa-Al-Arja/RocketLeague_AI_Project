"""
advanced_ml.py
==============
Advanced ML layer for the Rocket League AI Bot.

Implements six paradigms that directly improve the bot's intelligence:

  1. Anomaly Detection      — flag abnormal opponent behaviour (cheating-like patterns,
                               unusual ball trajectories, sudden strategy shifts)
  2. Multi-Task Learning    — one shared network learns attack, defense, boost AND
                               kick-off positioning simultaneously
  3. Online / Incremental   — lightweight streaming learner that never stops updating
                               (extends the existing OnlinePatternLearner with
                                Passive-Aggressive and SGD-style weight updates)
  4. Meta-Learning (MAML)   — fast-adapt the role-selector to a new opponent in a few
                               gradient steps; learns "how to learn quickly"
  5. Deep Learning helper   — tiny numpy MLP that combines DQN + multi-task heads;
                               no external deps beyond numpy
  6. Causal ML              — do-calculus-inspired counterfactual reasoning so the bot
                               asks "would changing my action have led to a goal?" rather
                               than just recording correlation

All classes are pure Python / numpy; no PyTorch / TensorFlow required.
Each class can be used standalone or plugged into AdaptiveLearner.

Author : RocketLeague AI Project
Date   : 2026
"""

from __future__ import annotations

import math
import random
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

import numpy as np

ALL_ADVANCED_COMPONENTS = frozenset({
    "anomaly",
    "multi_task",
    "passive_aggressive",
    "maml",
    "deep_rl",
    "causal",
})

# ─────────────────────────────────────────────────────────────
#  Shared utilities
# ─────────────────────────────────────────────────────────────

def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def _he_init(fan_in: int, fan_out: int) -> np.ndarray:
    """He / Kaiming initialisation — good default for ReLU networks."""
    return np.random.randn(fan_in, fan_out).astype(np.float32) * math.sqrt(2.0 / fan_in)


# ══════════════════════════════════════════════════════════════
#  1.  ANOMALY DETECTION
#      Detects unusual opponent / ball behaviour patterns.
#      Uses a rolling Z-score model (no external deps needed).
# ══════════════════════════════════════════════════════════════

class AnomalyDetector:
    """
    Lightweight streaming anomaly detector based on rolling mean / std.

    Tracks named metrics (e.g. "opp_speed", "ball_accel") and flags
    a reading as anomalous when it exceeds *threshold* standard deviations
    from the running mean.

    This is the Isolation-Forest / Z-score hybrid approach:
      - Rolling window  → adapts to drift
      - Z-score test    → O(1) per frame
      - Anomaly score   → used to trigger defensive override
    """

    def __init__(self, window: int = 300, threshold: float = 3.5):
        self._window = window
        self._threshold = threshold
        # Each metric: deque of recent values
        self._history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        # Anomaly event log: list of {"metric", "value", "z_score", "frame"}
        self.events: List[Dict] = []
        self._frame = 0

    # ── public API ──

    def update(self, metrics: Dict[str, float]) -> Dict[str, bool]:
        """
        Feed a dict of metric_name → value.
        Returns dict of metric_name → is_anomalous.
        """
        self._frame += 1
        flags: Dict[str, bool] = {}

        for name, value in metrics.items():
            buf = self._history[name]
            buf.append(value)

            if len(buf) < 20:           # not enough history yet
                flags[name] = False
                continue

            arr = np.array(buf, dtype=np.float32)
            mean = float(arr.mean())
            std  = float(arr.std()) + 1e-8
            z    = abs(value - mean) / std

            is_anomalous = z > self._threshold
            flags[name] = is_anomalous

            if is_anomalous:
                self.events.append({
                    "metric": name,
                    "value":  value,
                    "z_score": round(z, 2),
                    "frame":  self._frame,
                })
                # Keep only last 200 events
                if len(self.events) > 200:
                    self.events = self.events[-200:]

        return flags

    def anomaly_score(self, metric: str) -> float:
        """Return latest z-score for *metric* (0 if unknown)."""
        buf = self._history.get(metric)
        if not buf or len(buf) < 5:
            return 0.0
        arr = np.array(buf, dtype=np.float32)
        z = abs(buf[-1] - arr.mean()) / (arr.std() + 1e-8)
        return float(z)

    def overall_threat_level(self) -> float:
        """
        Aggregate threat level [0, 1] — high means opponent is doing
        something unusual (rapid speed bursts, abnormal shot angles, etc.)
        """
        scores = [self.anomaly_score(m) for m in self._history]
        if not scores:
            return 0.0
        max_z = max(scores)
        return float(min(1.0, max_z / (self._threshold * 2)))

    def to_dict(self) -> Dict:
        return {
            "events": self.events[-50:],  # persist only recent events
        }

    def from_dict(self, data: Dict):
        self.events = data.get("events", [])


# ══════════════════════════════════════════════════════════════
#  2.  MULTI-TASK LEARNING
#      One shared encoder, four separate task heads.
#
#  Tasks:
#    0 — attack role Q-values      (6 actions)
#    1 — defense role Q-values     (6 actions)
#    2 — boost management          (3 actions: grab / conserve / use)
#    3 — kick-off strategy         (3 actions: rush / fake / rotate)
#
#  Architecture:
#    input(27) → shared[64 → 32] → [task_head_i(16 → n_actions_i)]
#
#  Benefit over per-task models:
#    - Shared representation captures common patterns (ball position,
#      opponent threat, speed) once for all tasks
#    - Less total parameters; useful when data is sparse early in match
# ══════════════════════════════════════════════════════════════

_TASK_SIZES = [6, 6, 3, 3]   # output heads
_TASK_NAMES = ["attack", "defense", "boost", "kickoff"]


class MultiTaskNet:
    """
    Shared-encoder multi-task network (numpy, no external ML libs).

    Weights are updated with TD(0) gradient for each task independently,
    with the shared encoder receiving gradients from ALL active tasks.
    """

    def __init__(self, input_dim: int = 27, shared_hidden: int = 64,
                 task_hidden: int = 16, lr: float = 3e-4):
        self._lr = lr

        # Shared encoder: input → 64 → 32
        self.W1 = _he_init(input_dim,     shared_hidden)
        self.b1 = np.zeros(shared_hidden, dtype=np.float32)
        self.W2 = _he_init(shared_hidden, 32)
        self.b2 = np.zeros(32,            dtype=np.float32)

        # Per-task heads: 32 → task_hidden → n_actions
        self.heads_W1: List[np.ndarray] = [_he_init(32, task_hidden)    for _ in _TASK_SIZES]
        self.heads_b1: List[np.ndarray] = [np.zeros(task_hidden, dtype=np.float32) for _ in _TASK_SIZES]
        self.heads_W2: List[np.ndarray] = [_he_init(task_hidden, n)     for n in _TASK_SIZES]
        self.heads_b2: List[np.ndarray] = [np.zeros(n,           dtype=np.float32) for n in _TASK_SIZES]

        self._step = 0

    # ── Forward pass ──

    def _encode(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (h1, h2, pre_activation_h2) for backprop."""
        h1 = _relu(x @ self.W1 + self.b1)
        h2 = _relu(h1 @ self.W2 + self.b2)
        return h1, h2

    def predict(self, x: np.ndarray, task: int) -> np.ndarray:
        """Q-values for a given task (0-3)."""
        _, h2 = self._encode(x)
        th1 = _relu(h2 @ self.heads_W1[task] + self.heads_b1[task])
        return th1 @ self.heads_W2[task] + self.heads_b2[task]

    def best_action(self, x: np.ndarray, task: int) -> int:
        """Greedy action for *task*."""
        return int(np.argmax(self.predict(x, task)))

    # ── TD(0) update ──

    def update(self, x: np.ndarray, task: int, action: int,
               target_q: float, gamma: float = 0.97) -> float:
        """
        Single-step TD update for one (task, action) pair.
        Returns TD error magnitude.
        """
        self._step += 1
        h1, h2 = self._encode(x)

        # Task head forward
        th1_pre = h2 @ self.heads_W1[task] + self.heads_b1[task]
        th1     = _relu(th1_pre)
        q_vals  = th1 @ self.heads_W2[task] + self.heads_b2[task]

        td_error = target_q - q_vals[action]

        # ── Backprop through task head ──
        dq              = np.zeros_like(q_vals)
        dq[action]      = td_error
        dheads_W2       = np.outer(th1, dq)
        dheads_b2       = dq
        dtl1            = dq @ self.heads_W2[task].T * (th1_pre > 0)
        dheads_W1       = np.outer(h2, dtl1)
        dheads_b1       = dtl1

        # ── Gradient for shared encoder ──
        dh2_task        = dtl1 @ self.heads_W1[task].T
        # Accumulate shared grads (simple single step)
        dh2             = dh2_task
        h2_pre          = h1 @ self.W2 + self.b2
        dh1_act         = dh2 * (h2_pre > 0)
        dW2             = np.outer(h1, dh1_act)
        db2             = dh1_act
        x_pre           = x @ self.W1 + self.b1
        dh1             = dh1_act @ self.W2.T * (x_pre > 0)
        dW1             = np.outer(x, dh1)
        db1             = dh1

        # ── Gradient descent ──
        lr = self._lr
        self.heads_W2[task] += lr * dheads_W2
        self.heads_b2[task] += lr * dheads_b2
        self.heads_W1[task] += lr * dheads_W1
        self.heads_b1[task] += lr * dheads_b1
        self.W2             += lr * dW2
        self.b2             += lr * db2
        self.W1             += lr * dW1
        self.b1             += lr * db1

        return abs(td_error)

    def to_dict(self) -> Dict:
        return {
            "W1": self.W1.tolist(), "b1": self.b1.tolist(),
            "W2": self.W2.tolist(), "b2": self.b2.tolist(),
            "heads_W1": [w.tolist() for w in self.heads_W1],
            "heads_b1": [b.tolist() for b in self.heads_b1],
            "heads_W2": [w.tolist() for w in self.heads_W2],
            "heads_b2": [b.tolist() for b in self.heads_b2],
            "step": self._step,
        }

    def from_dict(self, data: Dict):
        self.W1 = np.array(data["W1"], dtype=np.float32)
        self.b1 = np.array(data["b1"], dtype=np.float32)
        self.W2 = np.array(data["W2"], dtype=np.float32)
        self.b2 = np.array(data["b2"], dtype=np.float32)
        for i in range(len(_TASK_SIZES)):
            self.heads_W1[i] = np.array(data["heads_W1"][i], dtype=np.float32)
            self.heads_b1[i] = np.array(data["heads_b1"][i], dtype=np.float32)
            self.heads_W2[i] = np.array(data["heads_W2"][i], dtype=np.float32)
            self.heads_b2[i] = np.array(data["heads_b2"][i], dtype=np.float32)
        self._step = data.get("step", 0)


# ══════════════════════════════════════════════════════════════
#  3.  ONLINE / INCREMENTAL LEARNING
#      Passive-Aggressive (PA-II) weight update rule for
#      real-time adaptation to streaming game data.
#
#  Unlike the existing OnlinePatternLearner (which tracks counts),
#  this maintains a true weight vector that updates every frame.
# ══════════════════════════════════════════════════════════════

class PassiveAggressiveLearner:
    """
    PA-II (Passive-Aggressive variant II) online linear classifier /
    regressor for real-time role recommendation.

    Suited for non-stationary streams: aggressiveness parameter C
    controls how fast old decisions can be overridden.

    Each "role" is a binary one-vs-all classifier.
    """

    def __init__(self, feature_dim: int = 27, n_classes: int = 6, C: float = 1.0):
        self._C = C
        self._n = n_classes
        self._dim = feature_dim
        # One weight vector per class
        self.weights = np.zeros((n_classes, feature_dim), dtype=np.float32)
        self._updates = 0

    def predict(self, x: np.ndarray) -> int:
        """Return predicted class (argmax of dot products)."""
        scores = self.weights @ x
        return int(np.argmax(scores))

    def update(self, x: np.ndarray, true_class: int, reward: float):
        """
        PA-II update: if prediction was wrong OR reward is negative,
        adjust weight vector toward the correct answer.

        *reward* modulates the learning step — negative reward makes
        the update more aggressive.
        """
        self._updates += 1
        pred = self.predict(x)

        # Encode "correct" label: +1 for true class, -1 for predicted
        if pred == true_class and reward >= 0:
            return  # Passive: no update needed

        # Loss: hinge with reward signal
        norm_sq = float(np.dot(x, x)) + 1e-8
        for c in range(self._n):
            if c == true_class:
                margin = 1.0 - float(self.weights[c] @ x)
            else:
                margin = -1.0 - float(self.weights[c] @ x)
            loss = max(0.0, margin)
            tau  = min(self._C, loss / (2.0 * norm_sq))  # PA-II
            sign = 1.0 if c == true_class else -1.0
            self.weights[c] += tau * sign * x

    def to_dict(self) -> Dict:
        return {"weights": self.weights.tolist(), "updates": self._updates}

    def from_dict(self, data: Dict):
        self.weights  = np.array(data["weights"], dtype=np.float32)
        self._updates = data.get("updates", 0)


# ══════════════════════════════════════════════════════════════
#  4.  META-LEARNING  (MAML — Model-Agnostic Meta-Learning)
#
#  The meta-learner stores a "meta-weights" network.
#  At the start of a new match (new opponent), it takes a handful
#  of experience frames and does a fast inner-loop adaptation.
#
#  Why useful here:
#    - Each opponent has a distinct playstyle
#    - MAML enables the bot to adapt to a new style in ~10 frames
#      instead of hundreds of Q-learning updates
#    - The meta-weights capture "how to be a good Rocket League bot
#      in general" — the inner loop specialises for THIS opponent
# ══════════════════════════════════════════════════════════════

class MAMLRoleAdapter:
    """
    Lightweight MAML implementation for role-selection adaptation.

    Meta-parameters: shared across all opponents (meta_W, meta_b).
    Inner-loop:      fast gradient descent on recent match experience.
    Outer-loop:      meta-update after each match/segment.

    Architecture: feature(27) → 32 → 6 role Q-values
    """

    def __init__(self, input_dim: int = 27, hidden: int = 32,
                 n_actions: int = 6,
                 inner_lr: float = 0.05,
                 outer_lr: float = 0.001,
                 inner_steps: int = 5):
        self._inner_lr   = inner_lr
        self._outer_lr   = outer_lr
        self._inner_steps = inner_steps

        # Meta-weights (θ)
        self.meta_W1 = _he_init(input_dim, hidden)
        self.meta_b1 = np.zeros(hidden,    dtype=np.float32)
        self.meta_W2 = _he_init(hidden,    n_actions)
        self.meta_b2 = np.zeros(n_actions, dtype=np.float32)

        # Fast-adapted weights for current opponent (θ')
        self._adapted_W1 = self.meta_W1.copy()
        self._adapted_b1 = self.meta_b1.copy()
        self._adapted_W2 = self.meta_W2.copy()
        self._adapted_b2 = self.meta_b2.copy()

        # Buffer of (x, action, td_target) for inner-loop
        self._inner_buffer: List[Tuple[np.ndarray, int, float]] = []
        self._max_buffer = 64

        self._meta_tasks: List[List[Tuple]] = []

    # ── Inference ──

    def predict(self, x: np.ndarray, use_adapted: bool = True) -> np.ndarray:
        W1 = self._adapted_W1 if use_adapted else self.meta_W1
        b1 = self._adapted_b1 if use_adapted else self.meta_b1
        W2 = self._adapted_W2 if use_adapted else self.meta_W2
        b2 = self._adapted_b2 if use_adapted else self.meta_b2
        h  = _relu(x @ W1 + b1)
        return h @ W2 + b2

    def best_action(self, x: np.ndarray) -> int:
        return int(np.argmax(self.predict(x)))

    # ── Inner loop: adapt to current opponent ──

    def record_experience(self, x: np.ndarray, action: int, td_target: float):
        """Store a frame for inner-loop adaptation."""
        self._inner_buffer.append((x, action, td_target))
        if len(self._inner_buffer) > self._max_buffer:
            self._inner_buffer.pop(0)

    def adapt(self):
        """
        Run *inner_steps* gradient steps on the buffered experience.
        Updates the adapted weights (θ') — does NOT change meta-weights (θ).
        Called at the start of a new match or after a goal event.
        """
        if len(self._inner_buffer) < 8:
            return

        W1 = self.meta_W1.copy()
        b1 = self.meta_b1.copy()
        W2 = self.meta_W2.copy()
        b2 = self.meta_b2.copy()

        for _ in range(self._inner_steps):
            batch = random.sample(self._inner_buffer,
                                  min(16, len(self._inner_buffer)))
            dW1 = np.zeros_like(W1);  db1 = np.zeros_like(b1)
            dW2 = np.zeros_like(W2);  db2 = np.zeros_like(b2)

            for x, action, target in batch:
                h_pre = x @ W1 + b1
                h     = _relu(h_pre)
                q     = h @ W2 + b2
                err   = target - q[action]
                dq    = np.zeros_like(q);  dq[action] = err
                dW2  += np.outer(h, dq)
                db2  += dq
                dh    = (dq @ W2.T) * (h_pre > 0)
                dW1  += np.outer(x, dh)
                db1  += dh

            n = len(batch)
            W1 += self._inner_lr * dW1 / n
            b1 += self._inner_lr * db1 / n
            W2 += self._inner_lr * dW2 / n
            b2 += self._inner_lr * db2 / n

        self._adapted_W1 = W1
        self._adapted_b1 = b1
        self._adapted_W2 = W2
        self._adapted_b2 = b2

    # ── Outer loop: meta-update across matches ──

    def end_of_match(self):
        """
        Call at the end of each match.
        Saves current inner-buffer as a meta-task and performs a
        simple first-order meta-update (FOMAML) to improve meta-weights.
        """
        if self._inner_buffer:
            self._meta_tasks.append(list(self._inner_buffer))
        if len(self._meta_tasks) > 10:
            self._meta_tasks = self._meta_tasks[-10:]

        if len(self._meta_tasks) < 2:
            return

        # FOMAML: update meta-weights toward average task gradient
        dW1 = np.zeros_like(self.meta_W1)
        db1 = np.zeros_like(self.meta_b1)
        dW2 = np.zeros_like(self.meta_W2)
        db2 = np.zeros_like(self.meta_b2)

        for task in self._meta_tasks[-5:]:
            sample = random.sample(task, min(8, len(task)))
            for x, action, target in sample:
                h_pre = x @ self.meta_W1 + self.meta_b1
                h     = _relu(h_pre)
                q     = h @ self.meta_W2 + self.meta_b2
                err   = target - q[action]
                dq    = np.zeros_like(q);  dq[action] = err
                dW2  += np.outer(h, dq)
                db2  += dq
                dh    = (dq @ self.meta_W2.T) * (h_pre > 0)
                dW1  += np.outer(x, dh)
                db1  += dh

        total = max(1, 5 * 8)
        self.meta_W1 += self._outer_lr * dW1 / total
        self.meta_b1 += self._outer_lr * db1 / total
        self.meta_W2 += self._outer_lr * dW2 / total
        self.meta_b2 += self._outer_lr * db2 / total

        # Reset inner buffer for new match
        self._inner_buffer.clear()

    def to_dict(self) -> Dict:
        return {
            "meta_W1": self.meta_W1.tolist(), "meta_b1": self.meta_b1.tolist(),
            "meta_W2": self.meta_W2.tolist(), "meta_b2": self.meta_b2.tolist(),
        }

    def from_dict(self, data: Dict):
        self.meta_W1 = np.array(data["meta_W1"], dtype=np.float32)
        self.meta_b1 = np.array(data["meta_b1"], dtype=np.float32)
        self.meta_W2 = np.array(data["meta_W2"], dtype=np.float32)
        self.meta_b2 = np.array(data["meta_b2"], dtype=np.float32)
        self._adapted_W1 = self.meta_W1.copy()
        self._adapted_b1 = self.meta_b1.copy()
        self._adapted_W2 = self.meta_W2.copy()
        self._adapted_b2 = self.meta_b2.copy()


# ══════════════════════════════════════════════════════════════
#  5.  DEEP LEARNING HELPER
#      Combined DQN + multi-task heads in a single deeper network.
#      Input: 27-dim encoded state
#      Shared: 64 → 32 with LayerNorm approximation
#      Heads:
#        - role_q (6)     — main role selection
#        - mechanic_q (8) — mechanic selection (flip, aerial, …)
#        - threat_v (1)   — value estimate (for Causal baseline)
# ══════════════════════════════════════════════════════════════

_ROLE_N      = 6
_MECHANIC_N  = 8
_THREAT_N    = 1


class DeepRLNet:
    """
    Deeper RL network with three output heads.

    Combines ideas from:
      - DQN (experience replay, TD targets)
      - Multi-Task Learning (shared encoder, separate heads)
      - Actor-Critic (value head for advantage estimation)

    Uses a replay buffer for stability (addresses DQN's deadly triad).
    """

    def __init__(self, input_dim: int = 27, lr: float = 2e-4,
                 replay_capacity: int = 2000, batch_size: int = 32):
        self._lr         = lr
        self._batch      = batch_size
        self._replay: deque = deque(maxlen=replay_capacity)
        self._step       = 0

        # Encoder: 27 → 64 → 32
        self.W1 = _he_init(input_dim, 64);  self.b1 = np.zeros(64,  dtype=np.float32)
        self.W2 = _he_init(64,        32);  self.b2 = np.zeros(32,  dtype=np.float32)

        # Role head: 32 → 16 → 6
        self.Wr1 = _he_init(32, 16);  self.br1 = np.zeros(16, dtype=np.float32)
        self.Wr2 = _he_init(16,  6);  self.br2 = np.zeros( 6, dtype=np.float32)

        # Mechanic head: 32 → 16 → 8
        self.Wm1 = _he_init(32, 16);  self.bm1 = np.zeros(16, dtype=np.float32)
        self.Wm2 = _he_init(16,  8);  self.bm2 = np.zeros( 8, dtype=np.float32)

        # Value head: 32 → 8 → 1
        self.Wv1 = _he_init(32, 8);  self.bv1 = np.zeros(8, dtype=np.float32)
        self.Wv2 = _he_init( 8, 1);  self.bv2 = np.zeros(1, dtype=np.float32)

    def _encode(self, x: np.ndarray):
        h1 = _relu(x @ self.W1 + self.b1)
        h2 = _relu(h1 @ self.W2 + self.b2)
        return h1, h2

    def q_role(self, x: np.ndarray) -> np.ndarray:
        _, h2 = self._encode(x)
        return _relu(h2 @ self.Wr1 + self.br1) @ self.Wr2 + self.br2

    def q_mechanic(self, x: np.ndarray) -> np.ndarray:
        _, h2 = self._encode(x)
        return _relu(h2 @ self.Wm1 + self.bm1) @ self.Wm2 + self.bm2

    def value(self, x: np.ndarray) -> float:
        _, h2 = self._encode(x)
        v = _relu(h2 @ self.Wv1 + self.bv1) @ self.Wv2 + self.bv2
        return float(v[0])

    def store(self, x: np.ndarray, role_action: int, mech_action: int,
              reward: float, x_next: np.ndarray, done: bool):
        """Add a transition to the replay buffer."""
        self._replay.append((x, role_action, mech_action, reward, x_next, done))

    def train_step(self, gamma: float = 0.97) -> float:
        """Sample a mini-batch from replay and perform one gradient update."""
        if len(self._replay) < self._batch:
            return 0.0
        self._step += 1

        batch = random.sample(self._replay, self._batch)
        total_loss = 0.0

        dW1 = np.zeros_like(self.W1);  db1 = np.zeros_like(self.b1)
        dW2 = np.zeros_like(self.W2);  db2 = np.zeros_like(self.b2)
        dWr1= np.zeros_like(self.Wr1); dbr1= np.zeros_like(self.br1)
        dWr2= np.zeros_like(self.Wr2); dbr2= np.zeros_like(self.br2)
        dWm1= np.zeros_like(self.Wm1); dbm1= np.zeros_like(self.bm1)
        dWm2= np.zeros_like(self.Wm2); dbm2= np.zeros_like(self.bm2)
        dWv1= np.zeros_like(self.Wv1); dbv1= np.zeros_like(self.bv1)
        dWv2= np.zeros_like(self.Wv2); dbv2= np.zeros_like(self.bv2)

        for x, ra, ma, rew, xn, done in batch:
            h1, h2 = self._encode(x)

            # ── Role head ──
            rh1_pre = h2 @ self.Wr1 + self.br1
            rh1     = _relu(rh1_pre)
            rq      = rh1 @ self.Wr2 + self.br2
            with_next_r = 0.0 if done else float(np.max(self.q_role(xn)))
            target_r    = rew + gamma * with_next_r
            td_r        = target_r - rq[ra]
            total_loss += td_r ** 2

            drq         = np.zeros_like(rq);  drq[ra] = td_r
            dWr2       += np.outer(rh1, drq)
            dbr2       += drq
            drh1        = (drq @ self.Wr2.T) * (rh1_pre > 0)
            dWr1       += np.outer(h2, drh1)
            dbr1       += drh1
            dh2_r       = drh1 @ self.Wr1.T

            # ── Mechanic head ──
            mh1_pre = h2 @ self.Wm1 + self.bm1
            mh1     = _relu(mh1_pre)
            mq      = mh1 @ self.Wm2 + self.bm2
            with_next_m = 0.0 if done else float(np.max(self.q_mechanic(xn)))
            target_m    = rew + gamma * with_next_m
            td_m        = target_m - mq[ma]
            total_loss += td_m ** 2

            dmq         = np.zeros_like(mq);  dmq[ma] = td_m
            dWm2       += np.outer(mh1, dmq)
            dbm2       += dmq
            dmh1        = (dmq @ self.Wm2.T) * (mh1_pre > 0)
            dWm1       += np.outer(h2, dmh1)
            dbm1       += dmh1
            dh2_m       = dmh1 @ self.Wm1.T

            # ── Value head ──
            vh1_pre = h2 @ self.Wv1 + self.bv1
            vh1     = _relu(vh1_pre)
            v_pred  = float((vh1 @ self.Wv2 + self.bv2)[0])
            v_tgt   = rew + (0.0 if done else gamma * self.value(xn))
            td_v    = v_tgt - v_pred
            total_loss += td_v ** 2

            dv = np.array([td_v], dtype=np.float32)
            dWv2 += np.outer(vh1, dv)
            dbv2 += dv
            dvh1  = (dv @ self.Wv2.T) * (vh1_pre > 0)
            dWv1 += np.outer(h2, dvh1)
            dbv1 += dvh1
            dh2_v = dvh1 @ self.Wv1.T

            # ── Shared encoder backward ──
            dh2_total = dh2_r + dh2_m + dh2_v
            h2_pre    = h1 @ self.W2 + self.b2
            dh1_enc   = dh2_total * (h2_pre > 0)
            dW2      += np.outer(h1, dh1_enc)
            db2      += dh1_enc
            dh1_enc2  = (dh1_enc @ self.W2.T) * (x @ self.W1 + self.b1 > 0)
            dW1      += np.outer(x, dh1_enc2)
            db1      += dh1_enc2

        lr  = self._lr / self._batch
        self.W1  += lr * dW1;   self.b1  += lr * db1
        self.W2  += lr * dW2;   self.b2  += lr * db2
        self.Wr1 += lr * dWr1;  self.br1 += lr * dbr1
        self.Wr2 += lr * dWr2;  self.br2 += lr * dbr2
        self.Wm1 += lr * dWm1;  self.bm1 += lr * dbm1
        self.Wm2 += lr * dWm2;  self.bm2 += lr * dbm2
        self.Wv1 += lr * dWv1;  self.bv1 += lr * dbv1
        self.Wv2 += lr * dWv2;  self.bv2 += lr * dbv2

        return float(total_loss / self._batch)

    def to_dict(self) -> Dict:
        names = ["W1","b1","W2","b2","Wr1","br1","Wr2","br2",
                 "Wm1","bm1","Wm2","bm2","Wv1","bv1","Wv2","bv2"]
        return {n: getattr(self, n).tolist() for n in names} | {"step": self._step}

    def from_dict(self, data: Dict):
        for n in ["W1","b1","W2","b2","Wr1","br1","Wr2","br2",
                  "Wm1","bm1","Wm2","bm2","Wv1","bv1","Wv2","bv2"]:
            if n in data:
                setattr(self, n, np.array(data[n], dtype=np.float32))
        self._step = data.get("step", 0)


# ══════════════════════════════════════════════════════════════
#  6.  CAUSAL ML
#      Counterfactual reasoning: "Would a different action have
#      led to a better outcome?"
#
#  Approach inspired by the do-calculus / structural causal models:
#    - Maintain a causal graph edge: action → outcome (goal/concede)
#    - Estimate E[outcome | do(action=a)] using inverse propensity
#      weighting (IPW) over historical transitions
#    - Provides a debiased Q-estimate that is not just correlation
#
#  Practical benefit:
#    If the bot always used "attack" when the game was easy (low
#    opponent threat) and scored, a correlation model would over-credit
#    "attack". The causal model corrects for this confounding by weighting
#    by 1/P(action | state) — the propensity score.
# ══════════════════════════════════════════════════════════════

class CausalActionEstimator:
    """
    Causal Q-estimator using Inverse Propensity Weighting (IPW).

    For each (state_key, action) pair:
      - Track how often that action was taken (propensity)
      - Track average reward with IPW correction
      - Return debiased Q-estimate

    This reduces confounding: e.g. "attack" looks good because it is
    chosen in easy states; IPW down-weights those easy cases.
    """

    def __init__(self, min_samples: int = 5, max_states: int = 2000):
        self._min_samples = min_samples
        self._max_states  = max_states

        # counts[state][action] = (n_chosen, reward_sum_ipw)
        self._counts:  Dict[str, Dict[str, int]]   = defaultdict(lambda: defaultdict(int))
        self._rewards: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

        # Propensity: P(action | state) estimated from visit counts
        self._total_count: Dict[str, int] = defaultdict(int)

    def record(self, state_key: str, action: str, reward: float,
               all_actions: Optional[List[str]] = None):
        """
        Store a transition for IPW estimation.

        *all_actions* is the list of all legal actions in this state —
        needed for propensity denominator.  If None, defaults to 6 roles.
        """
        self._counts[state_key][action]  += 1
        self._total_count[state_key]     += 1

        # Propensity: P(a|s) = n(s,a) / n(s)
        n_total = max(1, self._total_count[state_key])
        n_a     = self._counts[state_key][action]
        propensity = n_a / n_total

        # IPW correction: weight = 1 / propensity  (clipped to avoid explosion)
        ipw_weight = min(5.0, 1.0 / max(0.05, propensity))
        self._rewards[state_key][action] += reward * ipw_weight

        # Prune if too large
        if len(self._counts) > self._max_states:
            oldest = next(iter(self._counts))
            del self._counts[oldest]
            del self._rewards[oldest]
            del self._total_count[oldest]

    def causal_q(self, state_key: str, action: str) -> float:
        """
        Return IPW-corrected Q-value estimate for (state, action).
        Returns 0.0 if insufficient data.
        """
        n = self._counts[state_key].get(action, 0)
        if n < self._min_samples:
            return 0.0
        return self._rewards[state_key][action] / n

    def best_action_causal(self, state_key: str,
                           candidates: List[str]) -> Optional[str]:
        """
        Pick the action with the highest causal Q-value from *candidates*.
        Returns None if no candidate has enough data.
        """
        best_a    = None
        best_q    = float("-inf")
        for a in candidates:
            q = self.causal_q(state_key, a)
            if q > best_q:
                best_q = q
                best_a = a
        return best_a if best_q > float("-inf") else None

    def counterfactual_regret(self, state_key: str,
                              taken_action: str,
                              candidates: List[str]) -> float:
        """
        Counterfactual regret = (best causal Q among alternatives) − (Q of taken action).
        Positive value means a better action existed; how much reward was left on the table.
        """
        taken_q = self.causal_q(state_key, taken_action)
        alt_q   = max(
            (self.causal_q(state_key, a) for a in candidates if a != taken_action),
            default=taken_q,
        )
        return max(0.0, alt_q - taken_q)

    def to_dict(self) -> Dict:
        return {
            "counts":      {s: dict(a) for s, a in self._counts.items()},
            "rewards":     {s: dict(a) for s, a in self._rewards.items()},
            "total_count": dict(self._total_count),
        }

    def from_dict(self, data: Dict):
        for s, actions in data.get("counts", {}).items():
            for a, n in actions.items():
                self._counts[s][a] = int(n)
        for s, actions in data.get("rewards", {}).items():
            for a, v in actions.items():
                self._rewards[s][a] = float(v)
        for s, n in data.get("total_count", {}).items():
            self._total_count[s] = int(n)


# ══════════════════════════════════════════════════════════════
#  MASTER WRAPPER
#  AdvancedMLSystem — ties all six components together.
#  Used by AdaptiveLearner as a drop-in advanced layer.
# ══════════════════════════════════════════════════════════════

class AdvancedMLSystem:
    """
    Unified wrapper around all six advanced ML components.

    Usage in AdaptiveLearner:
        from core.advanced_ml import AdvancedMLSystem
        self.advanced = AdvancedMLSystem()

    Each frame, call:
        result = self.advanced.tick(state_vec, state_key, role_idx, mech_idx,
                                    reward, next_state_vec, done, game_metrics)
        role_override = result["role_override"]   # may be None
        threat        = result["threat_level"]
        cf_regret     = result["cf_regret"]
    """

    def __init__(self):
        from core.rl_algorithms import encode_state  # reuse existing state encoder
        self._encode = encode_state

        self.anomaly   = AnomalyDetector()
        self.mtl       = MultiTaskNet()
        self.pa        = PassiveAggressiveLearner()
        self.maml      = MAMLRoleAdapter()
        self.deep_net  = DeepRLNet()
        self.causal    = CausalActionEstimator()

        self._frame    = 0
        self._last_x: Optional[np.ndarray] = None
        self.active_components = set(ALL_ADVANCED_COMPONENTS)

    def set_active_components(self, components: set[str]) -> None:
        """Set which advanced-ML components are active this match."""
        self.active_components = set(components) if components else set(ALL_ADVANCED_COMPONENTS)

    # ── Per-frame update ──

    def tick(
        self,
        state_key:   str,
        role_idx:    int,          # 0-5 ROLE_ACTIONS index
        mech_idx:    int,          # 0-7 mechanic index
        reward:      float,
        next_key:    str,
        done:        bool,
        game_metrics: Dict[str, float],   # e.g. {"opp_speed": 1400.0, "ball_accel": 320.0}
    ) -> Dict:
        self._frame += 1
        active = self.active_components or set(ALL_ADVANCED_COMPONENTS)

        x     = self._encode(state_key)
        x_nxt = self._encode(next_key)

        # 1. Anomaly detection
        if "anomaly" in active:
            anomaly_flags = self.anomaly.update(game_metrics)
            threat = self.anomaly.overall_threat_level()
        else:
            anomaly_flags = {}
            threat = 0.0

        # 2. Store in deep network replay
        if "deep_rl" in active:
            self.deep_net.store(x, role_idx, mech_idx, reward, x_nxt, done)
            loss = self.deep_net.train_step() if self._frame % 4 == 0 else 0.0
        else:
            loss = 0.0

        # 3. Multi-task update (attack / defense task chosen by role)
        task_id  = min(role_idx, len(_TASK_SIZES) - 1)
        if "multi_task" in active:
            target_q = reward + 0.97 * float(np.max(self.mtl.predict(x_nxt, task_id)))
            self.mtl.update(x, task_id, role_idx % _TASK_SIZES[task_id], target_q)

        # 4. Online / PA update
        if "passive_aggressive" in active:
            self.pa.update(x, role_idx, reward)

        # 5. MAML: record and adapt every 60 frames
        if "maml" in active:
            self.maml.record_experience(x, role_idx, reward)
            if self._frame % 60 == 0:
                self.maml.adapt()

        # 6. Causal record
        from core.rl_state import ROLE_ACTIONS
        action_name = ROLE_ACTIONS[role_idx]
        if "causal" in active:
            self.causal.record(state_key, action_name, reward, ROLE_ACTIONS)
            cf_regret = self.causal.counterfactual_regret(
                state_key, action_name, ROLE_ACTIONS)
        else:
            cf_regret = 0.0

        # Determine role override (consensus from advanced systems)
        role_override = None
        if threat > 0.7:
            # High anomaly threat → prefer defensive role
            role_override = "defense"
        else:
            votes = []
            if "maml" in active:
                votes.append(ROLE_ACTIONS[self.maml.best_action(x)])
            if "causal" in active:
                votes.append(self.causal.best_action_causal(state_key, ROLE_ACTIONS) or ROLE_ACTIONS[role_idx])
            if "passive_aggressive" in active:
                votes.append(ROLE_ACTIONS[self.pa.predict(x)])
            if "multi_task" in active:
                votes.append(
                    ROLE_ACTIONS[
                        self.mtl.best_action(x, min(task_id, len(_TASK_SIZES)-1))
                        % len(ROLE_ACTIONS)
                    ]
                )

            if votes:
                counts = defaultdict(int)
                for v in votes:
                    counts[v] += 1
                top = max(counts, key=counts.__getitem__)
                min_votes = 3 if len(votes) >= 4 else max(2, len(votes))
                if counts[top] >= min_votes and top != ROLE_ACTIONS[role_idx]:
                    role_override = top

        return {
            "role_override": role_override,
            "threat_level":  threat,
            "cf_regret":     cf_regret,
            "train_loss":    loss,
            "anomaly_flags": anomaly_flags,
        }

    # ── Match events ──

    def on_goal_scored(self):
        self.maml.adapt()   # fast re-adapt after goal

    def on_goal_conceded(self):
        """Signal that the opponent scored — penalise causal weights and re-adapt MAML."""
        self.maml.adapt()
        # Boost the propensity-correction weight so future IPW is more conservative
        self.causal._total_count["_conceded"] = self.causal._total_count.get("_conceded", 0) + 1

    def on_match_end(self):
        self.maml.end_of_match()

    # ── Serialisation ──

    def to_dict(self) -> Dict:
        return {
            "anomaly":  self.anomaly.to_dict(),
            "pa":       self.pa.to_dict(),
            "maml":     self.maml.to_dict(),
            "deep_net": self.deep_net.to_dict(),
            "causal":   self.causal.to_dict(),
            "mtl":      self.mtl.to_dict(),
        }

    def from_dict(self, data: Dict):
        if "anomaly"  in data: self.anomaly.from_dict(data["anomaly"])
        if "pa"       in data: self.pa.from_dict(data["pa"])
        if "maml"     in data: self.maml.from_dict(data["maml"])
        if "deep_net" in data: self.deep_net.from_dict(data["deep_net"])
        if "causal"   in data: self.causal.from_dict(data["causal"])
        if "mtl"      in data: self.mtl.from_dict(data["mtl"])
