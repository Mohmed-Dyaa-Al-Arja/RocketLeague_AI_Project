# Rocket League AI Project (Search Algorithms + Reinforcement Learning)

Academic project implementing an intelligent Rocket League bot using a hybrid AI architecture that combines **classical search algorithms**, **reinforcement learning**, and a **dynamic strategy system**.

The system is designed as a modular AI research project demonstrating decision-making, path planning, and adaptive learning inside the Rocket League environment.

---

# Algorithms Used

## Classical Search Algorithms

| Algorithm             | Evaluation Function             | Type        |
| --------------------- | ------------------------------- | ----------- |
| **A***                | f(n) = g(n) + h(n)              | Informed    |
| **BFS**               | Level-based (FIFO)              | Uninformed  |
| **UCS**               | f(n) = g(n)                     | Uninformed  |
| **Greedy Best First** | f(n) = h(n)                     | Informed    |
| **DFS**               | Depth-based (LIFO stack)        | Uninformed  |
| **Beam Search**       | Limited-width heuristic search  | Informed    |
| **IDA***              | Iterative Deepening A*          | Informed    |
| **Decision Tree**     | State-based decision evaluation | AI Planning |

---

## Reinforcement Learning Algorithms

| Algorithm           | Purpose                      |
| ------------------- | ---------------------------- |
| **Q-Learning**      | Role and action selection    |
| **SARSA**           | Opponent behavior modeling   |
| **DQN**             | Deep value estimation        |
| **PPO**             | Stable policy optimization   |
| **A2C**             | Actor-Critic policy learning |
| **Monte Carlo**     | Episode-based learning       |
| **Model-Based RL**  | Environment modeling         |
| **Policy Gradient** | Learning from demonstrations |
| **Actor-Critic**    | Control refinement           |
| **Ensemble Voting** | Combining algorithm outputs  |
| **Online Learning** | Continuous adaptation        |

---

# Project Architecture

```
RocketLeague_AI_Project/
│
├── core/
│   ├── all_algorithms.py
│   ├── rl_algorithms.py
│   ├── adaptive_learner.py
│   ├── reward_calculator.py
│   ├── rl_state.py
│
├── navigation/
│   └── path_planner.py
│
├── game_logic/
│   ├── decision_engine.py
│   ├── strategy_manager.py
│   └── mode_manager.py
│
├── visualization/
│   └── overlay_renderer.py
│
├── gui/
│   └── control_panel.py
│
├── model/
│   ├── search_knowledge.json
│   └── rl_model_data.json
│
├── runtime/
│   ├── run_match.py
│   ├── launcher.py
│   ├── rl_ai_bot.cfg
│   └── rl_ai_looks.cfg
│
├── documentation/
│   ├── algorithm_explanation.txt
│   ├── ai_architecture.md
│   └── reward_system.md
│
├── start_ai.bat
├── requirements.txt
└── README.md
```

---

# Core AI Systems

## Decision Engine

The **Decision Engine** is responsible for selecting actions and algorithms based on the current game state.

Decision pipeline:

```
Game State Detection
      ↓
Strategy Selection
      ↓
Algorithm Selection
      ↓
Path Planning
      ↓
Action Execution
      ↓
Reward Evaluation
      ↓
Learning Update
```

---

## Game State System

The bot detects multiple match states to improve decision accuracy.

Examples:

Attack State
Defense State
Counter Attack State
Possession State
Kickoff State
Recovery State
Demo Attack State
Goal Defense State
Goal Attack State
Boost Collection State
Aerial Attack State
Wall Play State
Midfield Control State

The system supports **30+ professional game states**.

---

## Reward System

The reinforcement learning system uses a detailed reward structure.

### Positive Rewards

| Action                    | Reward |
| ------------------------- | ------ |
| Score Goal                | +100   |
| Assist Goal               | +50    |
| Shot on Target            | +20    |
| Successful Pass           | +10    |
| Save Goal                 | +40    |
| Clear Ball                | +25    |
| Approach Ball             | +5     |
| Move Toward Opponent Goal | +5     |

### Demo Rewards

| Action                            | Reward |
| --------------------------------- | ------ |
| Destroy opponent with boost speed | +15    |
| Destroy opponent and gain ball    | +25    |
| Destroy defender near enemy goal  | +30    |

### Negative Rewards

| Behavior                | Penalty |
| ----------------------- | ------- |
| Idle too long           | -5      |
| Camping inside own goal | -10     |
| Moving away from ball   | -3      |
| Wasting boost           | -2      |

### Ball Direction Rewards

Ball moving toward opponent goal → +3
Ball moving toward own goal → -3

The system supports **100+ reward signals**.

---

# Strategy System

The bot includes an **AI Strategy Layer** that dynamically changes playstyle during the match.

Strategies include:

Aggressive Attack Mode
Defensive Mode
Counter Attack Mode
Possession Control Mode
Demo Strategy Mode
Boost Control Mode
Balanced Strategy

The strategy manager analyzes:

* ball position
* score difference
* time remaining
* opponent behavior
* boost availability

and automatically switches strategy during the match.

---

# Search-Based Path Planning

The bot performs pathfinding using classical search algorithms.

Features:

2D grid discretization of the field
8-directional movement
Euclidean movement cost
Heuristic evaluation

Algorithms are selected dynamically depending on the situation.

---

# In-Game Visual Overlay

The debugging overlay shows:

Green box → player car
Blue box → ball
Red box → opponent goal
Purple zone → defensive area

Yellow path line → current search path

Additional visualization:

algorithm currently active
ball prediction trajectory
decision engine output

---

# Real-Time Information Panel (HUD)

Displays:

current algorithm
algorithm usage statistics
current strategy
game state
path cost

---

# Keyboard Controls

| Key | Function                                     |
| --- | -------------------------------------------- |
| M   | Manual mode (player control, AI observation) |
| N   | Balanced mode                                |
| B   | Attack mode                                  |
| V   | Defense mode                                 |
| P   | Temporary reset (session-only learning)      |

---

# Persistent Knowledge System

Stored in:

```
model/search_knowledge.json
model/rl_model_data.json
```

Stores:

successful navigation paths
goal scoring patterns
algorithm performance statistics
learned strategy improvements

The knowledge base evolves over time.

---

# Quick Start

## One-Click Launch

```
start_ai.bat
```

---

## Manual Launch

```
python runtime/launcher.py
```

---

## Direct Match

```
python runtime/run_match.py --mode 1v1
python runtime/run_match.py --mode 2v2
python runtime/run_match.py --mode 3v3 --opponent-skill 0.8
```

---

# Docker (Development / Tests)

This project includes a lightweight Docker setup for running the full
Python test suite and validating that all algorithms and modules load
correctly. It is intended for development/CI, not for running the actual
Rocket League game client inside Docker.

Basic usage (from the project root):

```
docker build -t rl-ai .
docker run --rm -it rl-ai python tests/test_full_project.py
```

Or with Docker Compose:

```
docker compose up --build
```

For a detailed overview of all algorithms and the Docker setup, see:

```
PROJECT_DOCKER_AND_ALGORITHMS_OVERVIEW.txt
```

---

# Requirements

```
rlbot
websockets<14
numpy
```

---

# Notes

* Temporary learning mode (`P`) does not overwrite stored knowledge
* The bot periodically saves knowledge during runtime
* All AI modules are fully modular and extensible
* Designed for academic AI experimentation and research
