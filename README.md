# Exp3 and Exp4 Algorithms for Contextual Bandit Problems

This project implements the Exp3 and Exp4 algorithms, designed to address exploration and exploitation challenges in a grid-based environment with dynamic target distributions. The framework utilizes multiple contexts and experts, allowing efficient adaptation to non-stationary environments, where rewards are influenced by target movements.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dependencies](#dependencies)
3. [File Descriptions](#file-descriptions)
4. [How to Run](#how-to-run)
5. [Configurations](#configurations)
6. [Results](#results)

## Project Overview

This project implements:

- **Exp3 Algorithm**: A classic bandit algorithm used for balancing exploration and exploitation. It handles grid-based environments where an agent takes actions, receives rewards, and updates its weights based on the observed feedback.
- **Exp4 Algorithm**: Integrates predictions from multiple experts to decide the agent's next action.

The environment consists of a 6x6 grid, where biases (rewards) associated with each cell are updated dynamically. The agents attempt to minimizing regret with adversarial target distributions.

## Dependencies

You can install these dependencies via pip:

```bash
pip install numpy
```

## File Descriptions

### 1. `exp3.py`

Implements the **Exp3 algorithm**. The agent navigates through a grid, selecting actions based on a probability distribution over available actions. Weights are updated using a reward feedback mechanism.

**Key Functions:**

- `exp3`: Executes a single step of the Exp3 algorithm.
- `get_reward`: Retrieves the reward for the agent’s current position.
- `calculate_regret`: Computes the weak regret and regret bound.

### 2. `exp4.py`

Implements the **Exp4 algorithm**, where actions are selected based on predictions from multiple experts. Each expert uses the Exp3 algorithm with different exploration parameters.

**Key Functions:**

- `get_experts_prediction`: Combines predictions from experts.
- `select_action`: Chooses an action based on expert predictions.
- `update`: Updates weights and cumulative rewards.

### 3. `context_exp3.py`

Handles multiple contexts for the Exp3 algorithm. The grid is subdivided into different contexts, each of which is assigned an instance of the Exp3 algorithm. Contexts are dynamically updated based on the agent's location.

**Key Functions:**

- `run`: Executes the Exp3 algorithm across contexts.
- `get_context`: Determines the context based on the agent’s position.

### 4. `PRGM.py`

Contains the logic for updating the bias values, simulating target movement patterns in the grid environment.

## How to Run

To run the **Exp4** algorithm, execute the following:

```bash
python exp4.py
```

For the **Exp3** algorithm, use:

```bash
python exp3.py
```

Alternatively, you can run the Context Exp3 test using:

```bash
python context_exp3.py
```

## Configurations

You can modify key parameters such as:

- **Number of actions (`num_actions`)**
- **Number of experts (`num_experts`)**
- **Exploration rate (`gamma`)**
- **Number of contexts (`num_contexts`)**

These parameters are configurable within the respective script files (`exp3.py`, `exp4.py`, `context_exp3.py`).

## Results

Logs for regret and weights are stored in the `results` directory, with separate files for each experiment:
