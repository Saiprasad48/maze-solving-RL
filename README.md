# Maze Solving RL

## Overview
This project explores reinforcement learning (RL) algorithms for solving maze problems, comparing Q-learning, Monte Carlo, SARSA, and Dynamic Programming. A key novelty is the introduction of a curiosity factor in Q-learning and Monte Carlo to enhance exploration efficiency. The project was conducted by Sai Prasad Reddy Kukudala and Sandeep Ramesh.

### Objectives
- Compare Q-learning and Monte Carlo on a 10x10 grid, with and without a curiosity factor.
- Compare SARSA and Dynamic Programming on a 6x6 grid.
- Evaluate the impact of curiosity-driven exploration on RL performance.

### Key Findings
- Curiosity-driven exploration significantly improves convergence speed and path efficiency.
- Q-learning with curiosity outperforms other methods in terms of cumulative rewards and steps to goal.
- SARSA benefits from higher learning rates, and Dynamic Programming converges reliably with value iteration.

## Project Structure
```
Maze_Solving_RL/
├── documents/
│   └── Maze_Solving_RL_Report.pdf  # Project report with detailed methodology and results
├── src/
│   ├── main.py                            # Main script to run experiments
│   ├── maze_env.py                        # Maze environment setup using Gymnasium
│   ├── maze_sarsa_and_dynamic_programming.py  # SARSA and Dynamic Programming implementation
│   ├── monte_carlo.py                     # Monte Carlo algorithm implementation
│   ├── q_learning.py                      # Q-learning algorithm implementation
│   ├── sarsa_average_q_value_heatmap_1.py # Heatmap visualization for SARSA Q-values (variant 1)
│   ├── sarsa_average_q_value_heatmap_2.py # Heatmap visualization for SARSA Q-values (variant 2)
│   ├── sarsa_average_q_value_heatmap_3.py # Heatmap visualization for SARSA Q-values (variant 3)
│   ├── v_star_values_heatmap.py           # Heatmap for Dynamic Programming V* values
│   └── Maze_Solving_RL.py                 # Core RL logic and utilities
├── Results(Graphs, Plots)/
│   ├── dynamic_programming_heatmap.png    # Heatmap for Dynamic Programming
│   ├── heatmap_sarsa_500.png             # SARSA heatmap (500 episodes)
│   ├── heatmap_sarsa_2000.png            # SARSA heatmap (2000 episodes)
│   ├── heatmap_sarsa_2000(alpha=1).png   # SARSA heatmap (2000 episodes, alpha=1)
│   ├── monte_carlo_policy_with_curiosity.png  # Monte Carlo policy with curiosity
│   ├── monte_carlo_policy_without_curiosity.png  # Monte Carlo policy without curiosity
│   ├── optimal_algorithm_comparison.png   # Comparison of optimal algorithms
│   ├── optimal_path_sarsa.png             # Optimal path learned by SARSA
│   ├── path_lengths_comparison.png        # Comparison of path lengths
│   ├── q_values_goal_convergence.png      # Q-values convergence at goal state
│   ├── q_values_start_convergence.png     # Q-values convergence at start state
│   ├── q_learning_policy_with_curiosity.png   # Q-learning policy with curiosity
│   ├── q_learning_policy_without_curiosity.png  # Q-learning policy without curiosity
│   ├── rewards_comparison.png             # Cumulative rewards comparison
│   ├── steps_to_goal_comparison.png       # Steps to goal over episodes
│   └── value_iteration_convergence.png    # Value iteration convergence for Dynamic Programming
├── .gitignore                            # Ignore Python build files
└── README.md                             # Project documentation
```

## Setup Instructions
1. **Clone the Repository**:
   ```
   git clone https://github.com/Saiprasad48/Maze_Solving_RL.git
   ```
2. **Install Dependencies**:
   Ensure you have Python 3.x installed, then install the required packages:
   ```
   pip install gymnasium numpy matplotlib
   ```
3. **Run Experiments**:
   Navigate to the `src` folder and run the main script:
   ```
   cd src
   python main.py
   ```
4. **View Results**:
   - Check the generated plots in `Results(Graphs, Plots)/`.
   - Read the detailed report in `documents/Maze_Solving_RL_Report.pdf`.

## Dependencies
- Python 3.x
- Gymnasium
- NumPy
- Matplotlib

## Contributors
- **Sai Prasad Reddy Kukudala**: Implemented Q-learning and Monte Carlo, conducted experiments for hypotheses 1 and 3, reviewed literature on curiosity-driven exploration.
- **Sandeep Ramesh**: Implemented SARSA and Dynamic Programming, conducted experiments for hypothesis 2, reviewed literature on SARSA and value iteration.

## Future Work
- Test RL algorithms on larger mazes.
- Incorporate deep RL techniques.
- Compare RL methods with traditional search algorithms like BFS/DFS.
- Experiment with varying discount factors (gamma) to optimize rewards and convergence.

## References
Refer to the `Maze_Solving_RL_Report.pdf` for a full list of references, including works by Pathak et al. (2017) on curiosity-driven exploration and Zou et al. (2019) on SARSA with linear function approximation.