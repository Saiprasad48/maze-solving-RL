from maze_env import MazeEnv
from q_learning import train_q_learning
from monte_carlo import train_monte_carlo
import matplotlib.pyplot as plt
import numpy as np

def plot_rewards(q_rewards_with_curiosity, q_rewards_without_curiosity, mc_rewards_with_curiosity, mc_rewards_without_curiosity):
    plt.figure(figsize=(12, 6))
    plt.plot(q_rewards_with_curiosity, label='Q-Learning with Curiosity')
    plt.plot(q_rewards_without_curiosity, label='Q-Learning without Curiosity')
    plt.plot(mc_rewards_with_curiosity, label='Monte Carlo with Curiosity')
    plt.plot(mc_rewards_without_curiosity, label='Monte Carlo without Curiosity')
    plt.title('Rewards over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.savefig('rewards_comparison.png')
    plt.close()

def plot_path_lengths(q_path_lengths_with_curiosity, q_path_lengths_without_curiosity, mc_path_lengths_with_curiosity, mc_path_lengths_without_curiosity):
    plt.figure(figsize=(12, 6))
    plt.plot(q_path_lengths_with_curiosity, label='Q-Learning with Curiosity')
    plt.plot(q_path_lengths_without_curiosity, label='Q-Learning without Curiosity')
    plt.plot(mc_path_lengths_with_curiosity, label='Monte Carlo with Curiosity')
    plt.plot(mc_path_lengths_without_curiosity, label='Monte Carlo without Curiosity')
    plt.title('Path Lengths over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Path Length')
    plt.legend()
    plt.savefig('path_lengths_comparison.png')
    plt.close()

def plot_steps_to_goal(q_steps_to_goal_with_curiosity, q_steps_to_goal_without_curiosity, mc_steps_to_goal_with_curiosity, mc_steps_to_goal_without_curiosity):
    plt.figure(figsize=(12, 6))
    plt.plot([step for step in q_steps_to_goal_with_curiosity if step != -1], label='Q-Learning with Curiosity')
    plt.plot([step for step in q_steps_to_goal_without_curiosity if step != -1], label='Q-Learning without Curiosity')
    plt.plot([step for step in mc_steps_to_goal_with_curiosity if step != -1], label='Monte Carlo with Curiosity')
    plt.plot([step for step in mc_steps_to_goal_without_curiosity if step != -1], label='Monte Carlo without Curiosity')
    plt.title('Steps to Goal over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Steps to Goal')
    plt.legend()
    plt.savefig('steps_to_goal_comparison.png')
    plt.close()

def plot_optimal_algorithm(q_rewards_with_curiosity, q_rewards_without_curiosity, mc_rewards_with_curiosity, mc_rewards_without_curiosity):
    cumulative_rewards = {
        'Q-Learning with Curiosity': np.cumsum(q_rewards_with_curiosity),
        'Q-Learning without Curiosity': np.cumsum(q_rewards_without_curiosity),
        'Monte Carlo with Curiosity': np.cumsum(mc_rewards_with_curiosity),
        'Monte Carlo without Curiosity': np.cumsum(mc_rewards_without_curiosity)
    }
    plt.figure(figsize=(12, 6))
    for label, rewards in cumulative_rewards.items():
        plt.plot(rewards, label=label)
    plt.title('Cumulative Rewards over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.legend()
    plt.savefig('optimal_algorithm_comparison.png')
    plt.close()

def plot_policy(env, agent, title):
    policy = np.argmax(agent.q_table, axis=1).reshape(env.height, env.width)
    plt.figure(figsize=(10, 10))
    plt.imshow(policy, cmap='cool')
    cbar = plt.colorbar(ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(['Up', 'Down', 'Left', 'Right'])
    plt.title(title)
    for i in range(env.height):
        for j in range(env.width):
            plt.text(j, i, ['↑', '↓', '←', '→'][policy[i, j]], ha='center', va='center')
    plt.savefig(f'{title.lower().replace(" ", "_")}_policy.png')
    plt.close()

def calculate_goal_probability_percentage(steps_to_goal):
    successful_episodes = sum(1 for step in steps_to_goal if step != -1)
    probability = (successful_episodes / len(steps_to_goal)) * 100
    return probability

def main():
    env = MazeEnv()
    episodes = 20000

    # Train Q-Learning with and without curiosity
    q_rewards_with_curiosity, q_path_lengths_with_curiosity, q_steps_to_goal_with_curiosity, q_agent_with_curiosity = train_q_learning(env, episodes, curiosity=True)
    q_rewards_without_curiosity, q_path_lengths_without_curiosity, q_steps_to_goal_without_curiosity, q_agent_without_curiosity = train_q_learning(env, episodes, curiosity=False)

    # Train Monte Carlo with and without curiosity
    mc_rewards_with_curiosity, mc_path_lengths_with_curiosity, mc_steps_to_goal_with_curiosity, mc_agent_with_curiosity = train_monte_carlo(env, episodes, curiosity=True)
    mc_rewards_without_curiosity, mc_path_lengths_without_curiosity, mc_steps_to_goal_without_curiosity, mc_agent_without_curiosity = train_monte_carlo(env, episodes, curiosity=False)

    # Plot rewards comparison
    plot_rewards(q_rewards_with_curiosity, q_rewards_without_curiosity, mc_rewards_with_curiosity, mc_rewards_without_curiosity)

    # Plot path lengths comparison
    plot_path_lengths(q_path_lengths_with_curiosity, q_path_lengths_without_curiosity, mc_path_lengths_with_curiosity, mc_path_lengths_without_curiosity)

    # Plot steps to goal comparison
    plot_steps_to_goal(q_steps_to_goal_with_curiosity, q_steps_to_goal_without_curiosity, mc_steps_to_goal_with_curiosity, mc_steps_to_goal_without_curiosity)

    # Plot optimal algorithm comparison
    plot_optimal_algorithm(q_rewards_with_curiosity, q_rewards_without_curiosity, mc_rewards_with_curiosity, mc_rewards_without_curiosity)

    # Plot policies
    plot_policy(env, q_agent_with_curiosity, "Q-Learning Policy with Curiosity")
    plot_policy(env, q_agent_without_curiosity, "Q-Learning Policy without Curiosity")
    plot_policy(env, mc_agent_with_curiosity, "Monte Carlo Policy with Curiosity")
    plot_policy(env, mc_agent_without_curiosity, "Monte Carlo Policy without Curiosity")

    # Calculate and print goal probabilities as percentages
    q_prob_with_curiosity = calculate_goal_probability_percentage(q_steps_to_goal_with_curiosity)
    q_prob_without_curiosity = calculate_goal_probability_percentage(q_steps_to_goal_without_curiosity)
    mc_prob_with_curiosity = calculate_goal_probability_percentage(mc_steps_to_goal_with_curiosity)
    mc_prob_without_curiosity = calculate_goal_probability_percentage(mc_steps_to_goal_without_curiosity)

    print(f"Q-Learning with Curiosity Goal Probability: {q_prob_with_curiosity}%")
    print(f"Q-Learning without Curiosity Goal Probability: {q_prob_without_curiosity}%")
    print(f"Monte Carlo with Curiosity Goal Probability: {mc_prob_with_curiosity}%")
    print(f"Monte Carlo without Curiosity Goal Probability: {mc_prob_without_curiosity}%")
    print("Training complete. Graphs saved as PNG files.")

if __name__ == "__main__":
    main()