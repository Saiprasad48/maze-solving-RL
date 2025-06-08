import numpy as np
import random

class MonteCarloAgent:
    def __init__(self, state_space, action_space, episodes, curiosity):
        self.q_table = np.zeros((state_space, action_space))
        self.returns = {}
        self.visit_counts = np.zeros(state_space)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.curiosity_factor = 0.1 if curiosity else 0
        self.episodes = episodes
        self.recently_visited = set()
        self.loop_penalty_threshold = 5

    def get_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, 3)
        else:
            return np.argmax(self.q_table[state])

    def update(self, episode_data, episode):
        G = 0
        visited_state_actions = set()
        for t in reversed(range(len(episode_data))):
            state, action, reward = episode_data[t]
            # Check for loop penalty
            if state in self.recently_visited:
                reward -= 0.5 # Apply a penalty for revisiting
            self.recently_visited.add(state)
            if len(self.recently_visited) > self.loop_penalty_threshold:
                self.recently_visited.pop() # Remove the oldest state
            G = self.gamma*G+reward
            if (state, action) not in visited_state_actions:
                visited_state_actions.add((state, action))
                if (state, action) not in self.returns:
                    self.returns[(state, action)] = []
                self.returns[(state, action)].append(G)
                intrinsic_reward = self.curiosity_factor / np.sqrt(self.visit_counts[state] + 1)
                self.q_table[state, action] = np.mean(self.returns[(state, action)]) + intrinsic_reward
                self.visit_counts[state] += 1
        self.epsilon = max(self.epsilon_min, 1 - (episode / self.episodes))

    def train_monte_carlo(env, episodes, curiosity):
        agent = MonteCarloAgent(env.observation_space.n, env.action_space.n, episodes, curiosity)
        rewards_history = []
        path_lengths = []
        steps_to_goal = []
        for episode in range(episodes):
            agent.recently_visited.clear() # Clear the set at the start of each episode
            state, _ = env.reset()
            episode_data = []
            total_reward = 0
            done = False
            steps = 0
            while not done:
                action = agent.get_action(state)
                next_state, reward, done, _, _ = env.step(action)
                episode_data.append((state, action, reward))
                state = next_state
                total_reward += reward
                steps += 1
            agent.update(episode_data, episode)
            rewards_history.append(total_reward)
            path_lengths.append(steps)
            steps_to_goal.append(steps if done else -1) # Record steps if goal reached
            if episode % 1000 == 0:
                print(f"Monte Carlo Episode {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")
        return rewards_history, path_lengths, steps_to_goal, agent

def train_monte_carlo(env, episodes, curiosity):
    return MonteCarloAgent.train_monte_carlo(env, episodes, curiosity)