import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MazeEnv(gym.Env):
    def __init__(self):
        super(MazeEnv, self).__init__()
        self.height = 10
        self.width = 10
        self.action_space = spaces.Discrete(4) #Up, Down, Left, Right
        self.observation_space = spaces.Discrete(self.height * self.width)
        #maze layout (0: empty, 1: wall, 2: goal)
        self.maze = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 2]
        ])
        self.start_pos = (0, 0)
        self.current_pos = self.start_pos
        self.goal_pos = (9, 9)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_pos = self.start_pos
        return self._get_obs(), {}

    def step(self, action):
        x, y = self.current_pos
        if action == 0: # Up
            x = max(0, x - 1)
        elif action == 1: # Down
            x = min(self.height - 1, x + 1)
        elif action == 2: # Left
            y = max(0, y - 1)
        elif action == 3: # Right
            y = min(self.width - 1, y + 1)
        if self.maze[x, y] == 1: # If it's a wall
            reward = -1.0 # Penalty for hitting a wall
            self.current_pos = (x, y) # Stay at the same position
        else:
            self.current_pos = (x, y)
            if self.maze[x, y] == 2: # If it's the goal
                reward = 10.0 # High reward for reaching the goal
            elif self.is_difficult_path(x, y): # Reward for difficult paths
                reward = -0.05 # Less penalty for difficult paths
            else:
                reward = -0.1 # Small penalty for each step
        done = self.maze[x, y] == 2 # if goal reached
        return self._get_obs(), reward, done, False, {}

    def is_difficult_path(self, x, y):
        # Example: Consider paths near walls as difficult
        if (x == 1 or x == self.height - 2) or (y == 1 or y == self.width - 2):
            return True
        return False

    def _get_obs(self):
        return self.current_pos[0] * self.width + self.current_pos[1]