# import the numbers python documentation into the python code.
# Import random number generator as well.
import numpy as np
import random

# class function that makes the maze and performs the sarsa algorithm.
class Maze:

    # First initialize the maze with the grid defined from outside the definition functions.
    # This is the object constructor.
    def __init__(self, grid_layout):
        self.grid = grid_layout
        self.rows, self.columns = len(grid_layout), len(grid_layout[0])
        self.start = (0, 0) # Starting state for the agent. It's the top left corner of the grid.
        self.goal = (self.rows - 1, self.columns - 1) #rows -1, columns -1 makes the bottom right corner sector value. The termination state.
        self.state = self.start # initialize the maze with the agent at (0,0)
    
    # restarts the agent back to the starting state (0,0).
    def restart_agent(self):
        self.state = self.start
        return self.state
    
    # Determins if the location of the agent is in the valid place. 
    # True if sector is 0(opening). 
    # False if the sector value is 1(that's a wall) or if the row/column value is outside the grid.
    def is_valid(self, row, col):
         return (0 <= row < self.rows) and (0 <= col < self.columns) and (self.grid[row][col] == 0)
    
    def step(self, action):
        row, col = self.state
        if action == 0: # Moves the agent up
            new_state = (row - 1, col)
        elif action == 1: # Moves the agent down
            new_state = (row + 1, col)
        elif action == 2: # Moves the agent left
            new_state = (row, col - 1)
        elif action == 3: # Moves the agent right
            new_state = (row, col + 1)
            
        # First check if agent is in the appropriate spot.
        if self.is_valid(new_state[0], new_state[1]):
            # Update the agent with new location sector if the sector is valid
            self.state = new_state
            # If agent is at the termination state.
            if self.state == self.goal:
                return self.state, 100, True
            # Agent is not at termination state.
            else:
                return self.state, -1, False
        else:
            # Agent new spot not valid (either a wall or outside the grid). Return agent back to its last state and give a negative reward of -10.
            return self.state, -10, False

# Making the maze. 0's are a walkway/opening. 1's are the walls.
# It's a square that is 6 units long and 6 units wide. 
maze_grid_layout = [
    [0, 1, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 1],
    [0, 1, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 0],
    [0, 1, 0, 1, 1, 0],
    [0, 0, 0, 0, 1, 0]
]

# Calling the initialization constructor to create the maze.
maze = Maze(maze_grid_layout)

# Initialize Q-table and V-table
# Each tuple represents a maze row. 
# Each row of a tuple corresponds to the maze column.
# Each column of a tuple represents the direction action which is up, down, left, right in that order.
q_table = np.zeros((maze.rows, maze.columns, 4))
v_table = np.zeros((maze.rows, maze.columns, 4))

# SARSA parameters
learning_rate_alpha = 0.1 # Learning rate 
discount_factor_gamma = 0.9 # Discount factor
epsilon = 0.1 # Exploration-exploitation trade-off
iterations = 4000

# SARSA algorithm. The maze will be performed but will give q-values for each state that will determine the optimal 
# direction of each state.
for episode in range(iterations):
    state = maze.restart_agent()
    row, col = state
    action = np.argmax(q_table[row, col, :]) if random.uniform(0, 1) > epsilon else random.randint(0, 3)
    done = False
    
    while not done:
        next_state, reward, done = maze.step(action)
        next_row, next_col = next_state
        next_action = np.argmax(q_table[next_row, next_col, :]) if random.uniform(0, 1) > epsilon else random.randint(0, 3)
        
        # SARSA Q-formula. This will output tuples of q-values for all four actions in each state.
        q_table[row, col, action] += learning_rate_alpha * (reward + discount_factor_gamma * q_table[next_row, next_col, next_action] - q_table[row, col, action])

        # The general V-formula. By looking at the V-table in the program output, 
        # we can find the max value from each action after several iterations to determine V* for each open state.
        v_table[row, col, action] = reward + discount_factor_gamma * v_table[next_row, next_col, next_action]
        
        state = next_state
        row, col = state
        action = next_action

# Print the learned Q-table to show the progress of the learning.
# Also print out the variables such a learning rate, discount factor, and the number of iterations.
print("Number of iterations:", iterations)
print("Learning rate alpha:", learning_rate_alpha)
print("Discount factor gamma:", discount_factor_gamma)
print("Exploration-exploitation trade-off epsilon will always be 0.1")
print("Learned SARSA Q-table:")
print(q_table)
print("-------------------------------------------------------------------------------------------")
print("Learned V-table:")
print(v_table)

# Let the agent start from the origin state (0,0). That the top left corner of the grid. The terminal state is (5,5).
# This algorithm will give us the order of the states the agent will travel starting from the start to the goal.
state = maze.restart_agent()
path = [state]
done = False
while not done:
    row, col = state
    action = np.argmax(q_table[row, col, :])
    state, _, done = maze.step(action)
    path.append(state)

print("Optimal path:", path)