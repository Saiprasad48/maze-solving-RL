import matplotlib.pyplot as plt
import numpy as np

import matplotlib
import matplotlib as mpl

# maze rows.
maze_rows = ["0", "1", "2", "3", "4", "5"]

# maze columns
maze_columns = ["0", "1", "2", "3", "4", "5"]

# Put in a 6X6 matrix of the average q-values.
# Calculated using 500 iteration, alpha = 0.1, discount factor = 0.9, epsilon = 0.1
average_Q_values = np.array([[-10.18, 0, 1.635, 0, 5.49, -1.71],
                        [-7.44, 0, 16.73, 25.98, 26.37, 0],
                        [-6.3, 0, 13.99, 0, 37.77, 45.85],
                        [-4.8, 0, 8.91, -1.06, 0, 56.77],
                        [-2.71, 0, 6.43, 0, 0, 63.69],
                        [-1.23, 0.58, 2.09, -3.72, 0, 100]])

# Plot in the average q-values. Make the values show in the squares.
fig, ax = plt.subplots()
im = ax.imshow(average_Q_values)

# Show all labels in the values inside it. Rows are rows of the maze, Colums are columns of the maze 6X6 maze.
ax.set_xticks(range(len(maze_columns)), labels=maze_columns,
              rotation=45, ha="right", rotation_mode="anchor")
ax.set_yticks(range(len(maze_rows)), labels=maze_rows)

# Loop over data dimensions and create text annotations. Constructs the 6X6 maze matrix.
for i in range(len(maze_rows)):
    for j in range(len(maze_columns)):
        text = ax.text(j, i, average_Q_values[i, j],
                       ha="center", va="center", color="w")

# Set the title for the average q-values heat map. Ensure that the figure fits tightly and that it shows on run.
ax.set_title("Average Q-values for every empty space of the 6X6 maze. Iterations = 500, Learning rate alpha = 0.1, Discount factor gamma = 0.9, epsilon = 0.1")
fig.tight_layout()
plt.show()