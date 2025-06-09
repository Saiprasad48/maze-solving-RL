import matplotlib.pyplot as plt
import numpy as np

import matplotlib
import matplotlib as mpl

# maze rows.
maze_rows = ["0", "1", "2", "3", "4", "5"]

# maze columns
maze_columns = ["0", "1", "2", "3", "4", "5"]

# Put in a 6X6 matrix of the average q-values.
# Calculated using 2000 iteration, alpha = 1, discount factor = 0.9, epsilon = 0.1
average_Q_values = np.array([[-29.74, 0, -14.37, 0, -7.595, -25.29],
                        [-35.05, 0, -18.62, -0.31, 4.315, 0],
                        [-32.3, 0, -13.22, 0, 5.9, 34.93],
                        [-31.76, 0, -24.84, -30.17, 0, 64.46],
                        [-21.89, 0, -21.58, 0, 0, 70.99],
                        [-32.6, -26.99, -16.07, -28.19, 0, 100]])

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
ax.set_title("Average Q-values for every empty space of the 6X6 maze. Iterations = 2000, Learning rate alpha = 1, Discount factor gamma = 0.9, epsilon = 0.1")
fig.tight_layout()
plt.show()