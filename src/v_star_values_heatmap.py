import matplotlib.pyplot as plt
import numpy as np

import matplotlib
import matplotlib as mpl

# maze rows.
maze_rows = ["0", "1", "2", "3", "4", "5"]

# maze columns
maze_columns = ["0", "1", "2", "3", "4", "5"]

# Put in a 6X6 matrix of the v_star_values.
# Calculated using 4000 iteration, discount factor = 0.9
V_star_values = np.array([[8.344, 0, 42.61, 0, 54.95, 36.05],
                        [10.383, 0, 48.46, 54.95, 62.171, 0],
                        [12.65, 0, 42.61, 0, 70.19, 79.1],
                        [15.16, 0, 37.35, 32.616, 0, 89],
                        [17.961, 0, 32.616, 0, 0, 100],
                        [21.07, 24.519, 28.354, 24.519, 0, 100]])

# Plot in the v_star_values. Make the values show in the squares.
fig, ax = plt.subplots()
im = ax.imshow(V_star_values)

# Show all labels in the values inside it. Rows are rows of the maze, Colums are columns of the maze 6X6 maze.
ax.set_xticks(range(len(maze_columns)), labels=maze_columns,
              rotation=45, ha="right", rotation_mode="anchor")
ax.set_yticks(range(len(maze_rows)), labels=maze_rows)

# Loop over data dimensions and create text annotations. Constructs the 6X6 maze matrix.
for i in range(len(maze_rows)):
    for j in range(len(maze_columns)):
        text = ax.text(j, i, V_star_values[i, j],
                       ha="center", va="center", color="w")

# Set the title for the V_Star heat map. Ensure that the figure fits tightly and that it shows on run.
ax.set_title("V* heatmap of the simple 6X6 maze using 4000 iterations and discount factor gamma = 0.9")
fig.tight_layout()
plt.show()