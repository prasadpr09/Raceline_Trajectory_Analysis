import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# --- Map metadata from YAML ---
resolution = 0.05  # meters per pixel
origin_x, origin_y = -24.3, -9.81  # origin from YAML

# Load the map image
map_img = np.array(Image.open("race3.pgm").convert('L'))  # 'L' mode for grayscale

# Rotate the map 180 degrees to correct ROS flip
# map_img = np.rot90(map_img, 1)
# Flip the map vertically (180-degree rotation around X-axis)
map_img = np.flipud(map_img)


# Get dimensions after rotation
height, width = map_img.shape

# Compute extent (adjusted for 180-degree rotation)
xmin = origin_x
xmax = origin_x + width * resolution
ymin = origin_y
ymax = origin_y + height * resolution
extent = [xmax, xmin, ymax, ymin]  # [left, right, bottom, top]

# Load the midpoint trajectory
csv1 = pd.read_csv('optimals/optimal_w_lh.csv', sep=';')
csv1 = pd.read_csv('optimals/optimal_w_lh.csv', sep=';')


# Rotate the trajectory coordinates to match the map
# Compute the center of the map in meters
center_x = (xmin + xmax) / 2
center_y = (ymin + ymax) / 2

# Apply 180-degree rotation to the trajectory coordinates around the center
# For 180 degrees: new_x = center_x - (x - center_x), new_y = center_y - (y - center_y)
csv1['x_rot'] = 2 * center_x - csv1['x']
csv1['y_rot'] = 2 * center_y - csv1['y']

# --- Set up the plot ---
fig, ax = plt.subplots(figsize=(16, 16))

# Show the map
ax.imshow(map_img, extent=extent, origin='lower', cmap='gray')

# Plot the rotated midpoint trajectory
ax.plot(csv1['x_rot'], csv1['y_rot'], label='midpoint', color='cyan')

# --- Beautify the plot ---
ax.set_xlabel('X (meters)')
ax.set_ylabel('Y (meters)')
ax.set_title('Trajectories on Map')
ax.legend()
ax.grid(True)

# Display the plot
plt.show()