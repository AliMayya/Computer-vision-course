import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage.io import imread
from matplotlib.path import Path

def manual_freehand_contour_selection(image):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(image, cmap=plt.cm.gray)
    ax.set_xticks([]), ax.set_yticks([])
    plt.title("Draw a freehand contour. Press Enter when done.")

    # Initialize an empty list to store user-defined points
    points = []

    def ondrag(event):
        if event.button == 1:
            points.append((event.ydata, event.xdata))
            ax.plot(event.xdata, event.ydata, 'ro', markersize=2)
            fig.canvas.draw()

    fig.canvas.mpl_connect('motion_notify_event', ondrag)
    plt.show()

    return np.array(points)

# Load your cell image (replace 'cell.jpg' with your actual image path)
img = imread('cell.jpg', as_gray=True)

# Manually select the freehand contour
init_contour = manual_freehand_contour_selection(img)

# Apply the active contour model
snake = active_contour(img,
                       init_contour, alpha=0.015, beta=0.05, gamma=0.001,max_iterations=20)

# Display the results
fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(img, cmap=plt.cm.gray)
ax.plot(init_contour[:, 1], init_contour[:, 0], '--r', lw=3, label="Initial Contour")
ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3, label="Active Contour")
ax.set_xticks([]), ax.set_yticks([])
ax.axis([0, img.shape[1], img.shape[0], 0])
plt.legend()
plt.title("Active Contour for Cell Detection (Freehand Contour)")
plt.show()

# Create a path object from the final contour
path = Path(snake)

# Create a grid of points
grid = np.indices(img.shape).reshape(2, -1).T

# Use the path object to create a mask of the ROI
mask = path.contains_points(grid).reshape(img.shape)

# Extract the ROI from the original image
roi = np.where(mask, img, np.nan)

# Display the ROI
fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(roi, cmap=plt.cm.gray)
ax.set_xticks([]), ax.set_yticks([])
plt.title("Region of Interest (ROI)")
plt.show()
