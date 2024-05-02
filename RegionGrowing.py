import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from matplotlib.path import Path

def manual_seed_selection(image, num_seeds):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(image, cmap=plt.cm.gray)
    ax.set_xticks([]), ax.set_yticks([])
    plt.title("Select seed points. Press Enter when done.")

    # Use ginput to get seed points
    seeds = plt.ginput(num_seeds)

    plt.show()

    return seeds

def region_growing(img, seeds, threshold):
    # parameters
    neighbors = [(0,1), (1,0), (-1,0), (0,-1)]  # 4-connectivity

    height, width = img.shape
    visited = np.zeros_like(img, dtype=np.uint8)
    out_img = np.zeros_like(img, dtype=np.uint8)

    # stack of pixels
    stack = []
    for seed in seeds:
        stack.append((int(seed[1]), int(seed[0])))  # note the y,x order because ginput returns x,y

    while(len(stack) > 0):
        s = stack.pop()
        x, y = s

        # Convert the seed coordinates to integers
        seed_x, seed_y = int(seeds[0][1]), int(seeds[0][0])

        if(np.abs(int(img[x, y]) - int(img[seed_x, seed_y])) <= threshold):
            out_img[x, y] = 255
            visited[x, y] = 1

            # add neighbors to stack
            for dx, dy in neighbors:
                nx, ny = x + dx, y + dy
                if nx >= 0 and nx < height and ny >= 0 and ny < width:
                    if visited[nx, ny] == 0:
                        stack.append((nx, ny))

    return out_img, visited


# Load test image path
img = cv2.imread('cell.jpg', cv2.IMREAD_GRAYSCALE)

# Manually select the seed points
seeds = manual_seed_selection(img, 1)

# Apply region growing algorithm
out_img, mask = region_growing(img, seeds, 40)

# Extract the segmented parts from the original image
segmented = np.where(mask, img, 0)

# Display the results
plt.figure(figsize=(10, 10))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap=plt.cm.gray)
plt.title("Original Image")

plt.subplot(1, 3, 2)
plt.imshow(out_img, cmap=plt.cm.gray)
plt.title("Region Growing")

plt.subplot(1, 3, 3)
plt.imshow(segmented, cmap=plt.cm.gray)
plt.title("Segmented Parts")

plt.show()
