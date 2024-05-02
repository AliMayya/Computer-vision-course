import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('1.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Apply Mean Shift
mean_shifted = cv2.pyrMeanShiftFiltering(image, sp=15, sr=50)

# Create a subplot
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# Show original image
ax[0].imshow(image)
ax[0].set_title('Original Image')

# Show segmented image
ax[1].imshow(mean_shifted)
ax[1].set_title('Segmented Image')

plt.show()
