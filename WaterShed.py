import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image in grayscale for thresholding and in color for watershed
image_gray = cv2.imread('cell.jpg', 0)
image_color = cv2.imread('cell.jpg')
image_color = cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB)

# Apply adaptive thresholding
_, thresh = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 2)

# Sure background area
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform, 0.9*dist_transform.max(), 255, 0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# Marker labelling
_, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers+1

# Now, mark the region of unknown with zero
markers[unknown==255] = 0

# Apply watershed to the color image
markers = cv2.watershed(image_color, markers)
image_color[markers == -1] = [255,0,0]

# Create a subplot
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# Show original image
ax[0].imshow(cv2.cvtColor(cv2.imread('cell.jpg'), cv2.COLOR_BGR2RGB))
ax[0].set_title('Original Image')

# Show segmented image
ax[1].imshow(markers, cmap='jet')  # The segmented image
ax[1].set_title('Segmented Image')

plt.show()
