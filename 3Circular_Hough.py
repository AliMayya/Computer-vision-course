import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
image = cv2.imread("iris.jpg", 1)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
gray_blur = cv2.GaussianBlur(gray, (5, 5), 2, 2)

# Detect circles using Hough Circle Transform
circles = cv2.HoughCircles(
    gray_blur,
    cv2.HOUGH_GRADIENT,
    dp=1,
    minDist=gray.shape[0] // 16,
    param1=50,
    param2=30,
    minRadius=75,
    maxRadius=100,  # Adjust these parameters for larger circles
)

if circles is not None:
    circles = np.uint16(np.around(circles))
    for circle in circles[0]:
        center = (circle[0], circle[1])
        radius = circle[2]
        cv2.circle(image, center, radius, (255, 0, 255), 3)  # Draw the circle

# Display the result
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Hough Circle Transform")
plt.show()
