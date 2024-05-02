import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
image=cv2.imread("box.jpg",1)
# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, 150, 250, apertureSize=3)
minLineLength = 1000
maxLineGap = 20
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength, maxLineGap)

# Draw the detected lines
for line in lines:
    x1, y1, x2, y2 = line[0]  # Unpack the endpoints
    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw the line

# Display the result
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Hough Line Transform")
plt.show()

