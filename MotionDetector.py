import numpy as np
import cv2

# Create a background subtractor object
fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=60, history=20)

# Open a video capture stream (you can replace this with your own video file)
cap = cv2.VideoCapture('test.mp4')

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    # Apply background subtraction to get the foreground mask
    fgmask = fgbg.apply(frame)

    # Remove small regions using bwareaopen
    min_area_threshold = 20  # Adjust as needed

    # Find connected components and filter out small regions
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fgmask, connectivity=8)

    # Create an array of indices for regions to be removed
    small_regions_indices = np.where(stats[:, cv2.CC_STAT_AREA] < min_area_threshold)[0]

    # Set the pixels corresponding to small regions to 0
    fgmask_processed = np.isin(labels, small_regions_indices, invert=True).astype(np.uint8) * fgmask

    # Display the processed foreground mask
    cv2.imshow('Processed Foreground Mask', fgmask_processed)

    # Display the original frame
    cv2.imshow('Original Frame', frame)

    # Press 'q' to exit
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
