import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the Haar Cascade classifier for face detection from the local file
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Capture video from the default camera (0)
cap = cv2.VideoCapture(0)

plt.ion()  # Turn on interactive mode for Matplotlib

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()
    
    if not ret:
        break

    # Resize the frame to speed up processing
    frame_small = cv2.resize(frame, (640, 480))

    # Convert the frame to grayscale (face detection works better on grayscale images)
    gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Loop through the detected faces and apply blur
    for (x, y, w, h) in faces:
        # Scale back face coordinates to original frame size
        x = int(x * frame.shape[1] / frame_small.shape[1])
        y = int(y * frame.shape[0] / frame_small.shape[0])
        w = int(w * frame.shape[1] / frame_small.shape[1])
        h = int(h * frame.shape[0] / frame_small.shape[0])

        # Extract the region of interest (the face)
        roi = frame[y:y+h, x:x+w]

        # Apply Gaussian blur to the region of interest
        blurred_roi = cv2.GaussianBlur(roi, (99, 99), 30)

        # Replace the original face region with the blurred one
        frame[y:y+h, x:x+w] = blurred_roi

    # Convert the frame from BGR to RGB for Matplotlib
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Display the frame with the blurred faces using Matplotlib
    plt.imshow(frame_rgb)
    plt.axis('off')  # Hide the axis
    plt.draw()
    plt.pause(0.01)

    # Check for 'q' key press to exit
    if plt.waitforbuttonpress(0.01):
        break

# Release the video capture object
cap.release()
plt.close()  # Close the Matplotlib window
