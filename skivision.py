import os
import cv2
from centroid_tracker import CentroidTracker
from noise_filter import filter_noise_rectangles
import mediapipe as mp
import numpy as np

# Setup MediaPipe Pose and drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose_estimator = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.1)

# Input video path
video_path = "video/your_video_file_here.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties for output
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define output filename: same as input with '_pose' appended before the extension
base_name = os.path.splitext(os.path.basename(video_path))[0]
output_filename = f"{base_name}_pose.mp4"
output_path = os.path.join(os.path.dirname(video_path), output_filename)

# Create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Background subtractor and tracker setup
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)
tracker = CentroidTracker(maxDisappeared=20)

# Dictionary to store the last 4 centroids for each tracked object
object_history = {}

# Parameters for arrow drawing
arrow_scale = 5         # Scale factor for arrow length
max_displacement = 100  # Maximum displacement (in pixels) allowed to draw arrow

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Create foreground mask and reduce noise with morphological operations
    fgmask = fgbg.apply(frame)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    # Find contours on the mask
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:
            x, y, w, h = cv2.boundingRect(contour)
            if w < 30 or h < 50:
                continue
            rects.append((x, y, w, h))

    # Filter out detections in the noisy zones
    rects = filter_noise_rectangles(frame, rects, left_ignore_pct=0.2, right_ignore_pct=0.1, top_ignore_pct=0.1)

    # Update tracker with filtered bounding boxes
    objects = tracker.update(rects)

    # Process each bounding box for posture detection
    for (x, y, w, h) in rects:
        # Crop the region of interest (ROI)
        roi = frame[y:y+h, x:x+w]
        # Convert from BGR to RGB for MediaPipe
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        results = pose_estimator.process(roi_rgb)
        
        if results.pose_landmarks:
            # Draw pose landmarks on the ROI
            mp_drawing.draw_landmarks(
                roi, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
            )
            # Place the processed ROI back into the frame
            frame[y:y+h, x:x+w] = roi
        
        # Draw bounding box for visualization (blue)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # For each tracked object, update its history and draw the smoothed arrow
    for objectID, centroid in objects.items():
        # Initialize history list if necessary
        if objectID not in object_history:
            object_history[objectID] = []
        # Append the current centroid and keep only the last 4 positions
        object_history[objectID].append(centroid)
        if len(object_history[objectID]) > 4:
            object_history[objectID].pop(0)
        
        # Only draw arrow if we have at least 4 frames of history
        if len(object_history[objectID]) >= 4:
            old = object_history[objectID][0]
            dx = centroid[0] - old[0]
            dy = centroid[1] - old[1]
            displacement = np.sqrt(dx**2 + dy**2)
            if displacement > 0 and displacement < max_displacement:
                arrow_end = (int(centroid[0] + arrow_scale * dx), int(centroid[1] + arrow_scale * dy))
                cv2.arrowedLine(frame, (centroid[0], centroid[1]), arrow_end, (0, 255, 255), 2, tipLength=0.3)
    
    # Show the frame and write it to the output video
    cv2.imshow("Ski Video - Pose & Tracking", frame)
    out.write(frame)
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
out.release()
pose_estimator.close()
cv2.destroyAllWindows()
