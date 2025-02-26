import cv2
from centroid_tracker import CentroidTracker
from noise_filter import filter_noise_rectangles
import mediapipe as mp
import numpy as np

# Setup MediaPipe Pose and drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose_estimator = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

video_path = "video/ski_demo_cropped.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Background subtractor and tracker setup
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)
tracker = CentroidTracker(maxDisappeared=20)

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
        if area > 100:
            x, y, w, h = cv2.boundingRect(contour)
            if w < 30 or h < 50:
                continue
            rects.append((x, y, w, h))

    # Filter out detections in the noisy top-left and top-right zones
    rects = filter_noise_rectangles(frame, rects, left_ignore_pct=0.2, right_ignore_pct=0.2, top_ignore_pct=0.2, full_top_ignore_pct=0.2)

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

    # Draw tracking info (yellow circles and IDs)
    for objectID, centroid in objects.items():
        cx, cy = centroid
        cv2.circle(frame, (cx, cy), 6, (0, 255, 255), -1)
        cv2.putText(frame, f"Skier {objectID}", (cx - 20, cy - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, f"Skier {objectID}", (cx - 20, cy - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Ski Video - Pose & Tracking", frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
pose_estimator.close()
cv2.destroyAllWindows()
