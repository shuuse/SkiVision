import os
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
from noise_filter import filter_noise_rectangles
import mediapipe as mp
import numpy as np

# Setup MediaPipe Pose and drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose_estimator = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# Input video path
video_path = "video/ski_demo2.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define output file name:
base_name = os.path.splitext(os.path.basename(video_path))[0]
output_filename = f"{base_name}_deepsort_pose.mp4"
output_path = os.path.join(os.path.dirname(video_path), output_filename)

# Define codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Background subtractor setup
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=20)

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

    # Filter out detections in the noisy top-left, top-right, and top zones
    rects = filter_noise_rectangles(frame, rects, left_ignore_pct=0.2, right_ignore_pct=0.2,
                                    top_ignore_pct=0.2, full_top_ignore_pct=0.2)

    # Prepare detections for DeepSORT in the expected format:
    # Each detection is a tuple: ([x, y, w, h], confidence, "person")
    detections = []
    for (x, y, w, h) in rects:
        detections.append(([x, y, w, h], 1.0, "person"))

    # Update DeepSORT tracker with current detections
    tracks = tracker.update_tracks(detections, frame=frame)

    # For each confirmed track, perform pose detection and draw bounding box and ID
    for track in tracks:
        if not track.is_confirmed():
            continue
        bbox = track.to_ltrb()  # Returns [xmin, ymin, xmax, ymax]
        x, y, x2, y2 = map(int, bbox)
        w = x2 - x
        h = y2 - y

        # Crop the region of interest (ROI) for pose detection
        roi = frame[y:y+h, x:x+w]
        if roi.size == 0:
            continue  # Skip if ROI is empty
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        results = pose_estimator.process(roi_rgb)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                roi,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
            )
            frame[y:y+h, x:x+w] = roi

        # Draw the bounding box and tracker ID
        cv2.rectangle(frame, (x, y), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f"Skier {track.track_id}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Write the processed frame to the output video
    out.write(frame)

    cv2.imshow("Ski Video - DeepSORT Pose & Tracking", frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
out.release()
pose_estimator.close()
cv2.destroyAllWindows()
