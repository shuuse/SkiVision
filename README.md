# SkiVision: AI-Powered Skier Tracking

SkiVision is a computer vision project that tracks skiers in a video using object detection, DeepSORT tracking, and pose estimation with MediaPipe. The project processes a video, tracks each skier with a unique ID, and overlays posture keypoints for analysis.

## Features
- **Object Detection**: Identifies moving skiers using background subtraction.
- **DeepSORT Tracking**: Assigns a unique ID to each skier and maintains identity even when objects cross.
- **Pose Estimation**: Uses MediaPipe to detect skier posture and overlay keypoints.
- **Noise Filtering**: Removes false detections from camera movement.
- **Video Export**: Saves the processed video with tracking and pose overlays.

## Installation
### 1. Clone the repository:
```sh
git clone https://github.com/your-repo/SkiVision.git
cd SkiVision
```

### 2. Create and activate a virtual environment:
```sh
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
```

### 3. Install dependencies:
```sh
pip install -r requirements.txt
```

## Usage
Place your input video in the `video/` folder. Then, run the processing script:
```sh
python video_deepsort_pose.py
```

The processed video will be saved in the same folder as the input video, with `_deepsort_pose` added to the filename.

## File Structure
```
SkiVision/
│── video/                    # Folder for input/output videos
│── centroid_tracker.py        # Legacy tracker (replaced by DeepSORT)
│── noise_filter.py            # Function to remove false detections
│── video_deepsort_pose.py     # Main script with tracking & pose detection
│── requirements.txt           # Dependencies
│── README.md                  # Project documentation
```

## Dependencies
- OpenCV
- NumPy
- MediaPipe
- DeepSORT (with PyTorch)
- TorchVision

## Example Output
The processed video will display:
- **Bounding Boxes** (blue): Around detected skiers.
- **Pose Keypoints** (green/red): Showing skier posture.
- **Tracker IDs** (white): Unique IDs for each skier.

## License
This project is open-source and available under the MIT License.
