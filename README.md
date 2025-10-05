# Aqua Tilapia Detection and Tracking

This project provides a solution for detecting and tracking Tilapia fish in video streams, calculating their movement and speed. It leverages state-of-the-art object detection (YOLO) and multi-object tracking (SORT) algorithms to provide real-time insights into fish behavior.

## Features

- **Tilapia Detection:** Utilizes a pre-trained YOLO model (`best.pt`) to accurately detect Tilapia fish in video frames.
- **Multi-Object Tracking:** Employs the SORT (Simple Online and Realtime Tracking) algorithm to maintain consistent IDs for individual fish across frames, enabling trajectory and speed analysis.
- **Speed Calculation:** Calculates and displays the instantaneous speed of each tracked fish, as well as the average speed of all detected fish.
- **Data Logging:** Records fish positions over time to a CSV file.
- **Visualizations:** Generates plots for average fish speed over time and individual fish trajectories.
- **Real-time Visualization:** Displays the processed video with bounding boxes, fish IDs, and speed information.

## Technologies Used

- **Python:** The primary programming language.
- **Ultralytics YOLO:** For efficient and accurate object detection.
- **SORT (Simple Online and Realtime Tracking):** For robust multi-object tracking.
- **OpenCV (`cv2`):** For video processing and visualization.
- **Numpy:** For numerical operations.
- **Matplotlib:** For generating plots.

## Project Structure

```
aqua_tilapia_detection/
├── .gitignore
├── archive/                      # Potentially older versions or related utilities
│   ├── average_speed_over_time.png
│   ├── centroid.py
│   └── sort.py
├── assets/                       # Sample video assets
│   └── video_shorter.mp4
├── models/                       # Pre-trained YOLO model
│   └── best.pt
├── result/                       # Output directory for generated data and plots
│   ├── average_speed_over_time.png
│   ├── fish_positions.csv
│   └── fish_trajectories.png
└── src/                          # Source code
    ├── data_handler.py           # Handles data saving and plotting
    ├── main.py                   # Main script to run the detection and tracking
    ├── sort.py                   # Implementation of the SORT tracking algorithm
    ├── utils.py                  # Utility functions (e.g., distance calculation)
    └── video_processor.py        # Core logic for video processing, detection, and tracking
```

## Setup and Installation

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd aqua_nila/aqua_tilapia_detection
    ```

    (Note: Replace `<repository_url>` with the actual repository URL if this project is part of a larger Git repository.)

2.  **Install dependencies:**
    It is recommended to use a virtual environment.

    ```bash
    pip install -r requirements.txt
    ```

    (This will install necessary libraries such as Ultralytics YOLO, OpenCV, NumPy, and Matplotlib.)

3.  **Download the YOLO model:**
    Ensure that `models/best.pt` is available. If not, you will need to train your own YOLO model or download a pre-trained one compatible with your detection task.

## Usage

To run the fish detection and tracking:

```bash
python src/main.py
```

The script will:

1.  Process the video specified in `VIDEO_PATH` (default: `assets/video_shorter.mp4`).
2.  Display a real-time window showing the tracked fish.
3.  Save fish position data to `result/fish_positions.csv`.
4.  Generate `result/average_speed_over_time.png` and `result/fish_trajectories.png`.

## Configuration

You can modify the following constants in `src/main.py` to adjust the behavior:

- `TRACK_TIME_WINDOW`: The time window (in seconds) for calculating fish speed and trajectory trails.
- `SKIP_FRAME`: Number of frames to skip between processing to speed up inference.
- `VIDEO_PATH`: Path to the input video file.
- `MODEL_PATH`: Path to the YOLO model (`.pt` file).
- `OUTPUT_CSV`: Path for the CSV file logging fish positions.
- `OUTPUT_GRAPH_SPEED`: Path for the average speed graph.
- `OUTPUT_GRAPH_TRAJECTORY`: Path for the fish trajectories graph.
