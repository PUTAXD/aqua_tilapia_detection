import os
from video_processor import VideoProcessor
from data_handler import plot_average_speed, plot_fish_trajectories

# Constants
TRACK_TIME_WINDOW = 1.0  # seconds
SKIP_FRAME = 1
VIDEO_PATH = "assets/video_shorter.mp4"
MODEL_PATH = "models/best.pt"
OUTPUT_CSV = "result/fish_positions.csv"
OUTPUT_GRAPH_SPEED = "result/average_speed_over_time.png"
OUTPUT_GRAPH_TRAJECTORY = "result/fish_trajectories.png"


if __name__ == "__main__":
    os.makedirs('result', exist_ok=True)

    processor = VideoProcessor(
        model_path=MODEL_PATH,
        video_path=VIDEO_PATH,
        csv_output_path=OUTPUT_CSV,
        skip_frame=SKIP_FRAME,
        track_time_window=TRACK_TIME_WINDOW
    )

    fish_positions_log, speed_log = processor.process_video()

    # Generate plots after video processing
    plot_average_speed(speed_log, OUTPUT_GRAPH_SPEED)
    plot_fish_trajectories(OUTPUT_CSV, OUTPUT_GRAPH_TRAJECTORY)
