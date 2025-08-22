import os
from video_processor import VideoProcessor
from data_handler import save_fish_positions_to_csv, plot_average_speed, plot_fish_trajectories

# Constants
TRACK_TIME_WINDOW = 1.0  # seconds
SKIP_FRAME = 1
VIDEO_PATH = "video_shorter.mp4"
MODEL_PATH = "best.pt"
OUTPUT_CSV = "fish_positions.csv"
OUTPUT_GRAPH_SPEED = "average_speed_over_time.png"
OUTPUT_GRAPH_TRAJECTORY = "fish_trajectories.png"

if __name__ == "__main__":
    processor = VideoProcessor(
        model_path=MODEL_PATH,
        video_path=VIDEO_PATH,
        skip_frame=SKIP_FRAME,
        track_time_window=TRACK_TIME_WINDOW
    )

    fish_positions_log, speed_log = processor.process_video()

    save_fish_positions_to_csv(fish_positions_log, OUTPUT_CSV)
    plot_average_speed(speed_log, OUTPUT_GRAPH_SPEED)
    plot_fish_trajectories(OUTPUT_CSV, OUTPUT_GRAPH_TRAJECTORY)
