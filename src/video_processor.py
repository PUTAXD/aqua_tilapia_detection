import cv2
import time
import numpy as np
from ultralytics import YOLO
from sort import Sort
from utils import get_distance
import csv
import os


class VideoProcessor:
    def __init__(self, model_path, video_path, csv_output_path, max_age=5, min_hits=2, iou_threshold=0.2, skip_frame=5, track_time_window=1.0):
        self.model = YOLO(model_path)
        self.cap = cv2.VideoCapture(video_path)
        self.tracker = Sort(max_age=max_age, min_hits=min_hits,
                            iou_threshold=iou_threshold)
        self.skip_frame = skip_frame
        self.track_time_window = track_time_window
        self.object_tracks = {}
        self.speed_log = []
        self.fish_positions_log = []  # Still keep for plotting at the end
        self.last_log_time = time.time()
        self.frame_count = 0
        self.csv_output_path = csv_output_path

    def process_video(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("✅ Video ended.")
                break

            self.frame_count += 1
            if self.frame_count % (self.skip_frame + 1) != 0:
                continue

            frame, self.last_log_time = self._process_frame(frame)

            cv2.imshow("Fish Movement Tracker (SORT)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("🛑 Interrupted by user.")
                break

        self.cap.release()
        cv2.destroyAllWindows()
        # Save fish positions to CSV directly
        from data_handler import save_fish_positions_to_csv
        save_fish_positions_to_csv(
            self.fish_positions_log, self.csv_output_path)
        return self.fish_positions_log, self.speed_log

    def _process_frame(self, frame):
        now = time.time()
        results = self.model(frame, device="cpu", verbose=False, imgsz=480)[0]

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            detections.append([x1, y1, x2, y2, conf])

        if len(detections) > 0:
            detections_np = np.array(detections)
        else:
            detections_np = np.empty((0, 5))

        tracked_objects = self.tracker.update(detections_np)

        for x1, y1, x2, y2, obj_id in tracked_objects:
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            # Log to internal list for eventual plotting
            self.fish_positions_log.append((now, int(obj_id), cx, cy))

            if obj_id not in self.object_tracks:
                self.object_tracks[obj_id] = []
            self.object_tracks[obj_id].append((cx, cy, now))

            self.object_tracks[obj_id] = [
                pt for pt in self.object_tracks[obj_id] if now - pt[2] <= self.track_time_window
            ]

            cv2.rectangle(frame, (int(x1), int(y1)),
                          (int(x2), int(y2)), (255, 255, 0), 1)
            cv2.putText(frame, f"ID {int(obj_id)}", (int(x1), int(y1) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 255), 1)

            trail = self.object_tracks[obj_id]
            for i in range(1, len(trail)):
                cv2.line(frame, trail[i-1][:2], trail[i][:2], (0, 255, 0), 2)

            if len(trail) >= 2:
                dist = sum(get_distance(trail[i][:2], trail[i+1][:2])
                           for i in range(len(trail) - 1))
                duration = trail[-1][2] - trail[0][2]
                speed = dist / duration if duration > 0 else 0
                cv2.putText(frame, f"{speed:.1f} px/s", (int(x1), int(y2) + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        all_speeds = []
        for trail in self.object_tracks.values():
            if len(trail) >= 2 and now - trail[-1][2] <= self.track_time_window:
                dist = sum(get_distance(trail[i][:2], trail[i+1][:2])
                           for i in range(len(trail) - 1))
                duration = trail[-1][2] - trail[0][2]
                if duration > 0:
                    all_speeds.append(dist / duration)

        if all_speeds:
            avg_speed = sum(all_speeds) / len(all_speeds)
            cv2.putText(frame, f"Average Speed: {avg_speed:.1f} px/s", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            if now - self.last_log_time >= 1.0:
                self.speed_log.append((now, avg_speed))
                self.last_log_time = now
        return frame, self.last_log_time
