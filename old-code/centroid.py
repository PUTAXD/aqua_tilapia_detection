import numpy as np
import cv2
import os
import math
import time
import matplotlib.pyplot as plt
from ultralytics import YOLO
from sort import Sort  # memakai file sort.py lokal, bukan pip install

# Load YOLO model
model = YOLO(
    "/home/martian/data-science/Data_science/aqua_nila/aqua_besar/best.pt")
class_names = model.names

# Load video
video_path = "/home/martian/data-science/Data_science/aqua_nila/ikan.mkv"
cap = cv2.VideoCapture(video_path)

# Tracker
tracker = Sort(max_age=5, min_hits=2, iou_threshold=0.2)

# Tracking data
object_tracks = {}  # {id: [(x, y, timestamp)]}
TRACK_TIME_WINDOW = 1.0  # seconds
SKIP_FRAME = 1
frame_count = 0
speed_log = []
last_log_time = time.time()


def get_distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


while True:
    ret, frame = cap.read()
    if not ret:
        print("✅ Video ended.")
        break

    frame_count += 1
    if frame_count % (SKIP_FRAME + 1) != 0:
        continue

    now = time.time()
    results = model(frame, device="cpu", verbose=False, imgsz=480)[0]

    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        detections.append([x1, y1, x2, y2, conf])

    # Pastikan deteksi tidak kosong
    if len(detections) > 0:
        detections_np = np.array(detections)
    else:
        detections_np = np.empty((0, 5))

    tracked_objects = tracker.update(detections_np)

    for x1, y1, x2, y2, obj_id in tracked_objects:
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

        if obj_id not in object_tracks:
            object_tracks[obj_id] = []
        object_tracks[obj_id].append((cx, cy, now))

        # Hapus data lama
        object_tracks[obj_id] = [
            pt for pt in object_tracks[obj_id] if now - pt[2] <= TRACK_TIME_WINDOW
        ]

        # Gambar bounding box
        cv2.rectangle(frame, (int(x1), int(y1)),
                      (int(x2), int(y2)), (255, 255, 0), 1)
        cv2.putText(frame, f"ID {int(obj_id)}", (int(x1), int(y1) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 255), 1)

        # Trail
        trail = object_tracks[obj_id]
        for i in range(1, len(trail)):
            cv2.line(frame, trail[i-1][:2], trail[i][:2], (0, 255, 0), 2)

        # Hitung kecepatan per ikan
        if len(trail) >= 2:
            dist = sum(get_distance(trail[i][:2], trail[i+1][:2])
                       for i in range(len(trail) - 1))
            duration = trail[-1][2] - trail[0][2]
            speed = dist / duration if duration > 0 else 0
            cv2.putText(frame, f"{speed:.1f} px/s", (int(x1), int(y2) + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Rata-rata kecepatan semua ikan
    all_speeds = []
    for trail in object_tracks.values():
        if len(trail) >= 2 and now - trail[-1][2] <= TRACK_TIME_WINDOW:
            dist = sum(get_distance(trail[i][:2], trail[i+1][:2])
                       for i in range(len(trail) - 1))
            duration = trail[-1][2] - trail[0][2]
            if duration > 0:
                all_speeds.append(dist / duration)

    if all_speeds:
        avg_speed = sum(all_speeds) / len(all_speeds)
        cv2.putText(frame, f"Average Speed: {avg_speed:.1f} px/s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if now - last_log_time >= 1.0:
            speed_log.append((now, avg_speed))
            last_log_time = now

    cv2.imshow("Fish Movement Tracker (SORT)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("🛑 Interrupted by user.")
        break

cap.release()
cv2.destroyAllWindows()

# Plot hasil kecepatan rata-rata
if speed_log:
    times = [t - speed_log[0][0] for t, _ in speed_log]
    speeds = [s for _, s in speed_log]
    plt.figure(figsize=(10, 5))
    plt.plot(times, speeds, marker="o", color="blue", linewidth=2)
    plt.title("Average Speed Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Average Speed (px/s)")
    plt.grid(True)
    plt.tight_layout()

    # Simpan ke file, bukan plt.show()
    output_graph = "average_speed_over_time.png"
    plt.savefig(output_graph)
    print(f"📊 Grafik tersimpan di {output_graph}")
