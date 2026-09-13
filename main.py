import cv2
import numpy as np
import json
from pathlib import Path
import subprocess
import imageio_ffmpeg

# Watermarks
def add_watermark(frame, text="AeroScout"):
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = max(0.7, frame.shape[1] / 1800)
    thickness = max(1, int(round(font_scale * 2)))

    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness
    )

    margin = 20
    padding_x = 14
    padding_y = 10
    accent_width = 5

    x2 = frame.shape[1] - margin
    x1 = x2 - text_width - (2 * padding_x) - accent_width
    y1 = margin
    y2 = y1 + text_height + baseline + (2 * padding_y)

    # Transparent black background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    # Gold accent
    cv2.rectangle(
        frame,
        (x1, y1),
        (x1 + accent_width, y2),
        (0, 150, 210),
        -1
    )

    cv2.putText(
        frame,
        text,
        (x1 + accent_width + padding_x, y1 + padding_y + text_height),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )

# Footage is in the same directory as this script
BASE_DIR = Path(__file__).resolve().parent

INPUT_PATH = BASE_DIR / "footage.mp4"
TEMP_OUTPUT_PATH = BASE_DIR / "annotated_output_temp.mp4"
OUTPUT_PATH = BASE_DIR / "annotated_output.mp4"
LOG_PATH = BASE_DIR / "detections.json"

if not INPUT_PATH.exists():
    raise FileNotFoundError(f"Video not found: {INPUT_PATH}")

cap = cv2.VideoCapture(str(INPUT_PATH))

if not cap.isOpened():
    raise RuntimeError("OpenCV could not open the video")

# Pull the source video's properties so the output matches
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    str(TEMP_OUTPUT_PATH),
    fourcc,
    fps,
    (width, height)
)

# Tune these once you can see real footage
lower_orange_yellow = np.array([10, 100, 100])
upper_orange_yellow = np.array([35, 255, 255])

lower_red = np.array([0, 100, 100])
upper_red = np.array([10, 255, 255])

MIN_AREA = 5000 # 500 is too low 
kernel = np.ones((5, 5), np.uint8)

frame_num = 0
detections = []  # this is the log we'll export

while cap.isOpened():
    ret, frame = cap.read()   # ret = success bool, frame = the image array
    if not ret:
        break                 # video ended

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

    mask1 = cv2.inRange(hsv, lower_orange_yellow, upper_orange_yellow)
    mask2 = cv2.inRange(hsv, lower_red, upper_red)
    mask = cv2.bitwise_or(mask1, mask2)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        area = cv2.contourArea(c)

        if area > MIN_AREA:
            x, y, w, h = cv2.boundingRect(c)

            # Heuristic detection score
            box_area = max(w * h, 1)

            size_score = min(area / (MIN_AREA * 10), 1.0)

            fill_ratio = area / box_area
            density_score = min(fill_ratio / 0.5, 1.0)

            detection_score = int(round(
                100 * (0.55 * size_score + 0.45 * density_score)
            ))

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                4
            )

            label = "Suspected fire region" 

            cv2.putText(
                frame,
                label,
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3,
                cv2.LINE_AA
            )

            detections.append({
                "frame": frame_num,
                "timestamp_ms": timestamp_ms,
                "bbox": [int(x), int(y), int(w), int(h)],
                "area": float(area),
                "detection_score": detection_score
            })

    # Add the watermark before saving the frame
    add_watermark(frame)

    out.write(frame)

    cv2.imshow("Detection", frame)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_num += 1

cap.release()
out.release()
cv2.destroyAllWindows()

# Convert the temporary MP4V video into a PowerPoint-compatible H.264 MP4
ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

subprocess.run([
    ffmpeg_exe,
    "-y",
    "-i", str(TEMP_OUTPUT_PATH),
    "-c:v", "libx264",
    "-preset", "medium",
    "-crf", "22",
    "-pix_fmt", "yuv420p",
    str(OUTPUT_PATH)
], check=True)

TEMP_OUTPUT_PATH.unlink(missing_ok=True)

# Export the log for the .SRT cross-referencing step
with open(LOG_PATH, "w") as f:
    json.dump(detections, f, indent=2)

print(f"Logged {len(detections)} detections to detections.json")