import cv2
import numpy as np
import json

cap = cv2.VideoCapture("footage.mp4") # Placeholder footage

# Pull the source video's properties so the output matches
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("annotated_output.mp4", fourcc, fps, (width, height))

# Tune these once you can see real footage
lower_orange_yellow = np.array([10, 100, 100])
upper_orange_yellow = np.array([35, 255, 255])

lower_red = np.array([0, 100, 100])
upper_red = np.array([10, 255, 255])

MIN_AREA = 500
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
        if cv2.contourArea(c) > MIN_AREA:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Fire?", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            detections.append({
                "frame": frame_num,
                "timestamp_ms": timestamp_ms,
                "bbox": [int(x), int(y), int(w), int(h)],
                "area": float(cv2.contourArea(c))
            })

    out.write(frame)

    cv2.imshow("Detection", frame)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_num += 1

cap.release()
cv2.destroyAllWindows()

# Export the log for the .SRT cross-referencing step
with open("detections.json", "w") as f:
    json.dump(detections, f, indent=2)

print(f"Logged {len(detections)} detections to detections.json")