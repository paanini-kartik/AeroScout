import cv2
import numpy as np

def nothing(x):
    pass

cap = cv2.VideoCapture("footage.mp4")
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

cv2.namedWindow("Trackbars")
cv2.resizeWindow("Trackbars", 400, 400)

cv2.createTrackbar("OY_H_min", "Trackbars", 10, 179, nothing)
cv2.createTrackbar("OY_H_max", "Trackbars", 35, 179, nothing)
cv2.createTrackbar("OY_S_min", "Trackbars", 100, 255, nothing)
cv2.createTrackbar("OY_S_max", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("OY_V_min", "Trackbars", 100, 255, nothing)
cv2.createTrackbar("OY_V_max", "Trackbars", 255, 255, nothing)

cv2.createTrackbar("R_H_min", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("R_H_max", "Trackbars", 10, 179, nothing)
cv2.createTrackbar("R_S_min", "Trackbars", 100, 255, nothing)
cv2.createTrackbar("R_S_max", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("R_V_min", "Trackbars", 100, 255, nothing)
cv2.createTrackbar("R_V_max", "Trackbars", 255, 255, nothing)

frame_num = 0
playing = False
frame = None

print("Controls: [space]=play/pause  [d]=next frame  [a]=prev frame  [q]=quit")

while True:
    if playing or frame is None:
        ret, new_frame = cap.read()
        if not ret:
            # reached the end — stop and hold on the last frame
            playing = False
            frame_num = total_frames - 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, new_frame = cap.read()
        frame = new_frame
        frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    oy_lower = np.array([cv2.getTrackbarPos("OY_H_min", "Trackbars"),
                          cv2.getTrackbarPos("OY_S_min", "Trackbars"),
                          cv2.getTrackbarPos("OY_V_min", "Trackbars")])
    oy_upper = np.array([cv2.getTrackbarPos("OY_H_max", "Trackbars"),
                          cv2.getTrackbarPos("OY_S_max", "Trackbars"),
                          cv2.getTrackbarPos("OY_V_max", "Trackbars")])

    r_lower = np.array([cv2.getTrackbarPos("R_H_min", "Trackbars"),
                         cv2.getTrackbarPos("R_S_min", "Trackbars"),
                         cv2.getTrackbarPos("R_V_min", "Trackbars")])
    r_upper = np.array([cv2.getTrackbarPos("R_H_max", "Trackbars"),
                         cv2.getTrackbarPos("R_S_max", "Trackbars"),
                         cv2.getTrackbarPos("R_V_max", "Trackbars")])

    mask1 = cv2.inRange(hsv, oy_lower, oy_upper)
    mask2 = cv2.inRange(hsv, r_lower, r_upper)
    mask = cv2.bitwise_or(mask1, mask2)

    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Overlay frame number + play/pause state so you always know where you are
    status = "PLAYING" if playing else "PAUSED"
    display_frame = frame.copy()
    cv2.putText(display_frame, f"Frame {frame_num}/{total_frames}  [{status}]",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Original", display_frame)
    cv2.imshow("Mask", mask)
    cv2.imshow("Result", result)

    key = cv2.waitKey(30 if playing else 0) & 0xFF  # waitKey(0) = wait forever when paused

    if key == ord('q'):
        break
    elif key == ord(' '):
        playing = not playing
    elif key == ord('d') and not playing:  # step forward
        frame_num = min(frame_num + 1, total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
    elif key == ord('a') and not playing:  # step backward
        frame_num = max(frame_num - 1, 0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()

cap.release()
cv2.destroyAllWindows()