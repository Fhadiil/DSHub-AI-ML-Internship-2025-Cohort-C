"""
VillageMed AI Symptom Tracker - Week 1 Prototype
Author: Muhammad Aliyu Katagum
License: MIT License
"""

import cv2
import mediapipe as mp
import time
import numpy as np

# Initialize FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)
mp_drawing = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)

# FPS counter
prev_time = 0

# Log file for expressions
# log_file = open("data/logs/expression_log.csv", "w")
# log_file.write("timestamp,expression\n")

screenshot_count = 0

def detect_expression(landmarks):
    """
    Very simple expression detector based on mouth openness.
    Returns 'happy', 'neutral', or 'stressed'.
    """
    top_lip = landmarks[13].y
    bottom_lip = landmarks[14].y
    mouth_open = abs(top_lip - bottom_lip)

    if mouth_open > 0.05:
        return "happy"
    elif mouth_open < 0.02:
        return "stressed"
    else:
        return "neutral"

while True:
    success, frame = cap.read()
    if not success:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1),
            )

            # Detect expression
            expression = detect_expression(face_landmarks.landmark)
            cv2.putText(frame, f"Expression: {expression}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            # Log expression
            # log_file.write(f"{time.time()},{expression}\n")

            # Save 5 screenshots only
            if screenshot_count < 5:
                cv2.imwrite(f"data/screenshots/frame_{screenshot_count}.jpg", frame)
                screenshot_count += 1

    # FPS calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show video
    cv2.imshow("VillageMed Symptom Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
log_file.close()
cv2.destroyAllWindows()
