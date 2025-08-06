import cv2
import mediapipe as mp
import csv
import os
import numpy as np
from collections import defaultdict

# --- Settings ---
csv_filename = "group2_N_Z_dataset.csv"
max_samples_per_letter = 200
allowed_letters = list("NOPQRSTUVWXYZ")

# --- Directory ---
output_dir = os.path.dirname(csv_filename)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)

# --- MediaPipe Setup ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# --- Webcam ---
cap = cv2.VideoCapture(0)
print("📸 SHOW SIGN + PRESS KEY (N-Z) to save | Q to quit")

# --- Sample Counter ---
sample_counts = defaultdict(int)

# --- Load existing data if resuming ---
if os.path.exists(csv_filename):
    with open(csv_filename, 'r') as f:
        for row in csv.reader(f):
            if row:
                label = row[0].upper()
                if label in allowed_letters:
                    sample_counts[label] += 1

# --- CSV Writer ---
with open(csv_filename, mode='a', newline='') as f:
    writer = csv.writer(f)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        coords = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                h, w, _ = frame.shape
                for lm in hand_landmarks.landmark:
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    coords.extend([x, y])

        # Show status
        for idx, letter in enumerate(allowed_letters):
            count = sample_counts[letter]
            cv2.putText(frame, f"{letter}: {count}/{max_samples_per_letter}", (10, 30 + 25*idx),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("Press N-Z to save | Q to quit", frame)
        key = cv2.waitKey(1)

        if key != -1:
            key_char = chr(key).upper()

            if key_char in allowed_letters:
                if len(coords) == 42:
                    if sample_counts[key_char] < max_samples_per_letter:
                        writer.writerow([key_char] + coords)
                        sample_counts[key_char] += 1
                        print(f"✅ Saved: {key_char} ({sample_counts[key_char]}/{max_samples_per_letter})")
                    else:
                        print(f"⚠️ Limit reached for {key_char}")
                else:
                    print("❌ Incomplete hand keypoints!")

            elif key_char == 'Q':
                print("👋 Exiting.")
                break

cap.release()
cv2.destroyAllWindows()
