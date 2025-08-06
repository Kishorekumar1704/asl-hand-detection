import cv2
import numpy as np
import mediapipe as mp
import pickle
from tensorflow.keras.models import load_model

# Load trained model and encoder
model = load_model("keypoint_model.h5")
with open("keypoint_label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# MediaPipe hand detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
print("🖐️ MediaPipe hand tracking. Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            coords = []
            for lm in hand_landmarks.landmark:
                h, w, _ = frame.shape
                x, y = int(lm.x * w), int(lm.y * h)
                coords.extend([x, y])
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

            if len(coords) == 42:
                X_input = np.array(coords).reshape(1, -1)
                prediction = model.predict(X_input)
                pred_label = label_encoder.inverse_transform(np.argmax(prediction, axis=1))[0]

                cv2.putText(frame, f"Predicted: {pred_label}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)

    cv2.imshow("ASL Live Detection (MediaPipe)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
