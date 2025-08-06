import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import pickle

# --- Load both trained models ---
model1 = load_model("model_group1_final1.h5")  # A–M
model2 = load_model("model_group2_final1.h5")  # N–Z

# --- Load label encoders ---
with open("label_encoder_group1a.pkl", "rb") as f:
    encoder1 = pickle.load(f)

with open("label_encoder_group2a.pkl", "rb") as f:
    encoder2 = pickle.load(f)

# --- Setup MediaPipe ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# --- Start Webcam ---
cap = cv2.VideoCapture(0)
print("📷 Show any A–Z hand sign (Press Q to quit)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    coords = []
    prediction_text = ""

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=3),
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2)
            )

            h, w, _ = frame.shape
            for lm in hand_landmarks.landmark:
                x = int(lm.x * w)
                y = int(lm.y * h)
                coords.extend([x, y])

        if len(coords) == 42:
            X_input = np.array(coords).reshape(1, -1)

            # Predict using both models
            pred1 = model1.predict(X_input, verbose=0)
            pred2 = model2.predict(X_input, verbose=0)

            # Compare confidence
            max1 = np.max(pred1)
            max2 = np.max(pred2)

            if max1 > max2:
                label = encoder1.inverse_transform([np.argmax(pred1)])[0]
                confidence = max1
            else:
                label = encoder2.inverse_transform([np.argmax(pred2)])[0]
                confidence = max2

            prediction_text = f"Predicted: {label} ({confidence * 100:.1f}%)"
            cv2.putText(frame, prediction_text, (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    else:
        cv2.putText(frame, "Show your hand sign...", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    cv2.imshow("Unified A–Z Prediction", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("👋 Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
