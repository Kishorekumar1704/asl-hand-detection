import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import pickle

# 📦 Load the trained model and label encoder
model = load_model("best_mediapipe_model.h5")  # You can change to final_mediapipe_model.h5 if preferred
with open("mediapipe_label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# 🎯 Setup MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# 🎥 Open webcam
cap = cv2.VideoCapture(0)
print("📷 Webcam started. Show your hand signs (press Q to quit)...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    coords = []
    prediction_text = "Show your hand..."

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            h, w, _ = frame.shape
            coords = []
            for lm in hand_landmarks.landmark:
                x = int(lm.x * w)
                y = int(lm.y * h)
                coords.extend([x, y])

            # Predict only if all 21 keypoints detected (42 values)
            if len(coords) == 42:
                X_input = np.array(coords, dtype='float32').reshape(1, -1)
                X_input = X_input / np.max(X_input)  # Normalize

                prediction = model.predict(X_input, verbose=0)
                confidence = np.max(prediction)
                predicted_index = np.argmax(prediction)
                predicted_label = label_encoder.inverse_transform([predicted_index])[0]

                if confidence > 0.6:
                    prediction_text = f"Sign: {predicted_label} ({confidence*100:.1f}%)"
                else:
                    prediction_text = f"Low confidence..."

    # 🖼️ Show text
    cv2.putText(frame, prediction_text, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0) if "Sign" in prediction_text else (0, 0, 255), 3)

    cv2.imshow("🔤 Live Sign Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("👋 Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
