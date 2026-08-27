import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
from collections import deque, Counter

# ----------------------------
# Load Model & Label Encoder
# ----------------------------
model = tf.keras.models.load_model("model_groupAZ_final.h5")

with open("label_encoder_groupAZ.pkl", "rb") as f:
    le = pickle.load(f)

# ----------------------------
# MediaPipe
# ----------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.7,
)

mp_draw = mp.solutions.drawing_utils

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="ASL Sign Recognition", layout="centered")

st.title("🤚 ASL Sign Recognition")
st.write("Show a hand sign (A-M) to the webcam.")

run = st.checkbox("Start Webcam")

frame_placeholder = st.empty()
prediction_placeholder = st.empty()

# ----------------------------
# Prediction Stabilizer
# ----------------------------
label_history = deque(maxlen=10)

if run:

    cap = cv2.VideoCapture(0)

    while run:

        ret, frame = cap.read()

        if not ret:
            st.error("Unable to access webcam.")
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb)

        display_text = "❗ Hand not detected"

        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:

                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                h, w, _ = frame.shape

                coords = []

                # Pixel coordinates (same as training)
                for lm in hand_landmarks.landmark:
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    coords.extend([x, y])

                if len(coords) == 42:

                    X = np.array(coords, dtype=np.float32).reshape(1, -1)

                    pred = model.predict(X, verbose=0)

                    confidence = float(np.max(pred))

                    label = le.inverse_transform([np.argmax(pred)])[0]

                    if confidence >= 0.70:

                        label_history.append(label)

                        common_label, count = Counter(label_history).most_common(1)[0]

                        if count >= 5:
                            display_text = f"✅ {common_label} ({confidence:.2f})"
                        else:
                            display_text = f"⏳ {label} Stabilizing..."

                    else:
                        label_history.clear()
                        display_text = f"⚠ Low Confidence ({confidence:.2f})"

        else:
            label_history.clear()

        frame_placeholder.image(frame, channels="BGR")

        prediction_placeholder.markdown(
            f"## Prediction: {display_text}"
        )

    cap.release()

else:
    st.info("Click **Start Webcam** to begin.")