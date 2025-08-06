import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
import time

# Load model and label encoder
model = tf.keras.models.load_model("asl_model.h5")
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Streamlit layout
st.title("🤚 SignSpeak: ASL to Text Translator (Webcam)")
st.markdown("Hold your hand sign in front of the webcam. The model will predict the letter in real time.")

run = st.checkbox("Start Webcam")

frame_window = st.image([])  # Live webcam feed
prediction_text = st.empty()  # Show prediction

cap = None

if run:
    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Unable to access webcam.")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        pred = "✋ Show a hand sign"
        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

                # Extract 21 hand keypoints
                keypoints = []
                for lm in handLms.landmark:
                    keypoints.extend([lm.x, lm.y])

                if len(keypoints) == 42:
                    X = np.array(keypoints).reshape(1, -1)
                    y_pred = model.predict(X, verbose=0)
                    label = le.inverse_transform([np.argmax(y_pred)])[0]
                    pred = f"🧠 Predicted Sign: **{label}**"

        frame_window.image(frame, channels="BGR")
        prediction_text.markdown(pred)
        time.sleep(0.05)

    cap.release()
else:
    st.info("🟡 Click 'Start Webcam' to begin.")
