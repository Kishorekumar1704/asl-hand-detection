import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import pickle
from fuzzywuzzy import process
import pyttsx3
import time
from collections import deque, Counter
import threading
import speech_recognition as sr

# --- Load Model & Encoder ---
model = load_model("model_groupAZ_final1.h5")
with open("label_encoder_groupAZ.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# --- Load English Word List ---
with open("words_alpha.txt") as f:
    english_words = [line.strip() for line in f if len(line.strip()) >= 2]

# --- Text-to-Speech (threaded) ---
engine = pyttsx3.init()
def speak(text):
    def run_speak():
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=run_speak).start()

# --- Autocorrect using fuzzy matching ---
def autocorrect(word, word_list, threshold=80):
    match, score = process.extractOne(word, word_list)
    return match if score >= threshold else word

# --- Speech-to-Text ---
def listen_and_print():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    print("🎤 Listening... Speak now.")
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        print(f"👂 You said: {text}")
        return text
    except sr.UnknownValueError:
        print("⚠️ Could not understand audio.")
    except sr.RequestError:
        print("⚠️ Speech Recognition service error.")
    return ""

# --- MediaPipe Setup ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       model_complexity=0,
                       min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# --- Webcam Setup ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("📷 Sign-Speech Translator Active. SPACE=add, ENTER=finish word, R=listen reply, Q=quit")

sentence = ""
predicted_letters = []
stable_label = ""
letter_prediction = ""
last_added_time = 0
capture_delay = 0.8
label_history = deque(maxlen=10)
reply_text = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    coords = []
    confidence = 0
    current_time = time.time()

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=4),
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2)
            )

            h, w, _ = frame.shape
            for lm in hand_landmarks.landmark:
                x = int(lm.x * w)
                y = int(lm.y * h)
                coords.extend([x, y])

        if len(coords) == 42:
            X_input = np.array(coords).reshape(1, -1)
            prediction = model.predict(X_input, verbose=0)
            label = label_encoder.inverse_transform([np.argmax(prediction)])[0]
            confidence = np.max(prediction)

            if confidence >= 0.7:
                label_history.append(label)
                most_common_label, count = Counter(label_history).most_common(1)[0]
                if count >= 5:
                    stable_label = most_common_label
                    letter_prediction = f"{stable_label} (✔ stable)"
                else:
                    stable_label = ""
                    letter_prediction = f"{label} (wait...)"
            else:
                stable_label = ""
                label_history.clear()
                letter_prediction = "Low Confidence"
    else:
        stable_label = ""
        label_history.clear()
        letter_prediction = "❗ Hand not detected"

    # Display Text
    cv2.putText(frame, f"Letter: {letter_prediction}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Letter Buffer: {''.join(predicted_letters)}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f"Sentence: {sentence}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(frame, f"Reply: {reply_text}", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 255), 2)

    cv2.imshow("Sign-Speech Chat", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):
        if stable_label and (current_time - last_added_time >= capture_delay):
            predicted_letters.append(stable_label)
            print(f"📝 Added Letter: {stable_label}")
            last_added_time = current_time

    elif key == 13:
        word = ''.join(predicted_letters)
        corrected = autocorrect(word, english_words)
        if corrected != word:
            print(f"🧠 Suggestion: {corrected}")
        else:
            print(f"✅ Word Completed: {corrected}")
        sentence += corrected + " "
        speak(corrected)
        predicted_letters = []

    elif key == ord('r'):
        reply_text = listen_and_print()

    elif key == ord('q'):
        print("👋 Exiting...")
        break

    elif key in [8, 127] and predicted_letters:
        removed = predicted_letters.pop()
        print(f"❌ Removed: {removed}")

cap.release()
cv2.destroyAllWindows()