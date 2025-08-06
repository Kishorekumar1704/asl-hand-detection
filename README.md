# 🖐️ ASL Sign Language to Sentence Translator (Left Hand Only)

A real-time American Sign Language (ASL) recognition system using a webcam, deep learning, and MediaPipe. It detects hand signs for A–Z (trained on **left-hand** only) and converts them into full **English sentences** with **auto-correction** and **speech output**.

![Demo Screenshot](demo.png) <!-- Optional: Replace with your screenshot image -->

---

## 📌 Features

- 🧠 Real-time hand sign recognition (A–Z)
- ✋ Left-hand sign detection using MediaPipe
- 🔤 Sentence building with letter buffering
- ✅ Fuzzy matching for autocorrection (`fuzzywuzzy`)
- 🔊 Text-to-speech using `pyttsx3`
- 📷 Live webcam preview
- 📦 Lightweight and easy to use

---

## 📁 Project Structure

📦 asl-sign-language-left-hand/
├── realaz.py # Main app: Detects signs and forms sentences
├── model_groupAZ_final1.h5 # Trained Keras model for A–Z
├── label_encoder_groupAZ.pkl # Label encoder for character mapping
├── words_alpha.txt # English dictionary for autocorrect
├── requirements.txt # All dependencies
└── README.md # This file
