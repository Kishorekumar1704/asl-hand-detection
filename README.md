Sign Language Detection App 🤟

A real-time Sign Language to Text translator that uses hand landmark detection via MediaPipe and a custom-trained ML model to recognize ASL letters. Works on both webcam and phone camera.

📌 Overview

This project is designed to bridge the communication gap between sign language users and others by converting ASL hand gestures into text in real-time.

Detects 21 hand keypoints using MediaPipe Hands.

Uses custom-trained ML model based on landmark coordinates for high accuracy.

Works directly in the browser with no server dependency (lightweight and fast).

Ideal for accessibility-focused applications.

🚀 Features

📷 Real-time hand tracking using MediaPipe.

🔍 Keypoint-based classification (not raw images) → better performance.

📱 Works with webcam and mobile camera.

⚡ Lightweight → runs locally without heavy GPU.

📊 Trained on custom ASL dataset with landmark coordinates.

📝 Shows predicted letter on live camera feed.

🛠 Tech Stack

Python (Model training)

TensorFlow / Keras (Model building)

MediaPipe Hands (Hand landmark detection)

OpenCV (Camera feed processing)

JavaScript / HTML (Frontend real-time view)

NumPy & Pandas (Data processing)

Matplotlib (Training visualization)

📂 Project Structure
📦 Sign-Language-Detection
 ┣ 📂 dataset/                # Training dataset (landmark coordinates)
 ┣ 📂 models/                 # Saved ML model (.h5)
 ┣ 📂 scripts/                # Python training scripts
 ┣ 📂 webapp/                 # Web version with MediaPipe + JS
 ┣ 📜 README.md               # Project documentation
 ┣ 📜 requirements.txt        # Python dependencies
 ┗ 📜 license.txt             # License file

⚙️ Installation & Setup
1️⃣ Clone the repository
git clone https://github.com/yourusername/sign-language-detection.git
cd sign-language-detection

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run the training script
python train_model.py

4️⃣ Run the live detection app
python app.py

📊 How It Works

Data Collection → ASL images converted into landmark coordinates using MediaPipe.

Model Training → ML model learns patterns from X/Y coordinates.

Real-Time Detection → Webcam feed processed → Hand keypoints extracted → Model predicts the letter.

Output Display → Predicted letter shown live on video.

📷 Demo

(Add GIF or screenshot of the app running here)

📜 License

This project is licensed under the MIT License — meaning you are free to use, modify, and distribute it, even for commercial purposes, as long as you include the original license.

🙌 Acknowledgements

MediaPipe by Google for real-time hand tracking.

TensorFlow for model building.

Open-source contributors for datasets and tools.

If you like this project, ⭐ star this repository and feel free to contribute.