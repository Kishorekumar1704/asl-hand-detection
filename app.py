import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle

from collections import deque, Counter
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase


# ============================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ASL Sign Recognition",
    page_icon="🤚",
    layout="centered"
)


# ============================================================
# TITLE
# ============================================================

st.title("🤚 ASL Sign Recognition")

st.write(
    "Show an ASL hand sign (A–Z) to the webcam."
)

st.warning(
    "Allow camera permission when your browser asks for access."
)


# ============================================================
# LOAD MODEL AND LABEL ENCODER
# ============================================================

@st.cache_resource
def load_model_and_encoder():

    model = tf.keras.models.load_model(
        "model_groupAZ_final.h5"
    )

    with open(
        "label_encoder_groupAZ.pkl",
        "rb"
    ) as f:

        label_encoder = pickle.load(f)

    return model, label_encoder


# Load model
model, le = load_model_and_encoder()


# ============================================================
# MEDIAPIPE HAND CONFIGURATION
# ============================================================

mp_hands = mp.solutions.hands

mp_draw = mp.solutions.drawing_utils


# ============================================================
# ASL VIDEO PROCESSOR
# ============================================================

class ASLVideoProcessor(VideoProcessorBase):

    def __init__(self):

        # ----------------------------------------------------
        # MediaPipe Hands
        # ----------------------------------------------------

        self.hands = mp_hands.Hands(

            static_image_mode=False,

            max_num_hands=1,

            model_complexity=0,

            min_detection_confidence=0.7,

            min_tracking_confidence=0.7
        )


        # ----------------------------------------------------
        # Prediction History
        # ----------------------------------------------------

        self.label_history = deque(
            maxlen=10
        )


        # ----------------------------------------------------
        # Current Prediction
        # ----------------------------------------------------

        self.current_label = "Hand not detected"

        self.confidence = 0.0


    # ========================================================
    # PROCESS EACH VIDEO FRAME
    # ========================================================

    def recv(self, frame):

        # ----------------------------------------------------
        # Convert WebRTC frame to OpenCV
        # ----------------------------------------------------

        img = frame.to_ndarray(
            format="bgr24"
        )


        # ----------------------------------------------------
        # Flip image horizontally
        # ----------------------------------------------------

        img = cv2.flip(
            img,
            1
        )


        # ----------------------------------------------------
        # Convert BGR to RGB
        # ----------------------------------------------------

        rgb = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )


        # ----------------------------------------------------
        # MediaPipe Hand Detection
        # ----------------------------------------------------

        results = self.hands.process(
            rgb
        )


        # Default message

        display_text = "❗ Hand not detected"


        # ====================================================
        # HAND DETECTED
        # ====================================================

        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:


                # ------------------------------------------------
                # Draw hand landmarks
                # ------------------------------------------------

                mp_draw.draw_landmarks(

                    img,

                    hand_landmarks,

                    mp_hands.HAND_CONNECTIONS
                )


                # ------------------------------------------------
                # Get image dimensions
                # ------------------------------------------------

                h, w, _ = img.shape


                # ------------------------------------------------
                # Store coordinates
                # ------------------------------------------------

                coords = []


                # ------------------------------------------------
                # Extract 21 hand landmarks
                # ------------------------------------------------

                for lm in hand_landmarks.landmark:

                    x = int(
                        lm.x * w
                    )

                    y = int(
                        lm.y * h
                    )


                    coords.extend(
                        [x, y]
                    )


                # ------------------------------------------------
                # Verify 42 coordinates
                # ------------------------------------------------

                if len(coords) == 42:


                    # --------------------------------------------
                    # Convert to NumPy array
                    # --------------------------------------------

                    X = np.array(
                        coords,
                        dtype=np.float32
                    ).reshape(
                        1,
                        -1
                    )


                    # --------------------------------------------
                    # Model prediction
                    # --------------------------------------------

                    pred = model.predict(
                        X,
                        verbose=0
                    )


                    # --------------------------------------------
                    # Confidence
                    # --------------------------------------------

                    confidence = float(
                        np.max(pred)
                    )


                    # --------------------------------------------
                    # Get predicted class index
                    # --------------------------------------------

                    predicted_index = int(
                        np.argmax(pred)
                    )


                    # --------------------------------------------
                    # Convert index to label
                    # --------------------------------------------

                    label = le.inverse_transform(
                        [predicted_index]
                    )[0]


                    # =================================================
                    # CONFIDENCE CHECK
                    # =================================================

                    if confidence >= 0.70:


                        # ---------------------------------------------
                        # Add prediction to history
                        # ---------------------------------------------

                        self.label_history.append(
                            label
                        )


                        # ---------------------------------------------
                        # Find most common prediction
                        # ---------------------------------------------

                        common_label, count = Counter(
                            self.label_history
                        ).most_common(1)[0]


                        # ---------------------------------------------
                        # Stabilize prediction
                        # ---------------------------------------------

                        if count >= 5:

                            display_text = (
                                f"✅ {common_label} "
                                f"({confidence:.2f})"
                            )


                            self.current_label = (
                                common_label
                            )


                        else:

                            display_text = (
                                f"⏳ {label} "
                                "Stabilizing..."
                            )


                            self.current_label = (
                                label
                            )


                        self.confidence = (
                            confidence
                        )


                    # =================================================
                    # LOW CONFIDENCE
                    # =================================================

                    else:

                        self.label_history.clear()


                        display_text = (
                            f"⚠ Low Confidence "
                            f"({confidence:.2f})"
                        )


                        self.current_label = (
                            "Low confidence"
                        )


                        self.confidence = (
                            confidence
                        )


        # ====================================================
        # NO HAND DETECTED
        # ====================================================

        else:

            self.label_history.clear()


            self.current_label = (
                "Hand not detected"
            )


            self.confidence = 0.0


        # ====================================================
        # PREDICTION DISPLAY BOX
        # ====================================================

        cv2.rectangle(

            img,

            (10, 10),

            (450, 75),

            (0, 0, 0),

            -1
        )


        # ====================================================
        # DISPLAY TEXT
        # ====================================================

        cv2.putText(

            img,

            display_text,

            (20, 52),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            (255, 255, 255),

            2
        )


        # ====================================================
        # RETURN FRAME
        # ====================================================

        return frame.from_ndarray(

            img,

            format="bgr24"
        )


# ============================================================
# START WEBCAM
# ============================================================

webrtc_streamer(

    key="asl-sign-recognition",

    video_processor_factory=ASLVideoProcessor,

    media_stream_constraints={

        "video": True,

        "audio": False
    },

    async_processing=True
)


# ============================================================
# INFORMATION
# ============================================================

st.markdown("---")

st.subheader("How to use")

st.write(
    """
    1. Click **START** to enable the webcam.
    2. Allow camera permission.
    3. Show one hand clearly in front of the camera.
    4. Hold the ASL sign steady for a few moments.
    5. The prediction will appear on the video.
    """
)


st.subheader("Prediction Information")

st.write(
    """
    - 🟢 Confidence ≥ 70% → Prediction accepted
    - ⏳ Prediction is stabilized using recent frames
    - ⚠ Confidence < 70% → Low confidence
    - ❗ No hand detected → Show your hand clearly
    """
)