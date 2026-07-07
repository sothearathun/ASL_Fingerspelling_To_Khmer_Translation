import json
import os
import threading
import time

import av
import cv2
import numpy as np
import mediapipe as mp
import pyttsx3
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from tensorflow.keras.models import load_model

from translator import translate_to_khmer
from utils import normalize_landmarks

CONFIDENCE_THRESHOLD = 0.8
STABILITY_FRAMES = 12  # consecutive frames the same letter must hold before it's accepted
MAX_HISTORY_SHOWN = 5
SPEAK_DELAY = 2  # seconds between speech outputs


class SharedState:
    """Mutable state shared between the webrtc worker thread (video_frame_callback)
    and the main Streamlit script thread (buttons). Guarded by `lock` since the two
    threads read/write concurrently."""

    def __init__(self):
        self.lock = threading.Lock()
        self.current_word = ""
        self.last_prediction = None
        self.stable_count = 0
        self.accepted = False
        self.last_spoken_time = 0.0
        # Lazily created inside video_frame_callback (not via st.cache_resource),
        # because pyttsx3's SAPI5 backend is a COM object bound to the thread that
        # creates it -- it must be created on the same worker thread that calls it.
        self.tts_engine = None


@st.cache_resource
def load_asl_model():
    return load_model('asl_model.h5')


@st.cache_resource
def load_class_names():
    return np.load('class_names.npy', allow_pickle=True)


@st.cache_resource
def load_hands():
    mp_hands = mp.solutions.hands
    return mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)


@st.cache_resource
def load_model_accuracy():
    if os.path.exists('model_accuracy.json'):
        with open('model_accuracy.json') as f:
            return json.load(f).get('accuracy')
    return None


if "shared" not in st.session_state:
    st.session_state.shared = SharedState()
if "history" not in st.session_state:
    st.session_state.history = []

shared = st.session_state.shared
model = load_asl_model()
class_names = load_class_names()
hands = load_hands()
model_accuracy = load_model_accuracy()
mp_drawing = mp.solutions.drawing_utils
mp_hands_connections = mp.solutions.hands.HAND_CONNECTIONS


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    detected_letter = None
    detected_confidence = 0.0

    with shared.lock:
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands_connections)

                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                landmarks = normalize_landmarks(landmarks)

                prediction = model.predict(np.array([landmarks]), verbose=0)
                predicted_class = np.argmax(prediction)
                confidence = np.max(prediction)

                if confidence > CONFIDENCE_THRESHOLD:
                    detected_letter = class_names[predicted_class]
                    detected_confidence = confidence

                    if detected_letter == shared.last_prediction:
                        shared.stable_count += 1
                    else:
                        shared.last_prediction = detected_letter
                        shared.stable_count = 1
                        shared.accepted = False

                    if shared.stable_count >= STABILITY_FRAMES and not shared.accepted:
                        shared.current_word += detected_letter
                        shared.accepted = True

                        now = time.time()
                        if now - shared.last_spoken_time > SPEAK_DELAY:
                            if shared.tts_engine is None:
                                shared.tts_engine = pyttsx3.init()
                                shared.tts_engine.setProperty('rate', 150)
                            shared.tts_engine.say(detected_letter)
                            shared.tts_engine.runAndWait()
                            shared.last_spoken_time = now
                else:
                    shared.last_prediction = None
                    shared.stable_count = 0
                    shared.accepted = False
        else:
            shared.last_prediction = None
            shared.stable_count = 0
            shared.accepted = False

        current_word_snapshot = shared.current_word

    if detected_letter:
        cv2.putText(img, f"{detected_letter} ({detected_confidence:.2f})",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, f"Word: {current_word_snapshot}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    return av.VideoFrame.from_ndarray(img, format="bgr24")


st.title("ASL Fingerspelling → Khmer Translator")

col1, col2 = st.columns([2, 1])

with col1:
    webrtc_streamer(
        key="asl",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
    )
    if model_accuracy is not None:
        st.caption(f"Model accuracy: {model_accuracy * 100:.1f}%")
    st.caption("The current word being signed is shown live in the video overlay above.")

with col2:
    st.subheader("Controls")
    b1, b2 = st.columns(2)
    with b1:
        finalize = st.button("Finalize Word", use_container_width=True)
    with b2:
        delete_last = st.button("Delete Last Letter", use_container_width=True)

    b3, b4 = st.columns(2)
    with b3:
        clear_word = st.button("Clear Word", use_container_width=True)
    with b4:
        reset_history = st.button("Reset History", use_container_width=True)

    if finalize:
        with shared.lock:
            word = shared.current_word
            shared.current_word = ""
        if word:
            khmer = translate_to_khmer(word)
            st.session_state.history.append((word, khmer))
            st.session_state.history = st.session_state.history[-MAX_HISTORY_SHOWN:]

    if delete_last:
        with shared.lock:
            shared.current_word = shared.current_word[:-1]

    if clear_word:
        with shared.lock:
            shared.current_word = ""

    if reset_history:
        st.session_state.history = []

    st.subheader("History")
    if not st.session_state.history:
        st.caption("No finalized words yet.")
    for eng, khm in reversed(st.session_state.history):
        st.markdown(f"**{eng}**")
        st.write(khm)
        st.divider()
