import cv2
import numpy as np
import mediapipe as mp
import pyttsx3
import time
import json
import os
from tensorflow.keras.models import load_model
from utils import normalize_landmarks, draw_khmer_text
from translator import translate_to_khmer

engine = pyttsx3.init()
engine.setProperty('rate', 150)

# -------------------------------------------------------------
# NOTE:
# The trained model (asl_model.h5) is *not included* in the repo.
# Please generate it by running:
#   1. src/data_collection.py  → to collect hand landmark data
#   2. src/train_model.py       → to train and create asl_model.h5
# This ensures higher accuracy for your own camera & lighting setup.
# -------------------------------------------------------------

model = load_model('asl_model.h5')
class_names = np.load('class_names.npy', allow_pickle=True)

model_accuracy = None
if os.path.exists('model_accuracy.json'):
    with open('model_accuracy.json') as f:
        model_accuracy = json.load(f).get('accuracy')

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
last_spoken_time = 0
speak_delay = 2  # Seconds between speech outputs

CONFIDENCE_THRESHOLD = 0.8
STABILITY_FRAMES = 12  # consecutive frames the same letter must hold before it's accepted
HISTORY_PANEL_WIDTH = 350
MAX_HISTORY_SHOWN = 5

last_prediction = None
stable_count = 0
accepted = False

current_word = ""
last_khmer = ""
history = []  # list of (english_word, khmer_word)

prev_frame_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    detected_letter = None
    detected_confidence = 0.0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            landmarks = normalize_landmarks(landmarks)

            prediction = model.predict(np.array([landmarks]), verbose=0)
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction)

            if confidence > CONFIDENCE_THRESHOLD:
                detected_letter = class_names[predicted_class]
                detected_confidence = confidence

                if detected_letter == last_prediction:
                    stable_count += 1
                else:
                    last_prediction = detected_letter
                    stable_count = 1
                    accepted = False

                if stable_count >= STABILITY_FRAMES and not accepted:
                    current_word += detected_letter
                    accepted = True

                    current_time = time.time()
                    if current_time - last_spoken_time > speak_delay:
                        engine.say(detected_letter)
                        engine.runAndWait()
                        last_spoken_time = current_time
            else:
                last_prediction = None
                stable_count = 0
                accepted = False
    else:
        last_prediction = None
        stable_count = 0
        accepted = False

    # ---- FPS ----
    now = time.time()
    fps = 1.0 / max(now - prev_frame_time, 1e-6)
    prev_frame_time = now

    # ---- overlay on camera frame ----
    if detected_letter:
        cv2.putText(frame, f"{detected_letter} ({detected_confidence:.2f})",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Word: {current_word}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(frame, "Khmer:", (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    (prefix_w, _), _ = cv2.getTextSize("Khmer: ", cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    frame = draw_khmer_text(frame, last_khmer, (10 + prefix_w, 115), font_size=26, color=(255, 255, 0))
    cv2.putText(frame, f"FPS: {fps:.0f}  {time.strftime('%H:%M:%S')}",
                (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    if model_accuracy is not None:
        cv2.putText(frame, f"Model accuracy: {model_accuracy * 100:.1f}%",
                    (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # ---- side history panel ----
    panel = np.zeros((frame.shape[0], HISTORY_PANEL_WIDTH, 3), dtype=np.uint8)
    cv2.putText(panel, "History", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y = 70
    for eng, khm in reversed(history):
        cv2.putText(panel, eng, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        panel = draw_khmer_text(panel, khm, (10, y + 25), font_size=20, color=(0, 255, 255))
        y += 60

    display = np.hstack((frame, panel))
    cv2.imshow('ASL Recognition', display)

    # ---- keyboard controls ----
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == 32:  # SPACE: finalize word
        if current_word:
            last_khmer = translate_to_khmer(current_word)
            history.append((current_word, last_khmer))
            history = history[-MAX_HISTORY_SHOWN:]
            current_word = ""
    elif key in (8, 127):  # Backspace/Delete: drop last letter
        current_word = current_word[:-1]
    elif key == ord('c'):  # clear current word
        current_word = ""
    elif key == ord('r'):  # reset history
        history = []
        last_khmer = ""

cap.release()
cv2.destroyAllWindows()
