import argparse
import glob
import os
import re
import time

import cv2
import mediapipe as mp
import numpy as np
from utils import create_directory, normalize_landmarks

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

COUNTDOWN_SECONDS = 3


def _next_sample_index(sign_dir, sign_name):
    """Resume numbering from the highest existing sample index instead of
    always starting at 0, so re-running for a sign adds new samples instead
    of overwriting the ones already collected for it."""
    existing = glob.glob(os.path.join(sign_dir, f"{sign_name}_*.npy"))
    max_index = -1
    for path in existing:
        match = re.search(rf"{re.escape(sign_name)}_(\d+)\.npy$", os.path.basename(path))
        if match:
            max_index = max(max_index, int(match.group(1)))
    return max_index + 1


def collect_data(sign_name, num_samples=100):
    sign_dir = os.path.join('data', sign_name)
    create_directory('data')
    create_directory(sign_dir)

    start_index = _next_sample_index(sign_dir, sign_name)
    if start_index > 0:
        print(f"Found {start_index} existing sample(s) for '{sign_name}' - resuming from index {start_index}")


    cap = cv2.VideoCapture(0)
    print(f"Collecting data for: {sign_name}")
    print("Press 'q' to stop")

    countdown_start = time.time()
    sample_count = 0
    target_count = start_index + num_samples

    while start_index + sample_count < target_count:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        elapsed = time.time() - countdown_start
        remaining = COUNTDOWN_SECONDS - elapsed

        if remaining > 0:
            cv2.putText(frame, f"Get ready: {remaining:.1f}s", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
        else:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    landmarks = []
                    for landmark in hand_landmarks.landmark:
                        landmarks.extend([landmark.x, landmark.y, landmark.z])
                    landmarks = normalize_landmarks(landmarks)

                    index = start_index + sample_count
                    np.save(os.path.join(sign_dir, f"{sign_name}_{index}.npy"), np.array(landmarks))
                    sample_count += 1
                    time.sleep(0.1)

            cv2.putText(frame, f"Collecting {sign_name}: {sample_count}/{num_samples}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('ASL Data Collection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect labeled hand-landmark samples for one ASL sign.")
    parser.add_argument('--sign', help="Sign name, e.g. 'A', 'B', 'Hello'")
    parser.add_argument('--num-samples', type=int, default=100, help="Number of samples to collect")
    args = parser.parse_args()

    sign = args.sign or input("Enter ASL sign name (e.g., 'A', 'B', 'Hello'): ")
    collect_data(sign, args.num_samples)
