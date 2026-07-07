import cv2
import mediapipe as mp
import numpy as np
import os
import time
from utils import create_directory, normalize_landmarks

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def collect_data(sign_name, num_samples=100):
    create_directory('data')
    create_directory(f'data/{sign_name}')
    
    cap = cv2.VideoCapture(0)
    print(f"Collecting data for: {sign_name}")
    print("Press 'q' to stop")
    
    sample_count = 0
    while sample_count < num_samples:
        ret, frame = cap.read()
        if not ret:
            continue
            
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
                landmarks = normalize_landmarks(landmarks)

                np.save(f'data/{sign_name}/{sign_name}_{sample_count}.npy', np.array(landmarks))
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
    sign_name = input("Enter ASL sign name (e.g., 'A', 'B', 'Hello'): ")
    collect_data(sign_name)