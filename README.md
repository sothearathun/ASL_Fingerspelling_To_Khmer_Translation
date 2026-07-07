ASL Fingerspelling to Khmer Translator
-
A real-time American Sign Language (ASL) fingerspelling recognizer that builds English words from individual signed letters and translates them into Khmer — using MediaPipe hand landmarks, not images.

---

About the Project
-
This started as a final-year project to make communication more accessible. A webcam feed is passed through MediaPipe to extract 21 hand landmarks per frame, which a small neural network classifies into ASL letters. Individual letters are stabilized and assembled into words, which can then be translated into Khmer and spoken aloud.

Two interfaces are available: a native OpenCV desktop window with keyboard controls, and a Streamlit web app with clickable buttons.

> **Scope note:** Recognition works on *static, held handshapes* only (fingerspelling letters and similarly static word-signs). Signs that involve motion (most full ASL vocabulary words) are not supported by this single-frame architecture — see [How It Works](#how-it-works).

---
Features
-
1) Real-time ASL letter recognition via webcam, trained on hand-landmark data (not images)
2) Prediction stabilization — a letter is only accepted once held steadily for several frames, avoiding repeat spam
3) Word building with edit controls (finalize, delete last letter, clear, reset history)
4) English → Khmer translation of finalized words, spoken aloud via text-to-speech
5) Live on-screen confidence, FPS, and model accuracy
6) A running history of translated words
7) Two interchangeable UIs: OpenCV desktop window or Streamlit web app
8) Easy to retrain on your own hand/camera setup, or extend to new signs

---

Tech Stack
-
> Python

> MediaPipe (hand landmark detection)

> OpenCV

> TensorFlow / Keras (letter classification model)

> scikit-learn (train/test split, evaluation metrics)

> deep-translator (English → Khmer translation)

> uharfbuzz + freetype-py (correct Khmer script shaping for the OpenCV overlay)

> pyttsx3 (offline text-to-speech)

> Streamlit + streamlit-webrtc (web UI alternative)

---

Folder Structure
-
```
Real-Time-ASL-Gesture-Recognition/
├── data/                 # Landmark training data (.npy) - created by data_collection.py, gitignored
├── asl_model.h5          # Trained model - created by train_model.py, gitignored
├── class_names.npy       # Class labels - created by train_model.py, gitignored
├── model_accuracy.json   # Test accuracy - created by train_model.py, gitignored
├── models/
│   └── metadata.json     # Model input configuration (included)
├── src/
│   ├── data_collection.py   # Records labeled hand-landmark samples from your webcam
│   ├── train_model.py       # Trains the classifier; prints a classification report + confusion matrix
│   ├── asl_recognition.py   # OpenCV desktop app: recognition, word-building, Khmer overlay, history panel
│   ├── asl_app.py           # Streamlit web app: same pipeline, browser UI with buttons
│   ├── translator.py        # English -> Khmer translation (deep-translator)
│   └── utils.py             # Landmark normalization, Khmer text shaping, data loading helpers
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```
---

Getting Started
-
1. Install dependencies

   ```pip install -r requirements.txt```

2. Collect your own data and train

   The trained model isn't included in this repo — accuracy depends heavily on your own hand shape, lighting, and camera, so you train it locally instead:

   ```python src/data_collection.py```

   Follow the prompt to enter a sign name (e.g. `A`) and record samples via webcam. Repeat once per letter/sign you want recognized, then:

   ```python src/train_model.py```

   This creates `asl_model.h5`, `class_names.npy`, and `model_accuracy.json` in the project root, and prints a classification report + confusion matrix so you can see which signs (if any) are being confused with each other.

3. Run it — pick one interface:

   **OpenCV desktop window:**
   ```python src/asl_recognition.py```

   Controls:
   | Key | Action |
   |---|---|
   | `SPACE` | Finalize current word → translate to Khmer → add to history |
   | `Backspace` | Delete last letter |
   | `c` | Clear current word |
   | `r` | Reset history |
   | `q` | Quit |

   **Streamlit web app:**
   ```streamlit run src/asl_app.py```

   Same recognition pipeline, opened in your browser with clickable buttons (Finalize Word / Delete Last Letter / Clear Word / Reset History) instead of keyboard shortcuts.

---

How It Works
-
1) MediaPipe detects 21 hand landmarks per webcam frame
2) Landmarks are normalized (centered on the wrist, scaled by palm size) so predictions are invariant to hand position/distance from the camera — see `normalize_landmarks` in `src/utils.py`
3) A small dense neural network classifies the normalized landmarks into a letter
4) A letter is only **accepted** once the same prediction holds steady for several consecutive frames, preventing a held sign from being registered repeatedly
5) Accepted letters accumulate into a word; finalizing a word (SPACE / button) translates it to Khmer via Google Translate and speaks it aloud
6) Because each prediction is a single static frame with no concept of motion or sequence, this only works for signs that are one held handshape — signs requiring movement would need a sequence model (e.g. LSTM over a window of frames) instead, which isn't implemented here

---

LICENSE
-
> This project is licensed under the MIT License.
