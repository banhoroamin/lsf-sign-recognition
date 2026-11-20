import pickle
import time
from collections import Counter, deque

import cv2
import mediapipe as mp

from hand_detector import (
    build_landmarker,
    draw_landmarks,
    flatten_landmarks,
    load_reference_images,
    normalize_landmarks,
)

MODEL_PATH = "models/classifier.pkl"
HISTORY_SIZE = 10


def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def stabilize(history, new_label):
    history.append(new_label)
    return Counter(history).most_common(1)[0][0]


def main():
    clf = load_model()
    landmarker = build_landmarker()
    reference_images = load_reference_images()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Impossible d'ouvrir la webcam.")

    time.sleep(1.5)  # laisse le temps à la webcam de démarrer réellement

    history = deque(maxlen=HISTORY_SIZE)
    frame_index = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            timestamp_ms = int(frame_index * (1000 / 30))
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            frame_index += 1

            hand_landmarks = result.hand_landmarks[0] if result.hand_landmarks else None

            if hand_landmarks:
                draw_landmarks(frame, hand_landmarks)
                raw = flatten_landmarks(hand_landmarks)
                features = normalize_landmarks(raw)

                probabilities = clf.predict_proba([features])[0]
                best_index = probabilities.argmax()
                predicted_label = clf.classes_[best_index]
                confidence = probabilities[best_index]

                stable_label = stabilize(history, predicted_label)

                text = f"{stable_label} ({confidence * 100:.0f}%)"
                cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            else:
                cv2.putText(frame, "Aucune main detectee", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow("Reconnaissance LSF", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # Echap
                break
            elif 97 <= key <= 122:
                letter = chr(key).upper()
                if letter in reference_images:
                    cv2.imshow("Reference", reference_images[letter])
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()


if __name__ == "__main__":
    main()
