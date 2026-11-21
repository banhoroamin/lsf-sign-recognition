import pickle
import time
from collections import Counter, deque

import cv2
import mediapipe as mp

from hand_detector import (
    build_landmarker,
    draw_landmarks,
    build_features,
    flatten_landmarks,
    load_reference_images,
)

MODEL_PATH = "models/classifier.pkl"
HISTORY_SIZE = 10
CONFIDENCE_THRESHOLD = 0.70  # confiance minimale pour qu'une lettre compte
HOLD_SECONDS = 1.5           # duree pendant laquelle il faut tenir la lettre
GRACE_SECONDS = 0.4          # un trou plus court que ca n'interrompt pas le chrono


def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def stabilize(history, new_label):
    history.append(new_label)
    return Counter(history).most_common(1)[0][0]


class Dictation:
    """Transforme une suite de lettres reconnues en texte, en exigeant que chaque lettre soit tenue."""

    def __init__(
        self,
        threshold=CONFIDENCE_THRESHOLD,
        hold_seconds=HOLD_SECONDS,
        grace_seconds=GRACE_SECONDS,
    ):
        self.threshold = threshold
        self.hold_seconds = hold_seconds
        self.grace_seconds = grace_seconds
        self.text = ""
        self.candidate = None       # lettre en cours de maintien
        self.start_time = None      # heure a laquelle elle a commence
        self.last_good_time = None  # derniere image ou la lettre etait confiante
        self.validated = False      # deja ajoutee au texte ?

    def reset_candidate(self):
        self.candidate = None
        self.start_time = None
        self.last_good_time = None
        self.validated = False

    def update(self, label, confidence, now):
        """Met a jour le chrono. Retourne la progression de 0.0 a 1.0."""
        good = label is not None and confidence >= self.threshold

        if good:
            self.last_good_time = now
            if label != self.candidate:
                self.candidate = label
                self.start_time = now
                self.validated = False
        elif self.candidate is None or now - self.last_good_time > self.grace_seconds:
            # trou trop long (ou rien en cours) : on repart de zero
            self.reset_candidate()
            return 0.0

        if self.validated:
            return 1.0

        progress = (now - self.start_time) / self.hold_seconds
        if progress >= 1.0 and good:
            # on ne valide que sur une image confiante, jamais pendant un trou
            self.text += label
            self.validated = True
            return 1.0
        return min(progress, 0.99)

    def add_space(self):
        self.text += " "

    def backspace(self):
        self.text = self.text[:-1]

    def clear(self):
        self.text = ""


def overlay_reference(frame, image, max_size=220, margin=10):
    """Colle l'image de reference en haut a droite de la frame (modifie la frame sur place)."""
    h, w = image.shape[:2]
    scale = max_size / max(h, w)
    small = cv2.resize(image, (int(w * scale), int(h * scale)))
    sh, sw = small.shape[:2]
    frame_w = frame.shape[1]
    x0 = frame_w - sw - margin
    y0 = margin
    frame[y0:y0 + sh, x0:x0 + sw] = small
    cv2.rectangle(frame, (x0, y0), (x0 + sw, y0 + sh), (255, 255, 255), 2)


def draw_progress_bar(frame, progress, x=10, y=70, width=300, height=18):
    """Barre qui se remplit pendant que la lettre est tenue (verte une fois validee)."""
    cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 255, 255), 2)
    filled = int(width * progress)
    if filled > 0:
        color = (0, 255, 0) if progress >= 1.0 else (0, 200, 255)
        cv2.rectangle(frame, (x, y), (x + filled, y + height), color, -1)


def draw_text_zone(frame, text, height=60):
    """Bandeau sombre en bas de l'image avec le texte dicte (on garde la fin si c'est trop long)."""
    frame_h, frame_w = frame.shape[:2]
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, frame_h - height), (frame_w, frame_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    shown = text
    while shown and cv2.getTextSize(shown + "_", font, scale, thickness)[0][0] > frame_w - 20:
        shown = shown[1:]
    cv2.putText(frame, shown + "_", (10, frame_h - 20), font, scale, (255, 255, 255), thickness)


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
    selected_letter = None
    dictation = Dictation()

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
            now = time.monotonic()

            if hand_landmarks:
                draw_landmarks(frame, hand_landmarks)
                raw = flatten_landmarks(hand_landmarks)
                features = build_features(raw)

                probabilities = clf.predict_proba([features])[0]
                best_index = probabilities.argmax()
                predicted_label = clf.classes_[best_index]
                confidence = probabilities[best_index]

                stable_label = stabilize(history, predicted_label)
                progress = dictation.update(predicted_label, confidence, now)

                text = f"{stable_label} ({confidence * 100:.0f}%)"
                cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            else:
                progress = dictation.update(None, 0.0, now)
                cv2.putText(frame, "Aucune main detectee", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            if progress > 0:
                draw_progress_bar(frame, progress)
            draw_text_zone(frame, dictation.text)

            if selected_letter is not None:
                overlay_reference(frame, reference_images[selected_letter])

            cv2.imshow("Reconnaissance LSF", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # Echap
                break
            elif 97 <= key <= 122:
                letter = chr(key).upper()
                if letter in reference_images:
                    selected_letter = letter
            elif key == 32:  # Espace
                dictation.add_space()
            elif key in (8, 127):  # Retour arriere : efface la derniere lettre
                dictation.backspace()
            elif key in (13, 10):  # Entree : efface tout le texte
                dictation.clear()
            elif key == 9:  # Tab : masque l'image de reference
                selected_letter = None
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()


if __name__ == "__main__":
    main()
