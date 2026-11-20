import csv
import os
import time

import cv2
import mediapipe as mp

from hand_detector import (
    build_landmarker,
    draw_landmarks,
    flatten_landmarks,
    load_reference_images,
)

CSV_PATH = "data/landmarks.csv"
TARGET_SAMPLES = 150
CATEGORIES = ["statique", "vertical", "horizontal", "rotation"]


def choose_category():
    print("Categories disponibles:")
    for i, cat in enumerate(CATEGORIES, start=1):
        print(f"  {i}. {cat}")
    while True:
        choice = input("Quelle categorie pour cette session ? (1-4): ").strip()
        if choice in ("1", "2", "3", "4"):
            return CATEGORIES[int(choice) - 1]
        print("Choix invalide, reessaie.")


def save_samples(label, category, samples):
    file_exists = os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            header = ["label", "category"] + [f"f{i}" for i in range(63)]
            writer.writerow(header)
        for features in samples:
            writer.writerow([label, category] + features)


def main():
    category = choose_category()

    landmarker = build_landmarker()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Impossible d'ouvrir la webcam.")

    time.sleep(1.5)  # laisse le temps à la webcam de démarrer réellement

    reference_images = load_reference_images()

    recording_label = None
    is_recording = False
    recording_samples = []
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

            if recording_label is not None:
                if recording_label in reference_images:
                    cv2.imshow("Reference", reference_images[recording_label])

                if not is_recording:
                    text = f"[{category}] {recording_label} selectionne - Entree pour demarrer, Echap pour annuler"
                    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                else:
                    if hand_landmarks:
                        features = flatten_landmarks(hand_landmarks)
                        recording_samples.append(features)

                    text = f"[{category}] Enregistrement {recording_label}: {len(recording_samples)}/{TARGET_SAMPLES}"
                    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                    if len(recording_samples) >= TARGET_SAMPLES:
                        save_samples(recording_label, category, recording_samples)
                        print(f"{TARGET_SAMPLES} echantillons sauvegardes pour la lettre {recording_label} ({category})")
                        recording_label = None
                        is_recording = False
                        recording_samples = []
            else:
                cv2.putText(
                    frame,
                    f"[{category}] Appuie sur une lettre pour enregistrer, Echap pour quitter",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

            cv2.imshow("Collecte de donnees", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # Echap : annule la selection, ou quitte si rien n'est selectionne
                if recording_label is None:
                    break
                recording_label = None
                is_recording = False
                recording_samples = []
            elif not is_recording and 97 <= key <= 122:
                recording_label = chr(key).upper()
                recording_samples = []
            elif recording_label is not None and not is_recording and key in (13, 10):
                is_recording = True
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()


if __name__ == "__main__":
    main()