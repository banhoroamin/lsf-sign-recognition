"""
Étape 1 : tracking de la main (démo).

Ce script ouvre la webcam et affiche les 21 points de la main en direct,
en réutilisant la détection définie dans hand_detector.py.

Quitter avec la touche 'q'.
"""

import time

import cv2
import mediapipe as mp

from hand_detector import build_landmarker, draw_landmarks


def main():
    landmarker = build_landmarker()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Impossible d'ouvrir la webcam.")

    time.sleep(1.5)  # laisse le temps à la webcam de démarrer réellement

    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)  # effet miroir, plus naturel
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            timestamp_ms = int(frame_index * (1000 / 30))  # ~30 fps
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            frame_index += 1

            if result.hand_landmarks:
                draw_landmarks(frame, result.hand_landmarks[0])

            cv2.imshow("Tracking de la main", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()


if __name__ == "__main__":
    main()
