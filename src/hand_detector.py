import math
import os
from itertools import combinations

import cv2
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

MODEL_PATH = "models/hand_landmarker.task"
REFERENCE_DIR = "data/reference"

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # pouce
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (5, 9), (9, 10), (10, 11), (11, 12),   # majeur
    (9, 13), (13, 14), (14, 15), (15, 16), # annulaire
    (13, 17), (17, 18), (18, 19), (19, 20),# auriculaire
    (0, 17),                               # paume
]


def build_landmarker():
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
    )
    return vision.HandLandmarker.create_from_options(options)


def draw_landmarks(frame, hand_landmarks):
    h, w, _ = frame.shape
    points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (255, 255, 255), 2)
    for x, y in points:
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)


def load_reference_images():
    images = {}
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        path = os.path.join(REFERENCE_DIR, f"{letter}.jpg")
        if os.path.exists(path):
            images[letter] = cv2.imread(path)
    return images


def flatten_landmarks(hand_landmarks):
    """Convertit les landmarks MediaPipe (objets x/y/z) en une liste plate de 63 nombres bruts."""
    coords = []
    for lm in hand_landmarks:
        coords.extend([lm.x, lm.y, lm.z])
    return coords


def normalize_landmarks(raw):
    """Normalise une liste plate de 63 coordonnees brutes (translation + rotation + echelle)."""
    points = [raw[i:i + 3] for i in range(0, len(raw), 3)]
    wrist = points[0]

    # 1. Translation : le poignet devient l'origine.
    translated = [[p[0] - wrist[0], p[1] - wrist[1], p[2] - wrist[2]] for p in points]

    # 2. Rotation : on aligne le vecteur poignet -> base du majeur vers le haut.
    mcp = translated[9]
    current_angle = math.atan2(mcp[1], mcp[0])
    target_angle = -math.pi / 2
    rotation = target_angle - current_angle
    cos_r, sin_r = math.cos(rotation), math.sin(rotation)

    rotated = []
    for x, y, z in translated:
        new_x = x * cos_r - y * sin_r
        new_y = x * sin_r + y * cos_r
        rotated.append([new_x, new_y, z])

    # 3. Echelle : on divise par la distance poignet -> base du majeur.
    mcp_rotated = rotated[9]
    scale = (mcp_rotated[0] ** 2 + mcp_rotated[1] ** 2 + mcp_rotated[2] ** 2) ** 0.5
    if scale == 0:
        scale = 1e-6

    features = []
    for x, y, z in rotated:
        features.extend([x / scale, y / scale, z / scale])
    return features


FINGERTIPS = [4, 8, 12, 16, 20]   # pouce, index, majeur, annulaire, auriculaire
FINGER_PIPS = [6, 10, 14, 18]     # 2e articulation de l'index, majeur, annulaire, auriculaire

# (a, b, c) : on mesure l'angle en b, entre les directions vers a et vers c
ANGLE_TRIPLETS = [
    (0, 1, 2), (1, 2, 3), (2, 3, 4),          # pouce
    (0, 5, 6), (5, 6, 7), (6, 7, 8),          # index
    (0, 9, 10), (9, 10, 11), (10, 11, 12),    # majeur
    (0, 13, 14), (13, 14, 15), (14, 15, 16),  # annulaire
    (0, 17, 18), (17, 18, 19), (18, 19, 20),  # auriculaire
]


def get_points(features):
    """Decoupe la liste plate de 63 nombres en 21 points [x, y, z]."""
    return [features[i:i + 3] for i in range(0, len(features), 3)]


def angle(a, b, c):
    """Angle (en radians) au point b, entre les directions vers a et vers c."""
    v1 = [a[i] - b[i] for i in range(3)]
    v2 = [c[i] - b[i] for i in range(3)]
    norm1 = math.sqrt(sum(x * x for x in v1))
    norm2 = math.sqrt(sum(x * x for x in v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    dot = sum(x * y for x, y in zip(v1, v2))
    cosine = max(-1.0, min(1.0, dot / (norm1 * norm2)))
    return math.acos(cosine)


def distance_features(features):
    """14 distances : entre les 5 bouts de doigts, et du pouce vers l'articulation des 4 doigts."""
    points = get_points(features)
    result = [math.dist(points[i], points[j]) for i, j in combinations(FINGERTIPS, 2)]
    result += [math.dist(points[4], points[j]) for j in FINGER_PIPS]
    return result


def angle_features(features):
    """15 angles de flexion, un par articulation."""
    points = get_points(features)
    return [angle(points[a], points[b], points[c]) for a, b, c in ANGLE_TRIPLETS]


def build_features(raw):
    """Caracteristiques finales pour le classifieur : points normalises (63) + distances (14) + angles (15)."""
    normalized = normalize_landmarks(raw)
    return normalized + distance_features(normalized) + angle_features(normalized)
