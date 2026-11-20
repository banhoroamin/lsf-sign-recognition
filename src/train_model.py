import csv

import numpy as np

import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from hand_detector import normalize_landmarks

CSV_PATH = "data/landmarks.csv"
MODEL_PATH = "models/classifier.pkl"
TEST_CATEGORY = "rotation"


def load_dataset():
    X = []
    y = []
    categories = []
    with open(CSV_PATH) as f:
        reader = csv.reader(f)
        next(reader)  # ignore l'en-tete
        for row in reader:
            y.append(row[0])
            categories.append(row[1])
            raw = [float(v) for v in row[2:]]
            X.append(normalize_landmarks(raw))
    return np.array(X), np.array(y), np.array(categories)

def main():
    X, y, categories = load_dataset()

    test_mask = categories == TEST_CATEGORY
    X_train, y_train = X[~test_mask], y[~test_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    if len(X_test) == 0:
        raise RuntimeError(
            f"Aucun echantillon de categorie '{TEST_CATEGORY}' trouve. "
            "Enregistre les 4 categories avant de lancer l'entrainement."
        )

    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    labels = sorted(set(y))
    print("Matrice de confusion (lignes = vraie lettre, colonnes = predite):")
    print("Ordre des lettres:", labels)
    print(confusion_matrix(y_test, y_pred, labels=labels))

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)
    print(f"Modele sauvegarde dans {MODEL_PATH}")


if __name__ == "__main__":
    main()