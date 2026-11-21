import csv

import numpy as np

import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from hand_detector import build_features

CSV_PATH = "data/landmarks.csv"
MODEL_PATH = "models/classifier.pkl"
TEST_CATEGORY = "rotation"


def load_dataset(feature_fn=build_features):
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
            X.append(feature_fn(raw))
    return np.array(X), np.array(y), np.array(categories)


def make_model():
    return RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)


def main():
    X, y, categories = load_dataset()

    if TEST_CATEGORY not in categories:
        raise RuntimeError(
            f"Aucun echantillon de categorie '{TEST_CATEGORY}' trouve. "
            "Enregistre les 4 categories avant de lancer l'entrainement."
        )

    # 1. Evaluation honnete : chaque categorie est mise de cote a tour de role.
    print("Precision quand une categorie entiere est mise de cote:")
    for held_out in sorted(set(categories)):
        mask = categories == held_out
        model = make_model().fit(X[~mask], y[~mask])
        accuracy = (model.predict(X[mask]) == y[mask]).mean()
        print(f"  {held_out:<11} {accuracy:.1%}")

    # 2. Rapport detaille sur la categorie de test habituelle.
    test_mask = categories == TEST_CATEGORY
    clf = make_model().fit(X[~test_mask], y[~test_mask])
    y_test = y[test_mask]
    y_pred = clf.predict(X[test_mask])
    print(f"\nRapport detaille (categorie '{TEST_CATEGORY}' mise de cote):")
    print(classification_report(y_test, y_pred))

    labels = sorted(set(y))
    print("Matrice de confusion (lignes = vraie lettre, colonnes = predite):")
    print("Ordre des lettres:", labels)
    print(confusion_matrix(y_test, y_pred, labels=labels))

    # 3. Modele final : entraine sur TOUTES les categories (plus de donnees, plus de variete).
    final_model = make_model().fit(X, y)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(final_model, f)
    print(f"\nModele final (entraine sur les {len(set(categories))} categories) sauvegarde dans {MODEL_PATH}")


if __name__ == "__main__":
    main()
