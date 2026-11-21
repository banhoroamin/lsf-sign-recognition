import numpy as np
from sklearn.base import clone
from sklearn.model_selection import LeaveOneGroupOut

from hand_detector import angle_features, distance_features, normalize_landmarks
from train_model import load_dataset
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


CONFIDENCE_THRESHOLD = 0.70  # le meme seuil que pour la dictee


def add_features(X, use_distances, use_angles):
    """Construit la matrice de caracteristiques : base (63) + distances et/ou angles."""
    rows = []
    for row in X:
        base = row.tolist()
        features = list(base)
        if use_distances:
            features += distance_features(base)
        if use_angles:
            features += angle_features(base)
        rows.append(features)
    return np.array(rows)


def evaluate(model, X, y, groups):
    """Evaluation 'une categorie a tour de role'.

    Retourne (precision moyenne, taux correct-et-confiant moyen, reussite par lettre).
    """
    letters = sorted(set(y))
    accuracies = []
    confident_rates = []
    per_letter = {letter: [] for letter in letters}

    for train_idx, test_idx in LeaveOneGroupOut().split(X, y, groups):
        fold_model = clone(model)  # copie neuve : rien n'est retenu d'un tour a l'autre
        fold_model.fit(X[train_idx], y[train_idx])

        probabilities = fold_model.predict_proba(X[test_idx])
        predicted = fold_model.classes_[probabilities.argmax(axis=1)]
        confidence = probabilities.max(axis=1)
        correct = predicted == y[test_idx]

        accuracies.append(correct.mean())
        confident_rates.append((correct & (confidence >= CONFIDENCE_THRESHOLD)).mean())
        for letter in letters:
            mask = y[test_idx] == letter
            if mask.any():
                per_letter[letter].append(correct[mask].mean())

    recall = {letter: float(np.mean(scores)) for letter, scores in per_letter.items()}
    return float(np.mean(accuracies)), float(np.mean(confident_rates)), recall


def worst_letters(recall, n=3):
    """Les n lettres les moins bien reconnues, sous forme de texte."""
    ranked = sorted(recall.items(), key=lambda item: item[1])[:n]
    return ", ".join(f"{letter} {score:.0%}" for letter, score in ranked)

FEATURE_SETS = {
    "base": (False, False),
    "+distances": (True, False),
    "+angles": (False, True),
    "tout": (True, True),
}


def make_classifiers():
    return {
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "GradBoosting": HistGradientBoostingClassifier(random_state=42),
        "SVM": make_pipeline(StandardScaler(), SVC(probability=True, random_state=42)),
        "MLP": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42),
        ),
    }


def main():
    # On charge les points normalises (63) : add_features y ajoute ensuite distances et angles.
    X, y, groups = load_dataset(normalize_landmarks)

    print(f"{'caracteristiques':<17} {'modele':<13} {'precision':>9} {'confiant':>9}  pires lettres")
    results = []
    for feature_name, (use_distances, use_angles) in FEATURE_SETS.items():
        X_features = add_features(X, use_distances, use_angles)
        for model_name, model in make_classifiers().items():
            accuracy, confident, recall = evaluate(model, X_features, y, groups)
            print(
                f"{feature_name:<17} {model_name:<13} {accuracy:>9.1%} {confident:>9.1%}  {worst_letters(recall)}",
                flush=True,
            )
            results.append((confident, accuracy, feature_name, model_name))

    print("\nTop 5 par taux correct-et-confiant:")
    for confident, accuracy, feature_name, model_name in sorted(results, reverse=True)[:5]:
        print(f"  {feature_name} + {model_name}: correct et confiant {confident:.1%}, precision {accuracy:.1%}")


if __name__ == "__main__":
    main()