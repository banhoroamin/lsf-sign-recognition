# Reconnaissance de signes LSF par webcam

Reconnaissance en temps réel de l'alphabet dactylologique LSF (23 lettres statiques — J, P, Z exclues car dynamiques) via la webcam du Mac, en MediaPipe + scikit-learn.

## Pipeline

1. **Webcam** (OpenCV) → image brute.
2. **MediaPipe Hand Landmarker** → 21 points 3D de la main.
3. **Normalisation** (translation + rotation + échelle) → 63 nombres invariants à la position, l'inclinaison et la distance à la caméra.
4. **Random Forest** (scikit-learn) → classification des points en lettre.
5. **Affichage en direct** de la lettre prédite, lissée sur les 10 dernières images.

## Structure des fichiers

- `src/hand_detector.py` — brique partagée : détection MediaPipe, dessin du squelette, normalisation, chargement des images de référence.
- `src/hand_tracking.py` — démo initiale (webcam + squelette, sans reconnaissance).
- `src/collect_data.py` — interface de collecte : webcam + touches clavier + choix de catégorie de mouvement, sauvegarde des points bruts.
- `src/train_model.py` — entraînement et évaluation du classifieur.
- `src/recognize.py` — reconnaissance en temps réel.
- `data/landmarks.csv` — données collectées (label, catégorie, 63 coordonnées brutes).
- `data/reference/` — images de référence de l'alphabet LSF (Wikimedia Commons, CC BY-SA, voir `CREDITS.md`).
- `models/hand_landmarker.task` — modèle MediaPipe pré-entraîné (téléchargé).
- `models/classifier.pkl` — notre classifieur entraîné sur nos propres données.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --no-compile -r requirements.txt
```

Le modèle MediaPipe n'est pas versionné (trop volumineux) : il faut le re-télécharger après un clone.

```bash
curl -L -o models/hand_landmarker.task "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
```

## Utilisation

```bash
python src/collect_data.py   # enregistrer de nouveaux signes
python src/train_model.py    # ré-entraîner le classifieur
python src/recognize.py      # reconnaissance en direct
```

## Bilan — ce qu'on a appris sur ce projet

### Organisation d'un projet Python
- Environnement virtuel (`.venv`) + `requirements.txt` pour isoler et reproduire les dépendances.
- Factoriser le code commun à plusieurs scripts dans un module partagé (`hand_detector.py`) plutôt que le copier-coller.
- Séparer la **capture** des données brutes du **traitement** (normalisation) : stocker du brut permet de changer la méthode de normalisation sans tout réenregistrer.

### Machine learning
- Un classifieur apprend une règle générale à partir d'exemples étiquetés, pour prédire l'étiquette de nouveaux exemples.
- **Feature engineering** : transformer les points 3D bruts en une représentation invariante (translation, échelle, rotation) avant de les donner au modèle — souvent plus efficace que de complexifier le modèle lui-même.
- **Random Forest** : ensemble d'arbres de décision qui votent ensemble, robuste par défaut sans réglage fin.
- **Train/test split stratifié**, et surtout : la **fuite de données** (data leakage). Un découpage aléatoire d'images issues d'une même séance continue donne un score trompeur (100% au départ), parce que le "test" n'est pas vraiment inédit. Évaluer sur une catégorie de mouvement entièrement différente (jamais vue à l'entraînement) a donné un score honnête (59% au départ), qui a ensuite guidé une vraie amélioration (93% après normalisation de rotation).
- **Matrice de confusion** : outil pour diagnostiquer précisément où un modèle se trompe plutôt que de deviner.
- Mesurer avant de supposer : la dispersion de G a été vérifiée numériquement avant de l'accuser — fausse piste, ce qui a évité de perdre du temps sur la mauvaise cause.

### Débogage en conditions réelles
- Permissions caméra macOS, et la Continuity Camera de l'iPhone qui peut prendre la priorité sur la webcam du Mac.
- Bug de compilation `pip` sur un fichier de test corrompu du paquet `mediapipe` (`--no-compile`).
- Changement d'API MediaPipe (`solutions` → `tasks`) qui invalide une grande partie des tutoriels trouvés en ligne.
- Ordre d'exécution du code Python : un `if __name__ == "__main__":` placé avant la définition d'une fonction qu'il appelle casse tout.

### Propriété intellectuelle
- Vérifier la licence des ressources externes utilisées (images Wikimedia Commons, CC BY-SA) et créditer la source.

## Limites connues
- **J, P, Z non reconnus** : signes dynamiques, incompatibles avec une approche "photo instantanée" des points de la main.
- **O/C, F/T, I/Y, G/L/D** restent les lettres les plus sujettes à confusion — ressemblance géométrique réelle entre certains signes, ou occlusion du pouce par l'index (profondeur peu fiable).
- **Un seul signeur** : les données viennent d'une seule personne ; le modèle ne généralisera probablement pas bien à la main de quelqu'un d'autre sans données supplémentaires.
