# Reconnaissance de signes LSF par webcam

Reconnaissance en temps réel de l'alphabet dactylologique LSF (23 lettres statiques — J, P, Z exclues car dynamiques) via la webcam du Mac, en MediaPipe + scikit-learn, avec une **dictée par signes** : une lettre tenue assez longtemps s'écrit dans une zone de texte.

## Pipeline

1. **Webcam** (OpenCV) → image brute.
2. **MediaPipe Hand Landmarker** → 21 points 3D de la main.
3. **Caractéristiques** (92 nombres) :
   - les 21 points **normalisés** (translation + rotation + échelle) → invariants à la position, l'inclinaison et la distance à la caméra (63 nombres) ;
   - 14 **distances** (entre les bouts de doigts, du pouce vers les articulations) ;
   - 15 **angles de flexion** (un par articulation).
4. **Random Forest** (scikit-learn) → classification en lettre, avec une confiance.
5. **Affichage en direct** de la lettre prédite (lissée sur 10 images) et de l'image de référence du signe.
6. **Dictée** : une lettre est ajoutée au texte quand elle est tenue `HOLD_SECONDS` (1,5 s) avec une confiance ≥ `CONFIDENCE_THRESHOLD` (0,70). Une tolérance `GRACE_SECONDS` (0,4 s) évite qu'une image isolée sous le seuil remette le chrono à zéro.

## Structure des fichiers

- `src/hand_detector.py` — brique partagée : détection MediaPipe, dessin du squelette, normalisation, distances, angles (`build_features`), chargement des images de référence.
- `src/hand_tracking.py` — démo initiale (webcam + squelette, sans reconnaissance).
- `src/collect_data.py` — interface de collecte : webcam + touches clavier + choix de catégorie de mouvement, sauvegarde des points bruts.
- `src/train_model.py` — entraînement et évaluation du classifieur (une catégorie mise de côté à tour de rôle), puis modèle final entraîné sur toutes les catégories.
- `src/recognize.py` — reconnaissance en temps réel + dictée (classe `Dictation`).
- `src/experiment.py` — banc d'essai qui compare des jeux de caractéristiques et des classifieurs avec la même évaluation, sans toucher au vrai modèle.
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
python src/recognize.py      # reconnaissance en direct + dictée
python src/experiment.py     # comparer caractéristiques et classifieurs (plusieurs minutes)
```

Touches dans `recognize.py` :

| Touche | Action |
|---|---|
| lettre (a–z) | affiche l'image de référence de cette lettre |
| Tab | masque l'image de référence |
| Espace | ajoute une espace au texte dicté |
| Retour arrière | efface la dernière lettre |
| Entrée | efface tout le texte |
| Échap | quitte |

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
- **Évaluer « une catégorie à tour de rôle »** (`LeaveOneGroupOut`) donne un chiffre plus solide qu'un seul découpage : 93,7 % pour la base, 97 % avec les caractéristiques enrichies.
- **Les bonnes caractéristiques valent mieux qu'un modèle plus compliqué.** Les distances entre doigts ont amélioré les 5 classifieurs testés (Random Forest, Extra Trees, gradient boosting, SVM, réseau de neurones).
- **Confiance ≠ précision.** Le réseau de neurones affichait le meilleur taux « correct et confiant » (94,7 %) mais était surconfiant : 2,7 % de fausses lettres passaient le seuil de 0,70, contre 0,06 % pour le Random Forest, retenu. Il faut mesurer le taux de fausses lettres confiantes avant de choisir.
- **Les règles d'une interface comptent autant que le modèle.** Avec un seuil de 0,90, seules 38 % des bonnes images passaient : le chrono de la dictée se remettait à zéro en permanence. Un seuil de 0,70 et une tolérance de 0,4 s ont réglé le problème sans changer le modèle.
- **Entraîner le modèle final sur toutes les données** une fois l'évaluation honnête terminée.

### Débogage en conditions réelles
- Permissions caméra macOS, et la Continuity Camera de l'iPhone qui peut prendre la priorité sur la webcam du Mac.
- Bug de compilation `pip` sur un fichier de test corrompu du paquet `mediapipe` (`--no-compile`).
- Changement d'API MediaPipe (`solutions` → `tasks`) qui invalide une grande partie des tutoriels trouvés en ligne.
- Ordre d'exécution du code Python : un `if __name__ == "__main__":` placé avant la définition d'une fonction qu'il appelle casse tout.

### Propriété intellectuelle
- Vérifier la licence des ressources externes utilisées (images Wikimedia Commons, CC BY-SA) et créditer la source.

## Limites connues
- **J, P, Z non reconnus** : signes dynamiques, incompatibles avec une approche "photo instantanée" des points de la main.
- **O, F/T, I/Y** restent les lettres les plus fragiles (O notamment, souvent confondue avec C) — ressemblance géométrique réelle entre certains signes, ou occlusion du pouce par l'index (profondeur peu fiable).
- **Pas d'espace ni de ponctuation dans le signe** : l'espace et l'effacement se font au clavier.
- **Prochaine étape envisagée** : la reconnaissance dynamique (J, P, Z), qui demande un nouveau format de données (suites d'images) et des caractéristiques de trajectoire.
- **Un seul signeur** : les données viennent d'une seule personne ; le modèle ne généralisera probablement pas bien à la main de quelqu'un d'autre sans données supplémentaires.
