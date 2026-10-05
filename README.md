# Greener — classification des déchets par région

Pipeline de classification d'images de déchets. L'application encode les images avec DINOv2, stocke les vecteurs de référence dans Qdrant et retourne, pour une région donnée, le matériau reconnu, la couleur de poubelle associée et le score de similarité.

## Table des matières

- [Vue d'ensemble](#vue-densemble)
- [Structure attendue des données](#structure-attendue-des-données)
- [Prérequis](#prérequis)
- [OBLIGATOIRE — télécharger et décompresser le dataset](#obligatoire--télécharger-et-décompresser-le-dataset)
- [Configuration](#configuration)
- [Installation et utilisation locale](#installation-et-utilisation-locale)
- [Docker Compose](#docker-compose)
- [API HTTP](#api-http)
- [Architecture et flux](#architecture-et-flux)
- [Dépannage](#dépannage)
- [Dépendances](#dépendances)

## Vue d'ensemble

Au démarrage, l'application :

1. charge les fichiers JSON de régions depuis `DATA_DIR` ;
2. valide chaque entrée avec Pydantic (`nom`, `region`, `poubelle`) ;
3. parcourt les images locales sous `IMAGE_DIR`, regroupées par libellé ;
4. encode les images avec le modèle Hugging Face `facebook/dinov2-large` (ou le modèle configuré) ;
5. crée une collection Qdrant par région, avec des vecteurs de distance cosinus, puis indexe les références ;
6. expose l'API FastAPI.

Une requête d'inférence encode l'image envoyée, interroge la collection Qdrant de la région demandée et renvoie le meilleur résultat.

## Structure attendue des données

### Fichiers JSON de régions

`DATA_DIR` doit contenir directement des fichiers `*.json`. Le nom du fichier, sans extension, devient le nom de la collection Qdrant et la valeur de `region`. La racine de chaque JSON doit être une liste d'objets contenant au minimum :

```json
[
  {
    "nom": "bouteille en plastique",
    "poubelle": "jaune"
  }
]
```

Les entrées invalides sont ignorées et journalisées. Les champs `nom`, `region` et `poubelle` doivent être des chaînes non vides ; `region` est ajouté par le code à partir du nom du fichier.

### Images de référence

`IMAGE_DIR` est la racine du dataset d'images. Chaque sous-répertoire immédiat représente un libellé et contient des fichiers `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp` ou `.gif` :

```text
<IMAGE_DIR>/
├── bouteille_en_plastique/
│   ├── image-001.jpg
│   └── image-002.png
└── canette/
    └── image-001.jpg
```

Le code normalise les libellés en minuscules et remplace les caractères non alphanumériques par `_` lors de la résolution d'un libellé.

## Prérequis

- Python `>=3.13` ;
- `uv` pour synchroniser les dépendances (ou un environnement virtuel Python équivalent) ;
- Docker et Docker Compose v2 si Qdrant est exécuté avec Compose ;
- suffisamment d'espace disque pour le dataset, les poids DINOv2 et le stockage Qdrant ;
- accès réseau initial pour télécharger les dépendances et le modèle Hugging Face.

## OBLIGATOIRE — télécharger et décompresser le dataset

Le projet ne peut pas fonctionner sans le dataset. Avant toute installation ou tout lancement, télécharger l'archive TAR du dataset depuis :

<https://drive.google.com/file/d/1VuH2IHp0lqivDrCTyw7hUcUPawtd0JDd/view>

Après téléchargement, décompresser l'archive dans le répertoire attendu par le code. Le chemin dépend du mode d'exécution :

- **Exécution locale** : `IMAGE_DIR` vaut par défaut `./image_backup`, donc le dataset d'images doit être sous `./image_backup/` ;
- **Docker Compose** : le volume `./image_dir:/greener/image_dir` est déclaré pour le conteneur ; configurer `IMAGE_DIR=/greener/image_dir` dans l'environnement du service `web`.

Exemple :

```bash
tar -xf <archive-du-dataset>.tar -C ./
```

Vérifier ensuite que les sous-répertoires de libellés sont directement accessibles sous `./image_dir/`, et non dans un niveau de répertoire supplémentaire. Les fichiers JSON de régions sont distincts : ils doivent être directement sous `DATA_DIR` — `./data/` en local ou `/greener/data` dans le conteneur.

## Configuration

Les paramètres sont lus par `pydantic-settings` depuis `.env` et les variables d'environnement. Les noms ci-dessous sont les alias déclarés dans `configs.py`.

| Variable | Défaut | Rôle |
|---|---:|---|
| `AI_HOST` | `0.0.0.0` | Adresse d'écoute de l'API. |
| `AI_PORT` | `8000` | Port d'écoute de l'API. |
| `EMBEDDING_MODEL_NAME` | `facebook/dinov2-small` | Identifiant du modèle d'embedding Hugging Face. |
| `QDRANT_HOST` | `localhost` | Hôte Qdrant. En Compose, utiliser `qdrant`. |
| `QDRANT_PORT` | `6333` | Port HTTP Qdrant. |
| `DATA_DIR` | `./data` | Répertoire contenant directement les JSON de régions. |
| `IMAGE_DIR` | `./image_dir` | Répertoire racine des images de référence. |
| `DEVICE` | `cpu` | Périphérique PyTorch (`cpu` ou `cuda`). |
| `MAX_CONCURRENT_UPLOADS` | `10` | Taille du sémaphore utilisé pour les chargements d'images. |
| `IMAGES_PER_LABEL` | `3` | Nombre maximal d'images chargées par libellé lors de l'indexation. |
| `LOG_LEVEL` | `INFO` | Niveau des journaux applicatifs. |

Variables de l'ancienne version **qui ne sont plus utilisées par l'application** : `REQUEST_TIMEOUT`, `MAX_RETRIES`, `MAX_CONCURRENT_DOWNLOADS`, `SAVE_IMAGES`, `IMAGE_BACKUP_DIR`.

Variables utilisées uniquement par Compose et `qdrant.dockerfile` :

| Variable | Utilisation |
|---|---|
| `QDRANT_VERSION` | Argument de build de l'image Qdrant ; défaut `v1.18.2`. |
| `QDRANT_GRPC_PORT` | Port gRPC publié par Compose. |
| `QDRANT_LOG_LEVEL` | Niveau de journalisation de Qdrant. |

Exemple de `.env` pour un lancement local :

```dotenv
AI_HOST=0.0.0.0
AI_PORT=8000
EMBEDDING_MODEL_NAME=facebook/dinov2-large
QDRANT_HOST=localhost
QDRANT_PORT=6333
DATA_DIR=./data
IMAGE_DIR=./image_backup
DEVICE=cpu
MAX_CONCURRENT_UPLOADS=10
IMAGES_PER_LABEL=3
LOG_LEVEL=INFO
QDRANT_VERSION=v1.18.2
QDRANT_GRPC_PORT=6334
QDRANT_LOG_LEVEL=INFO
```

## Installation et utilisation locale

Prérequis : le dataset doit avoir été téléchargé et décompressé (voir la section obligatoire ci-dessus).

Depuis la racine du projet :

```bash
uv sync
```

Démarrer Qdrant sur `localhost:6333`, puis lancer l'API :

```bash
uv run python main.py
```

`main.py` lance Uvicorn avec `AI_HOST` et `AI_PORT`. L'indexation est exécutée au démarrage de FastAPI ; le premier lancement peut être long, car il télécharge le modèle d'embedding.

## Docker Compose

`docker-compose.yaml` déclare :

- `web`, construit avec `build: .`, exposé sur `${AI_PORT}` ;
- `qdrant`, construit depuis `qdrant.dockerfile`, avec stockage persistant dans le volume `qdrant_storage`.

Montages du service `web` : `./image_dir:/greener/image_dir`, `./image_dir:/greener/image_dir` et `./data:/greener/data`. Définir dans l'environnement du conteneur `web` :

```dotenv
DATA_DIR=/greener/data
IMAGE_DIR=/greener/image_dir
QDRANT_HOST=qdrant
```

Lancement :

```bash
docker compose up --build -d
```

## API HTTP

### `POST /greener/upload/dechets`

Requête `multipart/form-data` :

| Champ | Type | Obligatoire | Description |
|---|---|---:|---|
| `file` | fichier | Oui | Image lisible par Pillow. |
| `region` | chaîne | Non | Nom de la collection Qdrant ; défaut : `ile_de_france`. |

Exemple :

```bash
curl -X POST "http://localhost:8000/greener/upload/dechets?region=ile_de_france" \
  -F "file=@./exemples/dechet.jpg"
```

Réponse HTTP 200 :

```json
{
  "material_name": "bouteille en plastique",
  "bin_color": "jaune",
  "score": 0.87
}
```

Codes d'erreur :

| HTTP | Situation |
|---:|---|
| `400` | Le fichier envoyé n'est pas une image valide. |
| `404` | La région demandée n'existe pas (`UnknownRegionError`). |
| `422` | Aucune correspondance trouvée (`NoMatchError`). |

La documentation interactive est disponible sur `/docs` lorsque le serveur est actif.

## Architecture et flux

```text
main.py
└── Viewer (FastAPI)
    ├── Controller
    │   └── Model
    │       ├── ImageEmbedder (DINOv2 + PyTorch/Transformers)
    │       └── AsyncQdrantClient
    └── lifespan
        └── run_startup_indexing
            ├── regions.py        -> JSON de DATA_DIR
            ├── ImageFetcher      -> images de IMAGE_DIR
            ├── RegionIndexer
            ├── ImageEmbedder
            └── VectorRepository  -> collections Qdrant
```

| Module | Rôle |
|---|---|
| `schemas.py` | Modèles `WasteItem`, `QdrantPayload`, `QdrantPoint`. |
| `regions.py` | Découverte et validation des JSON de régions. |
| `fetcher.py` | Lecture des images locales par libellé (aucun téléchargement distant). |
| `embedder.py` | Production de vecteurs DINOv2 normalisés. |
| `indexer.py` | Création d'une collection par région, identifiants déterministes. |
| `repository.py` | Création de collection, vérification et upsert dans Qdrant. |
| `model.py` | Encodage de l'image reçue et recherche du meilleur point. |
| `controller.py` | Lien entre la vue et le modèle. |
| `viewer.py` | Construction de l'application et indexation au démarrage. |
| `logging_config.py` | Configuration des journaux sur la sortie standard. |

## Dépannage

**Le démarrage échoue sur `DATA_DIR`** : créer le répertoire et y placer directement les JSON de régions (`ls ./data/*.json`). En conteneur, vérifier `/greener/data` et le volume monté.

**Aucun vecteur n'est indexé** : vérifier que le dataset a bien été décompressé, que `IMAGE_DIR` contient un sous-répertoire par libellé, que les extensions sont supportées et que les noms de répertoires correspondent aux valeurs `nom` des JSON après normalisation.

**Qdrant n'est pas joignable** : vérifier `QDRANT_HOST` et `QDRANT_PORT`. Sous Compose, l'hôte doit être `qdrant`, pas `localhost`.

**Le modèle ne se charge pas** : vérifier l'accès réseau, l'espace disque et la valeur de `DEVICE`. Sans GPU, utiliser `DEVICE=cpu`.

## Dépendances

Dépendances d'exécution (`pyproject.toml`) : FastAPI, Keras, Pydantic, pydantic-settings, python-dotenv, python-multipart, qdrant-client, TensorFlow, PyTorch, TorchVision, Transformers, Uvicorn.

Groupe `dev` : Pillow, pip-audit, pytest, pytest-asyncio, pytest-cov, requests, Ruff.
