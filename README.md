# Projet Central de Notebooks - Computer Vision & NLP

Ce dépôt regroupe plusieurs notebooks Jupyter couvrant des tâches avancées en vision par ordinateur (segmentation, détection, tracking) et en traitement du langage naturel (ABSA, fine-tuning de modèles de génération de texte).

## Table des matières

- [Prérequis](#prérequis)
- [Structure des notebooks](#structure-des-notebooks)
- [Détails des notebooks](#détails-des-notebooks)
- [Utilisation générale](#utilisation-générale)
- [Auteurs](#auteurs)

---

## Prérequis

- Python 3.8+
- [Jupyter Notebook](https://jupyter.org/)
- GPU recommandé pour l'entraînement deep learning
- Accès à Google Colab (optionnel mais conseillé pour les notebooks lourds)
- Librairies principales :
  - `torch`, `torchvision`, `tensorflow`, `keras`
  - `transformers`, `datasets`, `sentencepiece`, `accelerate`, `rouge`
  - `detectron2`, `opencv-python`, `scikit-image`, `scikit-learn`
  - `nltk`, `textblob`, `pandas`, `matplotlib`
  - `ultralytics` (pour YOLO)
  - Autres : voir les cellules d'installation dans chaque notebook

## Liste des notebooks

- **Absa_Project.ipynb** : Extraction d'aspects et d'opinions (ABSA) sur des avis textuels.
- **Bio_cell_segmentation_&_Detection.ipynb** : Segmentation et détection de cellules en imagerie biomédicale.
- **Claps_Flexy_Wings.ipynb** : Détection d'objets sur images biomécaniques (flexion d'ailes).
- **ResNet_training.ipynb** : Classification d'images par fusion d'images et de points clés avec ResNet50 & MoveNet.
- **VGG16_train.ipynb** : Classification d'images par fusion d'images et de points clés avec VGG16 & MoveNet.
- **Segmentation_SAM_test.ipynb** : Segmentation d'images avec SAM et analyse multimodale.
- **Stable_Diffusion_Artefact_Detection.ipynb** : Détection d'artefacts sur des images générées par Stable Diffusion.
- **Stable_diffusion_train.ipynb** : Fine-tuning et génération d'images avec Stable Diffusion.
- **Ui_llava.ipynb** : Interface et tests pour LLaVA (Large Language and Vision Assistant).

## Détails des notebooks

### 1. ABSA (Aspect-Based Sentiment Analysis)
- **Fichier** : [Absa_Project.ipynb](Absa_Project.ipynb)
- **Objectif** : Extraction automatique d'aspects, opinions et polarités à partir de textes.
- **Modèle** : Fine-tuning de T5-small.
- **Dépendances** : `transformers`, `datasets`, `nltk`, `textblob`, `sklearn`, `pandas`.
- **Exécution** : Suivre les cellules d'installation et d'entraînement. Les résultats sont sauvegardés en CSV et JSON.

### 2. Segmentation et Tracking Cellulaire
- **Fichier** : [Bio_med_cell_segmentation.ipynb](Bio_med_cell_segmentation.ipynb)
- **Objectif** : Segmentation binaire, instance, tracking de cellules sur images biomédicales.
- **Modèles** : U-Net (Keras), Watershed, Detectron2.
- **Fonctionnalités** :
  - Prétraitement, augmentation de données, entraînement, évaluation (IoU, précision, rappel).
  - Tracking multi-frames avec assignation d'ID.
  - Génération de vidéos à partir des séquences segmentées.
- **Dépendances** : `tensorflow`, `keras`, `opencv-python`, `scikit-image`, `detectron2`, `matplotlib`.
- **Exécution** : Adapter les chemins de données, exécuter séquentiellement.

### 3. Détection d'objets et Segmentation
- **Fichiers** :
  - [Claps_Flexy_Wings.ipynb](Claps_Flexy_Wings.ipynb) : Détection d'objets sur images biomécaniques (YOLOv8).
  - [Segmentation_SAM_test.ipynb](Segmentation_SAM_test.ipynb) : Segmentation avec SAM et YOLOv8-seg.
- **Dépendances** : `ultralytics`, `opencv-python`, `PIL`, `matplotlib`.


### 4. Classification d'Images
- **Fichiers** : 
  - [ResNet_training.ipynb](ResNet_training.ipynb)
  - [VGG16_train.ipynb](VGG16_train.ipynb)
- **Objectif** : Entraînement de modèles CNN pour la classification d'images pour détecter les bonnes positions dans le cadre du dépistage du cancer du sein.
- **Dépendances** : `tensorflow`, `keras`, `sklearn`, `matplotlib`.

### 5. Interface Multimodale
- **Fichier** : [Ui_llava.ipynb](Ui_llava.ipynb)
- **Objectif** : Tester et utiliser LLaVA pour des tâches de vision et langage pour description de UI.
- **Dépendances** : `transformers`, `torch`, `PIL`, `requests`.

### 6. Détection d'Artefacts
- **Fichier** : [Stable_Diffusion_Artefact_Detection.ipynb](Stable_Diffusion_Artefact_Detection.ipynb)
- **Objectif** : Détection d'artefacts dans une generation de Stable diffusion.

---

## Utilisation générale

1. **Cloner le dépôt** et ouvrir dans Jupyter ou Google Colab.
2. **Installer les dépendances** en exécutant les cellules d'installation (`!pip install ...`) en haut de chaque notebook.
3. **Adapter les chemins** de données selon votre environnement (Google Drive, local, etc.).
4. **Exécuter les cellules** dans l'ordre pour chaque notebook.
5. **Sauvegarde des résultats** : Les modèles et outputs sont sauvegardés sur Google Drive ou en local.

---

## Auteurs

- Mohamed Sadadou 

---

## Remarques

- Certains notebooks nécessitent un accès GPU pour un temps d'exécution raisonnable.
- Les chemins d'accès (`/content/drive/...`) sont adaptés à Google Colab avec Google Drive monté.


---

## Licence

Ce projet est fourni à titre d'exemple et ne doit pas être réutilisé, modifié ou distribué sans autorisation explicite de l'auteur. Toute utilisation est strictement réservée à la consultation comme démonstration du travail réalisé. Pour toute autre utilisation, veuillez contacter l'auteur.
