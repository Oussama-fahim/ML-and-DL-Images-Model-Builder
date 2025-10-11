import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import base64
import json
import pickle
import os
import tempfile
from datetime import datetime
import cv2
from PIL import Image
import time
import shutil
from skimage.feature import hog
from skimage import exposure
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    classification_report, confusion_matrix, roc_curve, auc,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.utils import resample

# Modèles de classification
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Modèles de régression
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# XGBoost, LightGBM, CatBoost
try:
    import xgboost as xgb
    import lightgbm as lgb
    import catboost as cb
except ImportError:
    st.warning("Certaines bibliothèques (XGBoost, LightGBM, CatBoost) ne sont pas installées. Certains modèles ne seront pas disponibles.")

# Configuration de la page
st.set_page_config(
    page_title="ML/DL Image Model Builder",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== FONCTIONS UTILITAIRES ==========
def load_data():
    """Charge les données de configuration"""
    if not os.path.exists('data'):
        os.makedirs('data')
    
    config_file = 'data/app_config.json'
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            return json.load(f)
    else:
        # Configuration par défaut
        default_config = {
            "models": {},
            "current_model": None
        }
        with open(config_file, 'w') as f:
            json.dump(default_config, f)
        return default_config

def save_data(config):
    """Sauvegarde les données de configuration"""
    with open('data/app_config.json', 'w') as f:
        json.dump(config, f, indent=4)

def save_model(model, model_name, model_type, problem_type, metrics, model_config):
    """Sauvegarde un modèle entraîné"""
    if not os.path.exists('models'):
        os.makedirs('models')
    
    model_dir = f"models/{problem_type}_{model_type}_{model_name}"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Sauvegarder le modèle
    if hasattr(model, 'save'):
        # Modèle Keras
        model.save(f"{model_dir}/model.h5")
        model_type_saved = "keras"
    else:
        # Modèle scikit-learn
        with open(f"{model_dir}/model.pkl", 'wb') as f:
            pickle.dump(model, f)
        model_type_saved = "sklearn"
    
    # Sauvegarder les métriques
    with open(f"{model_dir}/metrics.json", 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Sauvegarder la configuration du modèle (en enlevant les objets non sérialisables)
    model_config_copy = model_config.copy()
    model_config_copy['model_type_saved'] = model_type_saved
    
    # Supprimer les objets non sérialisables avant de sauvegarder en JSON
    if 'label_encoder' in model_config_copy:
        # Sauvegarder le LabelEncoder séparément
        with open(f"{model_dir}/label_encoder.pkl", 'wb') as f:
            pickle.dump(model_config_copy['label_encoder'], f)
        del model_config_copy['label_encoder']  # Supprimer du JSON
    
    with open(f"{model_dir}/config.json", 'w') as f:
        json.dump(model_config_copy, f, indent=4)
    
    # Sauvegarder les préprocesseurs si existants
    if 'feature_extractor' in st.session_state:
        with open(f"{model_dir}/feature_extractor.pkl", 'wb') as f:
            pickle.dump(st.session_state.feature_extractor, f)
    
    if 'scaler' in st.session_state:
        with open(f"{model_dir}/scaler.pkl", 'wb') as f:
            pickle.dump(st.session_state.scaler, f)
    
    # Sauvegarder les informations du modèle
    model_info = {
        "name": model_name,
        "type": model_type,
        "problem_type": problem_type,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics,
        "config": model_config_copy
    }
    
    # Mettre à jour la configuration
    config = load_data()
    config["models"][model_name] = model_info
    save_data(config)
    
    return model_dir

def load_saved_model(model_name):
    """Charge un modèle sauvegardé"""
    config = load_data()
    if model_name not in config["models"]:
        return None, None, None
    
    model_info = config["models"][model_name]
    model_dir = f"models/{model_info['problem_type']}_{model_info['type']}_{model_name}"
    
    # Charger le modèle
    try:
        if model_info["config"].get("model_type_saved") == "keras":
            model = tf.keras.models.load_model(f"{model_dir}/model.h5")
        else:
            with open(f"{model_dir}/model.pkl", 'rb') as f:
                model = pickle.load(f)
        
        # Charger les métriques
        with open(f"{model_dir}/metrics.json", 'r') as f:
            metrics = json.load(f)
        
        # Charger la configuration
        with open(f"{model_dir}/config.json", 'r') as f:
            model_config = json.load(f)
        
        # Charger le LabelEncoder s'il existe
        label_encoder_path = f"{model_dir}/label_encoder.pkl"
        if os.path.exists(label_encoder_path):
            with open(label_encoder_path, 'rb') as f:
                model_config['label_encoder'] = pickle.load(f)
        
        # Charger les préprocesseurs si existants
        feature_extractor_path = f"{model_dir}/feature_extractor.pkl"
        scaler_path = f"{model_dir}/scaler.pkl"
        
        if os.path.exists(feature_extractor_path):
            with open(feature_extractor_path, 'rb') as f:
                model_config['feature_extractor'] = pickle.load(f)
        
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                model_config['scaler'] = pickle.load(f)
        
        return model, metrics, model_config
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle: {str(e)}")
        return None, None, None

def get_saved_models():
    """Récupère la liste des modèles sauvegardés"""
    config = load_data()
    return config["models"]

def plot_confusion_matrix(cm, classes):
    """Trace une matrice de confusion"""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    return fig

def plot_roc_curve(fpr, tpr, roc_auc):
    """Trace une courbe ROC"""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right")
    return fig

def plot_regression_results(y_true, y_pred):
    """Trace les résultats de régression"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Graphique de dispersion
    ax1.scatter(y_true, y_pred, alpha=0.5)
    ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    ax1.set_xlabel('True Values')
    ax1.set_ylabel('Predicted Values')
    ax1.set_title('True vs Predicted Values')
    
    # Histogramme des résidus
    residuals = y_true - y_pred
    ax2.hist(residuals, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--')
    ax2.set_xlabel('Residuals')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Residuals')
    
    plt.tight_layout()
    return fig

def check_class_balance(y):
    """Vérifie l'équilibre des classes"""
    class_counts = pd.Series(y).value_counts()
    balance_ratio = class_counts.min() / class_counts.max()
    return balance_ratio > 0.5, class_counts

def extract_hog_features(images):
    """Extrait les features HOG des images"""
    hog_features = []
    for img in images:
        # Convertir en niveaux de gris si nécessaire
        if len(img.shape) == 3:
            # Vérifier si c'est une image RGB (3 canaux)
            if img.shape[2] == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            elif img.shape[2] == 1:
                # Déjà en niveaux de gris, juste retirer la dimension du canal
                gray = img.squeeze()
            else:
                # Format inattendu, prendre le premier canal
                gray = img[:, :, 0]
        else:
            # Image déjà en 2D (niveaux de gris)
            gray = img
        
        # S'assurer que l'image est au bon format (uint8) pour HOG
        if gray.dtype != np.uint8:
            if gray.max() <= 1.0:
                gray = (gray * 255).astype(np.uint8)
            else:
                gray = gray.astype(np.uint8)
        
        # Extraire les features HOG
        features, hog_image = hog(
            gray, 
            orientations=9, 
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2), 
            transform_sqrt=True, 
            block_norm='L2-Hys',
            visualize=True
        )
        hog_features.append(features)
    
    return np.array(hog_features)

def extract_pca_features(images, n_components=0.95, pca_model=None):
    """Extrait les features PCA des images"""
    # Aplatir les images
    images_flat = images.reshape(images.shape[0], -1)
    
    if pca_model is None:
        # Appliquer PCA
        pca = PCA(n_components=n_components)
        pca_features = pca.fit_transform(images_flat)
        return pca_features, pca
    else:
        # Utiliser le modèle PCA existant
        pca_features = pca_model.transform(images_flat)
        return pca_features, pca_model

def apply_data_augmentation(images, labels, augmentation_factor=2, use_vgg=False):
    """Applique l'augmentation de données aux classes minoritaires"""
    unique_labels, counts = np.unique(labels, return_counts=True)
    max_count = counts.max()
    
    augmented_images = []
    augmented_labels = []
    
    # Configuration différente selon si on utilise VGG16 ou pas
    if use_vgg:
        # Pour VGG16, on garde les images en RGB
        datagen = ImageDataGenerator(
            rotation_range=30,
            zoom_range=0.2,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )
    else:
        # Pour les autres modèles, on utilise la configuration standard
        datagen = ImageDataGenerator(
            rotation_range=30,
            zoom_range=0.2,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )
    
    for label in unique_labels:
        class_images = images[labels == label]
        class_count = len(class_images)
        
        if class_count < max_count:
            # Calculer le nombre d'images à générer
            num_to_generate = min(max_count - class_count, class_count * augmentation_factor)
            
            # Générer des images augmentées
            for i in range(int(num_to_generate)):
                # Choisir une image aléatoire
                idx = np.random.randint(0, class_count)
                img = class_images[idx]
                
                # Appliquer une transformation aléatoire
                if len(img.shape) == 3:
                    img = img.reshape((1,) + img.shape)
                    # Utiliser la nouvelle méthode pour Keras
                    transformed = next(datagen.flow(img, batch_size=1))[0]
                else:
                    img = img.reshape((1,) + img.shape + (1,))
                    transformed = next(datagen.flow(img, batch_size=1))[0].reshape(img.shape[1:3])
                
                augmented_images.append(transformed)
                augmented_labels.append(label)
    
    if augmented_images:
        all_images = np.concatenate([images, np.array(augmented_images)])
        all_labels = np.concatenate([labels, np.array(augmented_labels)])
        return all_images, all_labels
    else:
        return images, labels

def create_cnn_model(input_shape, num_classes, architecture='simple', problem_type='classification', 
                    filters=[32, 64, 128], dense_units=[128, 64], dropout_rate=0.5):
    """Crée un modèle CNN selon l'architecture spécifiée pour classification ou régression"""
    model = Sequential()
    
    if architecture == 'simple':
        model.add(Conv2D(filters[0], (3, 3), activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(filters[1], (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(filters[2], (3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(dense_units[0], activation='relu'))
        model.add(Dropout(dropout_rate))
        
        # Couche de sortie selon le type de problème
        if problem_type == 'classification':
            if num_classes == 1:
                model.add(Dense(1, activation='sigmoid'))
            else:
                model.add(Dense(num_classes, activation='softmax'))
        else:  # régression
            model.add(Dense(1, activation='linear'))
    
    elif architecture == 'vgg_like':
        model.add(Conv2D(filters[0], (3, 3), activation='relu', padding='same', input_shape=input_shape))
        model.add(Conv2D(filters[0], (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(filters[1], (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(filters[1], (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(filters[2], (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(filters[2], (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(dense_units[0], activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(dense_units[1], activation='relu'))
        model.add(Dropout(dropout_rate))
        
        # Couche de sortie selon le type de problème
        if problem_type == 'classification':
            if num_classes == 1:
                model.add(Dense(1, activation='sigmoid'))
            else:
                model.add(Dense(num_classes, activation='softmax'))
        else:  # régression
            model.add(Dense(1, activation='linear'))
    
    elif architecture == 'advanced':
        # Utiliser un modèle pré-entraîné comme base
        # VGG16 nécessite des images 3 canaux
        if input_shape[2] == 1:
            # Convertir l'input shape pour 3 canaux
            input_shape_vgg = (input_shape[0], input_shape[1], 3)
            st.warning(f"VGG16 nécessite 3 canaux. Conversion de {input_shape} vers {input_shape_vgg}")
        else:
            input_shape_vgg = input_shape
            
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape_vgg)
        base_model.trainable = False
        
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(dense_units[0], activation='relu'),
            Dropout(dropout_rate),
            Dense(dense_units[1], activation='relu'),
            Dropout(dropout_rate)
        ])
        
        # Couche de sortie selon le type de problème
        if problem_type == 'classification':
            if num_classes == 1:
                model.add(Dense(1, activation='sigmoid'))
            else:
                model.add(Dense(num_classes, activation='softmax'))
        else:  # régression
            model.add(Dense(1, activation='linear'))
    
    elif architecture == 'regression_cnn':
        # Architecture spécialisée pour la régression
        model.add(Conv2D(filters[0], (3, 3), activation='relu', input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(filters[1], (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(filters[2], (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(GlobalAveragePooling2D())
        model.add(Dense(dense_units[0], activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(dense_units[1], activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(1, activation='linear'))  # Sortie linéaire pour régression
    
    return model

def create_mlp_model(input_dim, num_classes, hidden_layers=[128, 64, 32], dropout_rate=0.2, problem_type='classification'):
    """Crée un modèle MLP pour classification ou régression"""
    model = Sequential()
    
    # Première couche
    model.add(Dense(hidden_layers[0], activation='relu', input_dim=input_dim))
    model.add(Dropout(dropout_rate))
    
    # Couches cachées supplémentaires
    for units in hidden_layers[1:]:
        model.add(Dense(units, activation='relu'))
        model.add(Dropout(dropout_rate))
    
    # Couche de sortie selon le type de problème
    if problem_type == 'classification':
        if num_classes == 1:
            model.add(Dense(1, activation='sigmoid'))  # Classification binaire
        else:
            model.add(Dense(num_classes, activation='softmax'))  # Classification multi-classes
    else:  # régression
        model.add(Dense(1, activation='linear'))  # Régression
    
    return model

def preprocess_image(img, target_size=(64, 64), use_vgg=False):
    """Prétraite une image pour l'entraînement ou le test"""
    if isinstance(img, str):
        img = Image.open(img)
    elif isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    
    # Redimensionner
    img = img.resize(target_size)
    img_array = np.array(img)
    
    # Conversion différente selon si on utilise VGG16 ou pas
    if use_vgg:
        # Pour VGG16, on garde les images en RGB (3 canaux)
        if len(img_array.shape) == 2:
            # Image en niveaux de gris, convertir en RGB
            img_array = np.stack([img_array] * 3, axis=-1)
        elif len(img_array.shape) == 3 and img_array.shape[2] == 1:
            # Image avec canal unique, convertir en RGB
            img_array = np.concatenate([img_array] * 3, axis=-1)
        elif len(img_array.shape) == 3 and img_array.shape[2] == 4:
            # Image RGBA, convertir en RGB
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        # Si déjà RGB (3 canaux), on ne fait rien
    
    else:
        # Pour les autres modèles, convertir en niveaux de gris (1 canal)
        if len(img_array.shape) == 2:
            # Image déjà en niveaux de gris
            img_array = np.expand_dims(img_array, axis=-1)
        elif len(img_array.shape) == 3 and img_array.shape[2] == 3:
            # Convertir RGB en niveaux de gris
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            img_array = np.expand_dims(img_array, axis=-1)
        elif len(img_array.shape) == 3 and img_array.shape[2] == 4:
            # Image RGBA, convertir en RGB puis en niveaux de gris
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            img_array = np.expand_dims(img_array, axis=-1)
    
    # Normaliser
    if img_array.max() > 1:
        img_array = img_array / 255.0
    
    # Ajouter une dimension de batch si nécessaire
    if len(img_array.shape) == 3:
        img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def preprocess_for_classical_ml(images, feature_method='HOG', pca_model=None):
    """Prétraite les images pour les algorithmes classiques"""
    # Préparer les images pour l'extraction de features
    if feature_method == 'HOG':
        # Pour HOG, nous avons besoin d'images 2D
        images_for_features = []
        for img in images:
            if len(img.shape) == 3 and img.shape[2] == 1:
                # Image avec canal unique, convertir en 2D
                images_for_features.append(img.squeeze())
            elif len(img.shape) == 3:
                # Image avec multiple canaux, prendre le premier canal
                images_for_features.append(img[:, :, 0])
            else:
                # Image déjà en 2D
                images_for_features.append(img)
        
        images_for_features = np.array(images_for_features)
        features = extract_hog_features(images_for_features)
        feature_extractor = 'HOG'
    else:  # PCA
        # Aplatir les images
        images_flat = images.reshape(images.shape[0], -1)
        
        if pca_model is None:
            # Entraîner un nouveau PCA
            pca = PCA(n_components=0.95)
            features = pca.fit_transform(images_flat)
            feature_extractor = pca
        else:
            # Utiliser le PCA existant
            features = pca_model.transform(images_flat)
            feature_extractor = pca_model
    
    # Normaliser les features
    scaler = StandardScaler()
    if pca_model is None:
        # Entraîner un nouveau scaler
        features_scaled = scaler.fit_transform(features)
    else:
        # Utiliser le scaler existant
        features_scaled = scaler.transform(features)
    
    return features_scaled, feature_extractor, scaler

def extract_features_single_image(image, feature_method, feature_extractor=None, scaler=None):
    """Extrait les features d'une seule image pour la prédiction"""
    if feature_method == 'HOG':
        # Préparer l'image pour HOG
        if len(image.shape) == 3 and image.shape[2] == 1:
            img_for_features = image.squeeze()
        elif len(image.shape) == 3:
            img_for_features = image[:, :, 0]
        else:
            img_for_features = image
        
        # Extraire features HOG
        features = extract_hog_features([img_for_features])
    else:  # PCA
        # Aplatir l'image
        image_flat = image.reshape(1, -1)
        
        if feature_extractor is not None:
            # Utiliser le PCA existant
            features = feature_extractor.transform(image_flat)
        else:
            # Créer un nouveau PCA (ne devrait pas arriver normalement)
            pca = PCA(n_components=0.95)
            features = pca.fit_transform(image_flat)
    
    # Normaliser si un scaler est fourni
    if scaler is not None:
        features = scaler.transform(features)
    
    return features

def get_classification_explanation(predicted_class, class_names, probabilities=None, confidence=None):
    """Génère une explication textuelle pour les prédictions de classification"""
    if probabilities is not None and len(probabilities) > 1:
        # Classification multi-classes avec probabilités
        top_classes = np.argsort(probabilities)[-3:][::-1]  # Top 3 classes
        explanation = f"**Prédiction : {class_names[predicted_class]}**\n\n"
        explanation += "**Probabilités par classe :**\n"
        for i, class_idx in enumerate(top_classes):
            prob = probabilities[class_idx]
            explanation += f"- {class_names[class_idx]}: {prob:.3f} ({prob*100:.1f}%)\n"
        
        if confidence is not None:
            explanation += f"\n**Niveau de confiance global : {confidence:.3f}**"
            
    elif confidence is not None:
        # Classification binaire avec confiance
        explanation = f"**Prédiction : {class_names[predicted_class]}**\n\n"
        explanation += f"**Niveau de confiance : {confidence:.3f}**\n\n"
        
        if confidence > 0.8:
            explanation += "✅ **Confiance élevée** - La prédiction est très fiable."
        elif confidence > 0.6:
            explanation += "⚠️ **Confiance moyenne** - La prédiction est raisonnablement fiable."
        else:
            explanation += "❌ **Confiance faible** - La prédiction est incertaine, vérification recommandée."
    else:
        explanation = f"**Prédiction : {class_names[predicted_class]}**"
    
    return explanation

def get_regression_explanation(prediction, target_range=None, target_name="valeur cible"):
    """Génère une explication textuelle pour les prédictions de régression"""
    explanation = f"**{target_name} prédite : {prediction:.3f}**\n\n"
    
    if target_range is not None:
        min_val, max_val = target_range
        normalized = (prediction - min_val) / (max_val - min_val)
        
        explanation += f"**Position dans l'échelle :**\n"
        explanation += f"- Minimum historique : {min_val:.2f}\n"
        explanation += f"- Maximum historique : {max_val:.2f}\n"
        explanation += f"- Position relative : {normalized:.1%}\n\n"
        
        if normalized > 0.8:
            explanation += "📈 **Valeur élevée** - Proche du maximum observé."
        elif normalized < 0.2:
            explanation += "📉 **Valeur basse** - Proche du minimum observé."
        else:
            explanation += "📊 **Valeur moyenne** - Dans la plage centrale des observations."
    
    return explanation

# ========== INTERFACE STREAMLIT ==========
# Configuration de la barre latérale
with st.sidebar:
    st.title("🧠 ML/DL Image Model Builder")
    selected = option_menu(
        menu_title="Menu Principal",
        options=["🏗️ Construction des modèles", "📊 Évaluation et performance", "🧪 Test", "❓ Aide"],
        icons=["building", "graph-up", "flask", "question-circle"],
        menu_icon="cast",
        default_index=0
    )

# Page Construction des modèles
if selected == "🏗️ Construction des modèles":
    st.title("🏗️ Construction des Modèles ML/DL pour Images")
    
    # Initialisation de la session
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 1
    if 'model_config' not in st.session_state:
        st.session_state.model_config = {}
    if 'image_data' not in st.session_state:
        st.session_state.image_data = {}
    if 'dataset' not in st.session_state:
        st.session_state.dataset = None
    if 'uploaded_files_cache' not in st.session_state:
        st.session_state.uploaded_files_cache = set()
    
    # Barre de progression
    steps = ["Définition du problème", "Sélection de l'algorithme", "Collecte des images", 
             "Prétraitement", "Validation", "Entraînement", "Sauvegarde"]
    
    progress = st.session_state.current_step / len(steps)
    st.progress(progress)
    
    cols = st.columns(len(steps))
    for i, step in enumerate(steps):
        with cols[i]:
            if i + 1 < st.session_state.current_step:
                st.success(f"✓ {step}")
            elif i + 1 == st.session_state.current_step:
                st.info(f"▶ {step}")
            else:
                st.write(f"{step}")
    
    st.markdown("---")
    
    # Étape 1: Définition du problème
    if st.session_state.current_step == 1:
        st.subheader("🎯 Étape 1: Définition du problème")
        
        with st.form("problem_definition"):
            col1, col2 = st.columns(2)
            
            with col1:
                problem_type = st.selectbox(
                    "Type de problème *",
                    ["", "Classification", "Régression"],
                    help="Sélectionnez le type de problème que vous souhaitez résoudre"
                )
            
            with col2:
                if problem_type == "Classification":
                    num_classes = st.number_input(
                        "Nombre de classes *",
                        min_value=2,
                        max_value=100,
                        value=2,
                        help="Nombre de catégories à prédire"
                    )
                else:
                    num_classes = 1
            
            method_type = st.selectbox(
                "Méthode de résolution *",
                ["", "Algorithmes classiques", "Réseaux de neurones"],
                help="Choisissez la famille d'algorithmes à utiliser"
            )
            
            model_name = st.text_input(
                "Nom du modèle *",
                placeholder="Ex: Mon_modele_classification",
                help="Donnez un nom unique à votre modèle"
            )
            
            # Configuration spécifique selon le type de problème
            if problem_type == "Classification":
                st.subheader("Configuration des classes")
                class_names = []
                for i in range(num_classes):
                    class_name = st.text_input(
                        f"Nom de la classe {i+1} *",
                        placeholder=f"Ex: Classe_{i+1}",
                        key=f"class_name_{i}"
                    )
                    class_names.append(class_name)
                
                # Vérifier que tous les noms de classes sont remplis
                class_names_valid = all(class_names)
                if not class_names_valid and st.form_submit_button:
                    st.error("Veuillez remplir tous les noms de classes")
                
            else:  # Régression
                st.subheader("Configuration de la régression")
                target_name = st.text_input(
                    "Nom de la variable cible *",
                    placeholder="Ex: Prix, Score, Température",
                    help="Donnez un nom significatif à la variable que vous voulez prédire"
                )
                if not target_name and st.form_submit_button:
                    st.error("Veuillez remplir le nom de la variable cible")
            
            submitted = st.form_submit_button("Confirmer et continuer →")
            
            if submitted:
                if not all([problem_type, method_type, model_name]):
                    st.error("Veuillez remplir tous les champs obligatoires (*)")
                elif problem_type == "Classification" and not all(class_names):
                    st.error("Veuillez remplir tous les noms de classes")
                elif problem_type == "Régression" and not target_name:
                    st.error("Veuillez remplir le nom de la variable cible")
                else:
                    st.session_state.model_config.update({
                        "problem_type": problem_type,
                        "num_classes": num_classes,
                        "method_type": method_type,
                        "model_name": model_name
                    })
                    
                    if problem_type == "Classification":
                        st.session_state.model_config["class_names"] = class_names
                    else:  # Régression
                        st.session_state.model_config["target_name"] = target_name
                    
                    st.session_state.current_step = 2
                    st.rerun()
    
    # Étape 2: Sélection de l'algorithme
    elif st.session_state.current_step == 2:
        st.subheader("🔧 Étape 2: Sélection de l'algorithme")
        
        problem_type = st.session_state.model_config["problem_type"]
        method_type = st.session_state.model_config["method_type"]
        
        if method_type == "Algorithmes classiques":
            if problem_type == "Classification":
                algorithms = {
                    "Régression logistique": "logistic_regression",
                    "SVC (Support Vector Classifier)": "svc",
                    "Arbre de décision": "decision_tree",
                    "Random Forest": "random_forest",
                    "XGBoost": "xgboost",
                    "LightGBM": "lightgbm",
                    "CatBoost": "catboost",
                    "Naive Bayes": "naive_bayes",
                    "AdaBoost": "adaboost",
                    "KNN": "knn",
                    "Gradient Boosting": "gradient_boosting"
                }
            else:  # Régression
                algorithms = {
                    "Régression linéaire": "linear_regression",
                    "Ridge Regression": "ridge",
                    "Lasso Regression": "lasso",
                    "Elastic Net": "elastic_net",
                    "Random Forest": "random_forest",
                    "XGBoost": "xgboost",
                    "LightGBM": "lightgbm",
                    "CatBoost": "catboost",
                    "SVR": "svr",
                    "KNN": "knn",
                    "Gradient Boosting": "gradient_boosting"
                }
        else:  # Réseaux de neurones
            if problem_type == "Classification":
                algorithms = {
                    "MLP (Perceptron multicouche)": "mlp",
                    "CNN Simple": "cnn_simple",
                    "CNN Avancé (VGG-like)": "cnn_vgg",
                    "CNN Transfer Learning": "cnn_advanced"
                }
            else:  # Régression
                algorithms = {
                    "MLP (Perceptron multicouche)": "mlp",
                    "CNN pour Régression": "cnn_regression",
                    "CNN Avancé (VGG-like)": "cnn_vgg",
                    "CNN Transfer Learning": "cnn_advanced"
                }
        
        selected_algorithm = st.selectbox(
            "Sélectionnez l'algorithme *",
            [""] + list(algorithms.keys())
        )
        
        # Description de l'algorithme sélectionné
        if selected_algorithm:
            algorithm_descriptions = {
                "MLP (Perceptron multicouche)": "Réseau de neurones simple avec couches fully-connected. Idéal pour les features extraits ou images flatten.",
                "CNN Simple": "Architecture CNN basique avec 3 couches convolutives. Bon équilibre performance/complexité.",
                "CNN Avancé (VGG-like)": "Architecture profonde inspirée de VGG. Meilleure précision mais plus lent à entraîner.",
                "CNN Transfer Learning": "Utilise VGG16 pré-entraîné. Excellente précision, idéal pour petits datasets.",
                "CNN pour Régression": "Architecture spécialisée pour la régression avec normalisation de batch.",
                "Régression logistique": "Modèle linéaire simple et rapide pour classification.",
                "Random Forest": "Ensemble d'arbres de décision. Robuste et performant.",
                "XGBoost": "Algorithme gradient boosting très performant.",
                "SVC (Support Vector Classifier)": "Bon pour les problèmes complexes avec marge de séparation.",
                "Gradient Boosting": "Algorithme de boosting séquentiel avec arbres de décision."
            }
            
            if selected_algorithm in algorithm_descriptions:
                st.info(f"**{selected_algorithm}**: {algorithm_descriptions[selected_algorithm]}")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("← Retour à l'étape précédente"):
                st.session_state.current_step = 1
                st.rerun()
        
        with col2:
            if st.button("Continuer →") and selected_algorithm:
                st.session_state.model_config["algorithm"] = selected_algorithm
                st.session_state.model_config["algorithm_key"] = algorithms[selected_algorithm]
                st.session_state.current_step = 3
                st.rerun()
    
    # Étape 3: Collecte des images
    elif st.session_state.current_step == 3:
        st.subheader("📁 Étape 3: Collecte des images")
        
        problem_type = st.session_state.model_config["problem_type"]
        num_classes = st.session_state.model_config["num_classes"]
        
        st.info(f"Type de problème: {problem_type}")
        
        if problem_type == "Classification":
            class_names = st.session_state.model_config["class_names"]
            st.subheader("Collecte d'images par classe")
            
            # Créer des onglets pour chaque classe
            class_tabs = st.tabs([f"{class_names[i]}" for i in range(num_classes)])
            
            for i, class_tab in enumerate(class_tabs):
                with class_tab:
                    st.write(f"### {class_names[i]}")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Upload d'images pour cette classe
                        uploaded_files = st.file_uploader(
                            f"Téléchargez des images pour {class_names[i]}",
                            type=['jpg', 'png', 'jpeg'],
                            accept_multiple_files=True,
                            key=f"upload_class_{i}"
                        )
                        
                        if uploaded_files:
                            if f"class_{i}" not in st.session_state.image_data:
                                st.session_state.image_data[f"class_{i}"] = []
                            
                            new_images = 0
                            for uploaded_file in uploaded_files:
                                # Vérifier si le fichier n'est pas déjà dans le cache
                                file_signature = f"{uploaded_file.name}_{uploaded_file.size}"
                                if file_signature not in st.session_state.uploaded_files_cache:
                                    # Lire l'image
                                    image = Image.open(uploaded_file)
                                    st.session_state.image_data[f"class_{i}"].append({
                                        "image": np.array(image),
                                        "label": i,
                                        "filename": uploaded_file.name,
                                        "class_name": class_names[i]
                                    })
                                    st.session_state.uploaded_files_cache.add(file_signature)
                                    new_images += 1
                            
                            if new_images > 0:
                                st.success(f"{new_images} nouvelles images ajoutées à {class_names[i]}")
                            else:
                                st.info("Aucune nouvelle image ajoutée (doublons détectés)")
                    
                    with col2:
                        # Capture d'images pour cette classe
                        st.write("Capture d'image en direct")
                        camera_input = st.camera_input(f"Prendre une photo pour {class_names[i]}", key=f"camera_{i}")
                        
                        if camera_input:
                            image = Image.open(camera_input)
                            if f"class_{i}" not in st.session_state.image_data:
                                st.session_state.image_data[f"class_{i}"] = []
                            
                            # Générer un nom unique pour l'image capturée
                            capture_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"captured_{class_names[i]}_{capture_time}.jpg"
                            
                            st.session_state.image_data[f"class_{i}"].append({
                                "image": np.array(image),
                                "label": i,
                                "filename": filename,
                                "class_name": class_names[i]
                            })
                            
                            st.success("Image capturée et ajoutée!")
        
        else:  # Régression
            target_name = st.session_state.model_config["target_name"]
            st.subheader(f"Collecte d'images avec {target_name}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Upload d'images avec valeurs cibles
                uploaded_files = st.file_uploader(
                    "Téléchargez des images",
                    type=['jpg', 'png', 'jpeg'],
                    accept_multiple_files=True,
                    key="upload_regression"
                )
                
                if uploaded_files:
                    if "regression_images" not in st.session_state.image_data:
                        st.session_state.image_data["regression_images"] = []
                    
                    new_images = 0
                    for uploaded_file in uploaded_files:
                        file_signature = f"{uploaded_file.name}_{uploaded_file.size}"
                        if file_signature not in st.session_state.uploaded_files_cache:
                            image = Image.open(uploaded_file)
                            st.session_state.image_data["regression_images"].append({
                                "image": np.array(image),
                                "filename": uploaded_file.name,
                                "target": None
                            })
                            st.session_state.uploaded_files_cache.add(file_signature)
                            new_images += 1
                    
                    if new_images > 0:
                        st.success(f"{new_images} nouvelles images téléchargées")
                    else:
                        st.info("Aucune nouvelle image ajoutée (doublons détectés)")
            
            with col2:
                # Capture d'images avec valeurs cibles
                st.write("Capture d'image en direct")
                camera_input = st.camera_input("Prendre une photo", key="camera_regression")
                
                if camera_input:
                    image = Image.open(camera_input)
                    if "regression_images" not in st.session_state.image_data:
                        st.session_state.image_data["regression_images"] = []
                    
                    capture_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"captured_regression_{capture_time}.jpg"
                    
                    st.session_state.image_data["regression_images"].append({
                        "image": np.array(image),
                        "filename": filename,
                        "target": None
                    })
                    
                    st.success("Image capturée!")
            
            # Définir les valeurs cibles pour les images
            if st.session_state.image_data.get("regression_images"):
                st.subheader(f"Définir les valeurs de {target_name}")
                
                for i, img_data in enumerate(st.session_state.image_data["regression_images"]):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.image(img_data["image"], caption=img_data["filename"], width=150)
                    with col2:
                        target_value = st.number_input(
                            f"{target_name} pour {img_data['filename']}",
                            value=img_data.get("target", 0.0),
                            key=f"target_{i}"
                        )
                        img_data["target"] = target_value
        
        # Afficher le résumé des données collectées
        st.subheader("Résumé des données collectées")
        if st.session_state.image_data:
            total_images = 0
            class_counts = {}
            
            if problem_type == "Classification":
                for class_key, images in st.session_state.image_data.items():
                    class_num = int(class_key.split('_')[1])
                    count = len(images)
                    class_name = st.session_state.model_config["class_names"][class_num]
                    class_counts[class_name] = count
                    total_images += count
            else:
                valid_images = [img for img in st.session_state.image_data.get("regression_images", []) if img["target"] is not None]
                total_images = len(valid_images)
                class_counts = {f"Images avec {target_name}": total_images}
            
            if total_images > 0:
                st.success(f"Total d'images collectées: {total_images}")
                
                # Graphique de distribution
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.bar(class_counts.keys(), class_counts.values(), color='skyblue')
                ax.set_title('Distribution des images')
                ax.set_xlabel(problem_type)
                ax.set_ylabel("Nombre d'images")
                plt.xticks(rotation=45)
                st.pyplot(fig)
                
                # Vérifier l'équilibre des classes pour la classification
                if problem_type == "Classification":
                    all_labels = []
                    for class_imgs in st.session_state.image_data.values():
                        for img in class_imgs:
                            all_labels.append(img["label"])
                    
                    if len(all_labels) > 0:
                        is_balanced, _ = check_class_balance(all_labels)
                        if not is_balanced:
                            st.warning("⚠️ Les classes ne sont pas équilibrées!")
                
            else:
                st.info("Aucune image valide collectée pour le moment")
        else:
            st.info("Aucune image collectée pour le moment")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("← Retour à l'étape précédente"):
                st.session_state.current_step = 2
                st.rerun()
        
        with col2:
            has_data = bool(st.session_state.image_data)
            if problem_type == "Régression":
                # Vérifier qu'au moins une image a une valeur cible définie
                has_data = any(img["target"] is not None for img in st.session_state.image_data.get("regression_images", []))
            
            if st.button("Continuer →", disabled=not has_data):
                st.session_state.current_step = 4
                st.rerun()
    
    # Étape 4: Prétraitement
    elif st.session_state.current_step == 4:
        st.subheader("🔧 Étape 4: Prétraitement des images")
        
        if not st.session_state.image_data:
            st.error("Aucune image chargée. Veuillez retourner à l'étape précédente.")
        else:
            st.info("Prétraitement des images")
            
            problem_type = st.session_state.model_config["problem_type"]
            algorithm_key = st.session_state.model_config["algorithm_key"]
            method_type = st.session_state.model_config["method_type"]
            
            # Déterminer si on utilise VGG16 (nécessite RGB)
            use_vgg = algorithm_key == "cnn_advanced"
            
            # Taille fixe à 64x64
            target_size = (64, 64)
            st.session_state.model_config["image_target_size"] = target_size
            st.session_state.model_config["use_vgg"] = use_vgg
            
            if use_vgg:
                st.info("🔍 Utilisation de VGG16 - Les images seront conservées en RGB (3 canaux)")
            
            # Préparer les données d'images
            if problem_type == "Classification":
                images = []
                labels = []
                
                for class_key, class_images in st.session_state.image_data.items():
                    for img_data in class_images:
                        # Redimensionner et normaliser l'image
                        img = Image.fromarray(img_data["image"])
                        img = img.resize(target_size)
                        img_array = np.array(img)
                        
                        # Conversion différente selon si on utilise VGG16 ou pas
                        if use_vgg:
                            # Pour VGG16, on garde les images en RGB (3 canaux)
                            if len(img_array.shape) == 2:
                                # Image en niveaux de gris, convertir en RGB
                                img_array = np.stack([img_array] * 3, axis=-1)
                            elif len(img_array.shape) == 3 and img_array.shape[2] == 1:
                                # Image avec canal unique, convertir en RGB
                                img_array = np.concatenate([img_array] * 3, axis=-1)
                            elif len(img_array.shape) == 3 and img_array.shape[2] == 4:
                                # Image RGBA, convertir en RGB
                                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
                            # Si déjà RGB (3 canaux), on ne fait rien
                        else:
                            # Pour les autres modèles, convertir en niveaux de gris (1 canal)
                            if len(img_array.shape) == 2:
                                # Image déjà en niveaux de gris
                                img_array = np.expand_dims(img_array, axis=-1)
                            elif len(img_array.shape) == 3 and img_array.shape[2] == 3:
                                # Convertir RGB en niveaux de gris
                                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                                img_array = np.expand_dims(img_array, axis=-1)
                            elif len(img_array.shape) == 3 and img_array.shape[2] == 4:
                                # Image RGBA, convertir en RGB puis en niveaux de gris
                                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
                                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                                img_array = np.expand_dims(img_array, axis=-1)
                        
                        # Normaliser si nécessaire
                        if img_array.max() > 1:
                            img_array = img_array / 255.0
                        
                        images.append(img_array)
                        labels.append(img_data["label"])
                
                X = np.array(images)
                y = np.array(labels)
                
                # Vérifier l'équilibre des classes
                is_balanced, class_counts = check_class_balance(y)
                
                if not is_balanced:
                    st.warning("Classes déséquilibrées détectées. Application de l'augmentation de données...")
                    X, y = apply_data_augmentation(X, y, use_vgg=use_vgg)
                    st.success(f"Augmentation de données appliquée. Nouvelle taille: {len(X)} images")
                
                # Encodage des labels
                le = LabelEncoder()
                y_encoded = le.fit_transform(y)
                st.session_state.model_config["label_encoder"] = le
                
            else:  # Régression
                images = []
                targets = []
                
                for img_data in st.session_state.image_data.get("regression_images", []):
                    if img_data["target"] is not None:
                        img = Image.fromarray(img_data["image"])
                        img = img.resize(target_size)
                        img_array = np.array(img)
                        
                        # Conversion différente selon si on utilise VGG16 ou pas
                        if use_vgg:
                            # Pour VGG16, on garde les images en RGB (3 canaux)
                            if len(img_array.shape) == 2:
                                # Image en niveaux de gris, convertir en RGB
                                img_array = np.stack([img_array] * 3, axis=-1)
                            elif len(img_array.shape) == 3 and img_array.shape[2] == 1:
                                # Image avec canal unique, convertir en RGB
                                img_array = np.concatenate([img_array] * 3, axis=-1)
                            elif len(img_array.shape) == 3 and img_array.shape[2] == 4:
                                # Image RGBA, convertir en RGB
                                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
                        else:
                            # Pour les autres modèles, convertir en niveaux de gris (1 canal)
                            if len(img_array.shape) == 2:
                                # Image déjà en niveaux de gris
                                img_array = np.expand_dims(img_array, axis=-1)
                            elif len(img_array.shape) == 3 and img_array.shape[2] == 3:
                                # Convertir RGB en niveaux de gris
                                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                                img_array = np.expand_dims(img_array, axis=-1)
                            elif len(img_array.shape) == 3 and img_array.shape[2] == 4:
                                # Image RGBA, convertir en RGB puis en niveaux de gris
                                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
                                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                                img_array = np.expand_dims(img_array, axis=-1)
                        
                        if img_array.max() > 1:
                            img_array = img_array / 255.0
                        
                        images.append(img_array)
                        targets.append(img_data["target"])
                
                X = np.array(images)
                y = np.array(targets)
                
                # Sauvegarder la plage des valeurs cibles pour l'explication
                if len(y) > 0:
                    st.session_state.model_config["target_range"] = (y.min(), y.max())
            
            if len(X) > 0:
                st.success(f"Images chargées: {X.shape}")
                st.info(f"Format des images: {X.shape[1:]} (hauteur, largeur, canaux)")
                
                # Feature extraction pour les algorithmes classiques
                if method_type == "Algorithmes classiques":
                    st.subheader("Extraction de features")
                    
                    feature_method = st.selectbox(
                        "Méthode d'extraction de features",
                        ["HOG", "PCA"],
                        help="Choisissez la méthode pour extraire les features des images"
                    )
                    
                    if st.button("Appliquer l'extraction de features"):
                        with st.spinner("Extraction des features en cours..."):
                            X_processed, feature_extractor, scaler = preprocess_for_classical_ml(X, feature_method)
                            
                            st.session_state.feature_extractor = feature_extractor
                            st.session_state.scaler = scaler
                            st.session_state.model_config["feature_method"] = feature_method
                            
                            st.success(f"Features extraits: {X_processed.shape}")
                            st.info(f"Méthode utilisée: {feature_method}")
                            
                            # Mettre à jour le dataset
                            st.session_state.dataset = {
                                "X": X_processed,
                                "y": y,
                                "data_type": "features",
                                "target_size": target_size,
                                "original_images": X
                            }
                
                else:
                    # Pour les réseaux de neurones, utiliser les images directement
                    X_processed = X
                    st.session_state.dataset = {
                        "X": X_processed,
                        "y": y,
                        "data_type": "image",
                        "target_size": target_size,
                        "original_images": X
                    }
                
                st.success(f"✅ Prétraitement terminé: {len(X)} images préparées")
                
                # Aperçu des images prétraitées
                st.subheader("Aperçu des images prétraitées")
                n_preview = min(5, len(X))
                cols = st.columns(n_preview)
                
                for i in range(n_preview):
                    with cols[i]:
                        if method_type == "Algorithmes classiques":
                            # Pour les features, montrer l'image originale
                            st.image(X[i].squeeze(), caption=f"Image {i+1}", use_column_width=True)
                        else:
                            # Afficher l'image selon son format
                            img_to_show = X_processed[i]
                            if len(img_to_show.shape) == 3 and img_to_show.shape[2] == 1:
                                # Image en niveaux de gris
                                st.image(img_to_show.squeeze(), caption=f"Image {i+1}", use_column_width=True)
                            elif len(img_to_show.shape) == 3 and img_to_show.shape[2] == 3:
                                # Image RGB
                                st.image(img_to_show, caption=f"Image {i+1}", use_column_width=True)
                            else:
                                st.image(img_to_show, caption=f"Image {i+1}", use_column_width=True)
                
            else:
                st.error("Aucune image valide avec valeurs cibles définies!")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button("← Retour à l'étape précédente"):
                    st.session_state.current_step = 3
                    st.rerun()
            
            with col2:
                if st.button("Continuer →"):
                    st.session_state.current_step = 5
                    st.rerun()
    
    # Étape 5: Validation
    elif st.session_state.current_step == 5:
        st.subheader("📊 Étape 5: Méthode de validation")
        
        validation_method = st.radio(
            "Choisissez la méthode de validation",
            ["Split train/test", "Validation croisée (Cross-validation)"],
            help="Sélectionnez comment évaluer les performances du modèle"
        )
        
        if validation_method == "Split train/test":
            train_size = st.slider(
                "Pourcentage du jeu d'entraînement",
                min_value=50,
                max_value=90,
                value=80,
                help="Proportion des données utilisées pour l'entraînement"
            )
            test_size = 100 - train_size
            st.write(f"Répartition: {train_size}% entraînement, {test_size}% test")
            st.session_state.model_config["train_size"] = train_size / 100
        
        else:  # Validation croisée
            cv_folds = st.slider(
                "Nombre de folds",
                min_value=3,
                max_value=10,
                value=5,
                help="Nombre de partitions pour la validation croisée"
            )
            st.session_state.model_config["cv_folds"] = cv_folds
        
        st.session_state.model_config["validation_method"] = validation_method
        
        # Gestion du déséquilibre pour la classification
        if (st.session_state.model_config["problem_type"] == "Classification" and 
            st.session_state.dataset is not None):
            
            y = st.session_state.dataset["y"]
            is_balanced, class_counts = check_class_balance(y)
            
            if not is_balanced and st.session_state.model_config["method_type"] == "Algorithmes classiques":
                st.warning("Classes déséquilibrées - utilisation de class_weight='balanced'")
                st.session_state.model_config["use_class_weight"] = True
            else:
                st.session_state.model_config["use_class_weight"] = False
        
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("← Retour à l'étape précédente"):
                st.session_state.current_step = 4
                st.rerun()
        
        with col2:
            if st.button("Continuer →"):
                st.session_state.current_step = 6
                st.rerun()
    
    # Étape 6: Entraînement
    elif st.session_state.current_step == 6:
        st.subheader("🚀 Étape 6: Entraînement du modèle")
        
        if st.session_state.dataset is None:
            st.error("Aucune donnée disponible. Veuillez retourner aux étapes précédentes.")
            st.stop()
        
        algorithm_key = st.session_state.model_config["algorithm_key"]
        problem_type = st.session_state.model_config["problem_type"]
        method_type = st.session_state.model_config["method_type"]
        num_classes = st.session_state.model_config["num_classes"]
        use_vgg = st.session_state.model_config.get("use_vgg", False)
        
        st.info(f"Algorithme sélectionné: {st.session_state.model_config['algorithm']}")
        st.info(f"Type de méthode: {method_type}")
        st.info(f"Type de problème: {problem_type}")
        if use_vgg:
            st.info("🔍 Utilisation de VGG16 - Images en RGB")
        
        # Configuration des hyperparamètres
        st.subheader("⚙️ Configuration des hyperparamètres")
        
        param_method = st.radio(
            "Méthode de configuration des paramètres",
            ["Paramètres par défaut", "Configuration manuelle", "Grid Search"],
            help="Choisissez comment configurer les hyperparamètres du modèle"
        )
        
        st.session_state.model_config["param_method"] = param_method
        
        # Configuration spécifique selon le type de modèle et la méthode
        if param_method == "Configuration manuelle":
            st.subheader("🔧 Configuration manuelle des hyperparamètres")
            
            if method_type == "Algorithmes classiques":
                if problem_type == "Classification":
                    if algorithm_key == "logistic_regression":
                        col1, col2 = st.columns(2)
                        with col1:
                            C = st.slider("Paramètre de régularisation (C)", 0.01, 10.0, 1.0, 0.01)
                            solver = st.selectbox("Algorithme d'optimisation", ["lbfgs", "liblinear", "saga"])
                        with col2:
                            max_iter = st.slider("Nombre max d'itérations", 100, 2000, 1000)
                            penalty = st.selectbox("Type de régularisation", ["l2", "l1"])
                        
                        params = {
                            'C': C,
                            'solver': solver,
                            'max_iter': max_iter,
                            'penalty': penalty
                        }
                    
                    elif algorithm_key == "random_forest":
                        col1, col2 = st.columns(2)
                        with col1:
                            n_estimators = st.slider("Nombre d'arbres", 10, 500, 100)
                            max_depth = st.slider("Profondeur maximale", 3, 50, 10)
                        with col2:
                            min_samples_split = st.slider("Min samples split", 2, 20, 2)
                            min_samples_leaf = st.slider("Min samples leaf", 1, 10, 1)
                        
                        params = {
                            'n_estimators': n_estimators,
                            'max_depth': max_depth,
                            'min_samples_split': min_samples_split,
                            'min_samples_leaf': min_samples_leaf
                        }
                    
                    elif algorithm_key == "svc":
                        col1, col2 = st.columns(2)
                        with col1:
                            C = st.slider("Paramètre C", 0.1, 10.0, 1.0, 0.1)
                            kernel = st.selectbox("Noyau", ["rbf", "linear", "poly", "sigmoid"])
                        with col2:
                            gamma = st.selectbox("Gamma", ["scale", "auto", "value"])
                            if gamma == "value":
                                gamma_value = st.slider("Valeur de gamma", 0.001, 1.0, 0.1, 0.001)
                                gamma = gamma_value
                        
                        params = {
                            'C': C,
                            'kernel': kernel,
                            'gamma': gamma
                        }
                    
                    elif algorithm_key == "xgboost":
                        col1, col2 = st.columns(2)
                        with col1:
                            n_estimators = st.slider("Nombre d'arbres", 10, 500, 100)
                            max_depth = st.slider("Profondeur maximale", 3, 20, 6)
                            learning_rate = st.slider("Taux d'apprentissage", 0.01, 1.0, 0.3, 0.01)
                        with col2:
                            subsample = st.slider("Subsample", 0.5, 1.0, 1.0, 0.05)
                            colsample_bytree = st.slider("Colsample bytree", 0.5, 1.0, 1.0, 0.05)
                        
                        params = {
                            'n_estimators': n_estimators,
                            'max_depth': max_depth,
                            'learning_rate': learning_rate,
                            'subsample': subsample,
                            'colsample_bytree': colsample_bytree
                        }
                    
                    elif algorithm_key == "gradient_boosting":
                        col1, col2 = st.columns(2)
                        with col1:
                            n_estimators = st.slider("Nombre d'arbres", 10, 500, 100)
                            learning_rate = st.slider("Taux d'apprentissage", 0.01, 1.0, 0.1, 0.01)
                            max_depth = st.slider("Profondeur maximale", 3, 20, 3)
                        with col2:
                            subsample = st.slider("Subsample", 0.5, 1.0, 1.0, 0.05)
                            min_samples_split = st.slider("Min samples split", 2, 20, 2)
                        
                        params = {
                            'n_estimators': n_estimators,
                            'learning_rate': learning_rate,
                            'max_depth': max_depth,
                            'subsample': subsample,
                            'min_samples_split': min_samples_split
                        }
                    
                    elif algorithm_key == "knn":
                        col1, col2 = st.columns(2)
                        with col1:
                            n_neighbors = st.slider("Nombre de voisins", 1, 50, 5)
                            weights = st.selectbox("Poids", ["uniform", "distance"])
                        with col2:
                            algorithm = st.selectbox("Algorithme", ["auto", "ball_tree", "kd_tree", "brute"])
                            leaf_size = st.slider("Taille de feuille", 10, 100, 30)
                        
                        params = {
                            'n_neighbors': n_neighbors,
                            'weights': weights,
                            'algorithm': algorithm,
                            'leaf_size': leaf_size
                        }
                    
                    else:  # Algorithmes simples avec peu de paramètres
                        params = {}
                
                else:  # Régression
                    if algorithm_key == "random_forest":
                        col1, col2 = st.columns(2)
                        with col1:
                            n_estimators = st.slider("Nombre d'arbres", 10, 500, 100)
                            max_depth = st.slider("Profondeur maximale", 3, 50, 10)
                        with col2:
                            min_samples_split = st.slider("Min samples split", 2, 20, 2)
                            min_samples_leaf = st.slider("Min samples leaf", 1, 10, 1)
                        
                        params = {
                            'n_estimators': n_estimators,
                            'max_depth': max_depth,
                            'min_samples_split': min_samples_split,
                            'min_samples_leaf': min_samples_leaf
                        }
                    
                    elif algorithm_key == "xgboost":
                        col1, col2 = st.columns(2)
                        with col1:
                            n_estimators = st.slider("Nombre d'arbres", 10, 500, 100)
                            max_depth = st.slider("Profondeur maximale", 3, 20, 6)
                            learning_rate = st.slider("Taux d'apprentissage", 0.01, 1.0, 0.3, 0.01)
                        with col2:
                            subsample = st.slider("Subsample", 0.5, 1.0, 1.0, 0.05)
                            colsample_bytree = st.slider("Colsample bytree", 0.5, 1.0, 1.0, 0.05)
                        
                        params = {
                            'n_estimators': n_estimators,
                            'max_depth': max_depth,
                            'learning_rate': learning_rate,
                            'subsample': subsample,
                            'colsample_bytree': colsample_bytree
                        }
                    
                    elif algorithm_key == "svr":
                        col1, col2 = st.columns(2)
                        with col1:
                            C = st.slider("Paramètre C", 0.1, 10.0, 1.0, 0.1)
                            kernel = st.selectbox("Noyau", ["rbf", "linear", "poly"])
                        with col2:
                            epsilon = st.slider("Epsilon", 0.01, 1.0, 0.1, 0.01)
                            gamma = st.selectbox("Gamma", ["scale", "auto", "value"])
                            if gamma == "value":
                                gamma_value = st.slider("Valeur de gamma", 0.001, 1.0, 0.1, 0.001)
                                gamma = gamma_value
                        
                        params = {
                            'C': C,
                            'kernel': kernel,
                            'epsilon': epsilon,
                            'gamma': gamma
                        }
                    
                    else:  # Algorithmes simples
                        params = {}
            
            else:  # Réseaux de neurones
                if algorithm_key == "mlp":
                    col1, col2 = st.columns(2)
                    with col1:
                        hidden_layers = st.text_input("Architecture des couches cachées (séparées par des virgules)", "128,64,32")
                        learning_rate = st.selectbox("Taux d'apprentissage", [0.001, 0.01, 0.1, 0.0001], index=0)
                        dropout_rate = st.slider("Taux de dropout", 0.0, 0.5, 0.2, 0.05)
                    with col2:
                        epochs = st.slider("Nombre d'époques", 10, 500, 100)
                        batch_size = st.slider("Taille du batch", 16, 128, 32)
                        activation = st.selectbox("Fonction d'activation", ["relu", "tanh", "sigmoid"])
                    
                    params = {
                        'hidden_layers': hidden_layers,
                        'learning_rate': learning_rate,
                        'dropout_rate': dropout_rate,
                        'epochs': epochs,
                        'batch_size': batch_size,
                        'activation': activation
                    }
                
                elif algorithm_key.startswith("cnn"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        learning_rate = st.selectbox("Taux d'apprentissage", [0.001, 0.01, 0.1, 0.0001], index=0)
                        epochs = st.slider("Nombre d'époques", 10, 200, 50)
                        batch_size = st.slider("Taille du batch", 16, 128, 32)
                    with col2:
                        filters_1 = st.slider("Filtres couche 1", 16, 128, 32)
                        filters_2 = st.slider("Filtres couche 2", 16, 128, 64)
                        filters_3 = st.slider("Filtres couche 3", 16, 128, 128)
                    with col3:
                        dense_units_1 = st.slider("Neurones couche dense 1", 32, 256, 128)
                        dense_units_2 = st.slider("Neurones couche dense 2", 16, 128, 64)
                        dropout_rate = st.slider("Taux de dropout", 0.0, 0.7, 0.5, 0.05)
                    
                    params = {
                        'learning_rate': learning_rate,
                        'epochs': epochs,
                        'batch_size': batch_size,
                        'filters': [filters_1, filters_2, filters_3],
                        'dense_units': [dense_units_1, dense_units_2],
                        'dropout_rate': dropout_rate
                    }
            
            st.session_state.model_config["params"] = params
        
        elif param_method == "Grid Search":
            st.subheader("🔍 Configuration du Grid Search")
            st.info("Le Grid Search testera automatiquement différentes combinaisons d'hyperparamètres")
            
            if method_type == "Algorithmes classiques":
                if algorithm_key == "random_forest":
                    col1, col2 = st.columns(2)
                    with col1:
                        n_estimators_options = st.text_input("Nombre d'arbres (séparés par des virgules)", "50,100,200")
                        max_depth_options = st.text_input("Profondeurs max (séparées par des virgules)", "5,10,15")
                    with col2:
                        min_samples_split_options = st.text_input("Min samples split (séparés par des virgules)", "2,5,10")
                    
                    params = {
                        'n_estimators': [int(x) for x in n_estimators_options.split(',')],
                        'max_depth': [int(x) for x in max_depth_options.split(',')],
                        'min_samples_split': [int(x) for x in min_samples_split_options.split(',')]
                    }
                
                elif algorithm_key == "svc":
                    col1, col2 = st.columns(2)
                    with col1:
                        C_options = st.text_input("Valeurs de C (séparées par des virgules)", "0.1,1,10")
                        kernel_options = st.text_input("Noyaux (séparés par des virgules)", "rbf,linear")
                    with col2:
                        gamma_options = st.text_input("Valeurs de gamma (séparées par des virgules)", "scale,auto,0.1")
                    
                    gamma_list = []
                    for g in gamma_options.split(','):
                        if g in ['scale', 'auto']:
                            gamma_list.append(g)
                        else:
                            gamma_list.append(float(g))
                    
                    params = {
                        'C': [float(x) for x in C_options.split(',')],
                        'kernel': kernel_options.split(','),
                        'gamma': gamma_list
                    }
                
                else:
                    st.info(f"Grid Search non configuré pour {algorithm_key}. Utilisation des paramètres par défaut.")
                    params = {}
            
            else:
                st.info("Grid Search n'est pas disponible pour les réseaux de neurones dans cette version.")
                params = {}
            
            st.session_state.model_config["params"] = params
        
        else:  # Paramètres par défaut
            st.info("Utilisation des paramètres par défaut optimisés pour chaque algorithme")
            params = {}
            st.session_state.model_config["params"] = params
        
        # Bouton d'entraînement
        st.markdown("---")
        st.subheader("🎯 Démarrage de l'entraînement")
        
        if st.button("🚀 DÉMARRER L'ENTRAÎNEMENT", type="primary", use_container_width=True):
            with st.spinner("Entraînement du modèle en cours... Cette opération peut prendre plusieurs minutes"):
                try:
                    # Préparation des données
                    X = st.session_state.dataset["X"]
                    y = st.session_state.dataset["y"]
                    data_type = st.session_state.dataset["data_type"]
                    
                    # Division des données selon la méthode de validation
                    if st.session_state.model_config["validation_method"] == "Split train/test":
                        train_size = st.session_state.model_config["train_size"]
                        if problem_type == "Classification":
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, 
                                train_size=train_size,
                                random_state=42,
                                stratify=y
                            )
                        else:
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, 
                                train_size=train_size,
                                random_state=42
                            )
                        
                        st.success(f"Données divisées: {len(X_train)} entraînement, {len(X_test)} test")
                    
                    else:  # Validation croisée
                        cv_folds = st.session_state.model_config["cv_folds"]
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, 
                            test_size=0.2,
                            random_state=42,
                            stratify=y if problem_type == "Classification" else None
                        )
                        st.success(f"Validation croisée avec {cv_folds} folds")
                    
                    # CRÉATION ET ENTRAÎNEMENT DU MODÈLE
                    model = None
                    history = None
                    
                    # Algorithmes classiques
                    if method_type == "Algorithmes classiques":
                        if problem_type == "Classification":
                            if algorithm_key == "logistic_regression":
                                if param_method == "Grid Search":
                                    param_grid = {
                                        'C': params.get('C', [0.1, 1, 10]),
                                        'solver': params.get('solver', ['lbfgs', 'liblinear']),
                                        'max_iter': params.get('max_iter', [1000])
                                    }
                                    model = GridSearchCV(
                                        LogisticRegression(
                                            class_weight='balanced' if st.session_state.model_config.get("use_class_weight") else None,
                                            random_state=42
                                        ),
                                        param_grid,
                                        cv=min(5, cv_folds if 'cv_folds' in st.session_state.model_config else 5),
                                        scoring='accuracy'
                                    )
                                else:
                                    model = LogisticRegression(
                                        C=params.get('C', 1.0),
                                        solver=params.get('solver', 'lbfgs'),
                                        max_iter=params.get('max_iter', 1000),
                                        penalty=params.get('penalty', 'l2'),
                                        class_weight='balanced' if st.session_state.model_config.get("use_class_weight") else None,
                                        random_state=42
                                    )
                            
                            elif algorithm_key == "random_forest":
                                if param_method == "Grid Search":
                                    param_grid = {
                                        'n_estimators': params.get('n_estimators', [100, 200]),
                                        'max_depth': params.get('max_depth', [10, 20]),
                                        'min_samples_split': params.get('min_samples_split', [2, 5])
                                    }
                                    model = GridSearchCV(
                                        RandomForestClassifier(
                                            class_weight='balanced' if st.session_state.model_config.get("use_class_weight") else None,
                                            random_state=42
                                        ),
                                        param_grid,
                                        cv=min(5, cv_folds if 'cv_folds' in st.session_state.model_config else 5),
                                        scoring='accuracy'
                                    )
                                else:
                                    model = RandomForestClassifier(
                                        n_estimators=params.get('n_estimators', 100),
                                        max_depth=params.get('max_depth', 10),
                                        min_samples_split=params.get('min_samples_split', 2),
                                        min_samples_leaf=params.get('min_samples_leaf', 1),
                                        class_weight='balanced' if st.session_state.model_config.get("use_class_weight") else None,
                                        random_state=42
                                    )
                            
                            elif algorithm_key == "svc":
                                if param_method == "Grid Search":
                                    param_grid = {
                                        'C': params.get('C', [0.1, 1, 10]),
                                        'kernel': params.get('kernel', ['rbf', 'linear']),
                                        'gamma': params.get('gamma', ['scale', 'auto'])
                                    }
                                    model = GridSearchCV(
                                        SVC(
                                            probability=True,
                                            class_weight='balanced' if st.session_state.model_config.get("use_class_weight") else None,
                                            random_state=42
                                        ),
                                        param_grid,
                                        cv=min(5, cv_folds if 'cv_folds' in st.session_state.model_config else 5),
                                        scoring='accuracy'
                                    )
                                else:
                                    model = SVC(
                                        C=params.get('C', 1.0),
                                        kernel=params.get('kernel', 'rbf'),
                                        gamma=params.get('gamma', 'scale'),
                                        probability=True,
                                        class_weight='balanced' if st.session_state.model_config.get("use_class_weight") else None,
                                        random_state=42
                                    )
                            
                            elif algorithm_key == "xgboost":
                                model = xgb.XGBClassifier(
                                    n_estimators=params.get('n_estimators', 100),
                                    max_depth=params.get('max_depth', 6),
                                    learning_rate=params.get('learning_rate', 0.3),
                                    subsample=params.get('subsample', 1.0),
                                    colsample_bytree=params.get('colsample_bytree', 1.0),
                                    random_state=42
                                )
                            
                            elif algorithm_key == "gradient_boosting":
                                model = GradientBoostingClassifier(
                                    n_estimators=params.get('n_estimators', 100),
                                    learning_rate=params.get('learning_rate', 0.1),
                                    max_depth=params.get('max_depth', 3),
                                    subsample=params.get('subsample', 1.0),
                                    min_samples_split=params.get('min_samples_split', 2),
                                    random_state=42
                                )
                            
                            elif algorithm_key == "knn":
                                model = KNeighborsClassifier(
                                    n_neighbors=params.get('n_neighbors', 5),
                                    weights=params.get('weights', 'uniform'),
                                    algorithm=params.get('algorithm', 'auto'),
                                    leaf_size=params.get('leaf_size', 30)
                                )
                            
                            else:  # Algorithmes par défaut
                                if algorithm_key == "naive_bayes":
                                    model = GaussianNB()
                                elif algorithm_key == "adaboost":
                                    model = AdaBoostClassifier(random_state=42)
                                elif algorithm_key == "decision_tree":
                                    model = DecisionTreeClassifier(
                                        class_weight='balanced' if st.session_state.model_config.get("use_class_weight") else None,
                                        random_state=42
                                    )
                        
                        else:  # Régression
                            if algorithm_key == "random_forest":
                                model = RandomForestRegressor(
                                    n_estimators=params.get('n_estimators', 100),
                                    max_depth=params.get('max_depth', 10),
                                    min_samples_split=params.get('min_samples_split', 2),
                                    min_samples_leaf=params.get('min_samples_leaf', 1),
                                    random_state=42
                                )
                            
                            elif algorithm_key == "xgboost":
                                model = xgb.XGBRegressor(
                                    n_estimators=params.get('n_estimators', 100),
                                    max_depth=params.get('max_depth', 6),
                                    learning_rate=params.get('learning_rate', 0.3),
                                    subsample=params.get('subsample', 1.0),
                                    colsample_bytree=params.get('colsample_bytree', 1.0),
                                    random_state=42
                                )
                            
                            elif algorithm_key == "svr":
                                model = SVR(
                                    C=params.get('C', 1.0),
                                    kernel=params.get('kernel', 'rbf'),
                                    epsilon=params.get('epsilon', 0.1),
                                    gamma=params.get('gamma', 'scale')
                                )
                            
                            elif algorithm_key == "gradient_boosting":
                                model = GradientBoostingRegressor(
                                    n_estimators=params.get('n_estimators', 100),
                                    learning_rate=params.get('learning_rate', 0.1),
                                    max_depth=params.get('max_depth', 3),
                                    subsample=params.get('subsample', 1.0),
                                    min_samples_split=params.get('min_samples_split', 2),
                                    random_state=42
                                )
                            
                            else:  # Algorithmes par défaut
                                if algorithm_key == "linear_regression":
                                    model = LinearRegression()
                                elif algorithm_key == "ridge":
                                    model = Ridge(random_state=42)
                                elif algorithm_key == "lasso":
                                    model = Lasso(random_state=42)
                                elif algorithm_key == "elastic_net":
                                    model = ElasticNet(random_state=42)
                                elif algorithm_key == "knn":
                                    model = KNeighborsRegressor()
                    
                        # Entraînement pour algorithmes classiques
                        if hasattr(model, 'fit'):
                            model.fit(X_train, y_train)
                            
                            if param_method == "Grid Search" and hasattr(model, 'best_params_'):
                                st.success(f"Meilleurs paramètres: {model.best_params_}")
                                model = model.best_estimator_
                        
                        y_pred = model.predict(X_test)
                        if hasattr(model, 'predict_proba'):
                            y_pred_proba = model.predict_proba(X_test)
                        else:
                            y_pred_proba = None
                    
                    else:  # Réseaux de neurones
                        # Préparation des données pour les réseaux de neurones
                        if algorithm_key == "mlp":
                            if data_type == "features":
                                input_dim = X_train.shape[1]
                                X_train_mlp = X_train
                                X_test_mlp = X_test
                            else:
                                input_dim = X_train.shape[1] * X_train.shape[2] * X_train.shape[3]
                                X_train_mlp = X_train.reshape(X_train.shape[0], -1)
                                X_test_mlp = X_test.reshape(X_test.shape[0], -1)
                            
                            # Création du modèle MLP
                            hidden_layers = [int(x.strip()) for x in params.get('hidden_layers', '128,64,32').split(',')]
                            model = create_mlp_model(
                                input_dim, 
                                num_classes, 
                                hidden_layers, 
                                params.get('dropout_rate', 0.2),
                                problem_type.lower()
                            )
                            
                            # Compilation
                            if problem_type == "Classification":
                                if num_classes == 1:
                                    model.compile(
                                        optimizer=Adam(learning_rate=params.get('learning_rate', 0.001)),
                                        loss='binary_crossentropy',
                                        metrics=['accuracy']
                                    )
                                    y_train_keras = y_train
                                    y_test_keras = y_test
                                else:
                                    model.compile(
                                        optimizer=Adam(learning_rate=params.get('learning_rate', 0.001)),
                                        loss='categorical_crossentropy',
                                        metrics=['accuracy']
                                    )
                                    y_train_keras = to_categorical(y_train, num_classes)
                                    y_test_keras = to_categorical(y_test, num_classes)
                            else:
                                model.compile(
                                    optimizer=Adam(learning_rate=params.get('learning_rate', 0.001)),
                                    loss='mse',
                                    metrics=['mae', 'mse']
                                )
                                y_train_keras = y_train
                                y_test_keras = y_test
                            
                            # Callbacks
                            callbacks = [
                                EarlyStopping(patience=10, restore_best_weights=True),
                                ReduceLROnPlateau(factor=0.5, patience=5)
                            ]
                            
                            # Entraînement
                            history = model.fit(
                                X_train_mlp, y_train_keras,
                                epochs=params.get('epochs', 100),
                                batch_size=params.get('batch_size', 32),
                                validation_data=(X_test_mlp, y_test_keras),
                                callbacks=callbacks,
                                verbose=0
                            )
                            
                            # Prédictions
                            y_pred_proba = model.predict(X_test_mlp)
                            if problem_type == "Classification":
                                if num_classes == 1:
                                    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
                                else:
                                    y_pred = np.argmax(y_pred_proba, axis=1)
                            else:
                                y_pred = y_pred_proba.flatten()
                        
                        elif algorithm_key.startswith("cnn"):
                            # Architecture mapping
                            architecture_map = {
                                "cnn_simple": "simple",
                                "cnn_vgg": "vgg_like", 
                                "cnn_advanced": "advanced",
                                "cnn_regression": "regression_cnn"
                            }
                            
                            architecture = architecture_map.get(algorithm_key, "simple")
                            input_shape = X_train.shape[1:]
                            
                            # Si VGG16 et images en niveaux de gris, adapter l'input shape
                            if architecture == "advanced" and input_shape[2] == 1:
                                st.warning("Adaptation de l'input shape pour VGG16 (1 canal -> 3 canaux)")
                                # Pour l'entraînement, on va dupliquer les canaux
                                X_train_vgg = np.concatenate([X_train] * 3, axis=-1)
                                X_test_vgg = np.concatenate([X_test] * 3, axis=-1)
                                input_shape_vgg = (input_shape[0], input_shape[1], 3)
                            else:
                                X_train_vgg = X_train
                                X_test_vgg = X_test
                                input_shape_vgg = input_shape
                            
                            model = create_cnn_model(
                                input_shape_vgg, 
                                num_classes, 
                                architecture, 
                                problem_type.lower(),
                                params.get('filters', [32, 64, 128]),
                                params.get('dense_units', [128, 64]),
                                params.get('dropout_rate', 0.5)
                            )
                            
                            # Compilation
                            if problem_type == "Classification":
                                if num_classes == 1:
                                    model.compile(
                                        optimizer=Adam(learning_rate=params.get('learning_rate', 0.001)),
                                        loss='binary_crossentropy',
                                        metrics=['accuracy']
                                    )
                                    y_train_keras = y_train
                                    y_test_keras = y_test
                                else:
                                    model.compile(
                                        optimizer=Adam(learning_rate=params.get('learning_rate', 0.001)),
                                        loss='categorical_crossentropy',
                                        metrics=['accuracy']
                                    )
                                    y_train_keras = to_categorical(y_train, num_classes)
                                    y_test_keras = to_categorical(y_test, num_classes)
                            else:
                                model.compile(
                                    optimizer=Adam(learning_rate=params.get('learning_rate', 0.001)),
                                    loss='mse',
                                    metrics=['mae', 'mse']
                                )
                                y_train_keras = y_train
                                y_test_keras = y_test
                            
                            # Callbacks
                            callbacks = [
                                EarlyStopping(patience=10, restore_best_weights=True),
                                ReduceLROnPlateau(factor=0.5, patience=5)
                            ]
                            
                            # Entraînement
                            history = model.fit(
                                X_train_vgg, y_train_keras,
                                epochs=params.get('epochs', 50),
                                batch_size=params.get('batch_size', 32),
                                validation_data=(X_test_vgg, y_test_keras),
                                callbacks=callbacks,
                                verbose=0
                            )
                            
                            # Prédictions
                            y_pred_proba = model.predict(X_test_vgg)
                            if problem_type == "Classification":
                                if num_classes == 1:
                                    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
                                else:
                                    y_pred = np.argmax(y_pred_proba, axis=1)
                            else:
                                y_pred = y_pred_proba.flatten()
                    
                    # ÉVALUATION DU MODÈLE
                    st.subheader("📊 Résultats de l'entraînement")
                    
                    # Affichage de l'historique pour les réseaux de neurones
                    if history is not None:
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                        
                        # Loss
                        ax1.plot(history.history['loss'], label='Train Loss')
                        if 'val_loss' in history.history:
                            ax1.plot(history.history['val_loss'], label='Validation Loss')
                        ax1.set_title('Évolution de la loss')
                        ax1.set_xlabel('Époques')
                        ax1.set_ylabel('Loss')
                        ax1.legend()
                        
                        # Métrique
                        if 'accuracy' in history.history:
                            ax2.plot(history.history['accuracy'], label='Train Accuracy')
                            if 'val_accuracy' in history.history:
                                ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
                            ax2.set_title('Évolution de l\'accuracy')
                            ax2.set_ylabel('Accuracy')
                        elif 'mae' in history.history:
                            ax2.plot(history.history['mae'], label='Train MAE')
                            if 'val_mae' in history.history:
                                ax2.plot(history.history['val_mae'], label='Validation MAE')
                            ax2.set_title('Évolution du MAE')
                            ax2.set_ylabel('MAE')
                        else:
                            ax2.plot(history.history['mse'], label='Train MSE')
                            if 'val_mse' in history.history:
                                ax2.plot(history.history['val_mse'], label='Validation MSE')
                            ax2.set_title('Évolution du MSE')
                            ax2.set_ylabel('MSE')
                        
                        ax2.set_xlabel('Époques')
                        ax2.legend()
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                    # Calcul des métriques
                    if problem_type == "Classification":
                        accuracy = accuracy_score(y_test, y_pred)
                        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                        
                        metrics = {
                            "accuracy": accuracy,
                            "precision": precision,
                            "recall": recall,
                            "f1_score": f1
                        }
                        
                        # Affichage des métriques
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Accuracy", f"{accuracy:.3f}")
                        with col2:
                            st.metric("Precision", f"{precision:.3f}")
                        with col3:
                            st.metric("Recall", f"{recall:.3f}")
                        with col4:
                            st.metric("F1-Score", f"{f1:.3f}")
                        
                        # Matrice de confusion
                        st.subheader("📋 Matrice de confusion")
                        cm = confusion_matrix(y_test, y_pred)
                        class_names = st.session_state.model_config.get("class_names", range(len(np.unique(y_test))))
                        fig_cm = plot_confusion_matrix(cm, class_names)
                        st.pyplot(fig_cm)
                        
                        # Courbe ROC pour classification binaire
                        if len(np.unique(y_test)) == 2 and y_pred_proba is not None:
                            st.subheader("📈 Courbe ROC")
                            if method_type == "Réseaux de neurones":
                                if num_classes == 1:
                                    fpr, tpr, _ = roc_curve(y_test, y_pred_proba.flatten())
                                else:
                                    fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
                            else:
                                fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
                            
                            roc_auc = auc(fpr, tpr)
                            fig_roc = plot_roc_curve(fpr, tpr, roc_auc)
                            st.pyplot(fig_roc)
                            metrics["auc"] = roc_auc
                            st.metric("AUC-ROC", f"{roc_auc:.3f}")
                    
                    else:  # Régression
                        mse = mean_squared_error(y_test, y_pred)
                        mae = mean_absolute_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        
                        metrics = {
                            "mse": mse,
                            "mae": mae,
                            "r2": r2
                        }
                        
                        # Affichage des métriques
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("MSE", f"{mse:.3f}")
                        with col2:
                            st.metric("MAE", f"{mae:.3f}")
                        with col3:
                            st.metric("R²", f"{r2:.3f}")
                        
                        # Graphiques de régression
                        st.subheader("📊 Résultats de la régression")
                        fig_reg = plot_regression_results(y_test, y_pred)
                        st.pyplot(fig_reg)
                    
                    # Sauvegarde dans la session
                    st.session_state.trained_model = model
                    st.session_state.model_metrics = metrics
                    st.session_state.training_history = history
                    
                    st.success("✅ Modèle entraîné avec succès!")
                    
                except Exception as e:
                    st.error(f"Erreur lors de l'entraînement: {str(e)}")
                    import traceback
                    st.error(f"Détails de l'erreur: {traceback.format_exc()}")
        
        # Navigation
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("← Retour à l'étape précédente"):
                st.session_state.current_step = 5
                st.rerun()
        
        with col2:
            if 'trained_model' in st.session_state:
                if st.button("Continuer →"):
                    st.session_state.current_step = 7
                    st.rerun()
            else:
                st.button("Continuer →", disabled=True)
    
    # Étape 7: Sauvegarde
    elif st.session_state.current_step == 7:
        st.subheader("💾 Étape 7: Sauvegarde du modèle")
        
        if 'trained_model' not in st.session_state:
            st.error("Aucun modèle entraîné trouvé. Veuillez retourner à l'étape d'entraînement.")
        else:
            model = st.session_state.trained_model
            metrics = st.session_state.model_metrics
            model_config = st.session_state.model_config
            
            st.info("Résumé du modèle à sauvegarder:")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Configuration:**")
                st.write(f"- Type de problème: {model_config['problem_type']}")
                st.write(f"- Algorithme: {model_config['algorithm']}")
                st.write(f"- Nom du modèle: {model_config['model_name']}")
                st.write(f"- Méthode de validation: {model_config['validation_method']}")
                st.write(f"- Configuration paramètres: {model_config['param_method']}")
                st.write(f"- Taille des images: {model_config.get('image_target_size', 'N/A')}")
                
                if model_config['problem_type'] == "Classification":
                    st.write(f"- Classes: {', '.join(model_config['class_names'])}")
                else:
                    st.write(f"- Variable cible: {model_config['target_name']}")
            
            with col2:
                st.write("**Performances:**")
                for metric, value in metrics.items():
                    st.write(f"- {metric}: {value:.3f}")
            
            if st.button("💾 SAUVEGARDER LE MODÈLE", type="primary", use_container_width=True):
                try:
                    model_dir = save_model(
                        model, 
                        model_config['model_name'],
                        model_config['method_type'],
                        model_config['problem_type'],
                        metrics,
                        model_config
                    )
                    
                    st.success(f"✅ Modèle sauvegardé avec succès dans: {model_dir}")
                    
                    # Réinitialisation pour un nouveau modèle
                    time.sleep(2)
                    st.session_state.current_step = 1
                    st.session_state.model_config = {}
                    st.session_state.trained_model = None
                    st.session_state.dataset = None
                    st.session_state.image_data = {}
                    st.session_state.model_metrics = None
                    st.session_state.training_history = None
                    st.session_state.uploaded_files_cache = set()
                    if 'feature_extractor' in st.session_state:
                        del st.session_state.feature_extractor
                    if 'scaler' in st.session_state:
                        del st.session_state.scaler
                    
                    st.rerun()
                
                except Exception as e:
                    st.error(f"Erreur lors de la sauvegarde: {str(e)}")
            
            if st.button("🔄 Recommencer avec un nouveau modèle", use_container_width=True):
                st.session_state.current_step = 1
                st.session_state.model_config = {}
                st.session_state.trained_model = None
                st.session_state.dataset = None
                st.session_state.image_data = {}
                st.session_state.model_metrics = None
                st.session_state.training_history = None
                st.session_state.uploaded_files_cache = set()
                if 'feature_extractor' in st.session_state:
                    del st.session_state.feature_extractor
                if 'scaler' in st.session_state:
                    del st.session_state.scaler
                st.rerun()

# Page Évaluation et performance
elif selected == "📊 Évaluation et performance":
    st.title("📊 Évaluation et Performance des Modèles")
    
    saved_models = get_saved_models()
    
    if not saved_models:
        st.info("Aucun modèle sauvegardé. Veuillez d'abord créer et entraîner un modèle.")
    else:
        model_names = list(saved_models.keys())
        selected_model = st.selectbox(
            "Sélectionnez un modèle à évaluer",
            model_names
        )
        
        if selected_model:
            model_info = saved_models[selected_model]
            
            # Charger le modèle et ses métriques
            model, metrics, model_config = load_saved_model(selected_model)
            
            if model is None:
                st.error("Erreur lors du chargement du modèle")
            else:
                st.subheader(f"Évaluation du modèle: {selected_model}")
                
                # Informations générales
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Type de problème", model_info["problem_type"])
                with col2:
                    st.metric("Type de méthode", model_info["type"])
                with col3:
                    st.metric("Algorithme", model_info["config"].get("algorithm", "N/A"))
                with col4:
                    st.metric("Date de création", model_info["created_at"].split()[0])
                
                st.markdown("---")
                
                # Métriques de performance
                st.subheader("📈 Métriques de Performance")
                
                if model_info["problem_type"] == "Classification":
                    # Métriques de classification
                    metrics_data = {
                        "Métrique": ["Accuracy", "Precision", "Recall", "F1-Score"],
                        "Valeur": [
                            metrics.get("accuracy", 0),
                            metrics.get("precision", 0),
                            metrics.get("recall", 0),
                            metrics.get("f1_score", 0)
                        ]
                    }
                    
                    # Graphique en barres des métriques
                    fig = px.bar(
                        metrics_data, 
                        x="Métrique", 
                        y="Valeur",
                        title="Métriques de Classification",
                        color="Métrique",
                        text_auto='.3f'
                    )
                    fig.update_layout(yaxis_range=[0, 1])
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Affichage des valeurs numériques
                    cols = st.columns(4)
                    metric_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
                    metric_values = [
                        metrics.get("accuracy", 0),
                        metrics.get("precision", 0),
                        metrics.get("recall", 0),
                        metrics.get("f1_score", 0)
                    ]
                    
                    for i, (name, value) in enumerate(zip(metric_names, metric_values)):
                        with cols[i]:
                            st.metric(name, f"{value:.3f}")
                    
                    # Courbe ROC si disponible
                    if "auc" in metrics:
                        st.subheader("Courbe ROC")
                        st.info(f"AUC: {metrics['auc']:.3f}")
                        
                        # Recréer une courbe ROC simplifiée pour l'affichage
                        fig_roc, ax = plt.subplots(figsize=(8, 6))
                        fpr = np.linspace(0, 1, 100)
                        tpr = np.sin(fpr * np.pi / 2)  # Approximation pour l'affichage
                        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {metrics["auc"]:.3f})')
                        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                        ax.set_xlim([0.0, 1.0])
                        ax.set_ylim([0.0, 1.05])
                        ax.set_xlabel('False Positive Rate')
                        ax.set_ylabel('True Positive Rate')
                        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
                        ax.legend(loc="lower right")
                        st.pyplot(fig_roc)
                
                else:  # Régression
                    # Métriques de régression
                    metrics_data = {
                        "Métrique": ["MSE", "MAE", "R²"],
                        "Valeur": [
                            metrics.get("mse", 0),
                            metrics.get("mae", 0),
                            metrics.get("r2", 0)
                        ]
                    }
                    
                    # Graphique en barres des métriques
                    fig = px.bar(
                        metrics_data, 
                        x="Métrique", 
                        y="Valeur",
                        title="Métriques de Régression",
                        color="Métrique",
                        text_auto='.3f'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Affichage des valeurs numériques
                    cols = st.columns(3)
                    metric_names = ["MSE", "MAE", "R²"]
                    metric_values = [
                        metrics.get("mse", 0),
                        metrics.get("mae", 0),
                        metrics.get("r2", 0)
                    ]
                    
                    for i, (name, value) in enumerate(zip(metric_names, metric_values)):
                        with cols[i]:
                            st.metric(name, f"{value:.3f}")

# Page Test
elif selected == "🧪 Test":
    st.title("🧪 Test des Modèles")
    
    saved_models = get_saved_models()
    
    if not saved_models:
        st.info("Aucun modèle sauvegardé. Veuillez d'abord créer et entraîner un modèle.")
    else:
        model_names = list(saved_models.keys())
        selected_model = st.selectbox(
            "Sélectionnez un modèle à tester",
            model_names
        )
        
        if selected_model:
            model_info = saved_models[selected_model]
            model, metrics, model_config = load_saved_model(selected_model)
            
            if model is None:
                st.error("Erreur lors du chargement du modèle")
            else:
                st.info(f"Modèle sélectionné: {selected_model} ({model_info['problem_type']})")
                
                tab1, tab2 = st.tabs(["📤 Upload d'images de test", "📷 Capture en temps réel"])
                
                with tab1:
                    uploaded_files = st.file_uploader(
                        "Téléchargez des images à tester",
                        type=['jpg', 'png', 'jpeg'],
                        accept_multiple_files=True,
                        key="test_upload_images"
                    )
                    
                    if uploaded_files:
                        st.subheader("Résultats des prédictions")
                        
                        for uploaded_file in uploaded_files:
                            col1, col2 = st.columns([2, 3])
                            
                            with col1:
                                # Afficher l'image
                                image = Image.open(uploaded_file)
                                st.image(image, caption=uploaded_file.name, width=200)
                            
                            with col2:
                                # Déterminer si on utilise VGG16 pour le prétraitement
                                use_vgg = model_config.get("use_vgg", False)
                                target_size = model_config.get("image_target_size", (64, 64))
                                
                                # Prétraiter l'image comme lors de l'entraînement
                                processed_image = preprocess_image(image, target_size, use_vgg=use_vgg)
                                
                                # Prédiction
                                try:
                                    if model_info["problem_type"] == "Classification":
                                        if model_info["type"] in ["Réseaux de neurones"]:
                                            if model_config["algorithm_key"].startswith("cnn"):
                                                # Gestion spéciale pour VGG16 si nécessaire
                                                if model_config["algorithm_key"] == "cnn_advanced" and processed_image.shape[-1] == 1:
                                                    # Dupliquer les canaux pour VGG16
                                                    processed_image = np.concatenate([processed_image] * 3, axis=-1)
                                                
                                                prediction = model.predict(processed_image)
                                                if model_config["num_classes"] == 1:
                                                    predicted_class = 1 if prediction[0][0] > 0.5 else 0
                                                    confidence = prediction[0][0] if predicted_class == 1 else 1 - prediction[0][0]
                                                    probabilities = None
                                                else:
                                                    predicted_class = np.argmax(prediction[0])
                                                    probabilities = prediction[0]
                                                    confidence = np.max(prediction[0])
                                            else:  # MLP
                                                if model_config["method_type"] == "Algorithmes classiques":
                                                    # MLP avec features
                                                    if 'feature_extractor' in model_config:
                                                        features = extract_features_single_image(
                                                            processed_image.squeeze(),
                                                            model_config['feature_method'],
                                                            model_config['feature_extractor'],
                                                            model_config.get('scaler')
                                                        )
                                                        prediction = model.predict(features)
                                                        if model_config["num_classes"] == 1:
                                                            predicted_class = 1 if prediction[0][0] > 0.5 else 0
                                                            confidence = prediction[0][0] if predicted_class == 1 else 1 - prediction[0][0]
                                                            probabilities = None
                                                        else:
                                                            predicted_class = np.argmax(prediction[0])
                                                            probabilities = prediction[0]
                                                            confidence = np.max(prediction[0])
                                                else:
                                                    # MLP avec images flatten
                                                    processed_flat = processed_image.reshape(1, -1)
                                                    prediction = model.predict(processed_flat)
                                                    if model_config["num_classes"] == 1:
                                                        predicted_class = 1 if prediction[0][0] > 0.5 else 0
                                                        confidence = prediction[0][0] if predicted_class == 1 else 1 - prediction[0][0]
                                                        probabilities = None
                                                    else:
                                                        predicted_class = np.argmax(prediction[0])
                                                        probabilities = prediction[0]
                                                        confidence = np.max(prediction[0])
                                        else:
                                            # Algorithmes classiques avec features
                                            if 'feature_extractor' in model_config:
                                                features = extract_features_single_image(
                                                    processed_image.squeeze(),
                                                    model_config['feature_method'],
                                                    model_config['feature_extractor'],
                                                    model_config.get('scaler')
                                                )
                                                
                                                prediction = model.predict(features)
                                                probabilities = model.predict_proba(features) if hasattr(model, 'predict_proba') else None
                                                predicted_class = prediction[0]
                                                confidence = probabilities[0][predicted_class] if probabilities is not None else None
                                            else:
                                                # Fallback pour les modèles sans feature extractor
                                                processed_flat = processed_image.reshape(1, -1)
                                                prediction = model.predict(processed_flat)
                                                probabilities = model.predict_proba(processed_flat) if hasattr(model, 'predict_proba') else None
                                                predicted_class = prediction[0]
                                                confidence = probabilities[0][predicted_class] if probabilities is not None else None
                                        
                                        class_names = model_config.get("class_names", [f"Classe {i}" for i in range(model_config["num_classes"])])
                                        
                                        # Générer l'explication
                                        explanation = get_classification_explanation(
                                            predicted_class, class_names, probabilities, confidence
                                        )
                                        st.markdown(explanation)
                                    
                                    else:  # Régression
                                        if model_info["type"] in ["Réseaux de neurones"]:
                                            if model_config["algorithm_key"].startswith("cnn"):
                                                # Gestion spéciale pour VGG16 si nécessaire
                                                if model_config["algorithm_key"] == "cnn_advanced" and processed_image.shape[-1] == 1:
                                                    # Dupliquer les canaux pour VGG16
                                                    processed_image = np.concatenate([processed_image] * 3, axis=-1)
                                                
                                                prediction = model.predict(processed_image)[0][0]
                                            else:  # MLP
                                                if model_config["method_type"] == "Algorithmes classiques":
                                                    if 'feature_extractor' in model_config:
                                                        features = extract_features_single_image(
                                                            processed_image.squeeze(),
                                                            model_config['feature_method'],
                                                            model_config['feature_extractor'],
                                                            model_config.get('scaler')
                                                        )
                                                        prediction = model.predict(features)[0][0]
                                                    else:
                                                        processed_flat = processed_image.reshape(1, -1)
                                                        prediction = model.predict(processed_flat)[0][0]
                                                else:
                                                    processed_flat = processed_image.reshape(1, -1)
                                                    prediction = model.predict(processed_flat)[0][0]
                                        else:
                                            # Algorithmes classiques avec features
                                            if 'feature_extractor' in model_config:
                                                features = extract_features_single_image(
                                                    processed_image.squeeze(),
                                                    model_config['feature_method'],
                                                    model_config['feature_extractor'],
                                                    model_config.get('scaler')
                                                )
                                                prediction = model.predict(features)[0]
                                            else:
                                                processed_flat = processed_image.reshape(1, -1)
                                                prediction = model.predict(processed_flat)[0]
                                        
                                        # Générer l'explication
                                        target_range = model_config.get("target_range")
                                        target_name = model_config.get("target_name", "valeur cible")
                                        explanation = get_regression_explanation(prediction, target_range, target_name)
                                        st.markdown(explanation)
                                
                                except Exception as e:
                                    st.error(f"Erreur lors de la prédiction: {str(e)}")
                
                with tab2:
                    st.info("Capture d'image en temps réel pour le test")
                    
                    camera_input = st.camera_input("Prendre une photo pour le test", key="test_camera")
                    
                    if camera_input:
                        image = Image.open(camera_input)
                        st.image(image, caption="Image capturée", width=300)
                        
                        # Déterminer si on utilise VGG16 pour le prétraitement
                        use_vgg = model_config.get("use_vgg", False)
                        target_size = model_config.get("image_target_size", (64, 64))
                        
                        # Prétraiter l'image
                        processed_image = preprocess_image(image, target_size, use_vgg=use_vgg)
                        
                        # Prédiction
                        try:
                            if model_info["problem_type"] == "Classification":
                                if model_info["type"] in ["Réseaux de neurones"]:
                                    if model_config["algorithm_key"].startswith("cnn"):
                                        # Gestion spéciale pour VGG16 si nécessaire
                                        if model_config["algorithm_key"] == "cnn_advanced" and processed_image.shape[-1] == 1:
                                            # Dupliquer les canaux pour VGG16
                                            processed_image = np.concatenate([processed_image] * 3, axis=-1)
                                        
                                        prediction = model.predict(processed_image)
                                        if model_config["num_classes"] == 1:
                                            predicted_class = 1 if prediction[0][0] > 0.5 else 0
                                            confidence = prediction[0][0] if predicted_class == 1 else 1 - prediction[0][0]
                                            probabilities = None
                                        else:
                                            predicted_class = np.argmax(prediction[0])
                                            probabilities = prediction[0]
                                            confidence = np.max(prediction[0])
                                    else:  # MLP
                                        processed_flat = processed_image.reshape(1, -1)
                                        prediction = model.predict(processed_flat)
                                        if model_config["num_classes"] == 1:
                                            predicted_class = 1 if prediction[0][0] > 0.5 else 0
                                            confidence = prediction[0][0] if predicted_class == 1 else 1 - prediction[0][0]
                                            probabilities = None
                                        else:
                                            predicted_class = np.argmax(prediction[0])
                                            probabilities = prediction[0]
                                            confidence = np.max(prediction[0])
                                else:
                                    # Algorithmes classiques avec features
                                    if 'feature_extractor' in model_config:
                                        features = extract_features_single_image(
                                            processed_image.squeeze(),
                                            model_config['feature_method'],
                                            model_config['feature_extractor'],
                                            model_config.get('scaler')
                                        )
                                        
                                        prediction = model.predict(features)
                                        probabilities = model.predict_proba(features) if hasattr(model, 'predict_proba') else None
                                        predicted_class = prediction[0]
                                        confidence = probabilities[0][predicted_class] if probabilities is not None else None
                            
                                class_names = model_config.get("class_names", [f"Classe {i}" for i in range(model_config["num_classes"])])
                                
                                # Générer l'explication
                                explanation = get_classification_explanation(
                                    predicted_class, class_names, probabilities, confidence
                                )
                                st.markdown(explanation)
                            
                            else:  # Régression
                                if model_info["type"] in ["Réseaux de neurones"]:
                                    if model_config["algorithm_key"].startswith("cnn"):
                                        # Gestion spéciale pour VGG16 si nécessaire
                                        if model_config["algorithm_key"] == "cnn_advanced" and processed_image.shape[-1] == 1:
                                            # Dupliquer les canaux pour VGG16
                                            processed_image = np.concatenate([processed_image] * 3, axis=-1)
                                        
                                        prediction = model.predict(processed_image)[0][0]
                                    else:  # MLP
                                        processed_flat = processed_image.reshape(1, -1)
                                        prediction = model.predict(processed_flat)[0][0]
                                else:
                                    # Algorithmes classiques avec features
                                    if 'feature_extractor' in model_config:
                                        features = extract_features_single_image(
                                            processed_image.squeeze(),
                                            model_config['feature_method'],
                                            model_config['feature_extractor'],
                                            model_config.get('scaler')
                                        )
                                        prediction = model.predict(features)[0]
                                
                                # Générer l'explication
                                target_range = model_config.get("target_range")
                                target_name = model_config.get("target_name", "valeur cible")
                                explanation = get_regression_explanation(prediction, target_range, target_name)
                                st.markdown(explanation)
                        
                        except Exception as e:
                            st.error(f"Erreur lors de la prédiction: {str(e)}")

# Page Aide
elif selected == "❓ Aide":
    st.title("❓ Guide d'Utilisation")
    
    st.markdown("""
    ## 📚 Guide Complet de l'Application ML/DL Image Model Builder
    
    ### 🏗️ Construction des Modèles
    
    **Étape 1: Définition du problème**
    - **Classification**: Prédire des catégories (ex: chat/chien)
        - Spécifiez le nombre de classes
        - Donnez un nom significatif à chaque classe
    - **Régression**: Prédire des valeurs continues (ex: prix, score)
        - Donnez un nom à la variable cible
    - Sélectionnez la méthode: Algorithmes classiques ou Réseaux de neurones
    
    **Étape 2: Sélection de l'algorithme**
    - **Algorithmes classiques**: Random Forest, SVM, XGBoost, etc.
    - **Réseaux de neurones**: MLP ou CNN
    - **VGG16**: Modèle pré-entraîné pour transfer learning (nécessite images RGB)
    
    **Étape 3: Collecte des images**
    - Upload d'images ou capture en direct
    - Gestion automatique des doublons
    - Pour la classification: organisation par classe avec les noms spécifiés
    - Pour la régression: attribution des valeurs cibles avec le nom spécifié
    
    **Étape 4: Prétraitement** - **AMÉLIORÉE**
    - Redimensionnement automatique à 64×64 pixels
    - **VGG16**: Conservation des images en RGB (3 canaux)
    - **Autres modèles**: Conversion en niveaux de gris (1 canal)
    - Normalisation des valeurs des pixels
    - **Algorithmes classiques**: Extraction de features (HOG ou PCA)
    - **Réseaux de neurones**: Utilisation directe des images
    - Gestion du déséquilibre par Data Augmentation
    
    **Étape 5: Validation**
    - **Split Train/Test**: Contrôle du pourcentage d'entraînement
    - **Validation Croisée**: Choix du nombre de folds
    
    **Étape 6: Entraînement** - **AMÉLIORÉE**
    - **3 méthodes de configuration**:
      - **Paramètres par défaut**: Optimisés pour chaque algorithme
      - **Configuration manuelle**: Contrôle complet des hyperparamètres
      - **Grid Search**: Recherche automatique des meilleurs paramètres
    - **Hyperparamètres complets** pour tous les modèles
    - **VGG16**: Gestion automatique de l'input shape (adaptation 1→3 canaux si nécessaire)
    - Visualisation en temps réel de l'entraînement
    - Gestion robuste des erreurs
    
    **Étape 7: Sauvegarde**
    - Modèles classiques: format .pkl
    - Réseaux de neurones: format .h5
    - Métriques: format JSON
    
    ### 📊 Évaluation et Performance
    
    - Consultation des modèles sauvegardés
    - Visualisation détaillée des performances
    - Graphiques interactifs des métriques
    
    ### 🧪 Test - **AMÉLIORÉE**
    
    - Test sur nouvelles images
    - Upload ou capture en direct
    - **Prétraitement adaptatif**: RGB pour VGG16, niveaux de gris pour autres modèles
    - Prédictions avec niveaux de confiance
    - Explications détaillées des résultats utilisant les noms de classes ou variables cibles spécifiés
    - Support pour tous les types de modèles
    
    ### 🔧 Technologies Utilisées
    
    - **Streamlit**: Interface utilisateur
    - **Scikit-learn**: Algorithmes classiques
    - **TensorFlow/Keras**: Réseaux de neurones
    - **OpenCV/PIL**: Traitement d'images
    - **Plotly/Matplotlib**: Visualisations
    
    ### 🎯 Fonctionnalités Avancées
    
    **Gestion Intelligente des Canaux:**
    - **VGG16**: Conservation automatique des images en RGB
    - **Autres modèles**: Conversion optimisée en niveaux de gris
    - Adaptation automatique de l'input shape selon le modèle
    
    **Data Augmentation Corrigée:**
    - Utilisation de `next()` au lieu de `next()` obsolète
    - Configuration différenciée selon le type de modèle
    - Gestion robuste des formats d'image
    
    **Personnalisation Complète:**
    - **Classification**: Noms personnalisés pour chaque classe
    - **Régression**: Nom personnalisé pour la variable cible
    - Interface adaptative selon le type de problème
    
    **Architectures CNN Améliorées:**
    - **CNN Simple**: Architecture basique pour débuter
    - **CNN VGG-like**: Architecture profonde pour meilleure précision
    - **CNN Transfer Learning**: Utilisation de VGG16 pré-entraîné avec gestion automatique des canaux
    - **CNN Régression**: Architecture spécialisée pour les valeurs continues
    
    **Configuration Flexible des Hyperparamètres:**
    - **Paramètres par défaut**: Optimisés et testés
    - **Configuration manuelle**: Contrôle granulaire
    - **Grid Search**: Optimisation automatique
    
    **Explications des Prédictions:**
    - **Classification**: Probabilités par classe avec noms personnalisés, niveau de confiance
    - **Régression**: Valeur prédite avec contexte dans l'échelle historique et nom de variable
    - **Indicateurs de confiance**: Haut/Moyen/Faible selon le score
    
    ### 🔧 Corrections Apportées
    
    **Problème PCA résolu:**
    - Extraction de features PCA corrigée pour les prédictions
    - Gestion cohérente des transformateurs (PCA et StandardScaler)
    - Sauvegarde et chargement corrects des extracteurs de features
    - Fonction `extract_features_single_image` pour un traitement uniforme
    """)

# Style CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        color: #155724;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        color: #856404;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 1rem;
        color: #0c5460;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    .prediction-high {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #28a745;
    }
    .prediction-medium {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
    }
    .prediction-low {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# Pied de page
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "ML/DL Image Model Builder • Développé avec Streamlit • "
    f"© {datetime.now().year} Tous droits réservés"
    "</div>",
    unsafe_allow_html=True
)