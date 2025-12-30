# 📦 Modèle de Prédiction du Temps de Livraison

## 🎯 Vue d'ensemble du projet

Ce projet vise à développer un modèle de Machine Learning capable de prédire le temps total d'une livraison, depuis la commande jusqu'à la réception. L'objectif est de permettre à l'entreprise de logistique et de livraison d'anticiper les retards, d'informer les clients en temps réel et d'optimiser l'organisation des tournées.

### 🚀 Objectifs principaux

- Développer une preuve de concept fiable et automatisée
- Remplacer les estimations manuelles par des prédictions basées sur le ML
- Réduire les retards et améliorer la satisfaction client
- Optimiser les tournées de livraison

---

## 📊 Contexte et Problématique

### Situation actuelle

- ❌ Estimations de temps de livraison faites manuellement
- ❌ Pas de modèle de prédiction fiable
- ❌ Suivi des expérimentations absent
- ❌ Retards fréquents créant une insatisfaction client

### Variables d'entrée

Le modèle prend en compte les facteurs suivants :

| Variable | Type | Description |
|----------|------|-------------|
| `Distance_km` | Numérique | Distance entre le restaurant et l'adresse de livraison |
| `Traffic_Level` | Catégorie | Niveau de trafic (faible, moyen, élevé) |
| `Vehicle_Type` | Catégorie | Type de véhicule utilisé (vélo, scooter, voiture) |
| `Time_of_Day` | Catégorie | Heure de la journée (matin, midi, soir, nuit) |
| `Courier_Experience` | Numérique | Années d'expérience du livreur |
| `Weather` | Catégorie | Conditions météorologiques |
| `Preparation_Time` | Numérique | Temps de préparation de la commande |

### Variable cible

- `DeliveryTime` : Temps total de livraison (en minutes)

---

## 🛠️ Architecture du Projet

```
Pr-diction-du-Temps-de-Livraison/
├── dataset.csv
├── EDA.ipynb
├── pipeline.py
├── test_pipeline.py
├── .github/workflows/
│   └── python-tests.yml        
GitHub Actions
├── requirements.txt            
Dépendances du projet
└── README.md                   Configuration du projet
```

---

## 📋 Mission Détaillée

### Phase 1️⃣ : Analyse et Exploration des Données (EDA)

#### Visualisations requises

**1. Heatmap de corrélation**
- Identifier les variables numériques les plus corrélées avec `DeliveryTime`
- Détection de corrélation avec la cible
- Détection de multicolinéarité entre variables

**2. Countplots**
- Analyser la distribution des variables catégorielles
- `Traffic_Level`
- `Vehicle_Type`
- `Time_of_Day`
- `Weather`

**3. Boxplots**
- Analyser la relation entre la cible et les variables catégorielles
- Trafic vs Temps de livraison
- Type de véhicule vs Temps de livraison
- Heure de la journée vs Temps de livraison
- Conditions météorologiques vs Temps de livraison

**4. Distribution de la variable cible**
- Analyser la distribution de `DeliveryTime`
- Histogramme et statistiques descriptives
- Détection d'outliers
- Vérification de la normalité

#### Objectif
Identifier les variables les plus explicatives avant la modélisation.

---

### Phase 2️⃣ : Prétraitement et Sélection de Features

#### Prétraitement

**Variables numériques** : StandardScaler (normalisation)
- `Distance_km`
- `Courier_Experience`
- `Preparation_Time`

**Variables catégorielles** : OneHotEncoder (encodage one-hot)
- `Traffic_Level`
- `Vehicle_Type`
- `Time_of_Day`
- `Weather`

#### Sélection de Features

- **Méthode** : SelectKBest avec test statistique `f_regression`
- **Objectif** : Sélectionner les k meilleures features pour réduire la dimensionnalité
- **Validation** : Comparer les performances avec/sans sélection

---

### Phase 3️⃣ : Modélisation avec GridSearchCV

### **Modèles à tester**

- **RandomForestRegressor**

- **Support Vector Regressor (SVR)**


### **Métriques d'évaluation**

- **MAE (Mean Absolute Error)**
    - Métrique de sélection principale
    - Interprétable en minutes
    - Robuste aux outliers

- **R² (Coefficient de détermination)**
    - Évaluation supplémentaire
    - Proportion de variance expliquée
    - Cible : R² > 0.85

#### Justification du modèle final

- Comparer MAE et R² entre les deux modèles
- Analyser le temps de prédiction
- Évaluer la stabilité et la robustesse

---

### Phase 4️⃣ : Pipeline sklearn (Bonus)

#### Objectifs

- Automatiser le flux de traitement
- Éviter les fuites de données
- Faciliter le déploiement en production

---

### Phase 5️⃣ : Tests Automatisés

#### Tests de validation des données

```python
# test_data.py
- Vérifier le format des données (types, dimensions)

# test_mae
- Vérifier que la MAE maximale ne dépasse pas un seuil défini
```

---

### Phase 6️⃣ : CI/CD avec GitHub Actions (Bonus)

#### Objectif

Exécuter automatiquement les tests à chaque push sur le repository.

#### Fichier de configuration

**`.github/workflows/python-tests.yml`**

```yaml
name: Run Unit Tests
on:
  push:
    branches: [main]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout du code
        uses: actions/checkout@v4
      - name: Installer Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Installer les dépendances
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install pytest joblib scikit-learn pandas
      - name: Exécuter les tests
        run: pytest -v
```

#### Avantages

- ✅ Garantit que le code reste fonctionnel
- ✅ Détecte les régressions automatiquement
- ✅ Valide les PR avant merge
- ✅ Génère des rapports de couverture de code

---

## 📦 Dépendances

```
pandas
numpy
scikit-learn
matplotlib
seaborn
pytest
```

Installez-les avec :

```bash
pip install -r requirements.txt
```

---

## 🚀 Guide de démarrage rapide

### 1. Cloner le repository

```bash
git clone https://github.com/SaidaAourras/Pr-diction-du-Temps-de-Livraison.git
cd Pr-diction-du-Temps-de-Livraison
```

### 2. Créer un environnement virtuel

```bash
python -m venv venv
source venv/bin/activate  # Sur Windows : venv\Scripts\activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Exécuter les tests

```bash
pytest -v
```

---

## 📈 Résultats attendus

| Métrique | Cible |
|----------|-------|
| MAE (test) | < 7 minutes |
| R² (test) | > 0.78 |

---

## 📝 Résumé des livrables

- ✅ Analyse exploratoire complète avec visualisations
- ✅ Prétraitement des données automatisé
- ✅ Sélection de features basée sur la statistique
- ✅ Modélisation avec GridSearchCV (2 modèles)
- ✅ Pipeline sklearn automatisé
- ✅ Suite de tests complète
- ✅ CI/CD avec GitHub Actions
- ✅ Documentation du projet

---

## 👥 Contributeurs

- Data Scientist Junior : Saida Aourras

---
**Dernière mise à jour** : Octobre 2025  
