# Commandes à exécuter

## 1. Créer l'environnement virtuel
```bash
python -m venv .venv
```

## 2. Activer l'environnement Windows
```bash
.venv\Scripts\activate
```

## 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

## 4. Lancer Jupyter
```bash
jupyter notebook
```

## 5. Lancer MLflow UI
```bash
mlflow ui
```

## 6. Lancer l'entraînement automatisé
```bash
cd src
python train_model.py
```

## 7. Tester le serving MLflow
```bash
mlflow models serve -m runs:/ID_DU_RUN/model -p 5001 --no-conda
```

Puis dans un autre terminal :
```bash
cd src
python serve_test.py
```
