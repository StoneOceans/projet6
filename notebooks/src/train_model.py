import pandas as pd
import mlflow
import mlflow.sklearn
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, f1_score
from business_score import find_best_threshold

DATA_PATH = '../data/processed/dataset_model.csv'


def main():
    df = pd.read_csv(DATA_PATH)
    y = df['TARGET']
    X = df.drop(columns=['TARGET', 'SK_ID_CURR'], errors='ignore')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    mlflow.set_experiment('pret_a_depenser_scoring')

    with mlflow.start_run(run_name='LightGBM_script_training'):
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]
        threshold, cost = find_best_threshold(y_test, y_proba)
        y_pred = (y_proba >= threshold).astype(int)

        mlflow.log_param('model', 'LightGBM')
        mlflow.log_param('class_weight', 'balanced')
        mlflow.log_param('best_threshold', threshold)
        mlflow.log_metric('auc', roc_auc_score(y_test, y_proba))
        mlflow.log_metric('accuracy', accuracy_score(y_test, y_pred))
        mlflow.log_metric('recall', recall_score(y_test, y_pred))
        mlflow.log_metric('f1_score', f1_score(y_test, y_pred))
        mlflow.log_metric('business_cost', cost)
        mlflow.sklearn.log_model(model, 'model', registered_model_name='credit_scoring_lightgbm')

        print('Modèle entraîné et enregistré dans MLflow.')
        print('Seuil optimal :', threshold)
        print('Coût métier :', cost)


if __name__ == '__main__':
    main()
