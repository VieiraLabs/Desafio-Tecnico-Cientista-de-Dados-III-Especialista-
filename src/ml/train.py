"""
Módulo para treinamento de modelos de classificação e persistência do melhor modelo.
Compara Regressão Logística, Random Forest e XGBoost.
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE
import xgboost as xgb

from src.ml.preprocessing import preprocess_data
from src.config import MODEL_PATH, TARGET_COLUMN

def train_and_evaluate():
    """
    Treina diferentes modelos e escolhe o de melhor F1 / AUC, lidando com desbalanceamento.
    """
    print("Iniciando preprocessamento de dados...")
    X_train, X_test, y_train, y_test, feature_names, _ = preprocess_data(save_preprocessor=True)
    
    print("Balanceando os dados com SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    
    print("\nTreinando Modelos...")
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight='balanced'),
        "XGBoost": xgb.XGBClassifier(
            use_label_encoder=False, eval_metric='logloss', random_state=42, 
            n_estimators=100, max_depth=5, learning_rate=0.1
        )
    }
    
    best_model_name = None
    best_model = None
    best_auc = 0
    metrics_report = {}

    for name, model in models.items():
        # XGBoost pode ser sensível ao SMOTE e já lida bem com desbalanceamento,
        # Mas vamos treinar com dados balanceados ou com parâmetro de peso.
        # Aqui treinaremos com SMOTE para todos.
        model.fit(X_train_sm, y_train_sm)
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        auc = roc_auc_score(y_test, y_prob)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"\n[{name}] Performance:")
        print(f"ROC-AUC: {auc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
        
        metrics_report[name] = {"auc": auc, "f1": f1}
        
        if auc > best_auc:
            best_auc = auc
            best_model = model
            best_model_name = name

    print(f"\n=> Melhor Modelo Selecionado: {best_model_name} (AUC={best_auc:.4f})")
    print("Salvando artefato...")
    
    # Salvar juntamente os metadados (features)
    model_artifact = {
        "model": best_model,
        "feature_names": feature_names,
        "model_name": best_model_name
    }
    joblib.dump(model_artifact, MODEL_PATH)
    print(f"Artefato salvo em {MODEL_PATH}")

if __name__ == "__main__":
    train_and_evaluate()
