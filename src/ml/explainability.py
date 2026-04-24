"""
Módulo dedicado à explicabilidade do modelo usando construções baseadas em árvores 
como Feature Importance direta e SHAP (SHapley Additive exPlanations).
"""

import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any

from src.ml.predict import load_artifacts
from src.ml.preprocessing import feature_engineering

def get_feature_importance_ranking(top_n: int = 10) -> List[Dict[str, Any]]:
    """
    Retorna as top_n features globais do modelo baseado no atributo feature_importances_
    do XGBoost / Random Forest salvos.
    """
    _, model, feature_names = load_artifacts()
    
    if not hasattr(model, 'feature_importances_'):
        raise ValueError("O modelo salvo não suporta feature_importances_ diretamente.")
    
    importances = model.feature_importances_
    
    # Criar dict ordenado
    feat_imp = list(zip(feature_names, importances))
    feat_imp.sort(key=lambda x: x[1], reverse=True)
    
    return [{"feature": fname, "importance": round(float(imp), 4)} for fname, imp in feat_imp[:top_n]]

def explain_single_prediction_shap(employee_data_dict: Dict[str, Any], top_factors: int = 4) -> Dict[str, Any]:
    """
    Constrói as importâncias SHAP locais para uma única instância e retorna os maiores vetores de força
    (fatores que empurraram a predição para 'churn').
    """
    df_raw = pd.DataFrame([employee_data_dict])
    df_engineered = feature_engineering(df_raw)
    
    preprocessor, model, feature_names = load_artifacts()
    X_processed = preprocessor.transform(df_engineered)
    
    # Configurar explainer do SHAP
    # Nota: O explainer ideal em produção para XGB/RF é TreeExplainer. E para Reg. Logística é LinearExplainer.
    # Assumindo que o melhor modelo foi um modelo baseado em árvore:
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_processed)
        
        # Algumas implementações retornam matrizes diferentes. 
        # Para XGBoost, shape é (n_samples, n_features). Para Random Forest, pode ser lista [(n, f), (n, f)]
        if isinstance(shap_values, list):
            # Classe 1 é Churn (Sim)
            sv = shap_values[1][0] 
        elif len(shap_values.shape) == 3:
            # Shape (n_samples, n_features, n_classes) - extrair classe 1
            sv = shap_values[0, :, 1]
        else:
            sv = shap_values[0]
            
        # Parecer os SHAP values com os nomes das features
        sv_dict = dict(zip(feature_names, sv))
        
        # Obter top fatores POSITIVOS (que puxam o risco de churn prar cima)
        risk_drivers = sorted([(k, v) for k, v in sv_dict.items() if v > 0], key=lambda x: x[1], reverse=True)
        # Obter fatores NEGATIVOS (que ancoram/retem)
        retention_factors = sorted([(k, v) for k, v in sv_dict.items() if v < 0], key=lambda x: x[1])
        
        return {
            "top_risk_factors": [{"feature": k, "shap_value": float(v)} for k, v in risk_drivers[:top_factors]],
            "top_retention_factors": [{"feature": k, "shap_value": float(v)} for k, v in retention_factors[:top_factors]]
        }
        
    except Exception as e:
        print(f"Erro no SHAP explicador: {e}")
        # Call back seguro caso o modelo não seja suportado (ex: LR sem kernel/linear explainer pronto)
        return {
            "top_risk_factors": [],
            "top_retention_factors": [],
            "error": str(e)
        }
