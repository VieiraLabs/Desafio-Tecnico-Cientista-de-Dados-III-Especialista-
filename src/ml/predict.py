"""
Módulo para realizar inferências de forma simplificada, 
expondo funções utilitárias que a API ou Agente podem chamar.
"""

import pandas as pd
import joblib
from functools import lru_cache
from typing import Dict, Any, Tuple

from src.config import PREPROCESSOR_PATH, MODEL_PATH, TARGET_COLUMN
from src.ml.preprocessing import feature_engineering, load_data

@lru_cache(maxsize=1)
def load_artifacts() -> Tuple[Any, Any, list]:
    """
    Carrega o preprocessador e o modelo da memória de forma cacheada para não onerar inferências múltiplas.
    """
    try:
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        model_artifact = joblib.load(MODEL_PATH)
        model = model_artifact["model"]
        feature_names = model_artifact.get("feature_names", [])
        return preprocessor, model, feature_names
    except FileNotFoundError:
        raise Exception("Artefatos de modelo não encontrados. Rodar 'python -m src.ml.train' primeiro.")

def predict_single_employee(employee_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dada as features de um colaborador em formato dict, retorna o risco, score e nível.
    """
    df_raw = pd.DataFrame([employee_data])
    
    # Adicionar colunas de engineering
    df_engineered = feature_engineering(df_raw)
    
    preprocessor, model, _ = load_artifacts()
    
    # Processar os dados
    X_processed = preprocessor.transform(df_engineered)
    
    # Inferência
    score = model.predict_proba(X_processed)[0][1]
    prediction = model.predict(X_processed)[0]
    
    level = "high" if score > 0.6 else "medium" if score > 0.3 else "low"
    
    return {
        "risk_score": round(float(score), 4),
        "will_churn": bool(prediction),
        "risk_level": level
    }

def get_employee_by_id(employee_number: int) -> Dict[str, Any]:
    """
    Busca os dados de um colaborador no dataset (simulando uma query em banco).
    """
    df = load_data()
    # No CSV original existe a coluna EmployeeNumber
    employee_row = df[df['EmployeeNumber'] == employee_number]
    
    if employee_row.empty:
        raise ValueError(f"Colaborador com ID {employee_number} não encontrado.")
        
    return employee_row.drop(columns=[TARGET_COLUMN], errors='ignore').iloc[0].to_dict()
