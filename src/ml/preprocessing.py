"""
Módulo de pré-processamento de dados para o modelo de People Analytics.
Contém funções para carregar, limpar, feature engineering e preparar dados para treino.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

from src.config import (
    DATASET_PATH, TARGET_COLUMN, COLS_TO_DROP,
    CATEGORICAL_COLS, NUMERICAL_COLS, ORDINAL_COLS,
    PREPROCESSOR_PATH
)

def load_data() -> pd.DataFrame:
    """Carrega os dados brutos e faz log de propriedades."""
    df = pd.read_csv(DATASET_PATH)
    print(f"Dataframe carregado com shape: {df.shape}")
    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria novas features com base no conhecimento do domínio de RH.
    """
    df = df.copy()
    
    # Razão de estagnação: tempo sem promoção sobre tempo total
    df['YearsSincePromotionRatio'] = np.where(
        df['TotalWorkingYears'] == 0, 0, 
        df['YearsSinceLastPromotion'] / df['TotalWorkingYears']
    )
    
    # Estabilidade no cargo atual: tempo no cargo atual sobre tempo na empresa
    df['RoleStabilityRatio'] = np.where(
        df['YearsAtCompany'] == 0, 0,
        df['YearsInCurrentRole'] / df['YearsAtCompany']
    )
    
    # Fator de esgotamento: horas extras e nível de envolvimento
    # Tratando temporariamente OverTime como binario para feature engineering manual
    overtime_num = df['OverTime'].map({'Yes': 1, 'No': 0}).fillna(0)
    df['BurnoutRiskScore'] = overtime_num * (5 - df['JobInvolvement']) # Quanto menor involvement com OT, pior
    
    # Adicionar derivadas nas colunas numéricas
    new_num_cols = ['YearsSincePromotionRatio', 'RoleStabilityRatio', 'BurnoutRiskScore']
    for col in new_num_cols:
        if col not in NUMERICAL_COLS:
            NUMERICAL_COLS.append(col)
            
    return df

def get_preprocessor():
    """
    Constrói e retorna o pipeline do scikit-learn (ColumnTransformer)
    """
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Para ordinais, poderíamos usar OrdinalEncoder puro, mas vamos tratar como numérico escalar
    # mantendo os valores de 1 a 5 ou aplicar un imputer.
    ordinal_transformer = Pipeline(steps=[
         ('imputer', SimpleImputer(strategy='most_frequent')),
         ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERICAL_COLS),
            ('cat', categorical_transformer, CATEGORICAL_COLS),
            ('ord', ordinal_transformer, ORDINAL_COLS)
        ])
    
    return preprocessor

def preprocess_data(save_preprocessor: bool = True):
    """
    Executa o pipeline ponta a ponta: carrega, feature engineering, split e preprocessamento.
    Retorna X_train, X_test, y_train, y_test pré-processados e os nomes das features.
    """
    df = load_data()
    
    # Limpeza básica
    cols_to_drop = [c for c in COLS_TO_DROP if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    
    # Target
    y = df[TARGET_COLUMN].map({'Yes': 1, 'No': 0}).values
    X = df.drop(columns=[TARGET_COLUMN])
    
    # Modificação customizada nas features
    X = feature_engineering(X)
    
    # Divisão (Stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    preprocessor = get_preprocessor()
    
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Obter os nomes das features para uso posterior em explicabilidade
    num_features = NUMERICAL_COLS
    cat_features = preprocessor.transformers_[1][1].named_steps['onehot'].get_feature_names_out(CATEGORICAL_COLS).tolist()
    ord_features = ORDINAL_COLS
    feature_names = num_features + cat_features + ord_features
    
    if save_preprocessor:
        joblib.dump(preprocessor, PREPROCESSOR_PATH)
        print(f"Preprocessador salvo em {PREPROCESSOR_PATH}")
        
    return X_train_processed, X_test_processed, y_train, y_test, feature_names, X_train

if __name__ == "__main__":
    X_train_p, X_test_p, y_train, y_test, fnames, orig_xtrain = preprocess_data(save_preprocessor=False)
    print(f"Features extraídas ({len(fnames)}): {fnames[:5]}...")
    print(f"Shape de treino: {X_train_p.shape}, Teste: {X_test_p.shape}")
    print(f"Distribuição do Target treino: 0={np.sum(y_train==0)}, 1={np.sum(y_train==1)}")
