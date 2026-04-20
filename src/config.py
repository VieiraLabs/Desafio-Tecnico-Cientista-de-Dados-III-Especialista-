"""
Configurações centralizadas do projeto People Analytics.
Carrega variáveis de ambiente e define caminhos e parâmetros globais.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Carrega variáveis do arquivo .env
load_dotenv()

# === Diretórios do Projeto ===
# Diretório raiz do projeto (dois níveis acima deste arquivo)
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"

# === Dataset ===
# Nome do arquivo CSV original do dataset IBM HR
DATASET_FILENAME = "WA_Fn-UseC_-HR-Employee-Attrition.csv"
DATASET_PATH = DATA_RAW_DIR / DATASET_FILENAME

# === Variável Alvo ===
# Coluna binária que indica se o colaborador saiu (Yes) ou não (No)
TARGET_COLUMN = "Attrition"

# === LLM / Groq ===
# Chave de API do Groq para inferência de modelos de linguagem
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
# Modelo de linguagem utilizado (Llama 3.3 70B é recomendado)
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
# Temperatura para geração de texto (0 = determinístico, 1 = criativo)
LLM_TEMPERATURE = 0.1

# === API ===
# Configurações do servidor FastAPI
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# === Modelo ML ===
# Nome do arquivo do modelo final salvo
MODEL_FILENAME = "best_model.joblib"
MODEL_PATH = MODELS_DIR / MODEL_FILENAME
# Nome do arquivo do preprocessador (scaler, encoder, etc.)
PREPROCESSOR_FILENAME = "preprocessor.joblib"
PREPROCESSOR_PATH = MODELS_DIR / PREPROCESSOR_FILENAME

# === Colunas ===
# Colunas a serem removidas por não conterem informação útil
COLS_TO_DROP = [
    "EmployeeNumber",   # Identificador único, sem valor preditivo
    "EmployeeCount",    # Constante (sempre 1)
    "StandardHours",    # Constante (sempre 80)
    "Over18",           # Constante (sempre 'Y')
]

# Colunas categóricas que precisam de encoding
CATEGORICAL_COLS = [
    "BusinessTravel",
    "Department",
    "EducationField",
    "Gender",
    "JobRole",
    "MaritalStatus",
    "OverTime",
]

# Colunas numéricas para scaling
NUMERICAL_COLS = [
    "Age", "DailyRate", "DistanceFromHome", "HourlyRate",
    "MonthlyIncome", "MonthlyRate", "NumCompaniesWorked",
    "PercentSalaryHike", "TotalWorkingYears", "TrainingTimesLastYear",
    "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion",
    "YearsWithCurrManager",
]

# Colunas ordinais (escala tipo Likert, 1-4 ou 1-5)
ORDINAL_COLS = [
    "Education",            # 1-5
    "EnvironmentSatisfaction",  # 1-4
    "JobInvolvement",       # 1-4
    "JobLevel",             # 1-5
    "JobSatisfaction",      # 1-4
    "PerformanceRating",    # 1-4
    "RelationshipSatisfaction", # 1-4
    "StockOptionLevel",     # 0-3
    "WorkLifeBalance",      # 1-4
]
