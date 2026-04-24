"""
Ferramentas (Tools) do Agente LangChain.
Estas funções servem como "habilidades" para o LLM interagir com a API, 
com os dados em disco e com os modelos.
"""

import pandas as pd
from langchain.tools import tool
from typing import Dict, Any, List

from src.config import TARGET_COLUMN
from src.ml.predict import get_employee_by_id, predict_single_employee
from src.ml.explainability import explain_single_prediction_shap, get_feature_importance_ranking
from src.ml.preprocessing import load_data

@tool
def search_employee_tool(employee_id: int) -> str:
    """Busca o perfil completo e os dados biogáficos (rh) de um colaborador específico usando o seu ID numérico (employee_id). Útil para conhecer o histórico, cargo e departamento do funcionário."""
    try:
        data = get_employee_by_id(employee_id)
        # Limpar um pouco para não estourar contexto e ficar amigável
        res = []
        for k, v in data.items():
            if k not in ['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber']:
                res.append(f"{k}: {v}")
        return "\n".join(res)
    except Exception as e:
        return f"Não foi possível encontrar o funcionário com ID {employee_id}. Erro: {str(e)}"

@tool
def calculate_churn_risk_tool(employee_id: int) -> str:
    """Aciona o modelo de Machine Learning preditivo para calcular o Risco de Saída (Churn) de um funcionário específico (via ID). Retorna o Risco, o Nível e a Decisão (sim ou não)."""
    try:
        data = get_employee_by_id(employee_id)
        prediction = predict_single_employee(data)
        
        return (f"Resultado do Modelo ML para o ID {employee_id}:\n"
                f"- Probabilidade de saída (Risk Score): {prediction['risk_score']*100:.2f}%\n"
                f"- Nível de Risco: {prediction['risk_level']}\n"
                f"- Vai sair? {'Sim' if prediction['will_churn'] else 'Não'}")
    except Exception as e:
        return f"Erro ao calcular predição pelo ID {employee_id}. Erro: {str(e)}"

@tool
def explain_churn_factors_tool(employee_id: int) -> str:
    """Extrai os Fatores-Chave do modelo de predição em cima dos dados de um funcionário (através do seu ID) usando o SHapley values (SHAP). Útil para descobrir o 'Por quê' o funcionário vai ou não embora."""
    try:
        data = get_employee_by_id(employee_id)
        shap_res = explain_single_prediction_shap(data, top_factors=3)
        
        if "error" in shap_res:
             return f"Não foi possível processar explicabilidade. Motivo: {shap_res['error']}"
        
        msg = f"Explicação do modelo preditivo para ID {employee_id}:\n\n"
        
        msg += "Fatores que EMPURRAM o usuário para fora (aumentam risco de churn):\n"
        for f in shap_res["top_risk_factors"]:
            msg += f"- {f['feature']} (peso {f['shap_value']:.4f})\n"
            
        msg += "\nFatores que MANTÉM o usuário na empresa (ancoram retenção):\n"
        for f in shap_res["top_retention_factors"]:
            msg += f"- {f['feature']} (peso {f['shap_value']:.4f})\n"
            
        return msg
    except Exception as e:
        return f"Erro obtendo fatores do SHAP. {str(e)}"

@tool
def get_department_stats_tool(department: str) -> str:
    """Busca estatísticas e indicadores globais de Attrition/Churn para o departamento especificado. Exemplos de departamento: 'Sales', 'Research & Development', 'Human Resources'."""
    try:
        df = load_data()
        
        if department not in df['Department'].unique():
            return f"Departamento '{department}' inexistente. Tente um destes: {list(df['Department'].unique())}"
            
        df_dept = df[df['Department'] == department]
        total = len(df_dept)
        
        churned = df_dept[df_dept[TARGET_COLUMN] == 'Yes']
        churn_rate = len(churned) / total * 100
        
        sal_avg = df_dept['MonthlyIncome'].mean()
        sal_churn = churned['MonthlyIncome'].mean() if len(churned) > 0 else 0
        stayed = df_dept[df_dept[TARGET_COLUMN] == 'No']
        sal_stayed = stayed['MonthlyIncome'].mean() if len(stayed) > 0 else 0
        
        return (f"Resumo do departamento '{department}':\n"
                f"Headcount: {total} colaboradores.\n"
                f"Taxa de Churn Atual: {churn_rate:.1f}%\n"
                f"Média Salarial Global do Dep: ${sal_avg:.2f}\n"
                f"Média Salarial dos que saem: ${sal_churn:.2f}\n"
                f"Média Salarial dos que ficam: ${sal_stayed:.2f}")
    except Exception as e:
        return str(e)

@tool
def get_demographic_churn_stats_tool(query_type: str = "general") -> str:
    """Busca estatísticas demográficas sobre evasão (churn). Útil para responder perguntas sobre idade média de evasão, sexo com maior saída, e outros dados demográficos de quem saiu da empresa."""
    try:
        df = load_data()
        churned = df[df[TARGET_COLUMN] == 'Yes']
        
        # Sexo com maior saída
        if 'Gender' in churned.columns:
            gender_counts = churned['Gender'].value_counts()
            top_gender = gender_counts.idxmax()
            top_gender_count = gender_counts.max()
            # Traduzir para pt-br caso seja Male/Female
            top_gender_pt = "Masculino" if top_gender == "Male" else "Feminino" if top_gender == "Female" else top_gender
            gender_info = f"O sexo com maior número de saídas é {top_gender_pt} ({top_gender_count} saídas)."
        else:
            gender_info = "Informação de sexo não disponível no dataset."
            
        # Idade média de evasão
        if 'Age' in churned.columns:
            avg_age_churn = churned['Age'].mean()
            age_info = f"A idade média de evasão (pessoas que saíram) é de {avg_age_churn:.1f} anos."
        else:
            age_info = "Informação de idade não disponível no dataset."
            
        # Estado civil com maior evasão
        if 'MaritalStatus' in churned.columns:
            marital_counts = churned['MaritalStatus'].value_counts()
            top_marital = marital_counts.idxmax()
            top_marital_count = marital_counts.max()
            
            # Traduzir MaritalStatus
            marital_map = {"Single": "Solteiro(a)", "Married": "Casado(a)", "Divorced": "Divorciado(a)"}
            top_marital_pt = marital_map.get(top_marital, top_marital)
            
            # Pegar estatísticas extras para explicar o 'por quê' (menor idade, menor renda, etc)
            df_marital = df[df['MaritalStatus'] == top_marital]
            avg_age_marital = df_marital['Age'].mean() if 'Age' in df_marital else 0
            avg_income_marital = df_marital['MonthlyIncome'].mean() if 'MonthlyIncome' in df_marital else 0
            
            marital_info = (f"O estado civil com maior evasão é {top_marital_pt} ({top_marital_count} saídas). "
                            f"Motivos prováveis (contexto): este grupo costuma ser mais jovem (idade média {avg_age_marital:.1f} anos) "
                            f"e possuir menor vínculo, além de uma renda média de ${avg_income_marital:.2f}, facilitando a troca de empregos.")
        else:
            marital_info = "Informação de estado civil não disponível no dataset."
            
        return f"Estatísticas Demográficas de Churn:\n- {gender_info}\n- {age_info}\n- {marital_info}"
    except Exception as e:
        return f"Erro ao obter estatísticas demográficas: {str(e)}"

@tool
def get_global_churn_reasons_tool(query_type: str = "general") -> str:
    """Retorna os maiores motivos/fatores globais de saída (churn) na empresa, com base no modelo de Machine Learning treinado (Feature Importance). Útil para responder 'quais os principais motivos de saída' ou 'o que mais impacta a evasão'."""
    try:
        top_factors = get_feature_importance_ranking(top_n=5)
        res = "Os 5 maiores motivos (variáveis) que impactam a saída de funcionários na empresa são:\n"
        for i, f in enumerate(top_factors):
            res += f"{i+1}º - {f['feature']} (Relevância/Peso: {f['importance']})\n"
        return res
    except Exception as e:
        return f"Erro ao consultar os maiores motivos de saída: {str(e)}"

# Lista consolidada de ferramentas para a engine principal
HR_TOOLS = [
    search_employee_tool,
    calculate_churn_risk_tool,
    explain_churn_factors_tool,
    get_department_stats_tool,
    get_demographic_churn_stats_tool,
    get_global_churn_reasons_tool
]
