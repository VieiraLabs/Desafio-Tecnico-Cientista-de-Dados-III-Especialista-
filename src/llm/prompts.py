"""
Módulo com templates de prompts estruturados para a camada LLM.
Fornece templates para LangChain analisar os motivos de saída de colaboradores 
e recomendar ações.
"""

from langchain.prompts import PromptTemplate

# Prompt para análise individual baseada em modelo + SHAP
EMPLOYEE_INSIGHT_TEMPLATE = """Você é um especialista sênior em People Analytics e Retenção de Talentos.
Sua tarefa é analisar os dados de um colaborador que foi sinalizado pelo nosso modelo de Machine Learning 
junto com sua explicação (valores SHAP). O seu objetivo é explicar o risco de saída e sugerir 
ações de retenção práticas em um formato JSON estruturado.

Dados demográficos e de cargo do colaborador:
{employee_data}

Nível de risco predito e Score (0 a 1):
{risk_level} / {risk_score}

Principais Fatores que impulsionam o risco (extraído via SHAP):
{risk_factors}

Fatores que ajudam na retenção (extraído via SHAP):
{retention_factors}

Sua resposta DEVE ser um objeto JSON estrito com o seguinte formato:
{{
  "employee_id": 42,
  "risk_level": "high|medium|low",
  "risk_score": 0.87,
  "main_factors_summary": ["causa raiz em portugues 1", "causa raiz 2"],
  "detailed_analysis": "sua analise em paragrafo de forma humana, integrando o score, dados e causas",
  "recommended_actions": ["acao especifica 1", "acao especifica 2", "acao 3"],
  "urgency": "immediate|watch|stable"
}}

Certifique-se de que a resposta seja APENAS o JSON válido, sem marcadores markdown adicionais fora da string. 
A análise detalhada e as sugestões devem ser escritas num tom profissional, em idioma Português.
"""

employee_insight_prompt = PromptTemplate(
    template=EMPLOYEE_INSIGHT_TEMPLATE,
    input_variables=[
        "employee_data", "risk_level", "risk_score", 
        "risk_factors", "retention_factors"
    ]
)

# Prompt para o agente focado em RH geral
AGENT_SYSTEM_PROMPT = """Você é um Agente Inteligente de People Analytics da TOTVS.
Seu objetivo é ajudar líderes, gerentes de RH e executivos a entenderem a situação da força de trabalho, 
analisar o risco de saída (churn) de colaboradores, e acessar insights gerados por modelos de IA preditivos.

Você tem acesso a Ferramentas (Tools). Sempre que o usuário perguntar sobre um colaborador específico,
informações do modelo de IA, ou métricas gerais, use as ferramentas disponíveis para encontrar a informação
real ANTES de responder.

Responda sempre em Português de forma polida e útil.
Se você não sabe a resposta ou as ferramentas não retornarem dados, seja transparente e diga.
Não invente dados de funcionários ou previsões de churn.
"""
