"""
Módulo para realizar invocações do modelo LLM (Groq) usando Langchain e 
gerar os JSONs formatados de insights com base nas saídas do ML (Predições + SHAP).
"""

import os
import json
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List

from src.config import GROQ_API_KEY, LLM_MODEL, LLM_TEMPERATURE
from src.llm.prompts import employee_insight_prompt

# Pydantic schema para parsear garantidamente
class InsightsSchema(BaseModel):
    employee_id: int = Field(description="O número/identificador do funcionário.")
    risk_level: str = Field(description="Nível de risco (high, medium, low).")
    risk_score: float = Field(description="O risco de churn medido de 0 a 1.")
    main_factors_summary: List[str] = Field(description="Resumo sucinto dos principais motivos de alerta (max 3 itens).")
    detailed_analysis: str = Field(description="Parágrafo que detalha a causa da fuga de talentos, cruzando com perfil do usuário.")
    recommended_actions: List[str] = Field(description="Iniciativas gerenciais plausíveis recomendadas para a retenção do funcionário.")
    urgency: str = Field(description="Nível de atenção para a RH: immediate, watch, stable.")

def get_llm():
    """Inicia e retorna o cliente do Groq Llama 3"""
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY não definida no ambiente.")
        
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=LLM_MODEL,
        temperature=LLM_TEMPERATURE
    )

def generate_employee_insights(
    employee_id: int, 
    employee_data: dict, 
    prediction_data: dict, 
    shap_data: dict
) -> dict:
    """
    Chama o LLM utilizando todas as informações (Brutas + Score + Explicações do SHAP),
    e exige json de resposta.
    """
    llm = get_llm()
    parser = JsonOutputParser(pydantic_object=InsightsSchema)
    
    # Adicionar instruções sintáticas do JSON Parser à template
    prompt_with_instructions = employee_insight_prompt.partial(
        format_instructions=parser.get_format_instructions()
    )
    
    # Criar a chain
    chain = prompt_with_instructions | llm | parser
    
    # Invocar a chain de pensamento
    try:
        response = chain.invoke({
            "employee_data": json.dumps({k: v for k, v in employee_data.items() if k not in ["EmployeeCount", "Over18", "StandardHours"]}), # Remover sujo
            "risk_level": prediction_data.get("risk_level", "unknown"),
            "risk_score": prediction_data.get("risk_score", 0.0),
            "risk_factors": json.dumps(shap_data.get("top_risk_factors", [])),
            "retention_factors": json.dumps(shap_data.get("top_retention_factors", []))
        })
        
        # Garante a chave ID de volta no response
        response["employee_id"] = employee_id
        return response
    
    except Exception as e:
        print(f"Erro na chain de LLM do insight individual: {e}")
        # Retornar formato base para não quebrar a UI em caso de erro na rede do LLM
        return {
            "employee_id": employee_id,
            "risk_level": prediction_data.get("risk_level", "unknown"),
            "risk_score": prediction_data.get("risk_score", 0.0),
            "main_factors_summary": ["Não foi possível gerar análise via IA para este colaborador."],
            "detailed_analysis": str(e),
            "recommended_actions": ["Recarregar ou analisar fatores de forma manual"],
            "urgency": "watch"
        }
