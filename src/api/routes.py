"""
Roteamento (Endpoints) para o servidor Web FastAPI.
Isola a lógica das requisições HTTP das regras de negócio.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any
import pandas as pd

from src.api.schemas import (
    PredictionResponse, EmployeeData, LLMInsightResponse, 
    AgentChatRequest, AgentChatResponse, DashboardSummaryResponse
)
from src.ml.predict import predict_single_employee, get_employee_by_id
from src.ml.explainability import explain_single_prediction_shap
from src.ml.preprocessing import load_data
from src.llm.insights import generate_employee_insights
from src.agent.orchestrator import run_agent_query
from src.config import TARGET_COLUMN

api_router = APIRouter()

@api_router.get("/health", summary="Checa status da Aplicação")
def health_check():
    return {"status": "ok", "message": "People Analytics Engine e LLM Operational."}

@api_router.get("/dashboard/summary", response_model=DashboardSummaryResponse, summary="Estatísticas macros do board")
def get_dashboard_summary():
    try:
        df = load_data()
        churn_count = len(df[df[TARGET_COLUMN] == 'Yes'])
        total = len(df)
        
        return {
            "total_employees": total,
            "total_churn": churn_count,
            "churn_rate": (churn_count / total * 100) if total > 0 else 0,
            "avg_salary": df["MonthlyIncome"].mean(),
            "departments": df["Department"].unique().tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/employees", summary="Lista dados de funcionários formatados")
def get_employees_list(limit: int = 50, skip: int = 0):
    try:
        df = load_data()
        
        # Limpar um pouco de colunas inuteis
        drop_cols = ["EmployeeCount", "StandardHours", "Over18"]
        df_clean = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
        
        chunk = df_clean.iloc[skip:skip+limit]
        return {"data": chunk.to_dict(orient='records'), "total": len(df)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/predict/{employee_id}", response_model=PredictionResponse, summary="Preditor Unitário ML")
def predict_employee_churn(employee_id: int):
    try:
        data = get_employee_by_id(employee_id)
        prediction = predict_single_employee(data)
        return {
            "employee_id": employee_id,
            "risk_score": prediction["risk_score"],
            "risk_level": prediction["risk_level"],
            "will_churn": prediction["will_churn"]
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@api_router.post("/insights/{employee_id}", response_model=LLMInsightResponse, summary="Gera o Insight Analítico com Groq LLM e SHAP")
def get_llm_churn_insights(employee_id: int):
    try:
        # Puxamos TUDO (E.T.L para RAG Dinâmico)
        data = get_employee_by_id(employee_id)
        prediction = predict_single_employee(data)
        shap_ex = explain_single_prediction_shap(data, top_factors=3)
        
        # Juntamos no Prompter Langchain + LLM Base 
        result_json = generate_employee_insights(
            employee_id=employee_id,
            employee_data=data,
            prediction_data=prediction,
            shap_data=shap_ex
        )
        return result_json
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/agent/chat", response_model=AgentChatResponse, summary="Interface interacional com HR Assistant")
def chat_with_hr_agent(req: AgentChatRequest):
    try:
        ai_response = run_agent_query(user_query=req.query, thread_id=req.session_id)
        return {"response": ai_response}
    except Exception as e:
         raise HTTPException(status_code=500, detail=str(e))
