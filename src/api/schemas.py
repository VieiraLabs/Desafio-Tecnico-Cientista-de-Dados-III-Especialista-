"""
Definição de esquemas (Schemas) Pydantic para validação de dados de Input (Requests) 
e Output (Responses) da API FastAPI. Padrão de tipagem forte e serialização.
"""

from pydantic import BaseModel, ConfigDict
from typing import List, Optional, Any, Dict

class PredictionResponse(BaseModel):
    employee_id: int
    risk_score: float
    risk_level: str
    will_churn: bool

# Como os dados brutos possuem 35 colunas, permitimos um InputDict genérico opcional
class EmployeeData(BaseModel):
    employee_id: int
    features: Optional[Dict[str, Any]] = None

class LLMInsightResponse(BaseModel):
    employee_id: int
    risk_level: str
    risk_score: float
    main_factors_summary: List[str]
    detailed_analysis: str
    recommended_actions: List[str]
    urgency: str

class AgentChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = "default_user_1"

class AgentChatResponse(BaseModel):
    response: str

class DashboardSummaryResponse(BaseModel):
    total_employees: int
    total_churn: int
    churn_rate: float
    avg_salary: float
    departments: List[str]
