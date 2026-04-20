"""
Arquivo Main FastAPI. Ponto de inicialização do Servidor Uvicorn e Injeção de Rotas.
Inclui CORS e padronização.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.config import API_PORT, API_HOST
from src.api.routes import api_router

def create_app() -> FastAPI:
    """Cria a instância do Framework e liga config e rotas"""
    app = FastAPI(
        title="TOTVS HR Analytics & Predictor API",
        description="API que agrega Inferência ML via XGBoost/RF, Explicabilidade com SHAP e Agentic LLM reasoning.",
        version="1.0.0",
        docs_url="/api/docs"
    )

    # Lidando com cross origins web
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router, prefix="/api/v1")
    return app

app = create_app()

if __name__ == "__main__":
    print(f"Subindo servidor na porta {API_PORT}...")
    uvicorn.run("src.api.main:app", host=API_HOST, port=API_PORT, reload=True)
