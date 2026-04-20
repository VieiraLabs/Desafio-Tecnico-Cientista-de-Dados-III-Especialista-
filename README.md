# Neural HR & Retention: TOTVS Data Science Challenge

Este projeto é uma solução fim-a-ponta de **People Analytics**, desenvolvida para o Desafio Técnico Cientista de Dados III (Especialista) da TOTVS. O sistema prevê o risco de rescisão (Churn) usando modelos em árvore, detalha o **motivo** com Valores Shapley e sugere ações de retenção via Agente ReAct Inteligente consumindo APIs Groq (LLaMA-3).

---

## 🌟 Principais Features Entregues

1. **Dashboard Premium Streamlit**: Interface *Dark Mode* ultra responsiva com painéis modulares e métricas analíticas da equipe/pessoas.
2. **Backend Scalável FastAPI**: API separada encapsulando inferência ML (XGBoost/RF), cálculos do SHAP Tree Explainer e Orquestrador IA *LangGraph*.
3. **Pilar Inteligente LLM / LangChain**: Prompters rígidos para respostas JSON e Chat Assistant nativo cruzando histórico de banco e features importances.
4. **Governança/MLOps Documentada**: Diretórios dedicados (`docs/`) endereçando a mentalidade Sr/Lead da liderança corporativa (Mentoria MLOps e Pipeline).

---

## 📂 Visão Rápida da Arquitetura
Para maiores detalhes leia o diagrama de sistema [exposto na documentação](docs/arquitetura.md).

```bash
/src
 ├── ml/             # Pré-processamento, Treino (XGBoost) e SHAP Explainer
 ├── llm/            # Integração Pydantic Parser + LangChain (Groq LLaMA-3)
 ├── api/            # FastAPI Schemas, Rotas Rest e Endpoint Manager
 └── agent/          # Orquestrador React Agent LangGraph com Tools Injetadas
/ui
 └── app.py          # Dashboard Premium Visual
```

---

## 🚀 Como Rodar o Projeto

Você tem duas vias de execução: via **Docker** (recomendado para revisar o projeto final entregue) ou de forma **Local** nativa.

### Premissa Obrigatória: Chave de API
Obtenha uma chave gratuita da [Groq Cloud (Console)](https://console.groq.com/keys). Renomeie o arquivo `.env.example` para `.env` e insira sua chave lá:

```env
GROQ_API_KEY=gsk_suachaveaqui
```

### Opção 1: Via Docker (Docker Compose) - ⭐ Mais Rápido

Na raiz do projeto rode o comando único:
```bash
docker-compose up --build
```
> O Docker efetuará os builds, instalará módulos Python e executará o script de ML. Ao subir, os serviços estão disponíveis em:
> - **Frontend Web (Dashboard):** `http://localhost:8501`
> - **API e Documentação Swagger:** `http://localhost:8000/api/docs`

### Opção 2: Local (Ambiente Python Nativo)

```bash
# 1. Crie e ative um ambiente virtual
python -m venv venv
.\venv\Scripts\Activate   # (Windows)
# source venv/bin/activate # (Unix)

# 2. Dependências
pip install -r requirements.txt

# 3. Baixe e Execute o Pipeline de ML Inicial (Gerar os Joblib Models)
python -m src.ml.train

# 4. Levante em Terminais/Bash's Adjacentes a API e o Streamlit
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
streamlit run ui/app.py
```

---

## 📖 Documentação de Negócio (Desafio Específico)

A avaliação solicitou textos estratégicos. Por favor consulte a leitura das sub-páginas contidas em `/docs/` para a avaliação da mentalidade e maturidade.

- [Liderança, Mentoria e Time](docs/lideranca_tecnica.md)
- [Julgamento de Custo e Decisões de ML](docs/decisoes_tecnicas.md)
- [Mermaid Pipeline Arquitetural](docs/arquitetura.md)

---
 *Construído e Arquitetado para escalonar impacto real (Developer e User Experience).* 
