# RH Neural: Motor de Retenção de Talentos (TOTVS Data Science Challenge)

Este projeto é uma solução de **People Analytics** *end-to-end*, desenvolvida para atender com 100% de aderência (incluindo todos os diferenciais extras recomendados) ao **Desafio Técnico Cientista de Dados III (Especialista)** da TOTVS. 

O sistema prevê o risco de rescisão (Churn) de colaboradores usando modelos em árvore (XGBoost/RandomForest) baseados no *IBM HR Analytics*, detalha o **motivo** exato das saídas usando *SHAP Values* e sugere ações de retenção ativas. Tudo isso operado por um **Agente ReAct Inteligente (LangChain + LLaMA-3)** capaz de buscar estatísticas demográficas ativamente, cruzar dados de departamentos e justificar os resultados preditivos na interface.

---

## 🌟 Atendimento Integral dos Requisitos e Diferenciais

1. **Machine Learning e Explicabilidade:** Treinamento estruturado, pipeline modular e justificativas claras de evasão utilizando `Feature Importance` e `SHAP`.
2. **Orquestração de LLMs e Agente Inteligente:** Um verdadeiro *Agentic Workflow* construído com LangChain. O chatbot do sistema possui "Tools" desenvolvidas sob medida para consultar salários de departamentos, demografia (idade, gênero e estado civil de churn) e rodar inferências ML em tempo real sob demanda.
3. **Dashboard Premium Streamlit (Ponto Extra):** Uma interface visual de alto padrão (Dark Mode em Azul Escuro e Ciano), totalmente focada em UX, unificando a experiência de consumo da predição.
4. **Backend Desacoplado (Produção):** O projeto não vive em notebooks. Roda em uma arquitetura de microsserviços via `FastAPI`, documentada em `Swagger` e com Deploy versionado.
5. **Visão de Liderança (Pilar Sênior):** Diretório estruturado `/docs` detalhando a arquitetura em Mermaid, as justificativas das escolhas de negócio, formação do Squad (Especialista, PM, DS Jr), visão de Mentoria (Code Review/Pair Programming) e CI/CD.

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

 *Criado pelo Cientista de Dados - David Vieira | [www.vieiralabs.ia.br](http://www.vieiralabs.ia.br)*
