# Decisões Técnicas Justificadas

As decisões de engenharia levadas para resolver este case técnico se concentraram em equilibrar robustez acadêmica com agilidade e "Developer Experience". 

Abaixo todas as apostas justificadas:

### 1) A Célula Preditora 
- **Linguagem / Pacotes:** Python (Pip / scikit / pandas / numpy). Ecosystem de prateleira (Padrão ouro mundial).
- **O Dataset: IBM HR:** Aconselhado pois ele cobre perfeitamente o mix numérico / categórico com viés relacional do Mundo Corporativo com o peso e distribuição correta (Altamente Imbued "attrição vs permanentes").
- **Balanceio e Técnicas:** Implementado SMOTE na etapa de reamostragem. Por os demitidos representarem minoria estatística, balancear os falsos negativos salva a empresa dos riscos silenciosos de churn preditivo.
- **Modelos Base:** Optamos por comparar *XGBoost*, *Random Forest* e *Logistic Regression*. Foi treinado XGB vs RF na prática. Random Forest como base sólida para interpretação em profundidade menor e XGBoost pela melhor aderabilidade multi-colunas. 

### 2) Camada Analítica/Invocacional de Backend
- **FastAPI / Uvicorn:** Permite abstrair a orquestração via ASGI (concorrente). Permite forte validação formal através dos models `Pydantic`. Garante que requisições (como IDs nulos ou features quebradas) emitam um Error Padrão REST 422 em vez de quebrar CÓDIGO DO MODELO!

### 3) LLM com Agentes e RAG Inerente
- **LangChain:** Empreguei as classes centrais do ecossistema. 
- **Groq LLaMA-3:** Em vez da clássica OpenAI de custo e lentidão restritivas, utilizei a *LPUs Groq hardware aceleratrixes*. Oferece latência de preenchimento (Tokens/s) na escala de centenas (Perfeito pra responder no ato as abstrações de RH da TOTVS). E um modelo state of truth, O LLaMA-3 aberto.
- **ReAct e Ferramentas (LangGraph/Agents):** Para cobrir o item arquitetural central ("Flow Agent do Desafio"), montei funções python cruas envoltas com decoradores `@tool`. O Agente usa um loop Reacting para: Pensar na dúvida -> Pesquisar ID correspondente (Tools)-> Gerar SHAP explicador -> Deduzir conclusão.

### 4) Explicabilidade Intuitiva 
- **Por que SHAP?:** Como exigido "Interpretabilidade alem de performance", integramos SHAP. Valores SHapley de Teoria dos Jogos fornecem atribuições exatas locais. Extraímos as frações mais determinantes em código python cru e forçamos o LLM a "digerir" e ler esse peso. Ele não inventa "baixo salário", ele visualiza na inferição que a feature `MonthlyRate` tem vetor shapley massivo negativo.

### 5) Apresentação Web em Streamlit "Custom CSS"
- Para fechar MVP em agilidade ignoramos Vite/React e usamos **Streamlit Premium**. O framework foi submetido a injeções nativas de CSS e componentes Altair/Plotly para forçar aparência Web "Profissional Darkglass" muito mais estetica que as tradicionais bibliotecas web brancas padronizadas de dados. Exclusivas Abas modularizam o painel de RH e o Agentic Chat de forma desacoplada!
