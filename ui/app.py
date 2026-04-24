"""
Aplicação Streamlit Premium para o People Analytics Dashboard
Apresenta um design Dark Mode moderno, CSS customizado, e 4 abas para 
demografia, preditor/shapley, Insights via agente LLM e métricas gerais.
"""

import streamlit as st
import pandas as pd
import requests
import json
import os
import sys
import base64
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any

# Adiciona a raiz do projeto no PYTHONPATH dinamicamente
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import API_PORT, API_HOST

# Endereço base da nossa API
API_URL = f"http://localhost:{API_PORT}/api/v1"

# =========================================================================
# Configuração, CSS "WOW" (Glassmorphism + Dark Mode + Gradient + Fonts Modernas)
# =========================================================================
favicon_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "imagens", "RH_Neural_simbol2.png"))

st.set_page_config(
    page_title="RH Neural", 
    page_icon=favicon_path, 
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def start_fastapi():
    """Inicia a FastAPI em background caso não esteja rodando."""
    import subprocess
    import time
    
    docs_url = f"http://{API_HOST}:{API_PORT}/api/docs"
    # Tenta verificar se a API já está online
    try:
        if requests.get(docs_url, timeout=1).status_code == 200:
            return True
    except requests.exceptions.RequestException:
        pass
        
    # Se não estiver, inicia o subprocesso
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.api.main:app", "--host", API_HOST, "--port", str(API_PORT)],
        cwd=project_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    
    # Aguarda a API subir (até 15 segundos)
    for _ in range(15):
        try:
            if requests.get(docs_url, timeout=1).status_code == 200:
                return True
        except requests.exceptions.RequestException:
            time.sleep(1)
            
    return False

start_fastapi()

def get_image_base64(path):
    try:
        with open(path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except Exception:
        return ""

def inject_premium_css():
    st.markdown("""
        <style>
        /* Importar fonte do Google: Inter e Roboto */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Roboto:wght@400;700&display=swap');

        /* Configuração Base para as Fontes e Background Padrão */
        html, body, [class*="css"]  {
            font-family: 'Inter', sans-serif !important;
            background-color: #002233; /* Deep Blue background */
            color: #E2E8F0;
        }

        /* Estilizando Títulos com Gradiente TOTVS Cyan -> Magenta */
        h1, h2, h3 {
            font-family: 'Roboto', sans-serif !important;
            font-weight: 700;
            background: -webkit-linear-gradient(45deg, #00C1D5, #E3006A);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        /* Metric Cards com Efeito TOTVS */
        [data-testid="stMetricValue"] {
            font-size: 2.2rem !important;
            font-weight: 700 !important;
            color: #ffffff !important;
        }
        [data-testid="stMetricLabel"] {
            font-size: 1rem !important;
            color: #A0AEC0 !important;
            font-weight: 400 !important;
        }
        
        div[data-testid="metric-container"] {
            background: rgba(26, 26, 26, 0.8);
            border-left: 4px solid #00C1D5;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
            transition: transform 0.2s ease-in-out;
        }
        div[data-testid="metric-container"]:hover {
            transform: translateY(-5px);
            border-left: 4px solid #E3006A;
        }
        
        /* Tabela Expansível estilizada */
        .stDataFrame {
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid #333333;
        }

        /* Barra Lateral e Header (Azul Escuro TOTVS) */
        [data-testid="stSidebar"], [data-testid="stHeader"] {
            background-color: #002233 !important;
            border-right: 1px solid #1A2E3B; /* Leve borda para separar do corpo, se necessário */
        }
        
        [data-testid="stHeader"]::before {
            content: "";
            display: flex;
            align-items: center;
            justify-content: center;
            width: 100%;
            height: 100%;
            position: absolute;
            left: 0;
            top: 0;
            z-index: 1000;
            pointer-events: none;
            background-size: auto 65%;
            background-repeat: no-repeat;
            background-position: center;
        }
        
        [data-testid="stSidebar"] [data-testid="stSidebarNav"] {
            display: none;
        }
        
        [data-testid="stSidebar"] .stButton > button {
            background-color: rgba(0, 0, 0, 0.15) !important;
            border: none !important;
            border-radius: 10px !important;
            color: white !important;
            font-weight: 600 !important;
            width: 100% !important;
            display: flex !important;
            justify-content: flex-start !important;
            padding: 12px 20px !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05) !important;
        }
        [data-testid="stSidebar"] .stButton > button:hover {
            background-color: rgba(0, 0, 0, 0.3) !important;
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.1) !important;
        }
        [data-testid="stSidebar"] .stButton > button p {
            font-size: 1.05rem !important;
            margin: 0 !important;
        }
        
        /* Botões Primários Estilo TOTVS */
        button[kind="primary"] {
            background: linear-gradient(90deg, #00C1D5 0%, #E3006A 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 6px !important;
            font-weight: bold !important;
        }
        
        /* Chat Agente (IA) */
        .stChatFloatingInputContainer {
            background: rgba(26, 26, 26, 0.9) !important;
            border-top: 1px solid #333333;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Injetar a logo do cabeçalho
    header_logo_path = os.path.join(os.path.dirname(__file__), "..", "imagens", "logo.png")
    header_logo_b64 = get_image_base64(header_logo_path)
    if header_logo_b64:
        st.markdown(f"""
            <style>
            [data-testid="stHeader"]::before {{
                background-image: url("data:image/png;base64,{header_logo_b64}");
            }}
            </style>
        """, unsafe_allow_html=True)

# Chamada do CSS
inject_premium_css()

# =========================================================================
# Callers Utilitários (Para isolar lógica HTTP)
# =========================================================================

@st.cache_data(ttl=60)
def fetch_dashboard_summary() -> dict:
    try:
         resp = requests.get(f"{API_URL}/dashboard/summary")
         resp.raise_for_status()
         return resp.json()
    except:
         return {"total_employees": 1470, "total_churn": 237, "churn_rate": 16.12, "avg_salary": 6500, "departments": []}

@st.cache_data(ttl=120)
def fetch_employees_sample() -> pd.DataFrame:
    try:
        resp = requests.get(f"{API_URL}/employees?limit=2000")
        resp.raise_for_status()
        data = resp.json()["data"]
        return pd.DataFrame(data)
    except:
        return pd.DataFrame()

def call_predict_churn(emp_id: int):
     try:
          resp = requests.post(f"{API_URL}/predict/{emp_id}")
          if resp.status_code == 200: return resp.json()
          return None
     except: return None
     
def call_llm_insights(emp_id: int):
     try:
          resp = requests.post(f"{API_URL}/insights/{emp_id}")
          if resp.status_code == 200: return resp.json()
          return None
     except: return None

# =========================================================================
# Elementos de Interface
# =========================================================================

# Inicialização de Estado da Navegação
if "current_page" not in st.session_state:
    st.session_state.current_page = "Visão Geral"

logo_path = os.path.join(os.path.dirname(__file__), "..", "imagens", "RH_Neural_simbol3.png")
logo_b64 = get_image_base64(logo_path)

# Logo na Sidebar
if logo_b64:
    st.sidebar.markdown(f'''
    <div style="display: flex; justify-content: center; margin-bottom: 20px;">
        <img src="data:image/png;base64,{logo_b64}" width="320" style="background: transparent;"/>
    </div>
    ''', unsafe_allow_html=True)

st.sidebar.markdown("### Menu Principal")

# Botões da Sidebar com ícones minimalistas
if st.sidebar.button("◈ Visão Geral", use_container_width=True):
    st.session_state.current_page = "Visão Geral"
if st.sidebar.button("▣ Dossiê", use_container_width=True):
    st.session_state.current_page = "Dossiê"
if st.sidebar.button("✨ Insight Generativo", use_container_width=True):
    st.session_state.current_page = "Insights"
if st.sidebar.button("💬 Agente Interativo", use_container_width=True):
    st.session_state.current_page = "Agente"

st.markdown("Plataforma de inteligência para acompanhamento preditivo de Evasão (Churn) e Retenção de Talentos, alimentada por **LLM** e **XGBoost**.")

# == ABA 1 : VISÃO GERAL ==
if st.session_state.current_page == "Visão Geral":
    df_emps = fetch_employees_sample()
    if not df_emps.empty:
         # Traduzir categorias para os gráficos em português
         df_emps["Department"] = df_emps["Department"].replace({
             "Sales": "Vendas",
             "Research & Development": "Pesquisa & Desenvolvimento",
             "Human Resources": "Recursos Humanos"
         })
         df_emps["Attrition"] = df_emps["Attrition"].replace({
             "Yes": "Sim",
             "No": "Não"
         })
         
         # Filtro Global da Aba
         departamentos = ["Todos"] + list(df_emps["Department"].unique())
         selected_dept = st.selectbox("Filtro por Departamento:", options=departamentos)
         
         if selected_dept != "Todos":
             df_emps = df_emps[df_emps["Department"] == selected_dept]

         st.markdown("### Resumo Executivo")
         
         total_emps = len(df_emps)
         total_churn = len(df_emps[df_emps["Attrition"] == "Sim"])
         churn_rate = (total_churn / total_emps * 100) if total_emps > 0 else 0
         avg_salary = df_emps["MonthlyIncome"].mean() if "MonthlyIncome" in df_emps.columns else 0

         c1, c2, c3, c4 = st.columns(4)
         c1.metric("Total de Colaboradores", f"{total_emps}")
         c2.metric("Desligamentos Previstos", f"{total_churn}")
         c3.metric("Taxa de Evasão", f"{churn_rate:.1f}%", delta="- Melhorar", delta_color="inverse")
         c4.metric("Média Salarial Mensal", f"R$ {avg_salary:,.2f}")
         
         st.markdown("---")
         st.markdown("### Densidade e Perfis")
         
         col_plot1, col_plot2 = st.columns(2)
         
         with col_plot1:
              st.markdown("**Esgotamento x Rotatividade (Por Departamento)**")
              fig1 = px.histogram(df_emps, x="Department", color="Attrition", barmode='group',
                                  labels={"Department": "Departamento", "Attrition": "Rotatividade"},
                                  color_discrete_map={"Sim": "#00C1D5", "Não": "#E3006A"},
                                  category_orders={"Attrition": ["Sim", "Não"]}, template="plotly_dark")
              fig1.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', yaxis_title="Contagem")
              st.plotly_chart(fig1, use_container_width=True)
              
         with col_plot2:
              st.markdown("**Risco Salarial: Pontuação x Nível do Cargo**")
              fig2 = px.box(df_emps, x="JobLevel", y="MonthlyIncome", color="Attrition",
                            labels={"JobLevel": "Nível do Cargo", "MonthlyIncome": "Renda Mensal", "Attrition": "Rotatividade"},
                            color_discrete_map={"Sim": "#00C1D5", "Não": "#E3006A"},
                            category_orders={"Attrition": ["Sim", "Não"]}, template="plotly_dark")
              fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
              st.plotly_chart(fig2, use_container_width=True)
    else:
         st.warning("Serviço de dados dos colaboradores indisponível (Certifique-se de ligar a FastAPI).")


# == ABA 2 : Dossiê Individual e SHAP ==
elif st.session_state.current_page == "Dossiê":
    st.markdown("### Busca Ativa Preditora")
    df_emps_local = fetch_employees_sample()
    
    if not df_emps_local.empty:
         # Traduzir categorias de Departamento para manter consistência
         df_emps_local["Department"] = df_emps_local["Department"].replace({
             "Sales": "Vendas",
             "Research & Development": "Pesquisa & Desenvolvimento",
             "Human Resources": "Recursos Humanos"
         })
         
         # Filtro de Departamento
         departamentos_dossie = ["Todos"] + list(df_emps_local["Department"].unique())
         selected_dept_dossie = st.selectbox("Filtro por Departamento:", options=departamentos_dossie, key="dept_dossie")
         
         if selected_dept_dossie != "Todos":
             df_emps_local = df_emps_local[df_emps_local["Department"] == selected_dept_dossie]

         st.markdown("Selecione um ID de colaborador para rodar a inferência ML em Tempo Real.")
         colA, colB = st.columns([1, 2])
         
         with colA:
              options_ids = df_emps_local['EmployeeNumber'].tolist()
              selected_id = st.selectbox("🆔 ID do Colaborador:", options=options_ids)
              
              if st.button("Executar Diagnóstico ML", type="primary", use_container_width=True):
                   with st.spinner("Convocando o Oráculo Baseado em Árvores..."):
                        pred_dict = call_predict_churn(selected_id)
                        if pred_dict:
                             st.session_state["last_pred"] = pred_dict
                             st.session_state["last_id"] = selected_id
                        else:
                             st.error("Erro ao comunicar com a API de Modelos.")
                             
              # Mostrar os dados crus debaixo 
              st.markdown("#### Metadados:")
              emp_info = df_emps_local[df_emps_local['EmployeeNumber'] == selected_id].iloc[0]
              emp_info_pt = emp_info[['Age', 'Department', 'JobRole', 'MonthlyIncome', 'YearsAtCompany', 'YearsSinceLastPromotion']].rename({'Age': 'Idade', 'Department': 'Departamento', 'JobRole': 'Cargo', 'MonthlyIncome': 'Renda Mensal', 'YearsAtCompany': 'Anos na Empresa', 'YearsSinceLastPromotion': 'Anos Desde Últ. Promoção'})
              st.dataframe(emp_info_pt, use_container_width=True)
              
         with colB:
              if "last_pred" in st.session_state and st.session_state.get("last_id") == selected_id:
                   pred = st.session_state["last_pred"]
                   lvl = pred["risk_level"].upper()
                   lvl_pt = {"HIGH": "ALTO", "MEDIUM": "MÉDIO", "LOW": "BAIXO"}.get(lvl, lvl)
                   score = pred["risk_score"] * 100
                   
                   color = "#E3006A" if lvl == "HIGH" else "#fbbf24" if lvl == "MEDIUM" else "#00C1D5"
                   
                   st.markdown(f"### Score Final: <span style='color:{color}'>{lvl_pt} ({score:.1f}%)</span>", unsafe_allow_html=True)
                   
                   # Gauge gráfico
                   fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = score,
                        title = {'text': "Risco de Abandonar a Empresa"},
                        gauge = {
                             'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                             'bar': {'color': color},
                             'bgcolor': "rgba(0,0,0,0)",
                             'steps': [
                                 {'range': [0, 30], 'color': "rgba(0, 193, 213, 0.3)"},
                                 {'range': [30, 65], 'color': "rgba(251, 191, 36, 0.3)"},
                                 {'range': [65, 100], 'color': "rgba(227, 0, 106, 0.3)"}],
                        }
                   ))
                   fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"}, height=300, margin=dict(l=20,r=20,t=40,b=20))
                   st.plotly_chart(fig_gauge, use_container_width=True)

# == ABA 3 : LLM Insight ==
elif st.session_state.current_page == "Insights":
    st.markdown("### Agente Estratega com Relatório SHAP (LLM)")
    if "last_pred" in st.session_state:
         cur_id = st.session_state["last_id"]
         st.info(f"O modelo Groq analisará o ID: **{cur_id}** cruzando histórico dele e o vetor SHAP em background.")
         
         if st.button("🧠 Gerar Plano de Ação Personalizado (Groq)"):
              with st.spinner("O LLM está lendo os dados extraídos pelo Explainer no backend..."):
                   ins = call_llm_insights(cur_id)
                   
                   if ins:
                        st.markdown("### Resumo Diagnóstico")
                        # Badge urgency
                        urgency_val = str(ins.get('urgency')).upper()
                        urgency_pt = {"IMMEDIATE": "IMEDIATA", "WATCH": "OBSERVAR", "LOW": "BAIXA", "NORMAL": "NORMAL"}.get(urgency_val, urgency_val)
                        u_color = "red" if ins.get('urgency') == 'immediate' else 'orange' if ins.get('urgency') == 'watch' else 'green'
                        st.markdown(f"**Urgência:** :{u_color}[{urgency_pt}]")
                        
                        st.write(ins.get("detailed_analysis", ""))
                        
                        col_r1, col_r2 = st.columns(2)
                        with col_r1:
                             st.markdown("#### Fatores de Impulso à Evasão (Alertas)")
                             for alert in ins.get("main_factors_summary", []):
                                  st.markdown(f"- ⚠️ {alert}")
                                  
                        with col_r2:
                             st.markdown("#### Plano de Retenção Recomendado")
                             for act in ins.get("recommended_actions", []):
                                  st.markdown(f"- ✅ {act}")
                                  
                   else:
                        st.error("Serviço LLM apresentou falha/timeout.")
    else:
         st.warning("Gere a predição local na aba 'Dossiê do Colaborador' antes de avançar para os insights LLM.")


# == ABA 4 : CHATBOT IA ==
elif st.session_state.current_page == "Agente":
    st.markdown("### Assistente Chatbot (ReAct Agent)")
    st.caption("Converse livremente sobre retenção. O agente usa ferramentas ReAct para buscar IDs e comparar grupos.")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [{"role": "assistant", "content": "Olá! Eu sou o assistente de RH. Pergunte qual o risco de um ID, por exemplo, ou qual a taxa de rotatividade da área corporativa de Vendas."}]

    # Mostrar histórico
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"], avatar="🤖" if msg["role"] == "assistant" else "👤"):
            st.markdown(msg["content"])

    # Capturar prompt
    if prompt := st.chat_input("Digite sua dúvida de RH inteligente..."):
        # Adiciona fala humana na tela
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        # Requisição
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Buscando metadados via Grafo LangGraph..."):
                 try:
                      res = requests.post(f"{API_URL}/agent/chat", json={"query": prompt, "session_id": "hr_web_1"})
                      ans = res.json().get("response", "Erro de leitura.")
                 except Exception as e:
                      ans = f"Ocorreu um erro no tráfego: {str(e)}"
                 st.markdown(ans)
                 
        st.session_state.chat_history.append({"role": "assistant", "content": ans})

# Footer da Tela / Liderança de Design
st.markdown("---")
st.markdown("<div style='text-align: center; color: gray; font-size: 0.8rem'>Criado pelo Cientista de Dados - David Vieira | www.vieiralabs.ia.br</div>", unsafe_allow_html=True)
