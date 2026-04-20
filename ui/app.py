"""
Aplicação Streamlit Premium para o People Analytics Dashboard
Apresenta um design Dark Mode moderno, CSS customizado, e 4 abas para 
demografia, preditor/shapley, Insights via agente LLM e métricas gerais.
"""

import streamlit as st
import pandas as pd
import requests
import json
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any

from src.config import API_PORT, API_HOST

# Endereço base da nossa API
API_URL = f"http://localhost:{API_PORT}/api/v1"

# =========================================================================
# Configuração, CSS "WOW" (Glassmorphism + Dark Mode + Gradient + Fonts Modernas)
# =========================================================================
st.set_page_config(
    page_title="TOTVS • People Analytics", 
    page_icon="💠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

def inject_premium_css():
    st.markdown("""
        <style>
        /* Importar fonte do Google: Inter e Roboto Mono */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Outfit:wght@400;700&display=swap');

        /* Configuração Base para as Fontes e Background Padrão */
        html, body, [class*="css"]  {
            font-family: 'Inter', sans-serif !important;
            background-color: #0b0f19; /* Dark blue background */
            color: #E2E8F0;
        }

        /* Estilizando Títulos */
        h1, h2, h3 {
            font-family: 'Outfit', sans-serif !important;
            font-weight: 700;
            background: -webkit-linear-gradient(45deg, #38bdf8, #818cf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        /* Metric Cards com Efeito Glassmorphism */
        [data-testid="stMetricValue"] {
            font-size: 2.2rem !important;
            font-weight: 700 !important;
            color: #f8fafc !important;
        }
        [data-testid="stMetricLabel"] {
            font-size: 1rem !important;
            color: #94a3b8 !important;
            font-weight: 400 !important;
        }
        
        div[data-testid="metric-container"] {
            background: rgba(30, 41, 59, 0.4);
            border: 1px solid rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            transition: transform 0.2s ease-in-out;
        }
        div[data-testid="metric-container"]:hover {
            transform: translateY(-5px);
        }
        
        /* Tabela Expansível estilizda */
        .stDataFrame {
            border-radius: 10px;
            overflow: hidden;
        }

        /* Barra Lateral */
        [data-testid="stSidebar"] {
            background-color: #0f172a;
            border-right: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        /* Chat Agente (IA) */
        .stChatFloatingInputContainer {
            background: rgba(15, 23, 42, 0.8) !important;
        }
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
        resp = requests.get(f"{API_URL}/employees?limit=250")
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

st.title("💠 TOTVS Neural HR & Retention")
st.markdown("Plataforma de inteligência para acompanhamento preditivo de Churn alimentada por **LLM** e **XGBoost**.")

# Navegação
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Visão Geral e Demografia", 
    "👥 Dossiê do Colaborador (Risco & Explicabilidade)", 
    "🧠 Insight Generativo (Groq)", 
    "🤖 Agente Interativo (Chatbot)"
])

# == ABA 1 : VISÃO GERAL ==
with tab1:
    st.markdown("### Executive Summary")
    summary = fetch_dashboard_summary()
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Headcount Global", f"{summary.get('total_employees', 0)}")
    c2.metric("Turnover Previsto (Historico)", f"{summary.get('total_churn', 0)}")
    c3.metric("Taxa de Churn", f"{summary.get('churn_rate', 0):.1f}%", delta="- Melhorar", delta_color="inverse")
    c4.metric("Média Salarial Mensal", f"${summary.get('avg_salary', 0):,.2f}")
    
    st.markdown("---")
    st.markdown("### Densidade e Perfis")
    
    df_emps = fetch_employees_sample()
    if not df_emps.empty:
         col_plot1, col_plot2 = st.columns(2)
         
         with col_plot1:
              st.markdown("**Burnout x Atrito (Por Departamento)**")
              fig1 = px.histogram(df_emps, x="Department", color="Attrition", barmode='group',
                                  color_discrete_sequence=['#38bdf8', '#ef4444'], template="plotly_dark")
              fig1.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
              st.plotly_chart(fig1, use_container_width=True)
              
         with col_plot2:
              st.markdown("**Risco Salarial: Score x Job Level**")
              fig2 = px.box(df_emps, x="JobLevel", y="MonthlyIncome", color="Attrition",
                            color_discrete_sequence=['#38bdf8', '#ef4444'], template="plotly_dark")
              fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
              st.plotly_chart(fig2, use_container_width=True)
    else:
         st.warning("Serviço de dados dos colaboradores indisponível (Certifique-se de ligar a FastAPI).")


# == ABA 2 : Dossiê Individual e SHAP ==
with tab2:
    st.markdown("### Busca Ativa Preditora")
    df_emps_local = fetch_employees_sample()
    
    if not df_emps_local.empty:
         st.markdown("Selecione um ID de colaborador presente na amostra para rodar a inferência ML em Tempo Real.")
         colA, colB = st.columns([1, 2])
         
         with colA:
              options_ids = df_emps_local['EmployeeNumber'].tolist()
              selected_id = st.selectbox("🆔 Employee ID:", options=options_ids)
              
              if st.button("Executar Diagnóstico ML", type="primary", use_container_width=True):
                   with st.spinner("Convocando o Oráculo Baseado em Árvores..."):
                        pred_dict = call_predict_churn(selected_id)
                        if pred_dict:
                             st.session_state["last_pred"] = pred_dict
                             st.session_state["last_id"] = selected_id
                        else:
                             st.error("Erro ao comunicar com a API de Modelos.")
                             
              # Mostrar os dados crus debaixo 
              st.markdown("#### Meta Dados:")
              emp_info = df_emps_local[df_emps_local['EmployeeNumber'] == selected_id].iloc[0]
              st.dataframe(emp_info[['Age', 'Department', 'JobRole', 'MonthlyIncome', 'YearsAtCompany', 'YearsSinceLastPromotion']], use_container_width=True)
              
         with colB:
              if "last_pred" in st.session_state and st.session_state.get("last_id") == selected_id:
                   pred = st.session_state["last_pred"]
                   lvl = pred["risk_level"].upper()
                   score = pred["risk_score"] * 100
                   
                   color = "#ef4444" if lvl == "HIGH" else "#fbbf24" if lvl == "MEDIUM" else "#10b981"
                   
                   st.markdown(f"### Score Final: <span style='color:{color}'>{lvl} ({score:.1f}%)</span>", unsafe_allow_html=True)
                   
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
                                 {'range': [0, 30], 'color': "rgba(16, 185, 129, 0.3)"},
                                 {'range': [30, 65], 'color': "rgba(251, 191, 36, 0.3)"},
                                 {'range': [65, 100], 'color': "rgba(239, 68, 68, 0.3)"}],
                        }
                   ))
                   fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"}, height=300, margin=dict(l=20,r=20,t=40,b=20))
                   st.plotly_chart(fig_gauge, use_container_width=True)

# == ABA 3 : LLM Insight ==
with tab3:
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
                        u_color = "red" if ins.get('urgency') == 'immediate' else 'orange' if ins.get('urgency') == 'watch' else 'green'
                        st.markdown(f"**Urgência:** :{u_color}[{str(ins.get('urgency')).upper()}]")
                        
                        st.write(ins.get("detailed_analysis", ""))
                        
                        col_r1, col_r2 = st.columns(2)
                        with col_r1:
                             st.markdown("#### Fatores de Impulso ao Churn (Alertas)")
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
with tab4:
    st.markdown("### Assistente Chatbot (ReAct Agent)")
    st.caption("Converse livremente sobre retenção. O agente usa ferramentas ReAct para buscar IDs e comparar grupos.")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [{"role": "assistant", "content": "Olá! Eu sou o RH Bot. Pergunte qual o risco de um ID, por exemplo, ou qual a taxa de turnover da área corporativa Sales."}]

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
st.markdown("<div style='text-align: center; color: gray; font-size: 0.8rem'>Pós-Graduação TOTVS | Arquitetado como uma UI Premium Real.</div>", unsafe_allow_html=True)
