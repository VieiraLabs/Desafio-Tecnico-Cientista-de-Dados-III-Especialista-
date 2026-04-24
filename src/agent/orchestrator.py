"""
Orquestrador do Agente baseado em Grafos ReAct (Reasoning and Acting) com memory persist.
Responsável por interpretar a intenção do usuário final e convocar as tools certas (Ex: ML, Consulta DBM) 
na ordem certa, unificando a inteligência final na resposta.
"""

from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from src.llm.insights import get_llm
from src.llm.prompts import AGENT_SYSTEM_PROMPT
from src.agent.tools import HR_TOOLS

class AgentState(TypedDict):
    """Estado do agente para persistência e rastreamento local das mesangens (Memória)"""
    messages: Annotated[Sequence[BaseMessage], operator.add]

def build_agent():
    """
    Constrói a instância Singleton do Agente ReAct + Tools conectando-o no LLM e injetando o System Prompt.
    """
    llm = get_llm()
    
    # Checkpointer na memória para rastrear e lembrar o que o usuário perguntou antes no mesmo chat
    memory = MemorySaver()
    
    # Criar a StateGraph simplificada do React Agent
    agent_executor = create_react_agent(
        llm, 
        HR_TOOLS, 
        state_modifier=AGENT_SYSTEM_PROMPT, 
        checkpointer=memory
    )
    
    return agent_executor

# Exportar a instância para fins táticos se necessário, podendo instanciar depois em memória
# Aqui, a invocação fica com a função abaixo ou via thread de chamadas da api

def run_agent_query(user_query: str, thread_id: str = "default_session_1") -> str:
    """
    Recebe a String do usuário final, injeta como input no agente 
    e extrai o conteúdo da resposta textual da IA.
    """
    agent = build_agent()
    
    config = {"configurable": {"thread_id": thread_id}}
    
    inputs = {
        "messages": [HumanMessage(content=user_query)]
    }
    
    try:
        # Puxamos o state após todas as chamadas iterativas (Thought -> Action -> Thought... Response)
        response_state = agent.invoke(inputs, config=config)
        
        # Última mensagem na stack
        return response_state["messages"][-1].content
    except Exception as e:
        print(f"Erro rodando agente LangGraph: {e}")
        return f"Desculpe, ocorreu um problema no orquestrador IA: {str(e)}"

if __name__ == "__main__":
    # Teste rápido executado como modulo (Mock Test)
    res = run_agent_query("Qual a taxa de churn atual do departamento Sales e porquê ele existe?")
    print("-" * 50)
    print("AGENTE: ", res)
