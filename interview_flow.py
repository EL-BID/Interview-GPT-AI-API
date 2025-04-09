import os
import sys
import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, TypedDict, Annotated
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from db_connection import get_db_connection

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InterviewState(TypedDict):
    """Estado de la entrevista."""
    messages: Annotated[List[BaseMessage], add_messages]  # Historial de mensajes
    current_question: Dict  # Pregunta actual y su contexto
    is_complete: bool  # Indica si la pregunta está completada
    question_number: int  # Número de pregunta actual
    total_questions: int  # Total de preguntas
    user_data: Optional[Dict[str, Any]]  # Datos del usuario

def get_llm():
    """Configuración del LLM."""
    return AzureChatOpenAI(
        deployment_name=os.getenv("AZURE_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=0.0
    )

def process_chunks(chunk: Dict) -> List[Dict]:
    """
    Procesa los chunks del agente y extrae los mensajes relevantes.
    
    Args:
        chunk (Dict): Chunk de datos del agente
        
    Returns:
        List[Dict]: Lista de mensajes procesados
    """
    processed_messages = []
    
    # Verificar si el chunk tiene la estructura esperada
    if "interviewer" in chunk and "messages" in chunk["interviewer"]:
        for message in chunk["interviewer"]["messages"]:
            # Solo procesar mensajes que no sean SystemMessage
            if not isinstance(message, SystemMessage):
                processed_messages.append({
                    "role": "assistant" if isinstance(message, AIMessage) else "user",
                    "content": message.content
                })
    
    return processed_messages

def interviewer_node(state: InterviewState) -> InterviewState:
    """Nodo principal que maneja la entrevista."""
    try:
        llm = get_llm()
        current_question = state["current_question"]
        total_questions = state["total_questions"]
        user_data = state.get("user_data", {})
        nombre_usuario = user_data.get("nombre", "").split()[0] if user_data and user_data.get("nombre") else ""
        
        # Preparar la sección del participante
        participante_section = f"PARTICIPANTE:\nNombre: {nombre_usuario}\n" if nombre_usuario else ""
        
        # Preparar la instrucción personalizada
        instruccion_personal = (
            f"- Usa el nombre de la persona ({nombre_usuario}) para hacer la conversación más personal"
            if nombre_usuario else
            "- Mantén un tono conversacional y cercano"
        )
        
        # Crear el prompt del sistema
        system_message = SystemMessage(
            content=f"""Eres un entrevistador amigable y conversacional, haciendo preguntas sobre un tema específico.
Tu objetivo es obtener información detallada y completa sobre el tema, manteniendo una conversación natural.

{participante_section}
CONTEXTO DE LA PREGUNTA:
Pregunta actual: {current_question['question']}
Numero de pregunta actual: {state['question_number']}
Total de preguntas: {state['total_questions']}
Aspectos a explorar: {current_question['context']}

INSTRUCCIONES:
1. {instruccion_personal}
2. Analiza las respuestas previas para esta pregunta
3. Evalúa si se han cubierto todos los aspectos importantes
4. Si falta información:
   - Identifica qué aspectos necesitan más detalle
   - Formula una pregunta de seguimiento natural y específica
   - Si despues de muchas preguntas (mas de 10) no se terminan de cubrir los aspectos importantes, agradece la participacion y dala por completada y responde con "COMPLETO".
5. Si el usuario responde que no sabe la respuesta, agradece la participacion y dala por completada y responde con "COMPLETO".
6. Si la respuesta está completa:
   - Basandote en la pregunta actual y el total de preguntas, si es la última pregunta, agradece la participación
   - Si no es la última pregunta, indica que continuaremos con la siguiente pregunta
   - Responde con: "COMPLETO: [resumen de la respuesta]"

IMPORTANTE:
- Mantén un tono conversacional y amigable
- Haz preguntas específicas pero naturales
- Tú decides cuándo se han cubierto todos los aspectos importantes
"""
        )
        
        # Si no hay mensajes previos o el primer mensaje no es el del sistema
        if not state["messages"] or not isinstance(state["messages"][0], SystemMessage):
            state["messages"] = [system_message] + state["messages"]
        
        # Obtener la respuesta del LLM
        response = llm.invoke(state["messages"])
        
        # Determinar si la respuesta está completa
        is_complete = "COMPLETO" in response.content
        
        return {
            **state,
            "messages": state["messages"] + [response],
            "is_complete": is_complete
        }
        
    except Exception as e:
        logger.error(f"Error en interviewer_node: {str(e)}")
        error_message = AIMessage(content=f"Lo siento, ha ocurrido un error: {str(e)}")
        return {
            **state,
            "messages": state["messages"] + [error_message],
            "is_complete": False
        }

def build_graph(checkpointer=None) -> StateGraph:
    """
    Construye el grafo de la entrevista.
    
    Args:
        checkpointer: El checkpointer para el grafo
        
    Returns:
        StateGraph: El grafo compilado
    """
    workflow = StateGraph(InterviewState)
    
    # Añadir el nodo principal
    workflow.add_node("interviewer", interviewer_node)
    
    # Definir el flujo simple: START -> interviewer -> END
    workflow.add_edge(START, "interviewer")
    workflow.add_edge("interviewer", END)
    
    return workflow.compile(checkpointer=checkpointer)

def get_interview_graph(checkpointer=None):
    """
    Obtiene el grafo compilado.
    
    Args:
        checkpointer: El checkpointer para el grafo
        
    Returns:
        StateGraph: El grafo compilado
    """
    graph = build_graph(checkpointer)
    return graph

async def run_interview_async(question: Dict = None, user_data: Dict = None, user_response: str = None, thread_id: str = "test-thread"):
    """
    Función principal que ejecuta la entrevista de forma asíncrona.
    
    Args:
        question (Dict): Pregunta actual y su contexto
        user_data (Dict): Datos del usuario
        user_response (str): Respuesta del usuario si existe
        thread_id (str): ID del hilo de la entrevista
        
    Returns:
        Dict: Resultados de la entrevista
    """
    try:
        # Formar el estado inicial
        state = {
            "messages": [HumanMessage(content=user_response)] if user_response else [],
            "current_question": question or {},
            "is_complete": False,
            "question_number": 1,
            "total_questions": 1,
            "user_data": user_data or {}
        }
        
        # Configuración para el checkpointer
        config = {"configurable": {"thread_id": thread_id}}
        
        # Obtener el checkpointer
        checkpointer, pool = await get_db_connection()
       
        
        try:
            # Obtener el grafo con el checkpointer
            graph = get_interview_graph(checkpointer=checkpointer)
            
            # Lista para almacenar los mensajes procesados
            processed_messages = []
            
            # Invocar el grafo con la configuración
            async for chunk in graph.astream(
                state,
                config
            ):
                logger.info(f"Chunk: {chunk}")
                # Procesar los chunks del agente
                chunk_messages = process_chunks(chunk)
                processed_messages.extend(chunk_messages)
            
            # Retornar el resultado usando la información de los chunks
            return {
                "status": "success",
                "thread_id": thread_id,
                "is_complete": any("COMPLETO" in msg["content"] for msg in processed_messages),
                "current_question": state["current_question"],
                "question_number": state["question_number"],
                "total_questions": state["total_questions"],
                "messages": processed_messages
            }
        finally:
            # Cerrar la conexión
            await pool.close()
            
    except Exception as e:
        logger.error(f"Error en run_interview: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        } 