import os
import sys
import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, TypedDict, Annotated, Literal
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
    validation_result: str  # Resultado de la validación de la respuesta
    question_number: int  # Número de pregunta actual
    total_questions: int  # Total de preguntas
    user_data: Optional[Dict[str, Any]]  # Datos del usuario
    description: str  # Descripción general de la entrevista (opcional)
    language: str  # Idioma en el que se realizará la entrevista (por defecto 'es')

def get_llm():
    """Configuración del LLM."""
    return AzureChatOpenAI(
        deployment_name=os.getenv("AZURE_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=0.0
    )

def process_chunks(chunk: Dict) -> Dict:
    """
    Procesa los chunks del agente y extrae los mensajes relevantes.
    
    Args:
        chunk (Dict): Chunk de datos del agente
        
    Returns:
        Dict: Diccionario con los mensajes procesados y el valor de is_complete
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
    elif "despedida" in chunk and "messages" in chunk["despedida"]:
        for message in chunk["despedida"]["messages"]:
            # Solo procesar mensajes que no sean SystemMessage
            if not isinstance(message, SystemMessage):
                processed_messages.append({
                    "role": "assistant" if isinstance(message, AIMessage) else "user",
                    "content": message.content
                })
    
    # Extraer el valor de is_complete y validation_result del chunk
    is_complete = chunk.get("interviewer", {}).get("is_complete", False) or chunk.get("despedida", {}).get("is_complete", False)
    validation_result = chunk.get("interviewer", {}).get("validation_result", "") or chunk.get("despedida", {}).get("validation_result", "")
    
    print(f"Valor de is_complete en process_chunks: {is_complete}")
    print(f"Valor de validation_result en process_chunks: {validation_result}")
    
    return {
        "messages": processed_messages,
        "is_complete": is_complete,
        "validation_result": validation_result
    }

def refrasear_mensaje(llm, messages, error_data, system_message=None):
    """
    Función auxiliar para refrasear mensajes cuando se detecta un error de filtro de contenido.
    
    Args:
        llm: Instancia del modelo de lenguaje
        messages: Lista de mensajes a procesar
        error_data: Datos del error de filtro de contenido
        system_message: Mensaje del sistema con las instrucciones originales (opcional)
        
    Returns:
        tuple: (mensajes actualizados, éxito)
    """
    try:
        # Obtener los resultados del filtro
        filter_result = error_data.get('error', {}).get('innererror', {}).get('content_filter_result', {})
        
        # Verificar qué categorías activaron el filtro
        triggered_categories = []
        for category, result in filter_result.items():
            if result.get('filtered', False):
                severity = result.get('severity', 'unknown')
                triggered_categories.append((category, severity))
        
        if triggered_categories:
            # Si hay un system_message, intentar refrasearlo primero
            if system_message:
                try:
                    # Crear un prompt para refraseo del system_message
                    rephrase_system_prompt = SystemMessage(content=f"""Por favor, reformula las siguientes instrucciones de una manera más apropiada, manteniendo el mismo significado y tono profesional pero evitando cualquier contenido que pueda ser considerado inapropiado:

Instrucciones originales: {system_message.content}

Reformula las instrucciones de manera que:
1. Mantenga el mismo estilo y tono profesional
2. Evite cualquier lenguaje que pueda ser considerado inapropiado
3. Conserve el mismo objetivo y significado de las instrucciones originales
4. Sea claro y directo, manteniendo un tono profesional""")
                    
                    # Obtener la versión refraseada del system_message
                    rephrased_system_response = llm.invoke([rephrase_system_prompt])
                    rephrased_system_content = rephrased_system_response.content
                    
                    # Actualizar el system_message
                    system_message = SystemMessage(content=rephrased_system_content)
                    logger.info(f"Prompt del sistema refraseado: {rephrased_system_content}")
                except Exception as e:
                    logger.error(f"Error al refrasear el prompt del sistema: {str(e)}")
                    return messages, False, None
            
            # Obtener el último mensaje del usuario
            last_user_message = None
            for msg in reversed(messages):
                if isinstance(msg, HumanMessage):
                    last_user_message = msg.content
                    break
            
            if last_user_message:
                # Crear un prompt para refraseo del mensaje del usuario
                rephrase_prompt = SystemMessage(content=f"""Por favor, reformula la siguiente respuesta de una manera más apropiada, manteniendo el mismo significado y tono conversacional pero evitando cualquier contenido que pueda ser considerado inapropiado:

Respuesta original: {last_user_message}

Reformula la respuesta de manera que:
1. Mantenga el mismo estilo y tono de la respuesta original
2. Evite cualquier lenguaje que pueda ser considerado inapropiado
3. Conserve el mismo objetivo y significado de la respuesta original
4. Sea natural y conversacional, sin ser excesivamente formal""")
                
                # Obtener la versión refraseada
                rephrased_response = llm.invoke([rephrase_prompt])
                rephrased_message = rephrased_response.content
                
                # Actualizar el mensaje en la lista
                for i, msg in enumerate(messages):
                    if isinstance(msg, HumanMessage) and msg.content == last_user_message:
                        messages[i] = HumanMessage(content=rephrased_message)
                        break
                
                logger.info(f"Respuesta refraseada: {rephrased_message}")
                
                # Si se proporcionó un system_message, intentar la llamada al LLM con las instrucciones originales
                if system_message:
                    try:
                        # Asegurarse de que el system_message esté al principio
                        if not isinstance(messages[0], SystemMessage):
                            messages = [system_message] + messages
                        elif messages[0].content != system_message.content:
                            messages[0] = system_message
                        
                        # Intentar la llamada al LLM con los mensajes actualizados
                        response = llm.invoke(messages)
                        logger.info(f"Llamada al LLM exitosa después del refraseo")
                        return messages, True, response
                    except Exception as e:
                        logger.error(f"Error al llamar al LLM después del refraseo: {str(e)}")
                        return messages, False, None
                
                return messages, True, None
        
        return messages, False, None
        
    except Exception as e:
        logger.error(f"Error al refrasear mensaje: {str(e)}")
        return messages, False, None

def interviewer_node(state: InterviewState) -> InterviewState:
    """Nodo principal que maneja la entrevista."""
    try:
        llm = get_llm()
        current_question = state["current_question"]
        user_data = state.get("user_data", {})
        nombre_usuario = user_data.get("nombre", "").split()[0] if user_data and user_data.get("nombre") else ""
        is_complete = state.get("is_complete", False)
        description = state.get("description", "")
        language = state.get("language", "es")
        
        logger.info(f"Estado is_complete en interviewer_node: {is_complete}")
        
        # Preparar la sección del participante
        participante_section = f"PARTICIPANTE:\nNombre: {nombre_usuario}\n" if nombre_usuario else ""
        
        # Determinar el estado actual para el prompt
        estado_actual = "COMPLETA" if is_complete is True else "NO SABE/NO RESPONDE" if is_complete == "NS-NR" else "INCOMPLETA"
        
        # Crear el prompt del sistema
        system_message = SystemMessage(
            content=f"""Eres un entrevistador profesional, amigable y cercano. Tu objetivo es hacer que el participante se sienta cómodo mientras obtienes una respuesta completa a la pregunta.

IMPORTANTE SOBRE EL IDIOMA:
1. DEBES RESPONDER EN EL MISMO IDIOMA EN QUE ESTÁ FORMULADA LA PREGUNTA
2. Si la pregunta está en un idioma específico, usa ese idioma para todas tus respuestas
3. Solo si la pregunta no tiene un idioma claro, usa el idioma por defecto: {language.upper()}

{participante_section}
PREGUNTA ACTUAL:
{current_question['question']}

CONTEXTO ESPECÍFICO A EXPLORAR:
{current_question['context']}

INFORMACIÓN DE LA ENTREVISTA:
Esta es una entrevista acerca de {description}
Pregunta actual: {current_question['question_number']} de {current_question['total_questions']}
Estado de la respuesta: {estado_actual}
Valor de is_complete: {is_complete}

INSTRUCCIONES:
1. Si lo tienes, usa el nombre del participante para hacer la conversación más personal
2. Mantén un tono profesional, amigable y cercano
3. No des opiniones, sugerencias ni interpretes las respuestas
4. Muestra interés genuino en las respuestas del participante
5. NUNCA cierres la conversación hasta que la respuesta esté completa o el participante indique que no sabe/no quiere responder
6. Mantén la conversación activa con preguntas de seguimiento relevantes al tema de {description}
   - EN CADA MENSAJE, haz SOLO UNA o DOS preguntas de seguimiento
   - Prioriza la pregunta más relevante para el contexto
7. NO TE DESVÍES DE LA PREGUNTA Y CONTEXTO ACTUAL:
   - Si el participante hace preguntas, responde amablemente que te enfocarás en su respuesta primero
   - Si el participante menciona temas no relevantes, redirígelo suavemente hacia la pregunta original
   - Si el participante quiere cambiar de tema, indícale amablemente que primero necesitas su respuesta sobre la pregunta actual
   - Mantén el foco en obtener una respuesta completa sobre la pregunta y contexto actual

MANEJO DE RESPUESTAS:
1. Si la respuesta es incoherente o no relacionada:
   - Amablemente señala que la respuesta no está relacionada
   - Reformula la pregunta de manera más clara
   - Haz una pregunta de seguimiento específica sobre el contexto

2. Si la respuesta es incompleta (is_complete = False) de acuerdo a {is_complete}:
   - Haz preguntas de seguimiento naturales sobre el contexto y el tema de {description}
   - Pide detalles y ejemplos concretos
   - Explora aspectos relevantes no mencionados
   - Continúa la conversación hasta obtener una respuesta completa

3. Si el participante no sabe o no quiere responder (is_complete = NS-NR) de acuerdo a {is_complete}:
   - Si es la última pregunta:
     * Agradece la participación y despídete amablemente
   - Si no es la última pregunta:
     * Indica amablemente que pasaremos a la siguiente pregunta

IMPORTANTE:
- Mantén la conversación activa hasta obtener una respuesta completa
- No cierres la conversación prematuramente
- Haz preguntas de seguimiento relevantes y específicas al tema de {description}
- Guía al participante de manera constructiva hacia una respuesta completa
- NO PERMITAS DESVIACIONES DE LA PREGUNTA Y CONTEXTO ACTUAL
- RESPONDE EN EL MISMO IDIOMA DE LA PREGUNTA, o en {language.upper()} si no es claro
"""
        )
        
        # Si no hay mensajes previos o el primer mensaje no es el del sistema
        if not state["messages"] or not isinstance(state["messages"][0], SystemMessage):
            state["messages"] = [system_message] + state["messages"]
        
        # Número máximo de intentos
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Obtener la respuesta del LLM
                response = llm.invoke(state["messages"])
              
                break
            except Exception as e:
                error_data = getattr(e, 'response', {}).json() if hasattr(e, 'response') else {}
                
                # Verificar si es un error de filtro de contenido
                if error_data.get('error', {}).get('code') == 'content_filter':
                    
                    
                    # Intentar refrasear el mensaje y obtener la respuesta del LLM
                    state["messages"], success, llm_response = refrasear_mensaje(llm, state["messages"], error_data, system_message)
                    
                    if success and llm_response:
                        response = llm_response
                        break
                    elif success:
                        retry_count += 1
                        continue
                
                # Si no es un error de filtro de contenido o no se pudo refrasear, lanzar la excepción
                raise
        
        return {
            **state,
            "messages": state["messages"] + [response]
        }
        
    except Exception as e:
        logger.error(f"Error en interviewer_node: {str(e)}")
        error_message = AIMessage(content=f"Lo siento, ha ocurrido un error: {str(e)}")
        return {
            **state,
            "messages": state["messages"] + [error_message]
        }

def validate_response(state: InterviewState) -> InterviewState:
    """Nodo que valida si la respuesta es completa según el contexto, considerando toda la conversación."""
    try:
        llm = get_llm()
        current_question = state["current_question"]
        messages = state["messages"]
        print(f"current_question: {current_question['context']}")
        
        # Solo validar si hay mensajes del usuario
        if not messages or not any(isinstance(msg, HumanMessage) for msg in messages):
            return state
            
        # Calcular el número de mensajes del usuario
        user_messages_count = len([msg for msg in messages if isinstance(msg, HumanMessage)])
            
        # Crear el prompt del sistema para validación
        system_message = SystemMessage(
            content=f"""Eres un analista experto en evaluar respuestas. Tu tarea es analizar TODA la conversación para determinar si se han cubierto TODOS los aspectos requeridos tanto de la pregunta como del contexto.

====================================================================
PREGUNTA A EVALUAR:
{current_question['question']}
====================================================================

====================================================================
CONTEXTO REQUERIDO:
{current_question['context']}
====================================================================

====================================================================
INFORMACIÓN DE LA CONVERSACIÓN:
Número de mensajes del usuario: {user_messages_count}
====================================================================

INSTRUCCIONES:

1. EVALUACIÓN DE LA CONVERSACIÓN:
   - Analiza TODOS los mensajes en conjunto, incluyendo respuestas y preguntas de seguimiento
   - Evalúa si la suma de todas las respuestas cubre completamente la pregunta y el contexto requerido
   - No evalúes mensaje por mensaje, sino la conversación como un todo
   - Si el usuario expresa que no tiene nada más que agregar, quiere terminar o pasar a la siguiente pregunta, marca la respuesta como COMPLETADO

2. CRITERIOS DE VALIDACIÓN:
   - La pregunta debe estar respondida en la conversación
   - Todos los aspectos del Contexto Requerido deben estar cubiertos, si no hay contexto solo se debe responder la pregunta
   - Si el usuario indica explícitamente que quiere terminar o pasar a la siguiente pregunta, marca como COMPLETADO

3. CASO ESPECIAL (Solo para primera respuesta):
   - IMPORTANTE: Este caso especial SOLO aplica si el número de mensajes del usuario es 1
   - Si hay exactamente 1 mensaje del usuario:
     * Marca como "NS-NR" si el participante indica que no sabe o no quiere responder
     * Marca como "COMPLETADO" si el usuario expresa que no tiene nada más que agregar o quiere terminar
   - Si hay más de 1 mensaje del usuario, ignora este caso especial y evalúa según los criterios normales

4. Responde EXACTAMENTE con uno de estos formatos:
   a) "NS-NR: [explicación]" - Solo para primera respuesta (cuando hay exactamente 1 mensaje del usuario)
   b) "COMPLETADO: [explicación de cómo la conversación cubrió todo]"
   c) "INCOMPLETO: [lista de aspectos faltantes]"

"""
        )
        
        # Obtener toda la conversación en formato texto
        conversation = "\n".join([
            f"{'Entrevistador' if isinstance(msg, AIMessage) else 'Participante'}: {msg.content}"
            for msg in messages
            if not isinstance(msg, SystemMessage)
        ])
        
        # Número máximo de intentos
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Invocar el LLM para validar
                validation_result = llm.invoke([
                    system_message, 
                    HumanMessage(content=f"Conversación completa a analizar:\n{conversation}")
                ])
                print(f"Resultado de validación: {validation_result.content}")
                break
            except Exception as e:
                error_data = getattr(e, 'response', {}).json() if hasattr(e, 'response') else {}
                
                # Verificar si es un error de filtro de contenido
                if error_data.get('error', {}).get('code') == 'content_filter':
                    logger.info(f"[ERROR] --> Filtro de contenido activado (Intento {retry_count + 1}/{max_retries})")
                    
                    # Intentar refrasear el mensaje y obtener la respuesta del LLM
                    messages, success, llm_response = refrasear_mensaje(llm, messages, error_data, system_message)
                    
                    if success and llm_response:
                        validation_result = llm_response
                        break
                    elif success:
                        retry_count += 1
                        continue
                
                # Si no es un error de filtro de contenido o no se pudo refrasear, lanzar la excepción
                raise
        
        # Determinar el estado de la respuesta
        if "NS-NR:" in validation_result.content and len([m for m in messages if isinstance(m, HumanMessage)]) <= 1:
            # Solo asignar NS-NR si es la primera respuesta del usuario
            is_complete = "NS-NR"
        elif "COMPLETADO:" in validation_result.content:
            is_complete = True
        else:
            is_complete = False
        
        logger.info(f"Estado establecido a: {is_complete}")
        
        return {
            **state,
            "is_complete": is_complete,
            "validation_result": validation_result.content
        }
        
    except Exception as e:
        logger.error(f"Error en validate_response: {str(e)}")
        return state

def despedida_node(state: InterviewState) -> InterviewState:
    """Nodo que maneja los mensajes de despedida cuando la respuesta está completa."""
    try:
        llm = get_llm()
        current_question = state["current_question"]
        user_data = state.get("user_data", {})
        nombre_usuario = user_data.get("nombre", "").split()[0] if user_data and user_data.get("nombre") else ""
        language = state.get("language", "es")
        
        # Obtener el último mensaje del participante
        last_user_message = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                last_user_message = msg.content
                break
        
        # Crear un prompt para generar el mensaje de despedida
        despedida_prompt = SystemMessage(
            content=f"""Eres un entrevistador profesional y amigable. Tu tarea es generar un mensaje de despedida apropiado basado en la siguiente información:

IMPORTANTE SOBRE EL IDIOMA:
1. DEBES RESPONDER EN EL MISMO IDIOMA EN QUE ESTÁ FORMULADA LA PREGUNTA
2. Si la pregunta está en un idioma específico, usa ese idioma para todas tus respuestas
3. Solo si la pregunta no tiene un idioma claro, usa el idioma por defecto: {language.upper()}

PREGUNTA ACTUAL:
{current_question['question']}

NOMBRE DEL PARTICIPANTE:
{nombre_usuario}

ES LA ÚLTIMA PREGUNTA:
{"Sí" if current_question['question_number'] == current_question['total_questions'] else "No"}

ÚLTIMA RESPUESTA DEL PARTICIPANTE:
{last_user_message if last_user_message else "No hay respuesta previa"}

INSTRUCCIONES:
1. Genera un mensaje de despedida que:
   - Sea personalizado usando el nombre del participante
   - Mantenga un tono profesional y amigable haciendo un comentario breve sobre la ultima respuesta del participante
   - Si es la última pregunta, incluya una despedida final y un agradecimiento por su tiempo
   - Si no es la última pregunta, indique que se pasará a la siguiente pregunta y un agradecimiento por su tiempo
2. El mensaje debe ser conciso (1-2 lineas máximo)
3. No debe incluir preguntas ni solicitudes de información adicional
4. Debe sonar natural y conversacional
5. DEBES responder en el mismo idioma de la pregunta, o en {language.upper()} si no es claro

IMPORTANTE:
- DEBES responder con el mensaje de despedida directamente
- NO incluyas ningún otro texto o formato
"""
        )
        
        # Obtener el mensaje de despedida del LLM
        response = llm.invoke([despedida_prompt])
        
        return {
            **state,
            "messages": state["messages"] + [response],
            "is_complete": True
        }
        
    except Exception as e:
        logger.error(f"Error en despedida_node: {str(e)}")
        error_message = AIMessage(content=f"Lo siento, ha ocurrido un error: {str(e)}")
        return {
            **state,
            "messages": state["messages"] + [error_message],
            "is_complete": True
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
    
    # Añadir los nodos
    workflow.add_node("validate_response", validate_response)
    workflow.add_node("interviewer", interviewer_node)
    workflow.add_node("despedida", despedida_node)
    
    # Definir el flujo: START -> validate_response -> conditional_edge -> interviewer/despedida -> END
    workflow.add_edge(START, "validate_response")
    
    # Función para determinar el siguiente nodo basado en is_complete
    def route_based_on_completion(state: InterviewState) -> Literal["despedida", "interviewer"]:
        if state.get("is_complete") is True:
            return "despedida"
        return "interviewer"
    
    # Añadir el conditional_edge
    workflow.add_conditional_edges(
        "validate_response",
        route_based_on_completion,
        {
            "despedida": "despedida",
            "interviewer": "interviewer"
        }
    )
    
    # Añadir las conexiones a END
    workflow.add_edge("interviewer", END)
    workflow.add_edge("despedida", END)
    
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

async def run_interview_async(question: Dict = None, user_data: Dict = None, user_response: str = None, thread_id: str = "test-thread", description: str = "", language: str = "es"):
    """
    Función principal que ejecuta la entrevista de forma asíncrona.
    
    Args:
        question (Dict): Pregunta actual y su contexto, incluyendo question_number y total_questions
        user_data (Dict): Datos del usuario
        user_response (str): Respuesta del usuario si existe
        thread_id (str): ID del hilo de la entrevista
        description (str): Descripción general de la entrevista (opcional)
        language (str): Idioma en el que se realizará la entrevista (por defecto 'es')
        
    Returns:
        Dict: Resultados de la entrevista
    """
    try:
        # Formar el estado inicial
        state = {
            "messages": [HumanMessage(content=user_response)] if user_response else [],
            "current_question": {
                "question": question.get("question", "") if question else "",
                "context": question.get("context", "") if question else "",
                "question_number": question.get("question_number", 1) if question else 1,
                "total_questions": question.get("total_questions", 1) if question else 1
            },
            "is_complete": False,
            "validation_result": "",
            "user_data": user_data or {},
            "description": description,
            "language": language
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
            
            # Variable para almacenar el último estado de is_complete y validation_result
            last_is_complete = False
            last_validation_result = ""
            
            # Invocar el grafo con la configuración
            async for chunk in graph.astream(
                state,
                config
            ):
                
                # Procesar los chunks del agente
                chunk_result = process_chunks(chunk)
                processed_messages.extend(chunk_result["messages"])
                last_is_complete = chunk_result["is_complete"]
                last_validation_result = chunk_result["validation_result"]
                
            
            # Retornar el resultado usando la información de los chunks
            return {
                "status": "success",
                "thread_id": thread_id,
                "is_complete": last_is_complete,
                "validation_result": last_validation_result,
                "current_question": state["current_question"],
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

async def get_checkpoints(thread_id: str):
    """
    Obtiene los checkpoints de una entrevista específica.
    
    Args:
        thread_id (str): ID del hilo de la entrevista
        
    Returns:
        Dict: Diccionario con los checkpoints y el último checkpoint
    """
    try:
        # Obtener el checkpointer
        checkpointer, pool = await get_db_connection()
        
        try:
            # Configuración para el checkpointer
            config = {"configurable": {"thread_id": thread_id}}
            
            # Obtener los checkpoints
            checkpoints = checkpointer.alist(config)
            checkpoints_list = []
            
            # Procesar los checkpoints
            async for checkpoint in checkpoints:
                checkpoint_data = checkpoint.checkpoint
                channel_values = checkpoint_data["channel_values"]
                current_question = channel_values.get("current_question", {})
                
                checkpoints_list.append({
                    "id": checkpoint_data["id"],
                    "timestamp": checkpoint_data["ts"],
                    "is_complete": channel_values.get("is_complete", False),
                    "current_question": {
                        "question": current_question.get("question", ""),
                        "context": current_question.get("context", ""),
                        "question_number": current_question.get("question_number", 1),
                        "total_questions": current_question.get("total_questions", 1)
                    },
                    "messages": [
                        {
                            "role": "user" if isinstance(msg, HumanMessage) else "assistant",
                            "content": msg.content
                        }
                        for msg in channel_values.get("messages", [])
                        if not isinstance(msg, SystemMessage)
                    ]
                })
            
            # Encontrar el último checkpoint basado en el timestamp
            last_checkpoint = max(checkpoints_list, key=lambda x: x["timestamp"]) if checkpoints_list else None
            
            return {
                "status": "success",
                "thread_id": thread_id,
                "checkpoints": checkpoints_list,
                "last_checkpoint": last_checkpoint
            }
            
        finally:
            # Cerrar la conexión
            await pool.close()
            
    except Exception as e:
        logger.error(f"Error al obtener checkpoints: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        } 