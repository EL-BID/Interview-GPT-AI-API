from urllib.request import Request
import azure.functions as func
import logging
import json
import asyncio
from interview_flow import run_interview_async
from db_connection import get_db_connection
from langchain_core.messages import HumanMessage

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

async def process_interview_request(req_body):
    """
    Función auxiliar que procesa la solicitud de entrevista.
    
    Args:
        req_body (dict): Cuerpo de la solicitud JSON
        
    Returns:
        tuple: (status_code, response_data)
    """
    try:
        # Verificar que thread_id esté presente (obligatorio)
        if 'thread_id' not in req_body:
            return 400, {"status": "error", "message": "thread_id es obligatorio"}
        
        thread_id = req_body['thread_id']
        logger.info(f"Procesando solicitud para thread_id: {thread_id}")
        
        # Obtener conexión a la base de datos
        logger.info("Obteniendo conexión a la base de datos...")
        pool, conn, memory = await get_db_connection()
        logger.info("Conexión a la base de datos establecida")
        
        try:
            # Formar el estado inicial
            state = {
                "messages": [],
                "current_question": req_body.get('question', {}),
                "is_complete": False,
                "question_number": 1,
                "total_questions": 1,
                "user_data": req_body.get('user_data', {})
            }
            logger.info(f"Estado inicial formado: {json.dumps(state)}")
            
            # Si hay user_response, actualizar el estado
            if 'user_response' in req_body:
                state["messages"] = [HumanMessage(content=req_body['user_response'])]
                logger.info(f"Respuesta del usuario agregada: {req_body['user_response']}")
            
            # Ejecutar la entrevista
            logger.info("Ejecutando entrevista...")
            result = await run_interview_async(
                state=state,
                thread_id=thread_id
            )
            logger.info(f"Entrevista ejecutada. Resultado: {json.dumps(result)}")
            
            # Verificar el resultado
            if result["status"] == "success":
                return 200, result
            else:
                return 500, result
        finally:
            # Cerrar la conexión
            logger.info("Cerrando conexión a la base de datos...")
            await conn.close()
            await pool.close()
            logger.info("Conexión a la base de datos cerrada")
        
    except Exception as e:
        logger.error(f"Error in process_interview_request: {str(e)}")
        return 500, {"status": "error", "message": str(e)}

async def get_interview_checkpoints(thread_id):
    """
    Función auxiliar que obtiene los checkpoints de una entrevista.
    
    Args:
        thread_id (str): ID del hilo de la entrevista
        
    Returns:
        tuple: (status_code, response_data)
    """
    try:
        # Configuración para el checkpointer
        config = {"configurable": {"thread_id": thread_id}} if thread_id else {}
        logger.info(f"Obteniendo checkpoints para thread_id: {thread_id}")
        
        # Obtener el checkpointer
        checkpointer = await get_db_connection()
        logger.info("Checkpointer inicializado correctamente")
        
        try:
            # Obtener los checkpoints usando el método alist
            checkpoints = []
            logger.info("Obteniendo checkpoints...")
            async for checkpoint in checkpointer.alist(config):
                checkpoints.append(checkpoint)
                logger.info(f"Checkpoint encontrado: {json.dumps(checkpoint)}")
            
            logger.info(f"Total de checkpoints encontrados: {len(checkpoints)}")
            return 200, {"status": "success", "checkpoints": checkpoints}
            
        except Exception as e:
            logger.error(f"Error obteniendo checkpoints: {str(e)}")
            return 500, {"status": "error", "message": str(e)}
        
    except Exception as e:
        logger.error(f"Error in get_interview_checkpoints: {str(e)}")
        return 500, {"status": "error", "message": str(e)}

@app.route(route="interview", methods=["POST"])
async def run_interview(req: Request) -> func.HttpResponse:
    """
    Función HTTP que maneja las solicitudes para ejecutar una entrevista.
    """
    try:
        logger.info("Recibida solicitud para run_interview")
        req_body = await req.json()
        logger.info(f"Cuerpo de la solicitud: {json.dumps(req_body)}")
        
        # Procesar la solicitud y obtener el resultado
        status_code, response_data = await process_interview_request(req_body)
        
        # Devolver la respuesta HTTP
        return func.HttpResponse(
            body=json.dumps(response_data),
            status_code=status_code,
            mimetype="application/json"
        )
        
    except Exception as e:
        logger.error(f"Error in run_interview: {str(e)}")
        return func.HttpResponse(
            body=json.dumps({"status": "error", "message": str(e)}),
            status_code=500,
            mimetype="application/json"
        )

@app.route(route="checkpoints", methods=["GET"])
async def get_checkpoints(req: Request) -> func.HttpResponse:
    """
    Función HTTP que maneja las solicitudes para obtener los checkpoints de una entrevista.
    """
    try:
        logger.info("Recibida solicitud para get_checkpoints")
        thread_id = req.params.get('thread_id')
        logger.info(f"thread_id: {thread_id}")
        
        if not thread_id:
            return func.HttpResponse(
                body=json.dumps({"status": "error", "message": "thread_id es obligatorio"}),
                status_code=400,
                mimetype="application/json"
            )
        
        status_code, response_data = await get_interview_checkpoints(thread_id)
        
        return func.HttpResponse(
            body=json.dumps(response_data),
            status_code=status_code,
            mimetype="application/json"
        )
        
    except Exception as e:
        logger.error(f"Error in get_checkpoints: {str(e)}")
        return func.HttpResponse(
            body=json.dumps({"status": "error", "message": str(e)}),
            status_code=500,
            mimetype="application/json"
        ) 