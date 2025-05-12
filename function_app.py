from urllib.request import Request
import azure.functions as func
import logging
import json
import asyncio
from interview_flow import run_interview_async, get_checkpoints

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

@app.route(route="interview_chat", methods=["POST"])
async def run_interview(req: func.HttpRequest) -> func.HttpResponse:
    """
    Función HTTP que maneja las solicitudes para ejecutar una entrevista.
    """
    try:
        logger.info("Recibida solicitud para run_interview")
        req_body = req.get_json()
        logger.info(f"Cuerpo de la solicitud: {json.dumps(req_body)}")
        
        # Verificar que thread_id esté presente (obligatorio)
        if 'thread_id' not in req_body:
            return func.HttpResponse(
                body=json.dumps({"status": "error", "message": "thread_id es obligatorio"}),
                status_code=400,
                mimetype="application/json"
            )
        
        thread_id = req_body['thread_id']
        logger.info(f"Procesando solicitud para thread_id: {thread_id}")
        
        try:
            # Ejecutar la entrevista
            logger.info("Ejecutando entrevista...")
            result = await run_interview_async(
                question=req_body.get('question'),
                user_data=req_body.get('user_data'),
                user_response=req_body.get('user_response'),
                thread_id=thread_id,
                description=req_body.get('description', ''),
                language=req_body.get('language', 'es')
            )
           
            
            # Verificar el resultado y devolver la respuesta HTTP
            status_code = 200 if result["status"] == "success" else 500
            return func.HttpResponse(
                body=json.dumps(result),
                status_code=status_code,
                mimetype="application/json"
            )
            
        except Exception as e:
            logger.error(f"Error en la ejecución de la entrevista: {str(e)}")
            return func.HttpResponse(
                body=json.dumps({"status": "error", "message": str(e)}),
                status_code=500,
                mimetype="application/json"
            )
        
    except Exception as e:
        logger.error(f"Error en run_interview: {str(e)}")
        return func.HttpResponse(
            body=json.dumps({"status": "error", "message": str(e)}),
            status_code=500,
            mimetype="application/json"
        )

@app.route(route="checkpoints", methods=["GET"])
async def get_interview_checkpoints(req: func.HttpRequest) -> func.HttpResponse:
    """
    Función HTTP que obtiene los checkpoints de una entrevista.
    """
    try:
        logger.info("Recibida solicitud para obtener checkpoints")
        
        # Obtener el thread_id de los parámetros de la URL
        thread_id = req.params.get('thread_id')
        if not thread_id:
            return func.HttpResponse(
                body=json.dumps({"status": "error", "message": "thread_id es obligatorio"}),
                status_code=400,
                mimetype="application/json"
            )
        
        logger.info(f"Obteniendo checkpoints para thread_id: {thread_id}")
        
        # Obtener los checkpoints
        result = await get_checkpoints(thread_id)
        
        # Verificar el resultado y devolver la respuesta HTTP
        status_code = 200 if result["status"] == "success" else 500
        return func.HttpResponse(
            body=json.dumps(result),
            status_code=status_code,
            mimetype="application/json"
        )
        
    except Exception as e:
        logger.error(f"Error al obtener checkpoints: {str(e)}")
        return func.HttpResponse(
            body=json.dumps({"status": "error", "message": str(e)}),
            status_code=500,
            mimetype="application/json"
        )

