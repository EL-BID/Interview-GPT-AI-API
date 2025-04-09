from urllib.request import Request
import azure.functions as func
import logging
import json
import asyncio
from interview_flow import run_interview_async

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
                thread_id=thread_id
            )
            logger.info(f"Entrevista ejecutada. Resultado: {json.dumps(result)}")
            
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

