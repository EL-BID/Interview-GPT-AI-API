import os
import openai
import azure.functions as func
import logging
import json
import asyncio
from azurefunctions.extensions.http.fastapi import Request, StreamingResponse, JSONResponse
from interview_flow import run_interview_async, get_checkpoints

from dotenv import load_dotenv

load_dotenv()

endpoint = os.environ["AZURE_OPEN_AI_ENDPOINT"]
api_key = os.environ["AZURE_OPENAI_API_KEY"]
api_version = os.environ["AZURE_OPENAI_API_VERSION"]
api_instance_name = os.environ["AZURE_OPENAI_API_INSTANCE_NAME"]
base_path = os.environ["AZURE_OPENAI_API_BASE_PATH"]

# Azure Open AI
deployment = os.environ["AZURE_DEPLOYMENT_NAME"]

client = openai.AsyncAzureOpenAI(
    azure_endpoint=endpoint,
    api_key=api_key,
    api_version=api_version
)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

# Langgraph endpoints for user interview chat

@app.route(route="interview_chat", methods=["POST"])
async def run_interview(req: Request) -> JSONResponse:
    """
    HTTP function that handles requests to run an interview.
    """
    try:
        logger.info("Received request for run_interview")
        req_body = await req.json()
        logger.info(f"Request body: {json.dumps(req_body)}")
        
      
        if 'thread_id' not in req_body:
            return JSONResponse(
                content={"status": "error", "message": "thread_id is required"},
                status_code=400
            )
        
        thread_id = req_body['thread_id']
        logger.info(f"Processing request for thread_id: {thread_id}")
        
        try:
            # Execute the interview
            logger.info("Executing interview...")
            result = await run_interview_async(
                question=req_body.get('question'),
                user_data=req_body.get('user_data'),
                user_response=req_body.get('user_response'),
                thread_id=thread_id,
                description=req_body.get('description', ''),
                language=req_body.get('language', 'es')
            )
           
            
           
            status_code = 200 if result["status"] == "success" else 500
            return JSONResponse(
                content=result,
                status_code=status_code
            )
            
        except Exception as e:
            logger.error(f"Error in interview execution: {str(e)}")
            return JSONResponse(
                content={"status": "error", "message": str(e)},
                status_code=500
            )
        
    except Exception as e:
        logger.error(f"Error in run_interview: {str(e)}")
        return JSONResponse(
            content={"status": "error", "message": str(e)},
            status_code=500
        )

@app.route(route="checkpoints", methods=["GET"])
async def get_interview_checkpoints(req: Request) -> JSONResponse:
    """
    HTTP function that gets interview checkpoints.
    """
    try:
        logger.info("Received request to get checkpoints")
        
      
        thread_id = req.query_params.get('thread_id')
        if not thread_id:
            return JSONResponse(
                content={"status": "error", "message": "thread_id is required"},
                status_code=400
            )
        
        logger.info(f"Getting checkpoints for thread_id: {thread_id}")
        
      
        result = await get_checkpoints(thread_id)
        
     
        status_code = 200 if result["status"] == "success" else 500
        return JSONResponse(
            content=result,
            status_code=status_code
        )
        
    except Exception as e:
        logger.error(f"Error getting checkpoints: {str(e)}")
        return JSONResponse(
            content={"status": "error", "message": str(e)},
            status_code=500
        )


# AI Endpoints for interview results in user side and Admin interview results (sumary and chat with interview)

async def stream_processor(response):
    async for chunk in response:
        if len(chunk.choices) > 0:
            delta = chunk.choices[0].delta
            if delta.content: # Get remaining generated response if applicable
                await asyncio.sleep(0.1)
                yield delta.content

# HTTP streaming Azure Function
@app.route(route="interview-gpt-openai", methods=["POST"])
async def stream_openai_text(req: Request) -> StreamingResponse:
   
    # Get variables from the http request body
    
    try: 
        body = await req.json()
        
        prompt = body.get('prompt')
        temperature = body.get('temperature')
        
        logging.info(f'Python HTTP request body: {prompt}')
        
        azure_open_ai_response = await client.chat.completions.create(
            model=deployment,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )

        return StreamingResponse(stream_processor(azure_open_ai_response), media_type="text/event-stream")
    
    except Exception as e:
        logging.error(f"Error: {e}")
        return JSONResponse(
            content={"error": "Internal Server Error"},
            status_code=500
        )
        
@app.route(route="chat_ia_interview", methods=["POST"])
async def chat_ia_interview(req: Request) -> StreamingResponse:
    logging.info('Python HTTP trigger function processed a request.')

    try:
        body = await req.json()
        input_user = body.get('inputUser')
        system_message_param = body.get('systemMessage')
        interview_data = body.get('interviewData')
        message_history = body.get('messageHistory', [])
        temperature = body.get('temperature')

        if not input_user or not interview_data:
            return JSONResponse(
                content={"error": "Invalid input"},
                status_code=400
            )

        system_message = f"You're an assistant who's good at answering questions. Always consider the chat history when answering. Here's the context for this interview: {json.dumps(interview_data)}"
        
        if system_message_param:
            system_message += f" {system_message_param}"
        
        messages = [
            {"role": "system", "content": system_message},
            *message_history,
            {"role": "user", "content": input_user}
        ]

        response = await client.chat.completions.create(
            model=deployment,
            messages=messages,
            temperature=temperature,
            stream=True
        )

        return StreamingResponse(
            stream_processor(response),
            media_type="text/event-stream",
            status_code=200
        )

    except Exception as e:
        logging.error(f"Error: {e}")
        return JSONResponse(
            content={"error": "Internal Server Error"},
            status_code=500
        )
