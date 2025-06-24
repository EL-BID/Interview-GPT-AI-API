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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InterviewState(TypedDict):
    """Interview state."""
    messages: Annotated[List[BaseMessage], add_messages]  # Message history
    current_question: Dict  # Current question and its context
    is_complete: bool  # Indicates if the question is completed
    validation_result: str  # Response validation result
    question_number: int  # Current question number
    total_questions: int  # Total questions
    user_data: Optional[Dict[str, Any]]  # User data
    description: str  # General interview description (optional)
    language: str  # Language in which the interview will be conducted (default 'es')

def get_llm():
    """LLM configuration."""
    return AzureChatOpenAI(
        deployment_name=os.getenv("AZURE_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=0.0
    )

def process_chunks(chunk: Dict) -> Dict:
    """
    Processes agent chunks and extracts relevant messages.
    
    Args:
        chunk (Dict): Agent data chunk
        
    Returns:
        Dict: Dictionary with processed messages and is_complete value
    """
    processed_messages = []
    
    # Check if chunk has expected structure
    if "interviewer" in chunk and "messages" in chunk["interviewer"]:
        for message in chunk["interviewer"]["messages"]:
            # Only process messages that are not SystemMessage
            if not isinstance(message, SystemMessage):
                processed_messages.append({
                    "role": "assistant" if isinstance(message, AIMessage) else "user",
                    "content": message.content
                })
    elif "farewell" in chunk and "messages" in chunk["farewell"]:
        for message in chunk["farewell"]["messages"]:
            # Only process messages that are not SystemMessage
            if not isinstance(message, SystemMessage):
                processed_messages.append({
                    "role": "assistant" if isinstance(message, AIMessage) else "user",
                    "content": message.content
                })
    
    # Extract is_complete and validation_result values from chunk
    is_complete = chunk.get("interviewer", {}).get("is_complete", False) or chunk.get("farewell", {}).get("is_complete", False)
    validation_result = chunk.get("interviewer", {}).get("validation_result", "") or chunk.get("farewell", {}).get("validation_result", "")
    
    print(f"is_complete value in process_chunks: {is_complete}")
    print(f"validation_result value in process_chunks: {validation_result}")
    
    return {
        "messages": processed_messages,
        "is_complete": is_complete,
        "validation_result": validation_result
    }

def rephrase_message(llm, messages, error_data, system_message=None):
    """
    Helper function to rephrase messages when content filter error is detected.
    
    Args:
        llm: Language model instance
        messages: List of messages to process
        error_data: Content filter error data
        system_message: System message with original instructions (optional)
        
    Returns:
        tuple: (updated messages, success)
    """
    try:
        # Get filter results
        filter_result = error_data.get('error', {}).get('innererror', {}).get('content_filter_result', {})
        
        # Check which categories triggered the filter
        triggered_categories = []
        for category, result in filter_result.items():
            if result.get('filtered', False):
                severity = result.get('severity', 'unknown')
                triggered_categories.append((category, severity))
        
        if triggered_categories:
            # If there's a system_message, try to rephrase it first
            if system_message:
                try:
                    # Prompt for system_message rephrasing
                    rephrase_system_prompt = SystemMessage(content=f"""Please reformulate the following instructions in a more appropriate way, maintaining the same meaning and professional tone but avoiding any content that could be considered inappropriate:

Original instructions: {system_message.content}

Reformulate the instructions so that:
1. Maintains the same style and professional tone
2. Avoids any language that could be considered inappropriate
3. Preserves the same objective and meaning of the original instructions
4. Is clear and direct, maintaining a professional tone""")
                    
                    # Get the rephrased version of the system_message
                    rephrased_system_response = llm.invoke([rephrase_system_prompt])
                    rephrased_system_content = rephrased_system_response.content
                    
                    # Update the system_message
                    system_message = SystemMessage(content=rephrased_system_content)
                    logger.info(f"Rephrased system prompt: {rephrased_system_content}")
                except Exception as e:
                    logger.error(f"Error rephrasing system prompt: {str(e)}")
                    return messages, False, None
            
            # Get the last user message
            last_user_message = None
            for msg in reversed(messages):
                if isinstance(msg, HumanMessage):
                    last_user_message = msg.content
                    break
            
            if last_user_message:
                # Prompt for user message rephrasing
                rephrase_prompt = SystemMessage(content=f"""Please reformulate the following response in a more appropriate way, maintaining the same meaning and conversational tone but avoiding any content that could be considered inappropriate:

Original response: {last_user_message}

Reformulate the response so that:
1. Maintains the same style and tone of the original response
2. Avoids any language that could be considered inappropriate
3. Preserves the same objective and meaning of the original response
4. Is natural and conversational, without being excessively formal""")
                
                # Get the rephrased version
                rephrased_response = llm.invoke([rephrase_prompt])
                rephrased_message = rephrased_response.content
                
                # Update the message in the list
                for i, msg in enumerate(messages):
                    if isinstance(msg, HumanMessage) and msg.content == last_user_message:
                        messages[i] = HumanMessage(content=rephrased_message)
                        break
                
                logger.info(f"Rephrased response: {rephrased_message}")
                
                # If a system_message was provided, try the LLM call with the original instructions
                if system_message:
                    try:
                        # Make sure the system_message is at the beginning
                        if not isinstance(messages[0], SystemMessage):
                            messages = [system_message] + messages
                        elif messages[0].content != system_message.content:
                            messages[0] = system_message
                        
                        # Try the LLM call with updated messages
                        response = llm.invoke(messages)
                        logger.info(f"LLM call successful after rephrasing")
                        return messages, True, response
                    except Exception as e:
                        logger.error(f"Error calling LLM after rephrasing: {str(e)}")
                        return messages, False, None
                
                return messages, True, None
        
        return messages, False, None
        
    except Exception as e:
        logger.error(f"Error rephrasing message: {str(e)}")
        return messages, False, None

def interviewer_node(state: InterviewState) -> InterviewState:
    """Main node that handles the interview."""
    try:
        llm = get_llm()
        current_question = state["current_question"]
        user_data = state.get("user_data", {})
        user_name = user_data.get("user_name", "").split()[0] if user_data and user_data.get("user_name") else ""
        is_complete = state.get("is_complete", False)
        description = state.get("description", "")
        language = state.get("language", "es")
        
        logger.info(f"is_complete state in interviewer_node: {is_complete}")
        
        # Prepare participant section
        participant_section = f"PARTICIPANT:\nName: {user_name}\n" if user_name else ""
        
        # Determine current state for prompt
        current_state = "COMPLETED" if is_complete is True else "DOESN'T KNOW/DOESN'T RESPOND" if is_complete == "NS-NR" else "INCOMPLETE"
        
        # System prompt
        system_message = SystemMessage(
            content=f"""You are a professional, friendly and approachable interviewer. Your goal is to make the participant feel comfortable while getting a complete answer to the question.

IMPORTANT ABOUT LANGUAGE:
1. YOU MUST RESPOND IN THE SAME LANGUAGE IN WHICH THE QUESTION IS FORMULATED
2. If the question is in a specific language, use that language for all your responses
3. Only if the question doesn't have a clear language, use the default language: {language.upper()}

{participant_section}
CURRENT QUESTION:
{current_question['question']}

SPECIFIC CONTEXT TO EXPLORE ABOUT CURRENT QUESTION:
{current_question['context']}

INTERVIEW INFORMATION:
This is an interview about {description}
Current question: {current_question['question_number']} of {current_question['total_questions']}
Response status: {current_state}
is_complete value: {is_complete}

INSTRUCTIONS:
1. If available, use the participant's name to make the conversation more personal
2. Maintain a professional, friendly and approachable tone
3. Do not give opinions, suggestions or interpret responses
4. Show genuine interest in the participant's responses
5. NEVER close the conversation until the response is complete or the participant indicates they don't know/don't want to respond
6. Keep the conversation active with relevant follow-up questions about the topic of {description}
   - IN EACH MESSAGE, ask ONLY ONE or TWO follow-up questions
   - Prioritize the most relevant question for the context
7. DO NOT DEVIATE FROM THE CURRENT QUESTION AND CONTEXT:
   - If the participant asks questions out of the current question and context, respond kindly that you will focus on their response first
   - If the participant mentions irrelevant topics, gently redirect them to the original question
   - If the participant wants to change topics, kindly indicate that you first need their response about the current question
   - Keep focus on getting a complete response about the current question and context

RESPONSE HANDLING:
1. If the response is incoherent or unrelated:
   - Kindly point out that the response is not related
   - Reformulate the question more clearly
   - Ask a specific follow-up question about the context

2. If the response is incomplete (is_complete = False) according to {is_complete}:
   - Ask natural follow-up questions about the context and topic of {description}
   - Ask for details and concrete examples
   - Explore relevant aspects not mentioned
   - Continue the conversation until you get a complete response

3. If the participant doesn't know or doesn't want to respond (is_complete = NS-NR) according to {is_complete}:
   - If it's the last question:
     * Thank them for participating and say goodbye kindly
   - If it's not the last question:
     * Kindly indicate that we will move to the next question

IMPORTANT:
- Keep the conversation active until you get a complete response
- Do not close the conversation prematurely
- Ask relevant and specific follow-up questions about the topic of {description}
- Guide the participant constructively toward a complete response
- DO NOT ALLOW DEVIATIONS FROM THE CURRENT QUESTION AND CONTEXT
- RESPOND IN THE SAME LANGUAGE AS THE QUESTION, or in {language.upper()} if not clear
"""
        )
        
        # If there are no previous messages or the first message is not the system message
        if not state["messages"] or not isinstance(state["messages"][0], SystemMessage):
            state["messages"] = [system_message] + state["messages"]
        
        # Maximum number of attempts
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Get LLM response
                response = llm.invoke(state["messages"])
              
                break
            except Exception as e:
                error_data = getattr(e, 'response', {}).json() if hasattr(e, 'response') else {}
                
                # Check if it's a content filter error
                if error_data.get('error', {}).get('code') == 'content_filter':
                    
                    
                    # Try to rephrase the message and get LLM response
                    state["messages"], success, llm_response = rephrase_message(llm, state["messages"], error_data, system_message)
                    
                    if success and llm_response:
                        response = llm_response
                        break
                    elif success:
                        retry_count += 1
                        continue
                
                # If it's not a content filter error or couldn't rephrase, raise the exception
                raise
        
        return {
            **state,
            "messages": state["messages"] + [response]
        }
        
    except Exception as e:
        logger.error(f"Error in interviewer_node: {str(e)}")
        error_message = AIMessage(content=f"Sorry, an error has occurred: {str(e)}")
        return {
            **state,
            "messages": state["messages"] + [error_message]
        }

def validate_response(state: InterviewState) -> InterviewState:
    """Node that validates if the response is complete according to context, considering the entire conversation."""
    try:
        llm = get_llm()
        current_question = state["current_question"]
        messages = state["messages"]
        print(f"current_question: {current_question['context']}")
        
        # Only validate if there are user messages
        if not messages or not any(isinstance(msg, HumanMessage) for msg in messages):
            return state
            
        # Calculate number of user messages
        user_messages_count = len([msg for msg in messages if isinstance(msg, HumanMessage)])
            
        # System prompt for validation
        system_message = SystemMessage(
            content=f"""You are an expert analyst in evaluating responses. Your task is to analyze the ENTIRE conversation to determine if ALL required aspects of both the question and context have been covered.

====================================================================
QUESTION TO EVALUATE:
{current_question['question']}
====================================================================

====================================================================
REQUIRED CONTEXT:
{current_question['context']}
====================================================================

====================================================================
CONVERSATION INFORMATION:
Number of user messages: {user_messages_count}
====================================================================

INSTRUCTIONS:

1. CONVERSATION EVALUATION:
   - Analyze ALL messages together, including responses and follow-up questions
   - Evaluate if the sum of all responses completely covers the question and required context
   - Do not evaluate message by message, but the conversation as a whole
   - If the user expresses that they have nothing more to add, want to finish or move to the next question, mark the response as COMPLETED

2. VALIDATION CRITERIA:
   - The question must be answered in the conversation
   - All aspects of the Required Context must be covered, if there's no context only the question should be answered
   - If the user explicitly indicates they want to finish or move to the next question, mark as COMPLETED

3. SPECIAL CASE (Only for first response):
   - IMPORTANT: This special case ONLY applies if the number of user messages is 1
   - If there are exactly 1 user message:
     * Mark as "NS-NR" if the participant indicates they don't know or don't want to respond
     * Mark as "COMPLETED" if the user expresses they have nothing more to add or want to finish
   - If there are more than 1 user message, ignore this special case and evaluate according to normal criteria

4. Respond EXACTLY with one of these formats:
   a) "NS-NR: [explanation]" - Only for first response (when there are exactly 1 user message)
   b) "COMPLETED: [explanation of how the conversation covered everything]"
   c) "INCOMPLETE: [list of missing aspects]"

"""
        )
        
        # Get the entire conversation in text format
        conversation = "\n".join([
            f"{'Interviewer' if isinstance(msg, AIMessage) else 'Participant'}: {msg.content}"
            for msg in messages
            if not isinstance(msg, SystemMessage)
        ])
        
        # Maximum number of attempts
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Invoke LLM for validation
                validation_result = llm.invoke([
                    system_message, 
                    HumanMessage(content=f"Complete conversation to analyze:\n{conversation}")
                ])
                print(f"Validation result: {validation_result.content}")
                break
            except Exception as e:
                error_data = getattr(e, 'response', {}).json() if hasattr(e, 'response') else {}
                
                # Check if it's a content filter error
                if error_data.get('error', {}).get('code') == 'content_filter':
                    logger.info(f"[ERROR] --> Content filter activated (Attempt {retry_count + 1}/{max_retries})")
                    
                    # Try to rephrase the message and get LLM response
                    messages, success, llm_response = rephrase_message(llm, messages, error_data, system_message)
                    
                    if success and llm_response:
                        validation_result = llm_response
                        break
                    elif success:
                        retry_count += 1
                        continue
                
                # If it's not a content filter error or couldn't rephrase, raise the exception
                raise
        
        # Determine response status
        if "NS-NR:" in validation_result.content and len([m for m in messages if isinstance(m, HumanMessage)]) <= 1:
            # Only assign NS-NR if it's the user's first response
            is_complete = "NS-NR"
        elif "COMPLETED:" in validation_result.content:
            is_complete = True
        else:
            is_complete = False
        
        logger.info(f"State set to: {is_complete}")
        
        return {
            **state,
            "is_complete": is_complete,
            "validation_result": validation_result.content
        }
        
    except Exception as e:
        logger.error(f"Error in validate_response: {str(e)}")
        return state

def farewell_node(state: InterviewState) -> InterviewState:
    """Node that handles farewell messages when the response is complete."""
    try:
        llm = get_llm()
        current_question = state["current_question"]
        user_data = state.get("user_data", {})
        user_name = user_data.get("user_name", "").split()[0] if user_data and user_data.get("user_name") else ""
        language = state.get("language", "es")
        
        # Get the last participant message
        last_user_message = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                last_user_message = msg.content
                break
        
        # Prompt to generate the farewell message
        farewell_prompt = SystemMessage(
            content=f"""You are a professional and friendly interviewer. Your task is to generate an appropriate farewell message based on the following information:

IMPORTANT ABOUT LANGUAGE:
1. YOU MUST RESPOND IN THE SAME LANGUAGE IN WHICH THE QUESTION IS FORMULATED
2. If the question is in a specific language, use that language for all your responses
3. Only if the question doesn't have a clear language, use the default language: {language.upper()}

CURRENT QUESTION:
{current_question['question']}

PARTICIPANT NAME:
{user_name}

IS THE LAST QUESTION:
{"Yes" if current_question['question_number'] == current_question['total_questions'] else "No"}

LAST PARTICIPANT RESPONSE:
{last_user_message if last_user_message else "No previous response"}

INSTRUCTIONS:
1. Generate a farewell message that:
   - Is personalized using the participant's name
   - Maintains a professional and friendly tone making a brief comment about the participant's last response
   - If it's the last question, include a final farewell and thanks for their time
   - If it's not the last question, indicate that we will move to the next question and thanks for their time
2. The message should be concise (1-2 lines maximum)
3. Should not include questions or requests for additional information
4. Should sound natural and conversational
5. YOU MUST respond in the same language as the question, or in {language.upper()} if not clear

IMPORTANT:
- YOU MUST respond with the farewell message directly
- DO NOT include any other text or format
"""
        )
        
        # Get the farewell message from LLM
        response = llm.invoke([farewell_prompt])
        
        return {
            **state,
            "messages": state["messages"] + [response],
            "is_complete": True
        }
        
    except Exception as e:
        logger.error(f"Error in farewell_node: {str(e)}")
        error_message = AIMessage(content=f"Sorry, an error has occurred: {str(e)}")
        return {
            **state,
            "messages": state["messages"] + [error_message],
            "is_complete": True
        }

def build_graph(checkpointer=None) -> StateGraph:
    """
    Builds the interview graph.
    
    Args:
        checkpointer: The graph checkpointer
        
    Returns:
        StateGraph: The compiled graph
    """
    workflow = StateGraph(InterviewState)
    
    # Add nodes
    workflow.add_node("validate_response", validate_response)
    workflow.add_node("interviewer", interviewer_node)
    workflow.add_node("farewell", farewell_node)
    
    # Define flow: START -> validate_response -> conditional_edge -> interviewer/farewell -> END
    workflow.add_edge(START, "validate_response")
    
    # Function to determine next node based on is_complete
    def route_based_on_completion(state: InterviewState) -> Literal["farewell", "interviewer"]:
        if state.get("is_complete") is True:
            return "farewell"
        return "interviewer"
    
    # Add conditional_edge
    workflow.add_conditional_edges(
        "validate_response",
        route_based_on_completion,
        {
            "farewell": "farewell",
            "interviewer": "interviewer"
        }
    )
    
    # Add connections to END
    workflow.add_edge("interviewer", END)
    workflow.add_edge("farewell", END)
    
    return workflow.compile(checkpointer=checkpointer)

def get_interview_graph(checkpointer=None):
    """
    Gets the compiled graph.
    
    Args:
        checkpointer: The graph checkpointer
        
    Returns:
        StateGraph: The compiled graph
    """
    graph = build_graph(checkpointer)
    return graph

async def run_interview_async(question: Dict = None, user_data: Dict = None, user_response: str = None, thread_id: str = "test-thread", description: str = "", language: str = "es"):
    """
    Main function that runs the interview asynchronously.
    
    Args:
        question (Dict): Current question and its context, including question_number and total_questions
        user_data (Dict): User data
        user_response (str): User response if exists
        thread_id (str): Interview thread ID
        description (str): General interview description (optional)
        language (str): Language in which the interview will be conducted (default 'es')
        
    Returns:
        Dict: Interview results
    """
    try:
        # Form initial state
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
        
        # Configuration for checkpointer
        config = {"configurable": {"thread_id": thread_id}}
        
        # Get checkpointer
        checkpointer, pool = await get_db_connection()
        
        try:
            # Get graph with checkpointer
            graph = get_interview_graph(checkpointer=checkpointer)
            
            # List to store processed messages
            processed_messages = []
            
            # Variable to store last is_complete and validation_result state
            last_is_complete = False
            last_validation_result = ""
            
            # Invoke graph with configuration
            async for chunk in graph.astream(
                state,
                config
            ):
                
                # Process agent chunks
                chunk_result = process_chunks(chunk)
                processed_messages.extend(chunk_result["messages"])
                last_is_complete = chunk_result["is_complete"]
                last_validation_result = chunk_result["validation_result"]
                
            
            # Return result using chunk information
            return {
                "status": "success",
                "thread_id": thread_id,
                "is_complete": last_is_complete,
                "validation_result": last_validation_result,
                "current_question": state["current_question"],
                "messages": processed_messages
                
            }
        finally:
            # Close connection
            await pool.close()
            
    except Exception as e:
        logger.error(f"Error in run_interview: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }

async def get_checkpoints(thread_id: str):
    """
    Gets checkpoints for a specific interview.
    
    Args:
        thread_id (str): Interview thread ID
        
    Returns:
        Dict: Dictionary with checkpoints and last checkpoint
    """
    try:
        # Get checkpointer
        checkpointer, pool = await get_db_connection()
        
        try:
            # Configuration for checkpointer
            config = {"configurable": {"thread_id": thread_id}}
            
            # Get checkpoints
            checkpoints = checkpointer.alist(config)
            checkpoints_list = []
            
            # Process checkpoints
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
            
            # Find last checkpoint based on timestamp
            last_checkpoint = max(checkpoints_list, key=lambda x: x["timestamp"]) if checkpoints_list else None
            
            return {
                "status": "success",
                "thread_id": thread_id,
                "checkpoints": checkpoints_list,
                "last_checkpoint": last_checkpoint
            }
            
        finally:
            # Close connection
            await pool.close()
            
    except Exception as e:
        logger.error(f"Error getting checkpoints: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        } 