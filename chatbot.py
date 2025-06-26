# uvicorn fastapi_app2:app --reload --port 8001
 
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status # Import status for better clarity
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field # Import Field for better validation
from typing import List, Dict, Optional # Import Optional for optional fields
 
import google.generativeai as genai
import logging # Import logging
 
# --- Logging Configuration (New) ---
logging.basicConfig(level=logging.INFO) # Set logging level to INFO
logger = logging.getLogger(__name__) # Get a logger for this module
 
# Load environment variables from .env file
load_dotenv()
 
# --- Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY not found in environment variables. Please set it in your .env file.")
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please set it in your .env file.")
 
# Configure the Gemini API
genai.configure(api_key=GEMINI_API_KEY)
 
# Initialize the Gemini model
model = None # Initialize as None
try:
    model = genai.GenerativeModel('gemini-2.0-flash')
    logger.info("Gemini model 'gemini-2.0-flash' initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing Gemini model 'gemini-2.0-flash': {e}")
    # In a production app, you might want to gracefully degrade or indicate service unavailability
    # For now, it will raise an error if model is called when None
 
# --- FastAPI App Initialization ---
app = FastAPI(
    title="GeminiSpeak Backend",
    description="FastAPI backend for Real-Time Gemini Chat Assistant"
)
 
# --- CORS Middleware ---
# Adjust allow_origins to your Streamlit frontend URL in production for security
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],  # Specific Streamlit default port. Adjust if yours is different.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
# --- Pydantic Models for Request/Response ---
# Enhanced Pydantic models for better validation and clarity
class ChatPart(BaseModel):
    text: str = Field(..., description="The text content of a message part.")
 
class ChatMessage(BaseModel):
    role: str = Field(..., description="The role of the message sender (e.g., 'user', 'model').")
    parts: List[ChatPart] = Field(..., description="A list of content parts, typically text.")
 
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000, description="The user's new message.")
    chat_history: List[ChatMessage] = Field(..., description="The full conversation history for context.")
 
class ChatResponse(BaseModel):
    ai_message: str = Field(..., description="The AI's generated response.")
 
# --- API Endpoints ---
@app.post("/chat", response_model=ChatResponse)
async def chat_with_gemini(request: ChatRequest):
    """
    Receives a user message and chat history, sends it to Gemini,
    and returns the AI's response.
    """
    if not model:
        logger.error("Attempted to call chat endpoint but Gemini model was not initialized.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Gemini model not initialized. Backend service unavailable."
        )
 
    # Basic input validation for the new message
    if not request.message.strip():
        logger.warning(f"Received empty message from frontend.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message cannot be empty."
        )
 
    try:
        # Prepare the chat history in the format Gemini expects (list of dicts)
        # Ensure 'parts' contains dicts with 'text' key
        formatted_history = []
        for msg in request.chat_history:
            # Re-format parts to be list of dictionaries for the Gemini API call if needed
            # The google-generativeai library's start_chat method can usually handle
            # {"role": "user", "parts": ["text content"]} directly if the SDK is recent.
            # However, being explicit with {"text": "..."} in parts is safer.
            formatted_parts = [{"text": part.text} for part in msg.parts]
            formatted_history.append({"role": msg.role, "parts": formatted_parts})
 
 
        logger.info(f"Received message: '{request.message}' with history length: {len(formatted_history)}")
        # logger.debug(f"Formatted history for Gemini: {formatted_history}") # Uncomment for deeper debugging
 
        # Initialize chat with existing history
        convo = model.start_chat(history=formatted_history)
 
        # Send the new message to the conversation
        # Using send_message_async for non-blocking I/O with FastAPI
        response = await convo.send_message_async(request.message)
 
        # Access the text from the response
        # Robustly check if content and parts exist
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            ai_text = response.candidates[0].content.parts[0].text
        else:
            ai_text = "Sorry, I couldn't generate a coherent response."
            logger.warning(f"Gemini returned an empty or malformed response for message: '{request.message}'")
 
        logger.info(f"Gemini responded: '{ai_text[:50]}...'") # Log first 50 chars
        return ChatResponse(ai_message=ai_text)
 
    except genai.types.BlockedPromptException as e:
        logger.warning(f"Prompt blocked by safety settings: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Your prompt was blocked by safety settings. Please try rephrasing: {e}"
        )
    except genai.types.APIError as e:
        logger.error(f"Gemini API Error: {e.status_code} - {e.message}")
        raise HTTPException(
            status_code=e.status_code if hasattr(e, 'status_code') else status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Gemini API error: {e.message}"
        )
    except Exception as e:
        logger.error(f"Unexpected error in chat_with_gemini: {e}", exc_info=True) # Log full traceback
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected internal server error occurred: {e}"
        )
 
@app.get("/")
async def root():
    return {"message": "Welcome to GeminiSpeak Backend! Visit /docs for API documentation."}
 
# Optional: Endpoint to list models (useful for debugging, can be removed in production)
@app.get("/list_models")
async def list_available_models():
    """
    Lists all Gemini models available with the configured API key
    and their supported generation methods.
    """
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="GEMINI_API_KEY not set.")
 
    try:
        available_models = []
        for m in genai.list_models():
            if "generateContent" in m.supported_generation_methods:
                available_models.append({
                    "name": m.name,
                    "displayName": m.display_name,
                    "supported_methods": list(m.supported_generation_methods)
                })
        logger.info(f"Successfully listed {len(available_models)} models.")
        return {"models": available_models}
    except Exception as e:
        logger.error(f"Error listing models: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error listing models: {e}")
 