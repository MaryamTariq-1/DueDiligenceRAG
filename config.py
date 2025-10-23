
import os
from dotenv import load_dotenv

path_env = "/content/DueDiligenceRAG/.env"
load_dotenv(path_env)

# ALL AVAILABLE APIs
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_AI_API_KEY")
GOOGLE_AI_STUDIO_API_KEY = os.getenv("GOOGLE_AI_STUDIO_API_KEY")

# Langfuse
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST")

# AWS
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
S3_BUCKET_NAME = "company-due-diligence-data-maryamtariq"
AWS_REGION = "eu-north-1"

# WORKING MODELS CONFIGURATION
MODELS = {
    # Groq Models (Fast & Free)
    "llama-3.1-8b-instant": {
        "provider": "groq",
        "api_key": GROQ_API_KEY,
        "base_url": "https://api.groq.com/openai/v1",
        "cost_input": 0.0,
        "cost_output": 0.0,
        "max_tokens": 2000,
        "description": "Llama 3.1 8B - Groq (Fastest)"
    },

    "llama-3.3-70b-versatile": {
        "provider": "groq",
        "api_key": GROQ_API_KEY,
        "base_url": "https://api.groq.com/openai/v1",
        "cost_input": 0.0,
        "cost_output": 0.0,
        "max_tokens": 2000,
        "description": "Llama 3.3 70B - Groq (Most Powerful)"
    },

    "mixtral-8x7b-32768": {
        "provider": "groq",
        "api_key": GROQ_API_KEY,
        "base_url": "https://api.groq.com/openai/v1",
        "cost_input": 0.0,
        "cost_output": 0.0,
        "max_tokens": 2000,
        "description": "Mixtral 8x7B - Groq (Good Balance)"
    },

    # DeepSeek Models (Free)
    "deepseek-chat": {
        "provider": "deepseek",
        "api_key": DEEPSEEK_API_KEY,
        "base_url": "https://api.deepseek.com/v1",
        "cost_input": 0.0,
        "cost_output": 0.0,
        "max_tokens": 2000,
        "description": "DeepSeek Chat - FREE"
    },

    # OpenRouter Free Models
    "mistralai/mistral-7b-instruct:free": {
        "provider": "openrouter",
        "api_key": OPENROUTER_API_KEY,
        "base_url": "https://openrouter.ai/api/v1",
        "cost_input": 0.0,
        "cost_output": 0.0,
        "max_tokens": 1000,
        "description": "Mistral 7B - OpenRouter FREE"
    },

    # Google AI Studio Models
    "gemini-1.5-flash": {
        "provider": "google",
        "api_key": GOOGLE_AI_STUDIO_API_KEY,
        "base_url": "https://generativelanguage.googleapis.com/v1beta/models/",
        "cost_input": 0.0,
        "cost_output": 0.0,
        "max_tokens": 2000,
        "description": "Gemini 1.5 Flash - Google"
    }
}

# Selected models for evaluation (4 models as per assignment)
EVALUATION_MODELS = [
    "llama-3.3-70b-versatile",  # Groq - Most powerful
    "deepseek-chat",            # DeepSeek - Free alternative
    "gemini-1.5-flash",         # Google - Different provider
    "mixtral-8x7b-32768"        # Groq - Good balance
]

# Prompt Styles
PROMPT_STYLES = ["factual", "analytical", "structured"]

# RAG Configuration
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K_RESULTS = 4
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# Evaluation Configuration
NUM_EVALUATION_QUESTIONS = 10