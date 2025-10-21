# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST")

# AWS
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
S3_BUCKET_NAME = "company-due-diligence-data-maryamtariq"
AWS_REGION = "eu-north-1"

# Model Configuration
MODELS = {
    "gpt-3.5-turbo": {  #  Start with cheapest option
        "provider": "openai",
        "cost_input": 0.0015,
        "cost_output": 0.002,
        "max_tokens": 1000
    }
}

# Prompt Styles
PROMPT_STYLES = {
    "factual": "Answer concisely based ONLY on the context. If unsure, say 'Not found in context'.",
    "analytical": "Analyze the context and provide reasoning for your answer.",
    "structured": "Provide answer in JSON format with keys: answer, confidence, evidence."
}

# RAG Configuration
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
TOP_K_RESULTS = 3
EMBEDDING_MODEL = "text-embedding-3-small"