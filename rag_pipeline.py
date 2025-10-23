# rag_pipeline.py - COMPLETE
import boto3
import pandas as pd
from langfuse import Langfuse
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from config import *
import time
import json
from typing import List, Dict, Tuple


class RAGPipeline:
    def __init__(self):
        self.s3 = boto3.client('s3')
        self.embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        self.vector_store = None
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.langfuse = Langfuse(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_HOST
        )

    def load_vector_store(self):
        """Load existing vector store"""
        try:
            self.vector_store = FAISS.load_local("vector_store", self.embeddings, allow_dangerous_deserialization=True)
            print("Vector store loaded from disk")
            return True
        except Exception as e:
            print(f"Error loading vector store: {e}")
            return False

    def retrieve_context(self, query: str, k: int = TOP_K_RESULTS) -> Tuple[str, List[str]]:
        """Retrieve relevant context for query"""
        if not self.vector_store:
            return "No vector store available", []

        docs = self.vector_store.similarity_search(query, k=k)
        context = "\n\n".join([doc.page_content for doc in docs])
        sources = [doc.metadata.get('source', 'unknown') for doc in docs]

        print(f"Retrieved {len(docs)} context chunks")
        return context, sources

    def create_prompt(self, question: str, context: str, prompt_type: str = "factual") -> str:
        """Create different prompt types"""
        base_prompts = {
            "factual": f"""Based ONLY on the following context from company documents, provide a concise factual answer to the question. If the answer cannot be found in the context, state "I cannot find this information in the provided documents."

CONTEXT:
{context}

QUESTION: {question}

CONCISE FACTUAL ANSWER:""",

            "analytical": f"""Analyze the following context from company documents and provide a comprehensive answer with your reasoning and interpretation.

CONTEXT:
{context}

QUESTION: {question}

Please structure your response as follows:
1. ANALYSIS: Your interpretation of the relevant information
2. KEY INSIGHTS: Important findings from the context
3. CONCLUSION: Your final answer based on the analysis

RESPONSE:""",

            "structured": f"""Based on the context provided, answer the question using a structured JSON format.

CONTEXT:
{context}

QUESTION: {question}

Provide your answer as a valid JSON object with exactly these keys:
- "answer": main answer string
- "confidence": "high/medium/low" 
- "key_points": array of bullet points
- "supporting_evidence": array of evidence snippets

JSON RESPONSE:"""
        }
        return base_prompts.get(prompt_type, base_prompts["factual"])

    def call_model(self, model_name: str, prompt: str, prompt_type: str) -> Tuple[str, float, float]:
        """Call appropriate model based on provider"""
        start_time = time.time()

        try:
            model_config = MODELS[model_name]
            provider = model_config["provider"]

            # Start Langfuse trace
            trace = self.langfuse.trace(
                name=f"RAG_{provider}_{prompt_type}",
                user_id="evaluation",
                metadata={
                    "model": model_name,
                    "provider": provider,
                    "prompt_type": prompt_type,
                    "timestamp": time.time()
                },
                tags=[model_name, provider, prompt_type, "free_model"]
            )

            # Log generation
            generation = trace.generation(
                name=f"generation_{provider}_{prompt_type}",
                input=prompt,
                metadata={
                    "model": model_name,
                    "provider": provider,
                    "prompt_type": prompt_type,
                    "prompt_length": len(prompt)
                }
            )

            # Call appropriate provider
            if provider == "groq":
                client = OpenAI(
                    api_key=model_config["api_key"],
                    base_url=model_config["base_url"]
                )
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500
                )
                answer = response.choices[0].message.content

            elif provider == "openrouter":
                client = OpenAI(
                    api_key=model_config["api_key"],
                    base_url=model_config["base_url"]
                )
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500
                )
                answer = response.choices[0].message.content

            elif provider == "deepseek":
                client = OpenAI(
                    api_key=model_config["api_key"],
                    base_url=model_config["base_url"]
                )
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500
                )
                answer = response.choices[0].message.content

            latency = time.time() - start_time
            cost = 0.0  # All models are FREE

            # Update generation
            generation.update(
                output=answer,
                metadata={
                    "latency_seconds": latency,
                    "cost_usd": cost,
                    "output_length": len(answer)
                }
            )

            # Add scores
            trace.score(name="generation_success", value=1.0)
            trace.score(name="latency", value=latency)
            trace.score(name="cost", value=cost)

            self.langfuse.flush()

            return answer, latency, cost

        except Exception as e:
            error_latency = time.time() - start_time
            print(f"Error calling {model_name}: {e}")
            return f"Error: {str(e)}", error_latency, 0.0

    def evaluate_similarity(self, reference: str, generated: str) -> float:
        """Evaluate similarity using sentence transformers"""
        if not reference or not generated:
            return 0.0

        try:
            ref_embedding = self.similarity_model.encode([reference])
            gen_embedding = self.similarity_model.encode([generated])
            from sklearn.metrics.pairwise import cosine_similarity
            similarity = cosine_similarity(ref_embedding, gen_embedding)[0][0]
            return float(similarity)
        except Exception as e:
            print(f"Similarity calculation error: {e}")
            return 0.0


# Test function
def test_rag_pipeline():
    print("Testing RAG Pipeline with FREE Models")
    print("=" * 50)

    rag = RAGPipeline()

    if not rag.load_vector_store():
        print("Please run document_loader.py first!")
        return

    test_question = "What is the company's approach to financial planning?"

    print(f"Question: {test_question}")

    # Retrieve context
    context, sources = rag.retrieve_context(test_question)
    print(f"Retrieved context from {len(sources)} sources")

    # Test first 3 models
    test_models = list(MODELS.keys())[:3]

    for model_name in test_models:
        print(f"\nTesting: {model_name}")
        print("-" * 40)

        for prompt_type in PROMPT_STYLES:
            print(f"Testing {prompt_type} prompt...")

            prompt = rag.create_prompt(test_question, context, prompt_type)
            answer, latency, cost = rag.call_model(model_name, prompt, prompt_type)

            print(f"Answer preview: {answer[:80]}...")
            print(f"Latency: {latency:.2f}s")
            print(f"Cost: ${cost} (FREE)")
            print("Check Langfuse dashboard for trace!")


if __name__ == "__main__":
    test_rag_pipeline()