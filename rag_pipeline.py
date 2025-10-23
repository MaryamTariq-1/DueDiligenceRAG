import boto3
import pandas as pd
from langfuse import Langfuse
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import google.generativeai as genai
import cohere
from config import *
import time
import json
from typing import List, Dict, Tuple


class RAGPipeline:
    def __init__(self):
        self.s3 = boto3.client('s3')
        # FIXED: Use correct HuggingFace embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )
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
            print("✓ Vector store loaded from disk")
            return True
        except Exception as e:
            print(f"✗ Error loading vector store: {e}")
            return False

    def retrieve_context(self, query: str, k: int = TOP_K_RESULTS) -> Tuple[str, List[str]]:
        """Retrieve relevant context for query"""
        if not self.vector_store:
            return "No vector store available", []

        docs = self.vector_store.similarity_search(query, k=k)
        context = "\n\n".join([doc.page_content for doc in docs])
        sources = [doc.metadata.get('source', 'unknown') for doc in docs]

        print(f"   Retrieved {len(docs)} context chunks")
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

            "structured": f"""Based on the context provided, answer the question using a structured format.

CONTEXT:
{context}

QUESTION: {question}

Provide your answer in this structured format:
- MAIN ANSWER: [your main answer]
- CONFIDENCE: [high/medium/low]
- KEY POINTS: [bullet points]
- EVIDENCE: [supporting evidence from context]

STRUCTURED RESPONSE:"""
        }
        return base_prompts.get(prompt_type, base_prompts["factual"])

    def call_model(self, model_name: str, prompt: str, prompt_type: str) -> Tuple[str, float, float]:
        """Call appropriate model based on provider"""
        start_time = time.time()

        try:
            model_config = MODELS[model_name]
            provider = model_config["provider"]
            api_key = model_config["api_key"]

            if not api_key:
                return f"Error: No API key for {model_name}", 0.0, 0.0

            # Start Langfuse trace
            trace = self.langfuse.trace(
                name=f"RAG_{model_name}_{prompt_type}",
                user_id="evaluation",
                metadata={
                    "model": model_name,
                    "provider": provider,
                    "prompt_type": prompt_type
                }
            )

            # Log generation
            generation = trace.generation(
                name=f"generation_{model_name}",
                input=prompt,
                metadata={
                    "model": model_name,
                    "provider": provider,
                    "prompt_type": prompt_type
                }
            )

            answer = ""

            # Call appropriate provider
            if provider == "groq":
                client = OpenAI(
                    api_key=api_key,
                    base_url=model_config["base_url"]
                )
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=model_config["max_tokens"],
                    temperature=0.1
                )
                answer = response.choices[0].message.content

            elif provider == "deepseek":
                client = OpenAI(
                    api_key=api_key,
                    base_url=model_config["base_url"]
                )
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=model_config["max_tokens"],
                    temperature=0.1
                )
                answer = response.choices[0].message.content

            elif provider == "openrouter":
                client = OpenAI(
                    api_key=api_key,
                    base_url=model_config["base_url"]
                )
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=model_config["max_tokens"],
                    temperature=0.1
                )
                answer = response.choices[0].message.content

            elif provider == "google":
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel(model_name.split('/')[-1] if '/' in model_name else model_name)
                response = model.generate_content(prompt)
                answer = response.text

            latency = time.time() - start_time
            cost = 0.0  # All models are FREE

            # Update generation with output
            generation.update(
                output=answer,
                metadata={
                    "latency_seconds": latency,
                    "cost_usd": cost
                }
            )

            self.langfuse.flush()

            return answer, latency, cost

        except Exception as e:
            error_latency = time.time() - start_time
            error_msg = f"Error calling {model_name}: {str(e)}"
            print(f"   {error_msg}")
            return error_msg, error_latency, 0.0

    def run_single_evaluation(self, question: str, models: List[str], prompt_types: List[str]) -> List[Dict]:
        """Run evaluation for a single question across models and prompt types"""
        results = []

        print(f"   Processing: {question[:80]}...")

        # Retrieve context once per question
        context, sources = self.retrieve_context(question)

        for model_name in models:
            for prompt_type in prompt_types:
                print(f"     Testing {model_name} with {prompt_type} prompt...")

                prompt = self.create_prompt(question, context, prompt_type)
                answer, latency, cost = self.call_model(model_name, prompt, prompt_type)

                result = {
                    'question': question,
                    'model': model_name,
                    'prompt_type': prompt_type,
                    'generated_answer': answer,
                    'latency': latency,
                    'cost': cost,
                    'context_sources': sources
                }

                results.append(result)

        return results


# Test function
def test_rag_pipeline():
    print("Testing RAG Pipeline with Available Models")
    print("=" * 50)

    rag = RAGPipeline()

    if not rag.load_vector_store():
        print("Please run document_loader.py first!")
        return

    test_question = "What is the company's approach to financial planning?"

    print(f"Question: {test_question}")

    # Test with evaluation models
    for model_name in EVALUATION_MODELS[:2]:  # Test first 2 models
        print(f"\nTesting: {model_name}")
        print("-" * 40)

        for prompt_type in PROMPT_STYLES[:1]:  # Test only factual first
            print(f"Testing {prompt_type} prompt...")

            context, sources = rag.retrieve_context(test_question)
            prompt = rag.create_prompt(test_question, context, prompt_type)
            answer, latency, cost = rag.call_model(model_name, prompt, prompt_type)

            print(f"Answer: {answer[:100]}...")
            print(f"Latency: {latency:.2f}s")
            print(f"Cost: ${cost}")
            print("✓ Check Langfuse dashboard for trace!")


if __name__ == "__main__":
    test_rag_pipeline()