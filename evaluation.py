
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langfuse import Langfuse
from config import *
from typing import Dict, List
import json


class Evaluator:
    def __init__(self):
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.langfuse = Langfuse(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_HOST
        )
        self.ground_truth_data = None

    def load_ground_truth_data(self) -> pd.DataFrame:
        """Load and return ground truth Q&A pairs"""
        try:
            # FIXED: Use correct path from document_loader
            df = pd.read_csv('data/training_data.csv')
            self.ground_truth_data = df
            print(f"✓ Loaded {len(df)} ground truth Q&A pairs")
            return df
        except Exception as e:
            print(f"✗ Error loading ground truth data: {e}")
            return None

    def get_reference_answer(self, question: str) -> str:
        """Get reference answer for a question"""
        if self.ground_truth_data is None:
            self.load_ground_truth_data()

        if self.ground_truth_data is not None:
            match = self.ground_truth_data[self.ground_truth_data['question'] == question]
            if not match.empty:
                return match.iloc[0]['answer']
        return ""

    def calculate_cosine_similarity(self, reference: str, generated: str) -> float:
        """Calculate cosine similarity between reference and generated answers"""
        if not reference or not generated:
            return 0.0

        try:
            # Generate embeddings
            ref_embedding = self.similarity_model.encode([reference])
            gen_embedding = self.similarity_model.encode([generated])

            # Calculate cosine similarity
            similarity = cosine_similarity(ref_embedding, gen_embedding)[0][0]
            return float(similarity)
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0.0

    def evaluate_and_log(self, result: Dict) -> float:
        """Evaluate a single result and log to Langfuse"""
        question = result['question']
        reference_answer = self.get_reference_answer(question)
        generated_answer = result['generated_answer']
        model_name = result['model']
        prompt_type = result['prompt_type']

        if not reference_answer:
            print(f"   Warning: No reference answer found for question: {question[:50]}...")
            return 0.0

        # Calculate accuracy score
        accuracy_score = self.calculate_cosine_similarity(reference_answer, generated_answer)

        # Log evaluation to Langfuse
        trace = self.langfuse.trace(
            name=f"Accuracy_Evaluation_{model_name}_{prompt_type}",
            user_id="accuracy_evaluator",
            metadata={
                "model": model_name,
                "prompt_type": prompt_type,
                "question": question,
                "reference_answer": reference_answer,
                "generated_answer": generated_answer[:500]  # First 500 chars
            }
        )

        # Log accuracy score
        trace.score(
            name="accuracy",
            value=accuracy_score,
            comment=f"Cosine similarity score for {model_name} with {prompt_type} prompt"
        )

        # Log other metrics
        trace.score(
            name="latency_seconds",
            value=result['latency']
        )

        trace.score(
            name="cost_usd",
            value=result['cost']
        )

        self.langfuse.flush()

        return accuracy_score

    def run_batch_evaluation(self, rag_results: List[Dict]) -> pd.DataFrame:
        """Evaluate all RAG results and return comprehensive results"""
        print("🧮 Starting batch evaluation...")

        evaluated_results = []

        for i, result in enumerate(rag_results):
            print(f"  Evaluating result {i + 1}/{len(rag_results)}...")

            # Add accuracy score to result
            accuracy = self.evaluate_and_log(result)
            result['accuracy_score'] = accuracy

            evaluated_results.append(result)

        # Create DataFrame
        results_df = pd.DataFrame(evaluated_results)

        # Save comprehensive results
        results_df.to_csv('comprehensive_evaluation_results.csv', index=False)
        print(f"✓ Evaluation completed! Saved {len(results_df)} results")

        return results_df

    def generate_summary_stats(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Generate summary statistics for results"""
        summary = results_df.groupby(['model', 'prompt_type']).agg({
            'accuracy_score': ['mean', 'std', 'count'],
            'latency': ['mean', 'std'],
            'cost': ['mean', 'sum']
        }).round(4)

        summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
        summary = summary.reset_index()

        return summary