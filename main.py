
from rag_pipeline import RAGPipeline
from evaluation import Evaluator
import pandas as pd
import time
from config import EVALUATION_MODELS, PROMPT_STYLES, NUM_EVALUATION_QUESTIONS


def main():
    print(" Due Diligence RAG Evaluation Framework")
    print("=" * 60)

    start_time = time.time()

    MODELS_TO_TEST = EVALUATION_MODELS

    PROMPT_TYPES = PROMPT_STYLES
    NUM_QUESTIONS = NUM_EVALUATION_QUESTIONS

    print("1. Initializing RAG Pipeline...")
    rag = RAGPipeline()

    if not rag.load_vector_store():
        print("Please run document_loader.py first to create the vector store")
        return

    print("2. Initializing Evaluator...")
    evaluator = Evaluator()

    ground_truth_df = evaluator.load_ground_truth_data()
    if ground_truth_df is None or len(ground_truth_df) == 0:
        print("No ground truth data available")
        return

    # Select questions for evaluation
    if len(ground_truth_df) > NUM_QUESTIONS:
        evaluation_questions = ground_truth_df.sample(NUM_QUESTIONS)['question'].tolist()
    else:
        evaluation_questions = ground_truth_df['question'].tolist()

    print(f"3. Running evaluation on {len(evaluation_questions)} questions...")
    print(f"   Models: {MODELS_TO_TEST}")
    print(f"   Prompt types: {PROMPT_TYPES}")
    total_runs = len(evaluation_questions) * len(MODELS_TO_TEST) * len(PROMPT_TYPES)
    print(f"   Total runs: {total_runs}")

    all_results = []

    for i, question in enumerate(evaluation_questions):
        print(f"\n[{i + 1}/{len(evaluation_questions)}] Processing: {question[:80]}...")

        # Run RAG pipeline for this question
        question_results = rag.run_single_evaluation(question, MODELS_TO_TEST, PROMPT_TYPES)
        all_results.extend(question_results)

        print(f"   Completed {len(question_results)} generations")

    print("\n4. Evaluating all results...")
    evaluated_df = evaluator.run_batch_evaluation(all_results)

    print("5. Generating summary statistics...")
    summary = evaluator.generate_summary_stats(evaluated_df)

    evaluated_df.to_csv('comprehensive_evaluation_results.csv', index=False)
    summary.to_csv('evaluation_summary.csv', index=False)

    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(" EVALUATION COMPLETED!")
    print("=" * 60)
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Total generations: {len(all_results)}")
    print(f"Results saved: comprehensive_evaluation_results.csv")
    print(f"Summary saved: evaluation_summary.csv")

    # Show top performers
    if not evaluated_df.empty:
        best_by_accuracy = evaluated_df.loc[evaluated_df['accuracy_score'].idxmax()]
        fastest = evaluated_df.loc[evaluated_df['latency'].idxmin()]
        cheapest = evaluated_df.loc[evaluated_df['cost'].idxmin()]

        print(f"\n BEST BY ACCURACY: {best_by_accuracy['model']} ({best_by_accuracy['prompt_type']})")
        print(f"   Accuracy: {best_by_accuracy['accuracy_score']:.3f}")
        print(f"   Latency: {best_by_accuracy['latency']:.2f}s")
        print(f"   Cost: ${best_by_accuracy['cost']:.6f}")

        print(f"\n FASTEST: {fastest['model']} ({fastest['prompt_type']})")
        print(f"   Latency: {fastest['latency']:.2f}s")

        print(f"\n CHEAPEST: {cheapest['model']} ({cheapest['prompt_type']})")
        print(f"   Cost: ${cheapest['cost']:.6f}")

    print(f"\n Check Langfuse dashboard for detailed traces and metrics!")
    print("   https://cloud.langfuse.com")


if __name__ == "__main__":
    main()