
import time
import os

def run_complete_evaluation():
    print(" DUE DILIGENCE RAG EVALUATION - COMPLETE FLOW")
    print("=" * 60)
    
    total_start_time = time.time()
    

    print("\n STEP 1: Document Ingestion")
    print("-" * 30)
    
   
    if os.path.exists('vector_store') and os.path.exists('data/training_data.csv'):
        print("✓ Vector store and training data already exist - Skipping document ingestion")
    else:
        print("Running document ingestion...")
        from document_loader import main as load_documents
        load_documents()
    
   
    print("\n🔍 STEP 2: Testing RAG Pipeline")
    print("-" * 30)
    
    try:
        from rag_pipeline import RAGPipeline
        from config import EVALUATION_MODELS, PROMPT_STYLES
        
        rag = RAGPipeline()
        if rag.load_vector_store():
            print("✓ Vector store loaded successfully")
             
            test_question = "What is the company's financial strategy?"
            print(f"Testing with: {test_question}")
            
            # Test first model
            test_model = EVALUATION_MODELS[0]
            test_prompt = PROMPT_STYLES[0]
            
            context, sources = rag.retrieve_context(test_question)
            prompt = rag.create_prompt(test_question, context, test_prompt)
            answer, latency, cost = rag.call_model(test_model, prompt, test_prompt)
            
            print(f"✓ Model test successful: {test_model}")
            print(f"  Answer preview: {answer[:100]}...")
            print(f"  Latency: {latency:.2f}s")
        else:
            print("✗ Failed to load vector store")
            return
    except Exception as e:
        print(f"✗ RAG test failed: {e}")
        return
    
    n
    print("\n STEP 3: Running Full Evaluation")
    print("-" * 30)
    
    try:
        from main import main as run_main_evaluation
        run_main_evaluation()
    except Exception as e:
        print(f"✗ Main evaluation failed: {e}")
        print("Trying alternative approach...")
        run_alternative_evaluation()
    
    total_time = time.time() - total_start_time
    print(f"\n COMPLETE EVALUATION FINISHED in {total_time:.2f} seconds!")
    print(" Check generated CSV files and Langfuse dashboard!")

def run_alternative_evaluation():
    """Alternative evaluation if main fails"""
    print("Running alternative evaluation...")
    
    from rag_pipeline import RAGPipeline
    from evaluation import Evaluator
    from config import EVALUATION_MODELS, PROMPT_STYLES, NUM_EVALUATION_QUESTIONS
    import pandas as pd
    
    
    rag = RAGPipeline()
    evaluator = Evaluator()
    
    if not rag.load_vector_store():
        return
    
    
    ground_truth_df = evaluator.load_ground_truth_data()
    if ground_truth_df is None:
        return
    
   
    test_questions = ground_truth_df.head(3)['question'].tolist()
    
    print(f"Testing with {len(test_questions)} questions, {len(EVALUATION_MODELS)} models, {len(PROMPT_STYLES)} prompt types")
    
    all_results = []
    
    for i, question in enumerate(test_questions):
        print(f"  Question {i+1}: {question[:60]}...")
        results = rag.run_single_evaluation(question, EVALUATION_MODELS, PROMPT_STYLES)
        all_results.extend(results)
    
   
    if all_results:
        evaluated_df = evaluator.run_batch_evaluation(all_results)
        evaluated_df.to_csv('quick_evaluation_results.csv', index=False)
        print(f"✓ Quick evaluation completed! {len(evaluated_df)} results saved")

if __name__ == "__main__":
    run_complete_evaluation()
