import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="Due Diligence RAG Analyzer",
    page_icon="-",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)


def load_evaluation_data():
    try:
        results_df = pd.read_csv('comprehensive_evaluation_results.csv')
        summary_df = pd.read_csv('evaluation_summary.csv')
        return results_df, summary_df
    except FileNotFoundError:
        st.error("Evaluation results not found. Please run the evaluation first.")
        return None, None


def main():
    st.markdown('<h1 class="main-header">Due Diligence RAG Analyzer</h1>', unsafe_allow_html=True)

    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose Mode",
        ["Dashboard", "Ask Questions", "Model Comparison", "Settings"]
    )

    if app_mode == "Dashboard":
        show_dashboard()
    elif app_mode == "Ask Questions":
        ask_questions()
    elif app_mode == "Model Comparison":
        model_comparison()
    elif app_mode == "Settings":
        show_settings()


def show_dashboard():
    st.header("Evaluation Dashboard")

    results_df, summary_df = load_evaluation_data()
    if results_df is None:
        return

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_questions = results_df['question'].nunique()
        st.metric("Total Questions", total_questions)

    with col2:
        total_models = results_df['model'].nunique()
        st.metric("Models Tested", total_models)

    with col3:
        total_evaluations = len(results_df)
        st.metric("Total Evaluations", total_evaluations)

    with col4:
        best_accuracy = results_df['accuracy_score'].max()
        st.metric("Best Accuracy", f"{best_accuracy:.2%}")

    st.subheader("Best Performing Model")
    best_result = results_df.loc[results_df['accuracy_score'].idxmax()]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Best Model</h4>
            <h3>{best_result['model']}</h3>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Best Prompt Style</h4>
            <h3>{best_result['prompt_type'].title()}</h3>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Accuracy Score</h4>
            <h3 style="color: #00aa00;">{best_result['accuracy_score']:.2%}</h3>
        </div>
        """, unsafe_allow_html=True)

    st.subheader("Performance Analysis")

    col1, col2 = st.columns(2)

    with col1:
        accuracy_chart_data = results_df.groupby(['model', 'prompt_type'])['accuracy_score'].mean().reset_index()
        fig = px.bar(accuracy_chart_data, x='model', y='accuracy_score', color='prompt_type',
                     title="Accuracy by Model and Prompt Type",
                     labels={'accuracy_score': 'Accuracy Score', 'model': 'Model'})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        latency_data = results_df.groupby('model')['latency'].mean().reset_index()
        fig = px.bar(latency_data, x='model', y='latency',
                     title="Average Latency by Model (seconds)",
                     labels={'latency': 'Latency (s)', 'model': 'Model'})
        st.plotly_chart(fig, use_container_width=True)


def ask_questions():
    st.header("Ask Due Diligence Questions")

    question = st.text_area(
        "Enter your due diligence question:",
        placeholder="e.g., What is the company's financial strategy?",
        height=100
    )

    col1, col2 = st.columns(2)

    with col1:
        model_choice = st.selectbox(
            "Select Model:",
            ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"]
        )

    with col2:
        prompt_style = st.selectbox(
            "Prompt Style:",
            ["factual", "analytical", "structured"]
        )

    if st.button("Get Answer", type="primary"):
        if question:
            with st.spinner("Retrieving information and generating answer..."):
                try:
                    from rag_pipeline import RAGPipeline

                    rag = RAGPipeline()
                    if rag.load_vector_store():
                        context, sources = rag.retrieve_context(question)
                        prompt = rag.create_prompt(question, context, prompt_style)
                        answer, latency, cost = rag.call_model(model_choice, prompt, prompt_style)

                        st.subheader("Answer:")
                        st.write(answer)

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Response Time", f"{latency:.2f}s")
                        with col2:
                            st.metric("Cost", f"${cost:.6f}")
                        with col3:
                            st.metric("Model", model_choice)

                        with st.expander("View Source Documents"):
                            st.write(f"Retrieved from {len(sources)} sources:")
                            for i, source in enumerate(sources, 1):
                                st.write(f"{i}. {source}")
                    else:
                        st.error("Failed to load RAG pipeline. Please check if vector store exists.")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter a question.")


def model_comparison():
    st.header("Model Comparison")

    results_df, summary_df = load_evaluation_data()
    if results_df is None:
        return

    working_results = results_df[results_df['accuracy_score'] > 0]

    if len(working_results) == 0:
        st.warning("No successful model evaluations found.")
        return

    st.subheader("Performance Metrics")

    metrics_df = working_results.groupby('model').agg({
        'accuracy_score': ['mean', 'std'],
        'latency': ['mean', 'std'],
        'cost': 'mean'
    }).round(4)

    metrics_df.columns = ['_'.join(col).strip() for col in metrics_df.columns.values]
    metrics_df = metrics_df.reset_index()

    st.dataframe(metrics_df, use_container_width=True)

    st.subheader("Detailed Results")
    st.dataframe(working_results[['question', 'model', 'prompt_type', 'accuracy_score', 'latency', 'cost']],
                 use_container_width=True)

    csv = working_results.to_csv(index=False)
    st.download_button(
        label="Download Full Results (CSV)",
        data=csv,
        file_name=f"due_diligence_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )


def show_settings():
    st.header("System Settings")

    st.subheader("API Configuration")
    st.info("API keys are managed in the .env file")

    st.subheader("Model Status")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Available Models:**
        - llama-3.3-70b-versatile (Groq)
        - llama-3.1-8b-instant (Groq)
        - mixtral-8x7b-32768 (Groq - Decommissioned)
        - deepseek-chat (Insufficient Balance)
        - gemini-1.5-flash (API Error)
        """)

    with col2:
        st.markdown("""
        **System Status:**
        - AWS S3 Connection
        - Vector Store
        - Langfuse Logging
        - Evaluation Pipeline
        """)

    st.subheader("Run Evaluation")
    if st.button("Run Full Evaluation", type="secondary"):
        with st.spinner("Running evaluation pipeline..."):
            try:
                from run_evaluation import run_complete_evaluation
                run_complete_evaluation()
                st.success("Evaluation completed successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Evaluation failed: {str(e)}")


if __name__ == "__main__":
    main()