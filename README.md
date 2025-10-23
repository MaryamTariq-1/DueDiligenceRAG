
# Due Diligence RAG System

A comprehensive Retrieval-Augmented Generation (RAG) system for company due diligence analysis, evaluating multiple LLM providers with automated performance tracking.

## Features

- **Multi-Model RAG Pipeline**: Compare 4+ LLM providers (Groq, DeepSeek, Google, OpenRouter)
- **Automated Evaluation**: Accuracy, latency, and cost metrics with Langfuse integration
- **AWS S3 Integration**: Secure document storage and retrieval
- **Interactive Frontend**: Streamlit-based UI for easy interaction
- **Free Tier Models**: Optimized for cost-effective analysis

##  Project Structure

```
DueDiligenceRAG/
├── app.py                 # Streamlit frontend
├── main.py               # Main evaluation pipeline
├── rag_pipeline.py       # Core RAG functionality
├── evaluation.py         # Model evaluation & metrics
├── document_loader.py    # S3 document processing
├── config.py            # API keys & model configurations
├── aws_setup.py         # AWS S3 configuration
├── run_evaluation.py    # Complete evaluation runner
└── data/                # Processed documents & results
```

### 1. Installation
```bash
git clone https://github.com/MaryamTariq-1/DueDiligenceRAG.git
cd DueDiligenceRAG
pip install -r requirements.txt
```

### 2. Environment Setup
Create `.env` file with your API keys:
```env
GROQ_API_KEY=your_groq_key
DEEPSEEK_API_KEY=your_deepseek_key
GOOGLE_AI_STUDIO_API_KEY=your_google_key
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
```

### 3. Run Complete Evaluation
```bash
python run_evaluation.py
```

### 4. Launch Web Interface
```bash
streamlit run app.py
```

## Supported Models

| Provider | Models | Cost | Best For |
|----------|--------|------|----------|
| **Groq** | Llama 3.3 70B, Mixtral 8x7B | Free | Speed & Accuracy |
| **DeepSeek** | DeepSeek Chat | Free | Cost-effective |
| **Google** | Gemini 1.5 Flash | Free | Balanced Performance |
| **OpenRouter** | Mistral 7B | Free | Alternative Testing |

## Evaluation Metrics

- **Accuracy**: Cosine similarity with ground truth
- **Latency**: Response time in seconds
- **Cost**: API usage costs (all models free)
- **Prompt Styles**: Factual, Analytical, Structured responses

## Configuration

Modify `config.py` to:
- Add/remove models from evaluation
- Adjust chunk sizes and overlap
- Change prompt styles
- Configure AWS S3 bucket

## Results

The system generates:
- `comprehensive_evaluation_results.csv` - Detailed results
- `evaluation_summary.csv` - Model performance summary
- **Langfuse Dashboard** - Real-time tracing & analytics

## Use Cases

- Company financial due diligence
- Document Q&A systems
- Multi-model LLM benchmarking
- RAG pipeline optimization

## Cost Optimization

All configured models use free tiers, making this ideal for:
- Academic research
- Proof-of-concepts
- Budget-conscious projects
- Multi-model testing

---

**Built by MARYAM TARIQ, for efficient due diligence analysis with modern RAG techniques.**
