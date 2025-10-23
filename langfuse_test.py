# lang_test.py
import os
from dotenv import load_dotenv

load_dotenv()


def test_langfuse_connection():
    print("Testing Langfuse Connection with NEW API Keys...")

    # Check if keys are loaded
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST")

    print(f"Public Key: {public_key}" if public_key else "MISSING")
    print(f"Secret Key: {secret_key[:15]}..." if secret_key else "MISSING")
    print(f"Host: {host}")

    if not public_key or not secret_key:
        print("ERROR: Langfuse keys not found in .env file")
        return False

    try:
        # Initialize Langfuse client - CORRECT WAY
        from langfuse import Langfuse
        langfuse = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host
        )
        print("✅ Langfuse client created successfully!")

        # Test basic functionality - SIMPLE VERSION
        print("Testing basic trace creation...")

        # Create a trace using the client method
        trace = langfuse.trace(
            name="DueDiligence-Test-Connection",
            user_id="maryam",
            metadata={"project": "DueDiligenceRAG", "test": True}
        )
        print("✅ Trace created")

        # Create a generation within the trace
        generation = trace.generation(
            name="test-generation",
            input="What is financial due diligence?",
            output="Financial due diligence involves analyzing a company's financial statements, assets, liabilities, and overall financial health.",
            metadata={
                "model": "test-model",
                "temperature": 0.1,
                "prompt_type": "factual"
            }
        )
        print("✅ Generation logged")

        # Create a score for evaluation
        trace.score(
            name="relevance-score",
            value=0.95,
            comment="Test evaluation of financial due diligence response"
        )
        print("✅ Score logged")

        # Flush data to ensure it's sent
        langfuse.flush()
        print("✅ Data sent to Langfuse")

        print("\n🎉 Langfuse integration successful!")
        print("📊 Check your 'DueDiligence-RAG' project in Langfuse dashboard")
        print("🌐 https://cloud.langfuse.com")
        return True

    except Exception as e:
        print(f"❌ Langfuse connection failed: {e}")
        import traceback
        print(f"Detailed error: {traceback.format_exc()}")
        return False


if __name__ == "__main__":
    test_langfuse_connection()