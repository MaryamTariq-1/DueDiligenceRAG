# working_langfuse_test.py
import os
from dotenv import load_dotenv

load_dotenv()


def test_langfuse_working():
    print(" WORKING LANGFUSE 3.8.0 TEST")
    print("=" * 50)

    try:
        from langfuse import Langfuse

        # Initialize client
        langfuse = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST")
        )
        print(" Langfuse client initialized")

        # Create trace using the correct v3.8.0 API
        trace = langfuse.trace(
            name="due-diligence-test",
            user_id="maryam",
            metadata={"project": "DueDiligenceRAG", "test": True}
        )
        print(" Trace created")

        # Create generation
        generation = trace.generation(
            name="test-generation",
            input="What is financial due diligence?",
            output="Financial due diligence involves analyzing company finances before acquisition.",
            metadata={
                "model": "test-model",
                "temperature": 0.1,
                "prompt_type": "factual"
            }
        )
        print(" Generation logged")

        # Add score
        trace.score(
            name="accuracy",
            value=0.95,
            comment="Test accuracy score for financial due diligence"
        )
        print(" Score added")

        # Flush data
        langfuse.flush()
        print(" Data flushed to Langfuse")

        print("\n SUCCESS! Langfuse 3.8.0 integration working!")
        print(" Check your dashboard: https://cloud.langfuse.com")
        return True

    except Exception as e:
        print(f" FAILED: {e}")
        import traceback
        print(f"Full error:\n{traceback.format_exc()}")
        return False


if __name__ == "__main__":
    test_langfuse_working()