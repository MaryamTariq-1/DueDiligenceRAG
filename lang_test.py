# langfuse_simple_test.py
import os
from dotenv import load_dotenv

load_dotenv()


def test_langfuse_simple():
    print("Testing Langfuse Connection (Simple Version)...")

    try:
        from langfuse import Langfuse

        langfuse = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST")
        )
        print("✅ Langfuse client created successfully!")

        # Simple test - create a generation
        generation = langfuse.start_generation(
            name="DueDiligence-Test",
            input="What is due diligence?",
            metadata={"test": True, "project": "DueDiligenceRAG"}
        )

        generation.end(
            output="Due diligence is the process of investigating a business before making an investment decision.")
        print("✅ Generation created and completed")

        # Flush and check
        langfuse.flush()
        print("✅ Data sent to Langfuse")

        print("\n🎉 Simple test successful!")
        return True

    except Exception as e:
        print(f"❌ Langfuse connection failed: {e}")
        return False


if __name__ == "__main__":
    test_langfuse_simple()