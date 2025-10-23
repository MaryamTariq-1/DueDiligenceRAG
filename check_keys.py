
import os
from dotenv import load_dotenv

load_dotenv()


def check_all_keys():
    print("Checking ALL API Keys...")
    print("=" * 50)

    # Langfuse Keys
    print("\nLANGFUSE KEYS:")
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")

    if public_key and public_key.startswith("pk-lf-"):
        print(f"SUCCESS - LANGFUSE_PUBLIC_KEY: {public_key[:15]}...")
    else:
        print("FAILED - LANGFUSE_PUBLIC_KEY: Missing or invalid")

    if secret_key and secret_key.startswith("sk-lf-"):
        print(f"SUCCESS - LANGFUSE_SECRET_KEY: {secret_key[:15]}...")
    else:
        print("FAILED - LANGFUSE_SECRET_KEY: Missing or invalid")

    # AWS Keys
    print("\nAWS KEYS:")
    aws_access = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY")

    if aws_access and aws_access.startswith("AKIA"):
        print(f"SUCCESS - AWS_ACCESS_KEY_ID: {aws_access[:10]}...")
    else:
        print("FAILED - AWS_ACCESS_KEY_ID: Missing or invalid")

    if aws_secret:
        print(f"SUCCESS - AWS_SECRET_ACCESS_KEY: {aws_secret[:10]}...")
    else:
        print("FAILED - AWS_SECRET_ACCESS_KEY: Missing")

    # FREE API Keys
    print("\nFREE API KEYS:")
    free_apis = {
        "GROQ_API_KEY": "gsk_",
        "HUGGINGFACE_API_KEY": "hf_",
        "DEEPSEEK_API_KEY": "sk-",
        "COHERE_API_KEY": "O",
        "OPENROUTER_AI_API_KEY": "sk-or-v1",
        "GOOGLE_AI_STUDIO_API_KEY": ""  # Any format
    }

    for key_name, prefix in free_apis.items():
        key_value = os.getenv(key_name)
        if key_value:
            if prefix and key_value.startswith(prefix):
                print(f"SUCCESS - {key_name}: {key_value[:15]}...")
            elif not prefix:
                print(f"SUCCESS - {key_name}: Present")
            else:
                print(f"WARNING - {key_name}: Present but format may be wrong")
        else:
            print(f"FAILED - {key_name}: Missing")

    print("\n" + "=" * 50)
    print("All keys checked!")


if __name__ == "__main__":
    check_all_keys()