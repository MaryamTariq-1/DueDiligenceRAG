# check_keys.py
import os
from dotenv import load_dotenv

load_dotenv()


def check_langfuse_keys():
    print("Checking Langfuse Keys...")
    print("=" * 40)

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")

    if not public_key:
        print(" LANGFUSE_PUBLIC_KEY is missing")
    else:
        if public_key.startswith("pk-lf-"):
            print(f" LANGFUSE_PUBLIC_KEY: {public_key[:15]}...")
        else:
            print(f" LANGFUSE_PUBLIC_KEY format wrong: {public_key[:20]}...")

    if not secret_key:
        print(" LANGFUSE_SECRET_KEY is missing")
    else:
        if secret_key.startswith("sk-lf-"):
            print(f" LANGFUSE_SECRET_KEY: {secret_key[:15]}...")
        else:
            print(f" LANGFUSE_SECRET_KEY format wrong: {secret_key[:20]}...")

    if public_key and secret_key:
        if public_key == secret_key:
            print(" ERROR: Public and Secret keys are the same!")
            print("   They should be different values")
        else:
            print(" Public and Secret keys are different (correct)")


if __name__ == "__main__":
    check_langfuse_keys()