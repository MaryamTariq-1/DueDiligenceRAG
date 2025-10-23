import subprocess
import sys
import os

def install_requirements():
    print("Installing frontend requirements...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements_frontend.txt"])

def run_streamlit():
    print("Starting Due Diligence RAG Frontend...")
    print("Open your browser and go to: http://localhost:8501")
    subprocess.call(["streamlit", "run", "app.py"])

if __name__ == "__main__":
    install_requirements()
    run_streamlit()