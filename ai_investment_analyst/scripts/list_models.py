"""
Quick script to list available Google AI models.
Run: python list_models.py
"""
import google.genai as genai
from app.core.config import settings

def main():
    try:
        client = genai.Client(api_key=settings.google_api_key)
        models = client.models.list()
        print("Available models:")
        for model in models:
            print(f"  - {model.name}: {model.description}")
    except Exception as e:
        print(f"Error: {e}")
        print("Check your GOOGLE_API_KEY in app/core/config.py or .env")

if __name__ == "__main__":
    main()