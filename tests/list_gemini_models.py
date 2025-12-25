# tests/list_gemini_models.py

import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

print("\nAvailable models:\n")

models = client.models.list()

for m in models:
    print(f"Model name: {m.name}")
    print(f"  Description: {getattr(m, 'description', 'N/A')}")
    print(f"  Input token limit: {getattr(m, 'input_token_limit', 'N/A')}")
    print(f"  Output token limit: {getattr(m, 'output_token_limit', 'N/A')}")
    print(f"  Supported methods: {getattr(m, 'supported_methods', 'N/A')}")
    print("-" * 60)
