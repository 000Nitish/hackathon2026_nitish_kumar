"""List available Gemini 2.5 Flash models."""
from dotenv import load_dotenv
load_dotenv(override=True)

import os
import google.genai as genai

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
models = client.models.list()
for m in models:
    if "flash" in m.name.lower():
        print(m.name)
