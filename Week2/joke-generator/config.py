# config.py
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Updated to current model
GROQ_MODEL = "llama-3.3-70b-versatile"  # Changed from llama-3.1-70b-versatile

MAX_TOKENS = 200
TEMPERATURE = 0.8

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found! Please create a .env file with your API key.\n"
        "Get your free key from: https://console.groq.com/keys"
    )