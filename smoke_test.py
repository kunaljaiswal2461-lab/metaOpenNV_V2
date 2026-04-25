import os, sys, torch, openai, fastapi, pydantic, yfinance
import openenv_core
import importlib.metadata
from dotenv import load_dotenv

print(f"Python: {sys.version}")
print(f"openenv: {importlib.metadata.version('openenv-core')}")
print(f"fastapi: {fastapi.__version__}")
print(f"pydantic: {pydantic.__version__}")

load_dotenv()
print(f"API_BASE_URL: {os.environ.get('API_BASE_URL')}")

if os.path.exists("inference.py"):
    print("inference.py located.")

print("PHASE 1 COMPLETE")
