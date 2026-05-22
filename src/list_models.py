"""
list_openrouter_models.py
Muestra los modelos gratuitos disponibles para tu cuenta de OpenRouter.
"""
import os
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

api_key = os.getenv("OPENROUTER_API_KEY")
headers = {"Authorization": f"Bearer {api_key}"}

resp = requests.get("https://openrouter.ai/api/v1/models", headers=headers)
modelos = resp.json().get("data", [])

print("Modelos GRATUITOS disponibles (pricing = 0):\n")
gratuitos = []
for m in modelos:
    precio = m.get("pricing", {})
    if str(precio.get("prompt", "1")) == "0" and str(precio.get("completion", "1")) == "0":
        gratuitos.append(m["id"])
        print(f"  {m['id']}")

print(f"\nTotal modelos gratuitos: {len(gratuitos)}")