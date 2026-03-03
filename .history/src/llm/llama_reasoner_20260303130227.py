import requests
import json


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"


def generate_ai_report(stress_level, shap_factors, routine_data, face_data):

    prompt = f"""
You are an AI psychological stress analysis assistant.

The system has detected:
Stress Level: {stress_level}

Top Influencing Factors:
{shap_factors}

Routine Data:
{routine_data}

Facial Metrics:
{face_data}

Explain in a professional, human-friendly way:

1. Why this stress level may have occurred
2. How routine and facial signals interact
3. Whether the stress seems acute or lifestyle-based
4. Provide structured recommendations
5. Keep it under 250 words
6. Do NOT mention machine learning or SHAP

Write as a professional AI analysis report.
"""

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload)

    result = response.json()

    return result["response"]