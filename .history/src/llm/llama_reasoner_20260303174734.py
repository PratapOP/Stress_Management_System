import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"


def generate_ai_report(name, age, stress_level, shap_factors, routine_data, face_data):

    prompt = f"""
You are an AI psychological stress analysis assistant.

User Name: {name}
User Age: {age}

Detected Stress Level: {stress_level}

Top Influencing Factors:
{shap_factors}

Routine Data:
{routine_data}

Facial Metrics:
{face_data}

Write a professional personalized stress analysis report.

Requirements:
- Greet the user by name at the beginning.
- Do not mention machine learning, SHAP, or models.
- Explain why this stress level may have occurred.
- Explain how lifestyle and facial signals interact.
- Indicate whether stress appears acute or lifestyle-based.
- Provide structured actionable recommendations.
- Keep it under 250 words.
- Tone: professional, supportive, intelligent.
"""

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload)
    result = response.json()

    return result["response"]