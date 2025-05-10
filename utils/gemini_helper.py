"""
Helper functions for using Google's Gemini Flash API for symptom analysis
"""

import os
import json
import google.generativeai as genai

# Get API key from environment variables
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# Configure the Gemini API
genai.configure(api_key=GOOGLE_API_KEY)

# Use Gemini Flash model
MODEL_NAME = "models/gemini-1.5-flash"


def analyze_symptoms(symptoms,
                     age,
                     gender,
                     duration,
                     severity,
                     additional_info=""):
    symptoms_text = ", ".join(symptoms)

    prompt = f"""
    Act as a medical advisor. Based on the following information, provide an analysis:

    Patient Information:
    - Age: {age}
    - Gender: {gender}
    - Symptoms: {symptoms_text}
    - Duration: {duration}
    - Severity: {severity}
    - Additional Information: {additional_info}

    Please analyze these symptoms and provide the following information in JSON format:

    {{
        "possible_conditions": [
            {{
                "name": "Condition name",
                "description": "Brief description",
                "common_symptoms": ["symptom1", "symptom2"],
                "diet_recommendations": ["recommendation1", "recommendation2"]
            }}
        ],
        "risk_level": "low/moderate/high",
        "seek_medical_attention": true/false,
        "general_advice": "General health advice text",
        "medical_sources": ["Source 1", "Source 2"]
    }}

    Include a clear disclaimer that this is not a medical diagnosis.
    """

    try:
        model = genai.GenerativeModel(MODEL_NAME)

        response = model.generate_content(
            contents=[{
                "role": "user",
                "parts": [prompt]
            }],
            generation_config={"temperature": 0.2})

        response_text = response.text

        try:
            result = json.loads(response_text)
        except json.JSONDecodeError:
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start >= 0 and end > start:
                result = json.loads(response_text[start:end])
            else:
                raise Exception(
                    "Failed to extract valid JSON from the response")

        return result

    except Exception as e:
        return {
            "error":
            True,
            "message":
            f"Error analyzing symptoms: {str(e)}",
            "possible_conditions": [],
            "risk_level":
            "unknown",
            "seek_medical_attention":
            True,
            "general_advice":
            "We encountered an error analyzing your symptoms. Please try again or consult a healthcare professional.",
            "medical_sources":
            ["https://www.who.int", "https://www.mayoclinic.org"]
        }


def get_symptom_conversation(symptoms, previous_conversation=None):
    if not previous_conversation:
        previous_conversation = []

    symptoms_text = ", ".join(symptoms)

    prompt = f"""
    I'm experiencing the following symptoms: {symptoms_text}. 
    Can you provide me with some friendly advice and reassurance?
    Keep your response conversational and supportive, like a caring nurse or doctor would.
    Do not try to diagnose me specifically, but provide general information and wellness tips.
    """

    try:
        model = genai.GenerativeModel(MODEL_NAME)

        conversation = previous_conversation.copy()
        conversation.append({"role": "user", "parts": [prompt]})

        response = model.generate_content(
            contents=conversation, generation_config={"temperature": 0.7})

        return response.text

    except Exception as e:
        return f"I'm sorry, I couldn't process your request at the moment. Please try again later. Error: {str(e)}"
