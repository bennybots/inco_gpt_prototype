import os
from dotenv import load_dotenv
import openai

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def classify_transaction(user_input):
    system_prompt = """You are an assistant classifying intercompany transaction types.
Classify the following user input into one of these categories:
1. Cost Plus
2. LRD

Only respond with one label: Cost Plus, LRD"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]
    
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=messages
    )

    return response.choices[0].message.content.strip()
