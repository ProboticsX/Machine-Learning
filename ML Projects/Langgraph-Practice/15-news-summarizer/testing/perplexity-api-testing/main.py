from openai import OpenAI
from dotenv import load_dotenv
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAI

load_dotenv()

YOUR_API_KEY = os.getenv("PERPLEXITY_API_KEY")

# messages = [
#     {
#         "role": "system",
#         "content": (
#             "You are an artificial intelligence assistant and you need to "
#             "engage in a helpful, detailed, polite conversation with a user."
#         ),
#     },
#     {   
#         "role": "user",
#         "content": (
#             "What are the top headlines in the business category for today?"
#         ),
#     },
# ]

# client = OpenAI(api_key=YOUR_API_KEY, base_url="https://api.perplexity.ai")

# # # chat completion without streaming
# response = client.chat.completions.create(
#     model="sonar",
#     messages=messages,
# )
# print(response)

import requests

url = "https://api.perplexity.ai/chat/completions"

payload = {
    "model": "sonar",
    "messages": [
        {
            "role": "system",
            "content": "Be precise and concise."
        },
        {
            "role": "user",
            "content": "Find images for Donald Trump"
        }
    ],
    "return_images": True,
    "return_related_questions": True,
}
headers = {
    "Authorization": f"Bearer {YOUR_API_KEY}",
    "Content-Type": "application/json"
}

response = requests.request("POST", url, json=payload, headers=headers)

try:
    response.raise_for_status()  # Raises an HTTPError for bad responses (4XX, 5XX)
    print(response.json())
except requests.exceptions.RequestException as e:
    print(f"Error making request: {e}")
except ValueError as e:
    print(f"Error parsing JSON response: {e}")