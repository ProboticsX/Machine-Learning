from elevenlabs.conversational_ai.conversation import ClientTools
from dotenv import load_dotenv
import os
import requests

load_dotenv()

perplexity_model = "sonar"
PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"

def get_perplexity_payload(question: str, return_images: bool = False):
    payload = {
        "model": perplexity_model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that answers questions and provides information."},
            {"role": "user", "content": question}
        ],
        "return_images": return_images,
    }
    return payload

def get_perplexity_headers():
    headers = {
        "Authorization": f"Bearer {os.getenv("PERPLEXITY_API_KEY")}",
        "Content-Type": "application/json"
    }
    return headers

def get_perplexity_response(question: str):
    """Use this tool to do any web search. Gets the response from the perplexity API.
    Args:
        question: The question to ask the perplexity API.
        return_images: Whether to return images in the response. If True, the response will contain the image URLs.
    Returns:
        str: The response from the perplexity API.
    """

    payload = {
        "model": perplexity_model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that answers questions and provides information."},
            {"role": "user", "content": question}
        ],
        "return_images": False,
    }
    headers = {
        "Authorization": f"Bearer {os.getenv("PERPLEXITY_API_KEY")}",
        "Content-Type": "application/json"
    }
    response = requests.request("POST", PERPLEXITY_API_URL, json=payload, headers=headers)
    return response.json()


client_tools = ClientTools()
client_tools.register("get_perplexity_response", get_perplexity_response)