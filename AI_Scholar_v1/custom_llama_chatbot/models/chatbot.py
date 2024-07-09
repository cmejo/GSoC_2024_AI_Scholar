import os
import requests
from transformers import AutoTokenizer, AutoModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ollama API endpoint and API key
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "https://api.ollama.com/inference")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "your_api_key")

# Load Hugging Face tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("models/fine_tuned_llama3")
model = AutoModel.from_pretrained("models/fine_tuned_llama3")

# Function to get chatbot response using Ollama
def ollama_rag_chatbot(question, api_url, api_key):
    response = requests.post(
        api_url,
        headers={"Authorization": f"Bearer {api_key}"},
        json={"question": question}
    )
    if response.status_code == 200:
        return response.json()["answer"]
    else:
        raise Exception(f"Failed to get response from Ollama: {response.status_code}")

# Example chatbot interaction using Ollama and Hugging Face
if __name__ == "__main__":
    question = input("Ask a question: ")
    answer = ollama_rag_chatbot(question, OLLAMA_API_URL, OLLAMA_API_KEY)
    print("Answer:", answer)
