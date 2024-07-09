# After deploying model on Ollama, you can integrate Hugging Face's capabilities for additional tasks such as custom embeddings.

from transformers import AutoTokenizer, AutoModel
import requests
import os

# Load environment variables
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "https://api.ollama.com/inference")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "your_api_key")

# Load Hugging Face tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("path_to_save_tokenizer")
model = AutoModel.from_pretrained("path_to_save_model")

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
