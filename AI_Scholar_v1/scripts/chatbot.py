import json
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np
import faiss
import torch
from dotenv import load_dotenv
from langchain.prompts import Prompt

# Load environment variables from .env file
load_dotenv()

# Load the pre-trained LLaMA model and tokenizer
model_name = os.getenv("MODEL_NAME", "meta-llama/llama3")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Load the FAISS index
index_path = os.getenv("INDEX_PATH", "models/faiss_index.bin")
index = faiss.read_index(index_path)

# Load the dataset
dataset_path = os.getenv("DATASET_PATH", "data/dataset.json")
with open(dataset_path, 'r') as f:
    dataset = json.load(f)

# Define a prompt template
prompt_template = """
You are an AI assistant specialized in answering questions based on the provided dataset. Given the following document text, provide a brief and accurate answer to the question:
Document: {document}
Question: {question}
Answer:
"""

# Function to find the most relevant document
def find_relevant_document(question, index, dataset, tokenizer, model):
    inputs = tokenizer(question, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    query_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy().reshape(1, -1)
    D, I = index.search(query_embedding, k=1)
    document = dataset[I[0][0]]["text"]
    return document

# Function to create the chatbot response
def rag_chatbot(question, index, dataset, tokenizer, model):
    document = find_relevant_document(question, index, dataset, tokenizer, model)
    prompt = prompt_template.format(document=document, question=question)
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# Main function to run the chatbot
if __name__ == "__main__":
    while True:
        question = input("Ask a question: ")
        if question.lower() in ["exit", "quit"]:
            break
        answer = rag_chatbot(question, index, dataset, tokenizer, model)
        print("Answer:", answer)

