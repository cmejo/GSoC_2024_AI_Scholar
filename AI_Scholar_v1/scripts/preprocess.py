import json
import os
from transformers import AutoTokenizer, AutoModel
import numpy as np
import faiss
import torch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load a pre-trained LLaMA model and tokenizer
model_name = os.getenv("MODEL_NAME", "meta-llama/llama3")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Load custom dataset
dataset_path = os.getenv("DATASET_PATH", "data/dataset.json")

def load_dataset(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

dataset = load_dataset(dataset_path)

# Generate embeddings for the dataset
def generate_embeddings(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

embeddings_list = [generate_embeddings(doc["text"], tokenizer, model) for doc in dataset]
embeddings_array = np.array(embeddings_list)

# Create a FAISS index
dimension = embeddings_array.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings_array)

# Save the index
index_path = os.getenv("INDEX_PATH", "models/faiss_index.bin")
faiss.write_index(index, index_path)
