import os
import json
from transformers import LlamaTokenizer

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Load the dataset
dataset_path = os.getenv("DATASET_PATH", "path/to/your/custom_dataset.json")
with open(dataset_path, 'r') as f:
    dataset = json.load(f)

# Initialize the tokenizer
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/llama3")

# Preprocess the dataset: Tokenize the texts
def preprocess(text):
    return tokenizer(text, truncation=True, padding="max_length", return_tensors="pt")

texts = [doc["text"] for doc in dataset]
tokenized_texts = [preprocess(text) for text in texts]
