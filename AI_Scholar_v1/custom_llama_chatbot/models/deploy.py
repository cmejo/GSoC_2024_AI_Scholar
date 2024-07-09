import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Model and tokenizer paths
model_path = "models/fine_tuned_llama3"

# Upload to Ollama
os.system(f"ollama upload --model {model_path} --name my_llama3_model")
