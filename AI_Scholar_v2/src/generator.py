from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_llama3_model(model_name="llama-3b"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_length=512):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_length=max_length)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response
