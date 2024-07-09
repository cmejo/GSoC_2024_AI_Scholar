from transformers import LlamaForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Load the dataset
dataset_path = os.getenv("DATASET_PATH", "path/to/your/custom_dataset.json")
train_dataset = load_dataset("json", data_files=dataset_path, split='train')

# Tokenize the dataset
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/llama3")
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding="max_length")

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)

# Load the model
model = LlamaForCausalLM.from_pretrained("meta-llama/llama3")

# Set training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("models/fine_tuned_llama3")
tokenizer.save_pretrained("models/fine_tuned_llama3")
