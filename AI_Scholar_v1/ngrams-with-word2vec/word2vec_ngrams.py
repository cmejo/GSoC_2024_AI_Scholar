import os
import json
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from gensim.models.phrases import Phrases, Phraser

# Load the dataset
dataset_path = os.getenv("DATASET_PATH", "/path.to.custom.dataset/home/cmejo/arxiv-dataset/custom_dataset.json")
with open(dataset_path, 'r') as f:
    dataset = json.load(f)

# Preprocess the dataset: Tokenize and clean text
def preprocess_text(text):
    return simple_preprocess(text, deacc=True)  # Tokenizes text and removes punctuation

texts = [preprocess_text(doc["text"]) for doc in dataset]

# Create bigrams and trigrams
bigram = Phrases(texts, min_count=5, threshold=100)
trigram = Phrases(bigram[texts], threshold=100)

bigram_mod = Phraser(bigram)
trigram_mod = Phraser(trigram)

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

data_bigrams = make_bigrams(texts)
data_trigrams = make_trigrams(texts)

# Train a Word2Vec model on the trigram data
model = Word2Vec(sentences=data_trigrams, vector_size=100, window=5, min_count=5, workers=4)

# Save the model
model_path = os.getenv("MODEL_PATH", "models/word2vec_trigrams.model")
model.save(model_path)

# Example: Get embedding for a specific trigram
example_trigram = trigram_mod[bigram_mod[preprocess_text("This is an example sentence for n-grams")]]
embedding = model.wv[example_trigram[0]]
print(f"Trigram: {example_trigram[0]}, Embedding: {embedding}")

