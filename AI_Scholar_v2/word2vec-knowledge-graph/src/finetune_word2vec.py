from gensim.models import Word2Vec
from src.text_preprocessing import preprocess_texts
from src.data_ingestion import pdf_to_text

def finetune_word2vec_model(existing_model_path, texts, vector_size=100, window=5, min_count=5, workers=4, epochs=10):
    # Load the existing Word2Vec model
    model = Word2Vec.load(existing_model_path)
    
    # Fine-tune the model
    model.train([text['tokens'] for text in texts], total_examples=len(texts), epochs=epochs)
    
    # Save the fine-tuned model
    model.save("fine_tuned_word2vec.model")
    return model

if __name__ == "__main__":
    # Load and preprocess the text data
    pdf_folder = '/home/cmejo/arxiv-dataset/pdf'
    texts = preprocess_texts(pdf_to_text(pdf_folder))
    
    # Fine-tune the existing Word2Vec model
    existing_model_path = "word2vec.model"
    model = finetune_word2vec_model(existing_model_path, texts)
    print("Fine-tuning complete.")
