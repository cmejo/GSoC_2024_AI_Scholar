import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text.lower())
    
    # Removing stopwords
    tokens = [word for word in tokens if word.isalnum()]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    return tokens

def preprocess_texts(texts):
    return [{'title': text['title'], 'tokens': preprocess_text(text['text'])} for text in texts]

if __name__ == "__main__":
    sample_text = "This is a sample text for preprocessing, including tokenization and stopword removal."
    print(preprocess_text(sample_text))
    
from src.finetune_word2vec import finetune_word2vec_model

fine_tuned_model = finetune_word2vec_model("word2vec.model", preprocessed_texts)
