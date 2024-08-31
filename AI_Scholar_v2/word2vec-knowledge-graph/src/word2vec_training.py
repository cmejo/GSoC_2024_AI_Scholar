from gensim.models import Word2Vec

def train_word2vec_model(texts, vector_size=100, window=5, min_count=5, workers=4):
    model = Word2Vec(sentences=[text['tokens'] for text in texts], vector_size=vector_size, window=window, min_count=min_count, workers=workers)
    model.save("word2vec.model")
    return model

if __name__ == "__main__":
    texts = preprocess_texts(pdf_to_text('../data/pdfs'))
    model = train_word2vec_model(texts)
    print(model.wv.most_similar('science'))
