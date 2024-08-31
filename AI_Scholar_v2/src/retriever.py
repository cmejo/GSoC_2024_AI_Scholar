from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

def create_vector_store(texts):
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vector_store = FAISS.from_documents(texts, embeddings)
    return vector_store

def get_retriever(vector_store):
    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
