from src.data_ingestion import pdf_to_text
from src.retriever import create_vector_store, get_retriever
from src.generator import load_llama3_model
from src.rag_chain import create_qa_chain

def main():
    pdf_folder = 'data/pdfs'
    texts = pdf_to_text(pdf_folder)
    vector_store = create_vector_store(texts)
    retriever = get_retriever(vector_store)
    model, tokenizer = load_llama3_model()
    qa_chain = create_qa_chain(retriever, model, tokenizer)

    print("Welcome to the AI Scholar Research Chatbot! Type 'exit' to quit.")
    while True:
        query = input("\nAsk a question: ")
        if query.lower() == "exit":
            break
        response = qa_chain.run(query)
        print("\nAnswer:", response)

if __name__ == "__main__":
    main()
