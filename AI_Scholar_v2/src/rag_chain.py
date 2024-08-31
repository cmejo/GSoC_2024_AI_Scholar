from langchain.chains import RetrievalQAChain
from langchain.llms import HuggingFaceLLM

def create_qa_chain(retriever, model, tokenizer):
    llm = HuggingFaceLLM(model=model, tokenizer=tokenizer, device="cuda" if torch.cuda.is_available() else "cpu")
    qa_chain = RetrievalQAChain(
        retriever=retriever,
        llm=llm,
        chain_type="stuff"
    )
    return qa_chain
