# AI Scholar RAG Chatbot v2

This Google Summer of Code 2024 project by Christopher Mejo includes scripts to preprocess data, create a chatbot using a retrieval-augmented generation (RAG) approach, build a knowledge graph, and visualize data.


## Project Structure

rag-chatbot/
├── README.md
├── LICENSE
├── .gitignore
├── setup.py
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── data_ingestion.py
│   ├── retriever.py
│   ├── generator.py
│   ├── rag_chain.py
│   ├── chatbot.py
│   ├── utils.py
├── config/
│   ├── __init__.py
│   ├── config.yaml
├── tests/
│   ├── __init__.py
│   ├── test_data_ingestion.py
│   ├── test_retriever.py
│   ├── test_generator.py
│   ├── test_rag_chain.py
│   ├── test_chatbot.py
└── notebooks/
    ├── data_exploration.ipynb
    ├── model_tuning.ipynb
    ├── chatbot_demo.ipynb
