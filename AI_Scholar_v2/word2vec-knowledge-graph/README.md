# Word2Vec Knowledge Graph from ArXiv Papers

This repository contains the code to create a knowledge graph from a large dataset of scientific papers extracted from ArXiv using Word2Vec.

## Features
- Ingest PDF documents from ArXiv.
- Preprocess text data (tokenization, stopword removal, etc.).
- Train a Word2Vec model on the preprocessed text.
- Create a knowledge graph using Word2Vec embeddings.

## Installation

```bash
git clone https://github.com/your-username/word2vec-knowledge-graph.git
cd word2vec-knowledge-graph
pip install -r requirements.txt
```
## Usage

1. To ingest data from PDFs:
```
python src/data_ingestion.py
```
2. To preprocess the text data:
```
python src/text_preprocessing.py

```
3. Training Word2Vec Model
```
python src/word2vec_training.py
```
4. To fine-tune the existing Word2Vec model:
```
python src/finetune_word2vec.py
```
5. To build and visualize the knowledge graph:

```
python src/knowledge_graph.py
```

## Project Structure

```
└── word2vec-knowledge-graph/
    ├── README.md
    ├── LICENSE
    ├── .gitignore
    ├── setup.py
    ├── requirements.txt
    ├── src/
    │   ├── __init__.py
    │   ├── data_ingestion.py
    │   ├── text_preprocessing.py
    │   ├── word2vec_training.py
    │   ├── knowledge_graph.py
    │   └── utils.py
    ├── config/
    │   ├── __init__.py
    │   └── config.yaml
    ├── notebooks/
    │   ├── data_exploration.ipynb
    │   ├── word2vec_training.ipynb
    │   └── knowledge_graph_creation.ipynb
    ├── tests/
    │   ├── __init__.py
    │   ├── test_data_ingestion.py
    │   ├── test_text_preprocessing.py
    │   ├── test_word2vec_training.py
    │   └── test_knowledge_graph.py
    └── data/
        └── pdfs/

```
## Contributing
Feel free to open issues or submit pull requests.

