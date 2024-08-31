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
```python src/knowledge_graph.py
```
## Contributing
```
#### `setup.py`

For packaging the project.

```python
from setuptools import setup, find_packages

setup(
    name='word2vec_knowledge_graph',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'gensim',
        'nltk',
        'networkx',
        'matplotlib',
        'PyPDF2',
    ],
    entry_points={
        'console_scripts': [
            'knowledge-graph=src.knowledge_graph:main',
        ],
    },
)
```
