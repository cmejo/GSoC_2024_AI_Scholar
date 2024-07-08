# AI Scholar RAG Chatbot

This Google Summer of Code 2024 project by Christopher Mejo includes scripts to preprocess data, create a chatbot using a retrieval-augmented generation (RAG) approach, build a knowledge graph, and visualize data.

## Setup

1. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

2. Create a `.env` file with the necessary environment variables:
    ```plaintext
    MODEL_NAME=meta-llama/llama3
    DATASET_PATH=data/dataset.json
    INDEX_PATH=models/faiss_index.bin
    ```

3. Run the preprocessing script to generate embeddings and create a FAISS index:
    ```bash
    python scripts/preprocess.py
    ```

4. Start the chatbot:
    ```bash
    python scripts/chatbot.py
    ```

5. Create the knowledge graph:
    ```bash
    python scripts/knowledge_graph.py
    ```

6. Visualize the data:
    ```bash
    Rscript scripts/visualize.R
    ```

## Project Structure

- `data/`: Contains the dataset.
- `models/`: Contains the saved FAISS index.
- `scripts/`: Contains the Python and R scripts.
- `.env`: Environment variables.
- `requirements.txt`: Python package dependencies.
- `README.md`: Project documentation.
- `.gitignore`: Files to ignore in git.
