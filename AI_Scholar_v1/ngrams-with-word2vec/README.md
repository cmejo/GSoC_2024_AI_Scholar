Explanation
Loading the Dataset:

The dataset is loaded from a JSON file specified by the DATASET_PATH environment variable.
Preprocessing the Text:

The preprocess_text function tokenizes and cleans the text, removing punctuation.
Creating Bigrams and Trigrams:

Bigrams and trigrams are created using the gensim.models.phrases.Phrases and Phraser classes.
make_bigrams and make_trigrams functions are used to generate bigrams and trigrams for the entire dataset.
Training the Word2Vec Model:

The Word2Vec model is trained on the trigram data.
The vector_size parameter specifies the dimensionality of the word vectors.
The window parameter specifies the maximum distance between the current and predicted word within a sentence.
The min_count parameter ignores all words with total frequency lower than this.
Saving the Model:

The trained Word2Vec model is saved to the path specified by the MODEL_PATH environment variable.
Generating Embeddings:

An example sentence is preprocessed into trigrams, and the embedding for the first trigram is printed.

Running the Script
Save the script as word2vec_ngrams.py and run it:


python word2vec_ngrams.py

.env file:

DATASET_PATH=/home/cmejo/arxiv-dataset/custom_dataset.json
MODEL_PATH=models/word2vec_trigrams.model
This script will preprocess your text data into bigrams and trigrams, train a Word2Vec model, and save the model for later use. You can then use this model to generate embeddings for any n-gram in your dataset.
