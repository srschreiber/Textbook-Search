import gensim
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
from config import Config
"""
This script will train a Word2Vec model on our text to generate word embeddings.
This will be used for query expansion to decide if a synonym for a word is related to the word given the context of the text.
This will help us to generate a more accurate query expansion and reduce noise in the search results.
"""
import os

nltk.download('punkt_tab')
# Load the text
class WordEmbedder:
    def __init__(self):
        self.cfg = Config()
        self.model: Word2Vec = None
    
    def load(self):
        self.model = Word2Vec.load(self.cfg.WORD2VEC_MODEL_PATH)
    
    def model_has_word(self, word):
        return word in self.model.wv.key_to_index

    def similarity(self, word1, word2):
        return self.model.wv.similarity(word1, word2)
    
    def train(self):
        # Load the text
        with open(self.cfg.TEXT_PATH, "r") as file:
            text = file.read()
            # Replace unwanted characters and normalize spaces
            text = ' '.join(text.split())  # Removes extra spaces
            text = text.replace('\n', ' ')  # Removes newlines
            text = text.replace('\t', ' ')  # Removes tabs
            text = text.replace('\r', ' ')   # Removes carriage returns
            text = text.replace('-', ' ')     # Replaces hyphens with spaces

            # Tokenize the text into sentences
            sentences = sent_tokenize(text)

            # Tokenize each sentence into words
            tokenized_sentences = [word_tokenize(s.lower()) for s in sentences]

            # Train the model
            self.model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4)

            # Save the model
            self.model.save(self.cfg.WORD2VEC_MODEL_PATH)

    
if __name__ == "__main__":
    # Train the model if doesn't exist

    word_embedder = WordEmbedder()
    if not os.path.exists(Config().WORD2VEC_MODEL_PATH):
        word_embedder.train()
    
    word_embedder.load()
    print(word_embedder.similarity("cell", "membrane"))
    