import gensim
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
from config import Config
"""
This script will train a Word2Vec model on our text to generate word embeddings.
This will be used for query expansion to decide if a synonym for a word is related to the word given the context of the text.
This will help us to generate a more accurate query expansion and reduce noise in the search results.
"""

# Load the text
class WordEmbedder:
    def __init__(self):
        self.cfg = Config()
        self.model: Word2Vec = None
    
    def load(self):
        self.model = Word2Vec.load(self.cfg.WORD2VEC_MODEL_PATH)
    
    def train(self):
        # Load the text
        with open(self.cfg.TEXT_PATH, "r") as file:
            text = file.read()
            # Tokenize the text
            tokens = word_tokenize(text)
            # to lowercase
            tokens = [w.lower() for w in tokens]
            # Train the model
            self.model = Word2Vec([tokens], vector_size=100, window=5, min_count=1, sg=1)
            # Save the model
            self.model.save(self.cfg.WORD2VEC_MODEL_PATH)
    
    