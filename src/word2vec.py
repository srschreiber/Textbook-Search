import gensim
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
from config import Config
import os

nltk.download('punkt')  # Ensure you have the necessary tokenizer downloaded

from nltk.corpus import stopwords

# Ensure stopwords are downloaded
nltk.download('stopwords')

class WordEmbedder:
    def __init__(self):
        self.cfg = Config()
        self.model: Word2Vec = None
        self.lemmatizer = nltk.WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))  # Load stop words
    
    def load(self):
        self.model = Word2Vec.load(self.cfg.WORD2VEC_MODEL_PATH)

    def train(self):
        with open(self.cfg.TEXT_PATH, "r") as file:
            text = file.read()
            # Clean the text (as before)
            # ...
            # Tokenize into sentences
            # remove all - replace with empty string
            text = text.replace("-", "")
            sentences = sent_tokenize(text.lower())

            
            # Tokenize each sentence into words, lemmatize, and remove stop words
            tokenized_sentences = []
            for sentence in sentences:
                tokens = word_tokenize(sentence)
                tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]  # Remove stop words
                tokenized_sentences.append(tokens)

            # Train the model
            self.model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=3, workers=4)

            # Save the model
            self.model.save(self.cfg.WORD2VEC_MODEL_PATH)


    def top_k_similar(self, word, k=5):
        if word in self.model.wv.key_to_index:
            return self.model.wv.most_similar(word, topn=k)
        else:
            return []  # Return empty if the word is not in the vocabulary

if __name__ == "__main__":
    word_embedder = WordEmbedder()
    if not os.path.exists(Config().WORD2VEC_MODEL_PATH):
        word_embedder.train()
    
    word_embedder.load()
    print(word_embedder.similarity("cell", "membrane"))
