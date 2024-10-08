"""
This creates a BM25 model for the given corpus.
It is then takes in a query, expands it, sets the weights for each term in the query, and ranks the documents in the corpus based on the BM25 score. 
"""
import os
import pickle
from collections import Counter
from math import log
from typing import List, Tuple
from query_expansion import WordExpansion
from tokenizer import load_spacy_output
from config import Config
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
import string


from rank_bm25 import BM25Okapi


class BM25Ranker:
    def __init__(self) -> None:
        self.WINDOWS_PATH = "data/windows.txt"
        self.WINDOW_STEP = 3
        self.WINDOW_SENTENCES = 4
        self.cfg = Config()
        self.lemmatizer = WordNetLemmatizer()
        self.doc = load_spacy_output()
        self.we = WordExpansion()
        self.bm25: BM25Okapi = self.get_bm25()

    def get_bm25(self) -> BM25Okapi:    
            
        # Load the windows
        self.windows = self.create_windows()
        model = model = BM25Okapi(self.windows)
        return model

    # tokenize into words and lemmatize
    def __tokenize_string(self, text: str) -> list[str]:
        # Process hyphens and tokenize the input text
        hypens = ["-", "—"]

        for hyphen in hypens:
            text = text.replace(hyphen, "")

        tokens = word_tokenize(text.lower())
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        return tokens

    def create_windows(self) -> Tuple[List[str], List[List[int]]]:
        print("Creating windows...")

        windows = []

        number_sentences = len(list(self.doc.sents))
        
        for i in range(0, number_sentences, self.WINDOW_STEP):
            window_sents = [sent.text for sent in list(self.doc.sents)[i:i+self.WINDOW_SENTENCES]]
            window = " ".join(window_sents)

            # Remove punctuation
            window = window.translate(str.maketrans('', '', string.punctuation))

            
            # Tokenize and keep only alphanumeric tokens
            tokens = self.__tokenize_string(window)
            # get lemmas for each word
            filtered_tokens = [word for word in tokens if word.isalnum()]

            windows.append(filtered_tokens)
        return windows
    
    
    def get_best_docs_for_query(self, query: str, top_n: int = 5) -> List[Tuple[str, float]]:
        # Normalize the query
        query = self.__tokenize_string(query)
        
        # Get BM25 scores for the query
        scores = self.bm25.get_scores(query)
        doc_scores = list(enumerate(scores))
        
        # Sort documents by BM25 score (high to low)
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return the top N documents and their scores
        top_docs = []
        for doc_id, score in doc_scores[:top_n]:
            content = " ".join(self.windows[doc_id])  # Map the document index to the window content
            top_docs.append((content, score))

        
        all_in_order_of_id = sorted(doc_scores, key=lambda x: x[0])

        print("All scores in order of document ID:")

        
        return top_docs
    

cfg = Config()
def make_or_load_bm25() -> BM25Ranker:
    if os.path.exists(cfg.BM25_MODEL_PATH):
        with open(cfg.BM25_MODEL_PATH, "rb") as file:
            print("Loading BM25 model...")
            bm25 = pickle.load(file)
    else:
        print("Creating BM25 model...")
        bm25 = BM25Ranker()
        with open(cfg.BM25_MODEL_PATH, "wb") as file:
            pickle.dump(bm25, file)
    return bm25

if __name__ == "__main__":
    bm25 = make_or_load_bm25()
    query = "golgi apparatus"
    ranked_sentences = bm25.get_best_docs_for_query(query, top_n=5)

    for i, (content, score) in enumerate(ranked_sentences):
        print(f"Document {i + 1} with score {score}:")
        print(content)
        print("=" * 80)


"""
from rank_bm25 import BM25Okapi

# Sample documents (lines of text)
documents = [
    "The cat in the hat.",
    "A cat is a fine pet.",
    "Dogs and cats make great pets.",
    "I love my pet cat."
]

# Preprocess the documents: split into tokens and remove punctuation
tokenized_docs = [doc.lower().replace('.', '').split() for doc in documents]

# Initialize the BM25 model
bm25 = BM25Okapi(tokenized_docs)

# Define a query
query = "cat pets"
tokenized_query = query.lower().split()

# Compute BM25 scores
scores = bm25.get_scores(tokenized_query)

# Display the scores
for i, score in enumerate(scores):
    print(f"Document {i}: {documents[i]} (Score: {score:.4f})")
"""