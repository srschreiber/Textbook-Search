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
from transformer import get_similarity
import threading
import multiprocessing
import time

# stopwords
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


from rank_bm25 import BM25Okapi

stop_words = set(stopwords.words('english'))  # Load stop words
class BM25Ranker:
    def __init__(self) -> None:
        self.WINDOWS_PATH = "data/windows.txt"
        self.WINDOW_STEP = 2
        self.WINDOW_SENTENCES = 5
        self.cfg = Config()
        self.lemmatizer = WordNetLemmatizer()
        self.doc = load_spacy_output()
        self.vocab = set()

        # load the vocab
        with open(self.cfg.VOCAB_PATH, "r") as file:
            for line in file:
                self.vocab.add(self.lemmatizer.lemmatize(line.strip()))
        

        self.we = WordExpansion(self.vocab)
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
        # remove stop words
        tokens = [word for word in tokens if word not in stop_words]
        return tokens

    def create_windows(self) -> Tuple[List[str], List[List[int]]]:
        print("Creating windows...")

        windows = []

        number_sentences = len(list(self.doc.sents))
        
        for i in range(0, number_sentences, self.WINDOW_STEP):
            window_sents = [sent.text for sent in list(self.doc.sents)[i:i+self.WINDOW_SENTENCES]]
            window = "".join(window_sents) 
            # remove newlines
            window = window.replace("\n", "")
            window = " ".join(window.split())
            tokens = self.__tokenize_string(window)
            # get lemmas for each word
            windows.append(tokens)
        return windows
    
    
    def get_best_docs_for_query(self, query: str, top_bm25: int = 10, top_out: int = 10, expand=True) -> List[Tuple[str, float]]:
        original_query = query
        # Normalize the query
        query = self.__tokenize_string(query)
        expanded_terms = []

        # expansion isnt helpful if the query is short because they are likely looking for something specific
        if expand and len(query) >= 3:
            expanded_terms.extend(self.we.expand_words(query))
                
        # now weight the terms based on the expansion scores
        term_weights = Counter()

        for term, _ in expanded_terms:
            term_weights[term] += 1

        p = .5
        # normalize so that original query takes up 70% of the weight
        total_expansion_weight = sum(term_weights.values())
        if total_expansion_weight >= 1:
            num_base = len(query)

            """
            x/(x + total_expansion_weight) = p
            x = p(x + total_expansion_weight)
            x = px + ptotal_expansion_weight
            x - px = ptotal_expansion_weight
            (1-p)x = ptotal_expansion_weight
            x = ptotal_expansion_weight / (1-p)
            """
            total_original_weight = p * total_expansion_weight / (1 - p)
            individual_original_weight = total_original_weight / num_base

            # add the individual weights for the original query
            for term in query:
                term_weights[term] += individual_original_weight
        else:
            # discount the expansion terms
            term_weights = Counter()
            for term in query:
                term_weights[term] += 1
        # now unroll the weights into the query
        query = []

        for term, weight in term_weights.items():
            query.extend([term] * round(weight))
        
        # Get BM25 scores for the query
        scores = self.bm25.get_scores(query)
        doc_scores = list(enumerate(scores))
        
        # Sort documents by BM25 score (high to low)
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return the top N documents and their scores
        # take a extra because we are going to re-rank them
        top_docs = doc_scores[:top_bm25*2]

        # capture original rankings (array position) because its interesting
        original_rankings = {doc: i for i, (doc, _) in enumerate(top_docs)}

        # now re-rank using the transformer
        similarities = []

        NUM_THREADS = multiprocessing.cpu_count()
        t_scores = [[] for _ in range(NUM_THREADS)]
        threads = []

        for i in range(NUM_THREADS):
            batch = top_docs[i * len(top_docs) // NUM_THREADS:(i + 1) * len(top_docs) // NUM_THREADS]
            def task(batch=batch, tid=i):
                for doc, _ in batch:
                    content = " ".join(self.windows[doc])
                    similarity = get_similarity(original_query, content)
                    t_scores[tid].append((doc, similarity, content, original_rankings[doc]))
            thread = threading.Thread(target=task)
            threads.append(thread)
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        for thread_score in t_scores:
            similarities.extend(thread_score)

        # sort by similarity
        top_docs = sorted(similarities, key=lambda x: x[1], reverse=True)
        
        top_docs_with_harmonic_mean = []

        for rank, (doc, score, content, original_ranking) in enumerate(top_docs):
            # compute harmonic mean of the two scores
            hm = 2 * (rank+1) * (original_ranking+1) / ((rank+1) + (original_ranking+1))
            top_docs_with_harmonic_mean.append((doc, score, content, original_ranking,rank, hm))
        
        # finally sort by the harmonic mean
        top_docs = sorted(top_docs_with_harmonic_mean, key=lambda x: x[5])[:top_out]


        
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
    query = "how dna mutations lead to malformed proteins"

    start_time = time.time()
    print(f"Querying BM25 for '{query}'...")
    # consider 300 bm25 results to rerank to top 20
    ranked_sentences = bm25.get_best_docs_for_query(query, top_bm25=300, top_out=20, expand=True)

    end_time = time.time()
    rank = 0
    for i, score, content, original_ranking, transformer_rank, harmonic_mean in ranked_sentences:
        print(f"Document {i + 1} rank={rank}, harmonic_mean={harmonic_mean},  bm25 rank={original_ranking},transformer_rank={transformer_rank} with score {score}:")
        print(content)
        print("=" * 80)
        rank += 1

    print(f"Query took {end_time - start_time} seconds to compute top-{len(ranked_sentences)} documents")
    
    


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