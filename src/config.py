"""
Config for global variables
"""

class Config:
    def __init__(self) -> None:
        self.TEXT_PATH = "data/text.txt"
        self.DEFAULT_BM25_INDEX_INPUT = "data/index_inputs/bm25.json"

        self.CRANFIELD_PATH = "data/cranfield/cranfield.txt"

        self.BM25_MODEL_PATH = "data/bm25.model"
        self.FAISS_MODEL_PATH = "data/faiss.model"
        self.BM25_CRANFIELD_MODELS_PATH = "data/cranfield/bm25_cranfield_model.model"
        self.FAISS_CRANFIELD_MODELS_PATH = "data/cranfield/cranfield_faiss_model.model"

        self.CRANFIELD_QREL_PATH = "data/cranfield/cranfield_qrels.txt"
        self.CRANFIELD_QUERIES_PATH = "data/cranfield/cranfield_queries.txt"
        self.CRANFIELD_QUERIES_SHORT_PATH = "data/cranfield/cranfield_queries_short.txt"
        self.CRANFIELD_QUERIES_LESS_SPECIFIC_PATH = "data/cranfield/cranfield_queries_less_specific.txt"

        self.CRANFIELD_SENTENCES_PATH = "data/cranfield/tokenized_cranfield_original_sentences.dat"
        self.CRANFIELD_LEMMATIZED_SENTENCES_PATH = "data/cranfield/tokenized_cranfield_lemmatized_sentences.dat"

        self.CRANFIELD_BM25_INDEX_INPUT = "data/cranfield/index_inputs/bm25_cranfield.json"
        self.CRANFIELD_SPACY_OUT = "data/cranfield/spacy_output.pkl.gz"
