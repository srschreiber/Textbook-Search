"""
Config for global variables
"""

class Config:
    def __init__(self) -> None:
        self.TEXT_PATH = "data/text.txt"
        self.BM25_MODEL_PATH = "data/bm25.model"
        self.FAISS_MODEL_PATH = "data/faiss.model"
