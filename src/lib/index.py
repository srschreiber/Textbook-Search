import os
import json
from tqdm import tqdm
from pyserini.search.lucene import LuceneSearcher
from transformers import AutoTokenizer, AutoModel
import faiss
import torch
import multiprocessing
from threading import Thread
from tqdm import tqdm
from lib.tokenizer import Tokenizer
import numpy as np
import subprocess

class Index():
    def __init__(self, index_dir, output_dir, tokenizer):
        self.index_dir = index_dir
        self.output_dir = output_dir
        self.tokenizer = tokenizer
    
    def get_original_sentence(self, doc_id):
        return self.tokenizer.get_original_sentence(doc_id)
    
    def build_index(self):
        index_dir = self.index_dir
        if os.path.exists(index_dir) and os.listdir(index_dir):
            print(f"Index already exists at {index_dir}. Skipping index building.")
            return

        cmd = [
            "python", "-m", "pyserini.index.lucene",
            "--collection", "JsonCollection",
            "--input", self.tokenizer.get_lemmatized_sentences_path(),
            "--index", index_dir,
            "--generator", "DefaultLuceneDocumentGenerator",
            "--threads", "1",
            "--storePositions", "--storeDocvectors", "--storeRaw"
        ]
        subprocess.run(cmd, check=True)
    
    def get_index(self):
        if not os.path.exists(self.index_dir) or not os.listdir(self.index_dir):
            self.build_index()
        return LuceneSearcher(self.index_dir)

    def search(searcher, query, top_k=10):
        hits = searcher.search(query, k=top_k)
        
        # map the hits to (doc_id, score) tuples and return
        return [(int(hit.docid), hit.score) for hit in hits]

