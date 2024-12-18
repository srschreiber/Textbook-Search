import os
from pyserini.search.lucene import LuceneSearcher
from lib.tokenizer import Tokenizer
import subprocess
import json


class Index():
    def __init__(self, index_path, index_input, tokenizer: Tokenizer):
        self.index_path = index_path
        self.tokenizer = tokenizer
        self.INPUT_PATH = index_input   
    
    # first, read the lemmatized sentences, write as json to the INPUT_PATH
    def prepare_index_inputs(self, windows_are_lines=False):
        if os.path.exists(self.INPUT_PATH):
            
            return
        os.makedirs(os.path.dirname(self.INPUT_PATH), exist_ok=True)

        # first, build the windows
        self.tokenizer.build_windows(windows_are_lines)

        sentences, _ = self.tokenizer.load_sentences()
        i = 0
        with open(self.INPUT_PATH, "w") as file:
            for sentence in sentences:
                file.write(json.dumps({"id": i, "contents": sentence}) + "\n")
                i += 1
    
    def build_index(self, windows_are_lines=False):
        if os.path.exists(self.index_path):
            print(f"Index already exists at {self.index_path}. Skipping index building.")
            return
        
        self.prepare_index_inputs(windows_are_lines)

        cmd = [
            "python", "-m", "pyserini.index.lucene",
            "--collection", "JsonCollection",
            "--input", os.path.dirname(self.INPUT_PATH),
            "--index", self.index_path,
            "--generator", "DefaultLuceneDocumentGenerator",
            "--threads", "1",
            "--storePositions", "--storeDocvectors", "--storeRaw"
        ]
        subprocess.run(cmd, check=True)
    
    def get_index(self):
        if not os.path.exists(self.index_path):
            self.build_index()
        return LuceneSearcher(self.index_path)

    def search(searcher, query, top_k=10):
        hits = searcher.search(query, k=top_k)
        
        # map the hits to (doc_id, score) tuples and return
        return [(int(hit.docid), hit.score) for hit in hits]