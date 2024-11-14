from lib.base_index import Index
from pyserini.search.lucene import querybuilder
from collections import Counter
from pyserini.search.lucene import LuceneSearcher

class BM25Index(Index):
    def __init__(self, index_dir, output_dir, tokenizer):
        super().__init__(index_dir, output_dir, tokenizer)
    
    def search(self, query, top_k=10):
        # search with BM25
        searcher: LuceneSearcher = self.get_index()
        searcher.set_rocchio()
        hits = searcher.search(query, k=top_k)
        # map the hits to (doc_id, score) tuples and return
        return hits
        

