from lib.base_index import Index
import os
import json
from tqdm import tqdm
# pip install pyserini
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
from sklearn.preprocessing import normalize

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
encoder = AutoModel.from_pretrained('bert-base-uncased')

class FaissIndex(Index):
    def __init__(self, index_path, output_path, tokenizer):
        super().__init__(index_path, output_path, tokenizer)
    
    def get_index(self, rebuild=False):
        if not os.path.exists(self.index_path) or rebuild:
            self.build_index(rebuild)
        return faiss.read_index(self.index_path)

    def embed_query(self, query: str):
        inputs = tokenizer(query, return_tensors='pt', truncation=True, max_length=512)
        
        with torch.no_grad():
            query_embedding = encoder(**inputs).last_hidden_state.mean(dim=1)
        
        query_embedding = query_embedding.squeeze()
        query_embedding = query_embedding / query_embedding.norm(p=2, dim=-1, keepdim=True)
        query_embedding_numpy = query_embedding.numpy()
        query_embedding_numpy = query_embedding_numpy.reshape(1, -1)
        return query_embedding_numpy

    def build_index(self, rebuild=False):
        if os.path.exists(self.index_path):
            if rebuild:
                print(f"Rebuilding index at {self.index_path}")
                # delete the existing index
                os.remove(self.index_path)
            else:
                print(f"Index already exists at {self.index_path}. Skipping index building.")
                return

        def encode_document(text):
            return self.embed_query(text)

        documents = []

        sents = self.tokenizer.get_original_sentences_path()

        with open(sents, "r") as file:
            for line in file:
                documents.append(line.strip())

        threads = []
        n_threads = multiprocessing.cpu_count()
        thread_outputs = [[] for _ in range(n_threads)]

        batches = []

        # Split documents into batches for multithreading
        batch_size = len(documents) // n_threads

        for i in range(n_threads):
            if i == n_threads - 1:
                batches.append(documents[i * batch_size:])
            else:
                batches.append(documents[i * batch_size:(i + 1) * batch_size])

        for i, batch in enumerate(batches):
            def encode_batch(pid=i, b=batch):
                output = thread_outputs[pid]
                for doc in tqdm(b):
                    output.append(encode_document(doc))

            t = Thread(target=encode_batch, args=(i, batch))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        embeddings = []
        for output in thread_outputs:
            embeddings.extend(output)


        consolidated_array = np.vstack(embeddings)
        d = consolidated_array.shape[1]
        index = faiss.IndexFlatIP(d)

        # Add document embeddings to the index
        index.add(consolidated_array)
        faiss.write_index(index, self.index_path)
        return index

    def search(self, query, top_k=10):
        index = self.get_index()
        query_embedding = self.embed_query(query)
        return index.search(query_embedding, top_k)

    # this is to rank a subset of documents
    def rank_doc_ids(self, query, doc_ids):
        # find the rankings of each doc_id
        index = self.get_index()
        query_embedding = self.embed_query(query)

        # create a temporary index with the right doc_ids and then query it
        temp_index = faiss.IndexFlatIP(index.d)

        embeddings = []
        for doc_id in doc_ids:
            # look up the embedding for the doc_id
            embedding = index.reconstruct(doc_id)
            embeddings.append(embedding)
        # stack embeddings to 2d array size len(doc_ids) x d
        embeddings = np.vstack(embeddings)
        temp_index.add(embeddings)
        
        hits = temp_index.search(query_embedding, len(doc_ids))
        
        reranked = [0] * len(doc_ids)
        ranks = hits[1].squeeze().tolist()
        scores = hits[0].squeeze().tolist()

        for i, (doc_id, score) in enumerate(zip(ranks, scores)):
            original_doc_id = doc_ids[doc_id]
            reranked[i] = (original_doc_id, score)
            
        return reranked
        
