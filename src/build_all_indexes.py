from lib.my_bm25_index import BM25Index
from config import Config
from lib.tokenizer import Tokenizer
from lib.my_faiss_index import FaissIndex
import math

def harmonic_mean_with_beta(p, r, beta=1):
    # Higher beta gives more weight to r, while lower beta gives more weight to p
    beta = max(beta, 0.0)
    beta = min(beta, 1.0)

    if p + r == 0:  
        return 0.0

    beta_squared = beta ** 2
    f_beta_score = (1 + beta_squared) * (p * r) / (beta_squared * p + r)
    return f_beta_score

def format_sentence(sentence: str):
    line_length = 100

    out = []

    for i, c in enumerate(sentence):
        if i > 0 and i % line_length == 0:
            # alnum?
            if c.isalnum(): 
                out.append("-")
            out.append("\n") 
        out.append(c)
    return "".join(out)

if __name__ == "__main__":
    config = Config()
    tokenizer = Tokenizer(config)
    index = BM25Index(config.BM25_MODEL_PATH, config.BM25_MODEL_PATH, tokenizer)
    faiss_index = FaissIndex(config.FAISS_MODEL_PATH, config.FAISS_MODEL_PATH, tokenizer)
    _, original_sentences = tokenizer.load_sentences()
\
    while True:
        doc_ids_to_rerank = []
        query = input("Enter query: ")

        if query == "exit":
            break

        #print("\n\nRESULTS FROM BM25")
        hits = index.search(query, 1000)
        for hit in hits:
            hit_doc_id = int(hit.docid)
            #sentence = original_sentences[hit_doc_id]
            #print(f"DOC {hit_doc_id}: {sentence}\n\n")
            doc_ids_to_rerank.append(hit_doc_id)
        
        if len(doc_ids_to_rerank) == 0:
            print("No results found.")
            continue

        #print("RESULTS FROM FAISS")
        # now rerank the results from bm25 with faiss
        faiss_reranked = faiss_index.rank_doc_ids(query, doc_ids_to_rerank)
        reranked_doc_ids = [doc_id for doc_id, _ in faiss_reranked]

        faiss_ranking = {}
        rank = 0
        for doc_id, _ in faiss_reranked:
            faiss_ranking[doc_id] = rank
            rank += 1
        
        bm25_ranking = {}
        for i, hit in enumerate(hits):
            doc_id = int(hit.docid)
            bm25_ranking[doc_id] = i
        
        # Now compute the harmonic mean of each document
        hmean = {}
        for doc_id in doc_ids_to_rerank:
            # Make sure faiss is more sensitive to lower ranks than bm25
            sensitivity_constant_faiss = len(doc_ids_to_rerank)//10
            sensitivity_constant_bm25 = len(doc_ids_to_rerank)//3
            p = (bm25_ranking[doc_id] + sensitivity_constant_bm25) / (len(doc_ids_to_rerank) + sensitivity_constant_bm25)
            r = (faiss_ranking[doc_id] + sensitivity_constant_faiss) / (len(doc_ids_to_rerank) + sensitivity_constant_faiss)
            # beta of 2 gives more weight to recall
            hmean[doc_id] = harmonic_mean_with_beta(p, r, beta=2)
        
        # sort by harmonic mean
        sorted_docs = sorted(hmean.keys(), key=lambda x: hmean[x], reverse=False)

        # todo: remove docs ids that are close together 
        top_k = 5

        print(f"Top {top_k} results for query '{query}':")
        for i, doc_id in enumerate(sorted_docs[:top_k]):
            sentence = original_sentences[doc_id]
            bm25_rank = bm25_ranking[doc_id]
            embedding_rank = faiss_ranking[doc_id]
            hmean_score = hmean[doc_id]
            formatted_sentence = format_sentence(sentence)
            print(f"DOC {doc_id}, hmean: {hmean_score}, bm25 rank: {bm25_rank}, embedding rank: {embedding_rank}: \n{formatted_sentence}\n\n")

    # weighted harmonic mean between bm25 and faiss scores


    


