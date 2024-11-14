from lib.base_index import Index
from lib.my_bm25_index import BM25Index
from config import Config
from lib.tokenizer import Tokenizer
from lib.my_faiss_index import FaissIndex

def harmonic_mean_with_beta(p, r, beta=1):
    # Higher beta gives more weight to r, while lower beta gives more weight to p
    beta = max(beta, 0.0)
    beta = min(beta, 1.0)

    if p + r == 0:  
        return 0.0

    beta_squared = beta ** 2
    f_beta_score = (1 + beta_squared) * (p * r) / (beta_squared * p + r)
    return f_beta_score

if __name__ == "__main__":
    config = Config()
    tokenizer = Tokenizer(config)
    index = BM25Index(config.BM25_MODEL_PATH, config.BM25_MODEL_PATH, tokenizer)
    faiss_index = FaissIndex(config.FAISS_MODEL_PATH, config.FAISS_MODEL_PATH, tokenizer)

    query = "What is transported into the cell membrane"
    _, original_sentences = tokenizer.load_sentences()
    doc_ids_to_rerank = []

    print("\n\nRESULTS FROM BM25")
    hits = index.search(query, 100)
    for hit in hits[:5]:
        hit_doc_id = int(hit.docid)
        sentence = original_sentences[hit_doc_id]
        print(f"DOC {hit_doc_id}: {sentence}\n\n")
        doc_ids_to_rerank.append(hit_doc_id)

    print("RESULTS FROM FAISS")
    # now rerank the results from bm25 with faiss
    faiss_reranked = faiss_index.rank_doc_ids(query, doc_ids_to_rerank)

    reranked_doc_ids = [doc_id for doc_id, _ in faiss_reranked]

    for doc_id, score in faiss_reranked[:5]:
        sentence = original_sentences[doc_id]
        print(f"HIT {doc_id}: {sentence}\n\n")


    # weighted harmonic mean between bm25 and faiss scores


    


