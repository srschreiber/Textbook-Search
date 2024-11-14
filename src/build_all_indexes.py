from lib.base_index import Index
from lib.my_bm25_index import BM25Index
from config import Config
from lib.tokenizer import Tokenizer
from lib.my_faiss_index import FaissIndex

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

    for doc_id, score in faiss_reranked[:5]:
        sentence = original_sentences[doc_id]
        print(f"HIT {doc_id}: {sentence}\n\n")

    


