from lib.my_bm25_index import BM25Index
from config import Config
from lib.tokenizer import Tokenizer
from lib.my_faiss_index import FaissIndex
import math
import numpy as np

def harmonic_mean_with_beta(p, r, beta=1, sensitivity_constant=.4):
    # Higher beta gives more weight to r, while lower beta gives more weight to p
    beta = max(beta, 0.0)

    p += sensitivity_constant
    r += sensitivity_constant

    if p + r == 0:  
        return 0.0

    beta_squared = beta ** 2
    f_beta_score = (1 + beta_squared) * (p * r) / (beta_squared * p + r)
    return f_beta_score

def geometric_mean(values, weights):
    values = np.array(values, dtype=np.float64)
    weights = np.array(weights)
    return np.prod(values ** (weights / np.sum(weights)))

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

def search_top_k(index, faiss_index, query, top_k=10, initial_search_k=1000, dont_rerank=False, score_function=lambda p, r: geometric_mean([p, r], [1, 1.5])):
    query_bm25 = tokenizer.lemmatize_query(query)
    hits = index.search(query_bm25, initial_search_k)
    doc_ids_to_rerank = []

    for hit in hits:
        hit_doc_id = int(hit.docid)
        doc_ids_to_rerank.append(hit_doc_id)
    
    if len(doc_ids_to_rerank) == 0:
        return
    
    bm25_ranking = {}
    for i, hit in enumerate(hits):
        doc_id = int(hit.docid)
        bm25_ranking[doc_id] = i

    if dont_rerank:
        return doc_ids_to_rerank[:top_k], bm25_ranking, None, None

    # now rerank the results from bm25 with faiss
    faiss_reranked = faiss_index.rank_doc_ids(query, doc_ids_to_rerank)
    reranked_doc_ids = [doc_id for doc_id, _ in faiss_reranked]

    faiss_ranking = {}
    rank = 0
    for doc_id, _ in faiss_reranked:
        faiss_ranking[doc_id] = rank
        rank += 1
    
    # Now compute the harmonic mean of each document
    scores = {}
    for doc_id in doc_ids_to_rerank:
        # Sensitivity constant is used to make small numbers less sensitive
        sensitivity_constant = 10
        p = bm25_ranking[doc_id] 
        r = faiss_ranking[doc_id] 
        scores[doc_id] = score_function(p, r)
    
    # sort by harmonic mean
    sorted_docs = sorted(scores.keys(), key=lambda x: scores[x], reverse=False)

    # todo: remove docs ids that are close together 
    return sorted_docs[:top_k], bm25_ranking, faiss_ranking, scores
# remember that there are 
def compute_ndcg(results, qrels, k=10):
    def dcg(relevances):
        dcg_simple = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances[:k]))
        return dcg_simple

    ndcg_scores = []
    for qid, query_results in results.items():
        if qid not in qrels:
            continue
        relevances_current = [qrels[qid].get(docid, 0) for _, docid in query_results]
        idcg = dcg(sorted(qrels[qid].values(), reverse=True))
        if idcg == 0:
            continue
        ndcg_scores.append(dcg(relevances_current) / idcg)

    if not ndcg_scores:
        return 0.0
    return np.mean(ndcg_scores)

def compute_mean_average_precision(results, qrels, k=10):
    average_precision_scores = []
    
    for qid, query_results in results.items():
        if qid not in qrels:
            continue
        
        relevant_docs = qrels[qid]
        hits = 0
        precision_at_k = []

        for i, (_, docid) in enumerate(query_results):
            if i >= k:
                break
            
            if docid in relevant_docs:
                hits += 1
                precision_at_k.append(hits / (i + 1))
        
        if precision_at_k:  # Avoid division by zero
            average_precision = sum(precision_at_k) / len(relevant_docs)
            average_precision_scores.append(average_precision)
        else:
            average_precision_scores.append(0)
    
    if average_precision_scores:
        return np.array(average_precision_scores).mean()
    else:
        return 0

if __name__ == "__main__":
    config = Config()
    # cranfield mode will evaluate ndcg over the cranfield dataset to help test the parameters,
    # although cranfield is much better tailored to the bm25 model with strong
    mode = "cranfield"

    print("Welcome to the BM25 + Faiss search demo!")
    print("It takes a couple hours to index the files in this project from scratch, so I have pre-processed the model files for you in this submission.")
    mode_str = input("Would you like to run in model evaluation mode? (n for the main program) (y/n): ").lower()
    if mode_str == "y":
        mode = "cranfield"
    else:
        mode = "interactive"
    
    query_start_id = 0
    knowledge_level = 0

    score_function_map = {
        0: lambda p, r: geometric_mean([p, r], [1, 1.5]),
        1: lambda p, r: p
    }

    if mode != "cranfield":
        # use default paths
        tokenizer = Tokenizer(config.TEXT_PATH)
        index = BM25Index(config.BM25_MODEL_PATH, config.DEFAULT_BM25_INDEX_INPUT, tokenizer)
        faiss_index = FaissIndex(config.FAISS_MODEL_PATH, tokenizer)
        index.build_index()
        faiss_index.build_index()
        _, original_sentences = tokenizer.load_sentences()
    else:
        # cranfield starts at 1 for qrels
        query_start_id = 1 
        tokenizer = Tokenizer(config.CRANFIELD_PATH, config.CRANFIELD_SENTENCES_PATH, config.CRANFIELD_LEMMATIZED_SENTENCES_PATH, config.CRANFIELD_SPACY_OUT)
        index = BM25Index(config.BM25_CRANFIELD_MODELS_PATH, config.CRANFIELD_BM25_INDEX_INPUT, tokenizer)
        faiss_index = FaissIndex(config.FAISS_CRANFIELD_MODELS_PATH, tokenizer)
        index.build_index()
        faiss_index.build_index()
        _, original_sentences = tokenizer.load_sentences()


    # compute ndcg if cranfield
    if mode == "cranfield":
        knowledge_level = 0
        queries_path = config.CRANFIELD_QUERIES_LESS_SPECIFIC_PATH
        qrels = {}
        with open(config.CRANFIELD_QREL_PATH) as file:
            for line in file:
                qid, docid, relevance = line.strip().split()
                qid = int(qid)
                docid = int(docid)
                relevance = int(relevance)
                if qid not in qrels:
                    qrels[qid] = {}
                qrels[qid][docid] = relevance

        results = {}
        top_k = 10
        with open(queries_path) as file:
            qid = query_start_id
            for line in file:
                query = line.strip()
                sorted_docs, bm25_ranking, faiss_ranking, hmean = search_top_k(index, faiss_index, query, top_k=top_k, initial_search_k=1000, dont_rerank=False, score_function=score_function_map[knowledge_level])
                results[qid] = [(qid, doc) for doc in sorted_docs]
                qid += 1

        # Note: I started off using a harmonic mean, but switched to a geometric mean because the ndcg was much better
        ndcg = compute_ndcg(results, qrels, k=top_k)
        print(f"Computed NDCG@{top_k}: {ndcg}")
        map_at_k = compute_mean_average_precision(results, qrels, k=top_k)
        print(f"Computed MAP@{top_k}: {map_at_k}")
    else:
        
        welcome_text = "Welcome to the BM25 + Faiss search demo! \n"\
            "You can enter a query and the system will return the top 5 results for that query.\n"\
            "There are two modes you can use by entering the following numbers:\n\n"\
            "1: The system will favor results that have a good BM25 rank or Embedding rank (Geometric mean of the ranks)\n"\
            "2: The system will favor results that have a good BM25 rank (BM25 only)\n\n"
        
        # make green
        welcome_text = f"\033[92m{welcome_text}\033[0m"
        print(welcome_text)

        while True:
            res = input("Please select your search mode 1-2: ")
            if not res.isdigit() or int(res) < 1 or int(res) > 2:
                print("Invalid knowledge level, defaulting to 1.")
                res = "1"

            knowledge_level = int(res) - 1

            doc_ids_to_rerank = []
            query = input("Enter query: ")

            if not query:
                continue

            if query == "exit":
                break
    
            top_k = 5
            sorted_docs, bm25_ranking, faiss_ranking, scores = search_top_k(index, faiss_index, query, top_k=top_k, initial_search_k=1000, score_function=score_function_map[knowledge_level])
            print('*' * 100)
            print(f"Top {top_k} results for query '{query}':")
            for i, doc_id in enumerate(sorted_docs[:top_k]):
                sentence = original_sentences[doc_id]
                bm25_rank = bm25_ranking[doc_id]
                embedding_rank = faiss_ranking[doc_id]
                score = scores[doc_id]
                formatted_sentence = format_sentence(sentence)
                # make formatted_sentence blue
                formatted_sentence = f"\033[94m{formatted_sentence}\033[0m"
                header = f"DOC {doc_id}, score: {score}, bm25 rank: {bm25_rank}, embedding rank: {embedding_rank}"
                # make header red
                header = f"\033[91m{header}\033[0m"
                print(f"{header}\n{formatted_sentence}\n\n")
            print('*' * 100)

    # weighted harmonic mean between bm25 and faiss scores


    


