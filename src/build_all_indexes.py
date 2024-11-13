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

    query = "In what follows, we acknowledge and thank all of the scientists whose suggestions have helped us to prepare this edition. (A combined list of those who helped with our first, second, third, fourth, fifth, and sixth editions is also provided.) General: Joseph Ahlander (Northeastern State University), Buzz Baum (Molecular Research Institute, United Kingdom), Michael Burns (Loyola University Chicago), Silvia C. Finnemann (Fordham University), Nora Goosen (Leiden University, The Netherlands), Harold Hoops (State University of New York, Buffalo), Joanna Norris (University of Rhode Island), Mark V. Reedy (Creighton University), Jeff Singer (Portland State University), Amy Springer (University of Massachusetts), Andreas Wodarz (University of Cologne, Germany). In addition, Tiago Barros produced new molecular models for the Seventh Edition. Chapter 1: Sage Arbor (Marian University Indianapolis), Stephen E. Asmus (Centre College), Jill Banfield (University of California, Berkeley), Zoé Burke (University of Bath, United Kingdom), Elizabeth Good (University of Illinois, Urbana-Champaign), Julian Guttman (Simon Fraser University, Canada), Sudhir Kuman (Temple University), Sue Hum-Musser (Western Illinois University), Brad Mehrtens (University of Illinois, Urbana-Champaign), Inaki Ruiz-Trillo (University of Barcelona, Spain), David Stern (Janelia Research Campus), Andrew Wood (Southern Illinois University), Lidan You (University of Toronto, Canada)."
    idx = faiss_index.get_index(True)
    query_embedding = faiss_index.embed_query(query).reshape(1, -1)
    hits = idx.search(query_embedding, 5)

    _, original_sentences = tokenizer.load_sentences()

    print("RESULTS FROM FAISS")
    for hit in hits[1][0]:
        sentence = original_sentences[hit]
        print(f"HIT: {sentence}\n\n")
    
    print("\n\nRESULTS FROM BM25")
    idx = index.get_index()
    hits = idx.search(query, 10)
    for hit in hits:
        hit_doc_id = int(hit.docid)
        sentence = original_sentences[hit_doc_id]
        print(f"HIT: {sentence}\n\n")

    


