import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import threading
import time
import multiprocessing
import pickle
import os

nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

# Load the pretrained model and tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


# Function to get embeddings
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt")
    # if dimension exceeds 512, truncate it. This is a limitation of the model but sliding overlapping windows might make this less of an issue
    if inputs["input_ids"].shape[1] > 512:
        inputs["input_ids"] = inputs["input_ids"][:, :512]
        inputs["attention_mask"] = inputs["attention_mask"][:, :512]
    with torch.no_grad():
        outputs = model(**inputs)
    # Return the mean of the last hidden states as the embedding
    return outputs.last_hidden_state.mean(dim=1).numpy()

def get_similarity(query: str, text: str):
    query_embedding = get_embedding(query)
    text_embedding = get_embedding(text)
    return cosine_similarity(query_embedding, text_embedding)[0][0]


cosin_similarities = {}

# load from file via pickle if exists
if os.path.exists("cosine_similarities.pkl"):
    with open("cosine_similarities.pkl", "rb") as file:
        cosin_similarities = pickle.load(file)

# todo: precompute and cache the word embeddings so we only need to compute the query embedding
def select_k_best_words(query: list[str], words, k=5):


    start_time = time.time()
    print(f"Selecting {k} best words from {len(words)} words")
    NUM_THREADS = multiprocessing.cpu_count()

    # one for each thread
    thread_scores = [[] for _ in range(NUM_THREADS)]
    threads = []
    scores = []
    # filter out words stop words
    words = [word for word in words if word[0] not in stop_words]
    # filter out stop words
    query = [word for word in query if word not in stop_words]
    query = "biological subjects related to this phrase: " + " ".join(query)
    query_embedding = get_embedding(query)
    # if query is list, join it into a string

    per_thread = len(words) // NUM_THREADS
    for i in range(NUM_THREADS):
        batch = words[i * per_thread:(i + 1) * per_thread]

        def task(batch=batch, tid=i):
            for word, penalty in batch:
                # Get embeddings
                key = "biological subjects related to the word: " + word
                if key in cosin_similarities:
                    combined_embedding = cosin_similarities[key]
                else:
                    combined_embedding = get_embedding("biological subjects related to the word: " + word)
                    cosin_similarities[key] = combined_embedding

                # Calculate cosine similarity
                similarity_score = cosine_similarity(query_embedding, combined_embedding)
                thread_scores[tid].append((word, penalty * similarity_score[0][0]))
        
        thread = threading.Thread(target=task)
        threads.append(thread)
    
    for thread in threads:
        thread.start()
    
    for thread in threads:
        thread.join()

    # write cache to file for future use
    with open("cosine_similarities.pkl", "wb") as file:
        pickle.dump(cosin_similarities, file)
    
    # combine into scores list 
    scores = []
    for thread_score in thread_scores:
        scores.extend(thread_score)
        
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[:k]
    print(f"Selected {k} best words in {time.time() - start_time} seconds")

    return scores

if __name__ == "__main__":
    words = ["cell", "membrane", "structure", "the alamo", "function", "biology", "chemistry", "physics", "math", "nucleus", "membrane", "computer", "science", "school", "city", "bus", "train", "car", "plane", "airport", "station", "building", "house", "apartment", "room", "kitchen", "living", "bedroom", "bathroom", "toilet", "sink", "shower", "mirror", "window", "door", "floor", "wall", "ceiling", "light", "lamp", "table", "chair", "sofa", "couch", "desk", "shelf", "book", "magazine", "newspaper", "television", "remote", "phone", "computer", "keyboard", "mouse", "monitor", "printer", "scanner", "speaker", "headphone", "microphone", "camera", "battery", "charger", "cable", "wire", "plug", "socket", "switch", "button", "knob", "handle", "lock", "key", "safe", "drawer", "cabinet", "closet", "wardrobe", "mirror", "picture", "frame", "clock", "watch", "jewelry", "ring", "necklace", "bracelet", "earring", "hat", "cap", "scarf", "glove", "mitten", "sock", "shoe", "boot", "sandal", "slipper", "shirt", "t-shirt", "sweater", "jumper", "jacket", "coat", "suit", "dress", "skirt", "trousers", "jeans", "shorts", "underwear", "bra", "panties", "boxers", "briefs", "vest", "tie", "belt", "wallet", "purse", "bag", "backpack", "suitcase", "umbrella", "glasses", "sunglasses", "hat", "cap", "scarf", "glove", "mitten", "sock", "shoe", "boot", "sandal", "slipper", "shirt", "t-shirt", "sweater", "jumper", "jacket", "coat", "suit", "dress", "skirt", "trousers", "jeans", "shorts", "underwear", "bra", "panties", "boxers", "briefs", "vest", "tie", "belt"]
    print(select_k_best_words("cell membrane structure", words, k=10))