import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import nltk

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
    with torch.no_grad():
        outputs = model(**inputs)
    # Return the mean of the last hidden states as the embedding
    return outputs.last_hidden_state.mean(dim=1).numpy()


def select_k_best_words(query, words, k=5):
    scores = []
    q_num_words = len(query)
    # filter out words stop words
    words = [word for word in words if word not in stop_words]

    if isinstance(query, list):
        # filter out stop words
        query = [word for word in query if word not in stop_words]
        query = "subjects closely related to these key terms " + "[" + ", ".join(query) + "]"
    query_embedding = get_embedding(query)
    # if query is list, join it into a string

    for word in words:
        # Get embeddings
        combined_embedding = get_embedding("subjects closely related to " + word)

        # Calculate cosine similarity
        similarity_score = cosine_similarity(query_embedding, combined_embedding)
        scores.append((word, similarity_score[0][0]))
    return sorted(scores, key=lambda x: x[1], reverse=True)[:k]

if __name__ == "__main__":
    words = ["cell", "membrane", "structure", "the alamo", "function", "biology", "chemistry", "physics", "math", "nucleus", "membrane", "computer", "science", "school", "city", "bus", "train", "car", "plane", "airport", "station", "building", "house", "apartment", "room", "kitchen", "living", "bedroom", "bathroom", "toilet", "sink", "shower", "mirror", "window", "door", "floor", "wall", "ceiling", "light", "lamp", "table", "chair", "sofa", "couch", "desk", "shelf", "book", "magazine", "newspaper", "television", "remote", "phone", "computer", "keyboard", "mouse", "monitor", "printer", "scanner", "speaker", "headphone", "microphone", "camera", "battery", "charger", "cable", "wire", "plug", "socket", "switch", "button", "knob", "handle", "lock", "key", "safe", "drawer", "cabinet", "closet", "wardrobe", "mirror", "picture", "frame", "clock", "watch", "jewelry", "ring", "necklace", "bracelet", "earring", "hat", "cap", "scarf", "glove", "mitten", "sock", "shoe", "boot", "sandal", "slipper", "shirt", "t-shirt", "sweater", "jumper", "jacket", "coat", "suit", "dress", "skirt", "trousers", "jeans", "shorts", "underwear", "bra", "panties", "boxers", "briefs", "vest", "tie", "belt", "wallet", "purse", "bag", "backpack", "suitcase", "umbrella", "glasses", "sunglasses", "hat", "cap", "scarf", "glove", "mitten", "sock", "shoe", "boot", "sandal", "slipper", "shirt", "t-shirt", "sweater", "jumper", "jacket", "coat", "suit", "dress", "skirt", "trousers", "jeans", "shorts", "underwear", "bra", "panties", "boxers", "briefs", "vest", "tie", "belt"]
    print(select_k_best_words("cell membrane structure", words, k=10))