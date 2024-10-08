from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
from datasets import Dataset
from tokenizer import load_spacy_output

# Load a pre-trained model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)  # Using AutoModel instead of AutoModelForSequenceClassification

# Function to get sentence embeddings
def get_sentence_vector(sentence):
    inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Take the mean of the token embeddings to get a single vector for the sentence
    return outputs.last_hidden_state.mean(dim=1).squeeze()

def generate_embeddings():
    # Load the data
    doc = load_spacy_output()
    sentence_objs = list(doc.sents)
    sentences = [sentence.text for sentence in sentence_objs]

    # Create a DataFrame
    df = pd.DataFrame(sentences, columns=["text"])

    # Compute embeddings for each sentence
    embeddings = []
    sentence_num = 0
    for sentence in df["text"]:
        sentence_obj = sentence_objs[sentence_num]

        # include the location of the sentence in the text with the embedding
        location = (sentence_obj.start, sentence_obj.end)
        vec = get_sentence_vector(sentence)
        embeddings.append((location, vec.numpy()))
        sentence_num += 1

    # Optionally save the embeddings and sentences
    df["embeddings"] = embeddings
    df.to_csv("sentence_embeddings.csv", index=False)

if __name__ == "__main__":
    generate_embeddings()  # Generate embeddings without needing to train
