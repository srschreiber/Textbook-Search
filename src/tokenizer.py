import spacy
import pickle
import gzip
from spacy.tokens.doc import Doc
from config import Config
from collections import Counter
import os

"""
pip install spacy
python -m spacy download en_core_web_sm  # Download the English model
"""
SPACY_OUT = "data/spacy_output.pkl.gz"
cfg = Config()

# download en_core_web_sm
# python -m spacy download en_core_web_sm

spacy.cli.download("en_core_web_sm")

def tokenize_text(text):
    # up the maximum length of text that can be processed to 10 gb
    # Load the spaCy model
    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = 10000000000
    IN_FILE = cfg.TEXT_PATH   

    # Read the text file
    with open(IN_FILE, "r") as file:
        text = file.read()
        # Process the text with spaCy
        doc = nlp(text)

        with gzip.open(SPACY_OUT, 'wb') as f:
            pickle.dump(doc, f)

def get_text_vocabulary(min_frequency=5) -> set:
    """
    Include each word in the vocabulary if it appears at least min_frequency times in the text
    """
    # if the vocabulary file exists, read it and return it
    if os.path.exists(cfg.VOCAB_PATH):
        with open(cfg.VOCAB_PATH, "r") as file:
            return {line.strip() for line in file}
        
    with open(cfg.TEXT_PATH, "r") as file:
        text = file.read()
        nlp = spacy.load("en_core_web_sm")
        # Process the text with spaCy
        nlp.max_length = 10000000000
        doc = nlp(text)
        # Get the vocabulary
        vocab = Counter()
        for word in doc:
            vocab[word.text.lower()] += 1
        # Filter the vocabulary
        vocab = {word for word, count in vocab.items() if count >= min_frequency}
        # write the vocabulary to a file
        with open(cfg.VOCAB_PATH, "w") as file:
            for word in vocab:
                file.write(word + "\n")
        return vocab


def load_spacy_output() -> Doc:
    with gzip.open(SPACY_OUT, 'rb') as f:
        doc: Doc = pickle.load(f)
    return doc

if __name__ == "__main__":
    print("Tokenizing text...")
    tokenize_text(cfg.TEXT_PATH)
    print("Tokenization complete")
    print("Loading text into spacy doc...")
    doc = load_spacy_output()
    print("Doc loaded")

    # print 100 sentences
    for i, sent in enumerate(doc.sents):
        print(sent.text)
        if i == 100:
            break
        
