import spacy
import pickle
import gzip
from spacy.tokens.doc import Doc
from config import Config
from collections import Counter
import os
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
import string
from nltk.corpus import stopwords
from tqdm import tqdm


stop_words = set(stopwords.words('english'))  # Load stop words


"""
pip install spacy
python -m spacy download en_core_web_sm  # Download the English model
"""


SPACY_OUT = "data/spacy_output.pkl.gz"
cfg = Config()
# download en_core_web_sm
# python -m spacy download en_core_web_sm

spacy.cli.download("en_core_web_sm")

class Tokenizer():
    def __init__(self, cfg: Config):
        # This will be used for the BM25 model
        self.LEMMATIZED_SENTENCES_PATH = "data/lemmatized_sentences.dat"
        # This will be used for the Faiss model
        self.ORIGINAL_SENTENCES_PATH = "data/original_sentences.dat"
        self.WINDOW_STEP = 2
        self.WINDOW_SENTENCES = 5
        self.lemmatizer = WordNetLemmatizer()
        self.lemmatized_sentences = []
        self.original_sentences = []
        self.cfg = cfg
    
    def get_lemmatized_sentences_dir(self):
        return os.path.dirname(self.LEMMATIZED_SENTENCES_PATH)

    def get_lemmatized_sentences_path(self):
        return self.LEMMATIZED_SENTENCES_PATH

    def get_original_sentences_path(self):
        return self.ORIGINAL_SENTENCES_PATH
    
    def get_original_sentence(self, doc_id):
        return self.original_sentences[doc_id]
    
    def create_sentences(self):
        windows = []
        doc = self.load_spacy_output()
        number_sentences = len(list(doc.sents))

        if os.path.exists(self.LEMMATIZED_SENTENCES_PATH) and os.path.exists(self.ORIGINAL_SENTENCES_PATH):
            # delete the files
            #os.remove(self.LEMMATIZED_SENTENCES_PATH)
            #os.remove(self.ORIGINAL_SENTENCES_PATH)
            return

        with open(self.LEMMATIZED_SENTENCES_PATH, "w") as file_lemmatized:
            with open(self.ORIGINAL_SENTENCES_PATH, "w") as file_original:
                for i in tqdm(range(0, number_sentences, self.WINDOW_STEP), "Creating windows"):
                    window_sents = [sent.text for sent in list(doc.sents)[i:i+self.WINDOW_SENTENCES]]
                    window = " ".join(window_sents) 
                    # remove newlines
                    window = window.replace("\n", "")
                    window = " ".join(window.split())
                    window_original = window
                    tokens = self.__tokenize_string_bm25(window)
                    # get lemmas for each word
                    windows.append(tokens)
                    file_lemmatized.write(" ".join(tokens) + "\n")
                    file_original.write(window_original + "\n")                    
        return windows
    
    def load_sentences(self):
        with open(self.LEMMATIZED_SENTENCES_PATH, "r") as file_lemmatized:
            self.lemmatized_sentences = [line.strip() for line in file_lemmatized]

        with open(self.ORIGINAL_SENTENCES_PATH, "r") as file_original:
            self.original_sentences = [line.strip() for line in file_original]
        return self.lemmatized_sentences, self.original_sentences

    def __tokenize_string_bm25(self, text: str) -> list[str]:
        # Process hyphens and tokenize the input text
        hypens = ["-", "—"]

        for hyphen in hypens:
            text = text.replace(hyphen, "")

        tokens = word_tokenize(text.lower())
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        # remove stop words
        tokens = [word for word in tokens if word not in stop_words]
        return tokens

    def tokenize_text(self):
        # up the maximum length of text that can be processed to 10 gb
        # Load the spaCy model
        nlp = spacy.load("en_core_web_sm")
        nlp.max_length = 10000000000
        IN_FILE = self.cfg.TEXT_PATH   

        if os.path.exists(SPACY_OUT):
            return 

        # Read the text file
        with open(IN_FILE, "r") as file:
            text = file.read()
            # Process the text with spaCy
            doc = nlp(text)

            with gzip.open(SPACY_OUT, 'wb') as f:
                pickle.dump(doc, f)
        return doc


    def load_spacy_output(self) -> Doc:
        with gzip.open(SPACY_OUT, 'rb') as f:
            doc: Doc = pickle.load(f)
        return doc
    
    def build_windows(self):
        print("Tokenizing text...")
        # 1. Tokenize the text
        self.tokenize_text()
        print("Tokenization complete")
        print("Creating windows...")
        # 2. Create the windows
        self.create_sentences()
        # 3. load the windows into memory
        self.load_sentences()

    """
    if __name__ == "__main__":
        # Load or create the spacy output, construct the vocabulary and windows
        print("Tokenizing text...")
        tokenize_text(self.cfg.TEXT_PATH)
        print("Tokenization complete")
        print("Loading text into spacy doc...")
        doc = load_spacy_output()
        print("Doc loaded")
    """
            
