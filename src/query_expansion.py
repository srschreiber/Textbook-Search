import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
from nltk.corpus.reader.wordnet import WordNetError
from nltk.stem import WordNetLemmatizer
from typing import List, Tuple
from tokenizer import get_text_vocabulary
from config import Config

nltk.download("wordnet")
nltk.download("wordnet_ic")

class WordExpansion:
    def __init__(self):
        self.wnl: WordNetLemmatizer = WordNetLemmatizer()
        self.brown_ic = wordnet_ic.ic("ic-brown.dat")
        self.cfg = Config()
    
    
    def vocab_to_lemmas(self, vocab: set) -> set:
        lemmas = set()
        for word in vocab:
            lemmas.add(self.wnl.lemmatize(word))
        return lemmas

    def expand_word(self, word: str, vocab: set = None, max_distance: int = 1) -> List[Tuple[str, float]]:
        # Get the root word
        root_word = self.wnl.lemmatize(word)
        
        queue = [(root_word, 1)]
        output = []
        processed = set([root_word])

        # Perform a BFS
        while queue:
            w, depth = queue.pop(0)

            if depth > max_distance:
                continue

            # Get the synsets for the word
            synsets = wn.synsets(w)

            for synset in synsets:
                # Get the lemmas for the synset
                if synset.lemmas():
                    lemmas = synset.lemmas()
                else:
                    continue

                # Filter out lemmas that are not in the vocab
                if vocab is not None:
                    lemmas = [lemma for lemma in lemmas if lemma.name() in vocab]

                for lemma in lemmas:
                    if lemma.name() not in processed:
                        # Calculate similarity if needed
                        node = (lemma.name(), depth)
                        output.append(node)
                        queue.append((lemma.name(), depth + 1))
                        processed.add(lemma.name())

        # Sort the synonyms by the depth or similarity
        output = sorted(output, key=lambda x: x[1])
        return output

# Test it out with a word
if __name__ == "__main__":
    we = WordExpansion()
    vocab = get_text_vocabulary()
    lemmas = we.vocab_to_lemmas(vocab)
    out = we.expand_word("run", vocab=lemmas, max_distance=1)
    print(out)