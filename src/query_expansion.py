import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
from nltk.corpus.reader.wordnet import WordNetError
from nltk.stem import WordNetLemmatizer
from typing import List, Tuple
from tokenizer import get_text_vocabulary
from config import Config
from word2vec import WordEmbedder
from collections import defaultdict
import heapq

nltk.download("wordnet")
nltk.download("wordnet_ic")

class WordExpansion:
    def __init__(self, sim_threshold: float = 0.6):
        self.wnl: WordNetLemmatizer = WordNetLemmatizer()
        self.brown_ic = wordnet_ic.ic("ic-brown.dat")
        self.cfg = Config()
        self.we = WordEmbedder()
        self.we.load()
        self.sim_threshold = sim_threshold
        self.expand_limit = 50
        
    
    def vocab_to_lemmas(self, vocab: set) -> set:
        lemmas = set()
        for word in vocab:
            lemmas.add(self.wnl.lemmatize(word))
        return lemmas

    def __related_synsets(self, syn):
        related_words = set()
        for lemma in syn.lemmas():
            related_words.add(lemma.name())

        # Hypernyms (more general terms)
        for hyper in syn.hypernyms():
            related_words.update(lemma.name() for lemma in hyper.lemmas())

        # Hyponyms (more specific terms)
        for hypo in syn.hyponyms():
            related_words.update(lemma.name() for lemma in hypo.lemmas())

        # Meronyms (parts of)
        for part in syn.part_meronyms():
            related_words.update(lemma.name() for lemma in part.lemmas())

        # Holonyms (wholes of)
        for whole in syn.member_holonyms():
            related_words.update(lemma.name() for lemma in whole.lemmas())
        
        return related_words
    
    def expand_word(self, word: str, max_distance: int = 3) -> List[Tuple[str, float]]:
        # Get the root word
        root_word = self.wnl.lemmatize(word)
        
        queue = [(root_word, 1)]
        output = []
        processed = set([root_word])

        # min heap of the k largest elements
        output = []

        # Perform a BFS
        while queue:
            w, depth = queue.pop()

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
                lemmas = self.__related_synsets(synset)

                for lemma in lemmas:
                    if lemma not in processed:
                        if not self.we.has_word(lemma):
                            continue

                        # Calculate similarity if needed
                        node = (lemma, depth + 1)
                        sim = self.we.similarity(root_word, lemma)

                        # Add to the output if the limit has not been reached
                        if len(output) < self.expand_limit and sim >= self.sim_threshold:
                            heapq.heappush(output, (sim, lemma))
                        # Replace the smallest element if the new element is larger
                        elif len(output) > 0 and sim > output[0][0]:
                            heapq.heappop(output)
                            heapq.heappush(output, (sim, lemma))


                        queue.append(node)
                        processed.add(lemma)
        
        sorted_out = []
        while output:
            sim, node = heapq.heappop(output)
            sorted_out.append((node, sim))
        return sorted_out
    
    def get_expanded_query(self, query: str) -> dict[str, set[str]]:
        # remove stopwords first
        removed_stopwords = [word for word in query.split() if word not in self.we.stop_words]

        expanded_query = defaultdict(set[str])
        for word in removed_stopwords:
            expanded = self.expand_word(word, max_distance=4)
            for w in expanded:
                expanded_query[word].add(w)

        return expanded_query

# Test it out with a word
if __name__ == "__main__":
    we = WordExpansion()

    query = "function of the cell"

    print("ORIGINAL QUERY:", query)
    print("EXPANDED QUERY:", we.get_expanded_query(query))
