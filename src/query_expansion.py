import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
from nltk.stem import WordNetLemmatizer
from typing import List, Tuple
from config import Config
from collections import defaultdict
import heapq
from transformer import select_k_best_words
import math

nltk.download("wordnet")
nltk.download("wordnet_ic")

EXPAND_LIMIT = 20
SIMILARITY_CUTOFF = 0.7
class WordExpansion:
    def __init__(self):
        self.wnl: WordNetLemmatizer = WordNetLemmatizer()
        self.brown_ic = wordnet_ic.ic("ic-brown.dat")
        self.cfg = Config()
        self.we.load()
        
    
    def vocab_to_lemmas(self, vocab: set) -> set:
        lemmas = set()
        for word in vocab:
            lemmas.add(self.wnl.lemmatize(word))
        return lemmas


    def __related_synsets(self, syn):
        related_words = set()

        # Add direct synonyms
        for lemma in syn.lemmas():
            related_words.add(lemma.name())

        # Add hyponyms (more specific terms)
        for hypo in syn.hyponyms():
            related_words.update(lemma.name() for lemma in hypo.lemmas())

        # Add part meronyms (parts of the term)
        for meronym in syn.part_meronyms():
            related_words.update(lemma.name() for lemma in meronym.lemmas())

        # Add synonyms from similar terms
        for similar in syn.similar_tos():
            related_words.update(lemma.name() for lemma in similar.lemmas())

        # Add hypernyms (general terms)
        for hyper in syn.hypernyms():
            related_words.update(lemma.name() for lemma in hyper.lemmas())

        # Add member holonyms (whole of)
        for holonym in syn.member_holonyms():
            related_words.update(lemma.name() for lemma in holonym.lemmas())

        return related_words

    
    def expand_words(self, query: list[str], max_distance: int = 2) -> List[Tuple[str, float]]:

        output = set()
        # explore synsets for each word in words
        for word in query:            
            # Get the root word
            root_word = self.wnl.lemmatize(word)
            
            queue = [(root_word, 1)]
            processed = set([root_word])


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
                            output.add(lemma)
                            queue.append(node)
                            processed.add(lemma)


        # find k best lemmas
        best = select_k_best_words(query, list(output), EXPAND_LIMIT)
        # cut off any words that are below the similarity cutoff
        best = [(word, score) for word, score in best if score >= SIMILARITY_CUTOFF]
        return best
    
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

    query = "amphibian metamorphosis thyroid hormone"

    print("ORIGINAL QUERY:", query)
    print("EXPANDED QUERY:", we.get_expanded_query(query))
