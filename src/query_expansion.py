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
from collections import deque


nltk.download("wordnet")
nltk.download("wordnet_ic")
stop_words = set(nltk.corpus.stopwords.words("english"))
EXPAND_LIMIT = 20
class WordExpansion:
    def __init__(self, vocab: set) -> None:
        self.wnl: WordNetLemmatizer = WordNetLemmatizer()
        self.brown_ic = wordnet_ic.ic("ic-brown.dat")
        self.cfg = Config()        
        self.vocab = self.vocab_to_lemmas(vocab)
    
    def vocab_to_lemmas(self, vocab: set) -> set:
        lemmas = set()
        for word in vocab:
            lemmas.add(self.wnl.lemmatize(word))
        return lemmas

    def __related_synsets(self, syn):
        related_words = set()

        # Get the POS tag of the original synset
        original_pos = syn.pos()

        def add_related_words(synset_list):
            """Helper function to add related words if POS matches the original."""
            for related_syn in synset_list:
                if related_syn.pos() == original_pos:  
                    for lemma in related_syn.lemmas():
                        related_words.add(lemma.name())

        # Add direct synonyms (same POS)
        add_related_words([syn])

        # Add hypernyms (only if POS matches)
        add_related_words(syn.hypernyms())

        # Add hyponyms (only if POS matches)
        add_related_words(syn.hyponyms())

        # Add holonyms (only if POS matches)
        add_related_words(syn.member_holonyms())

        # Add meronyms (only if POS matches)
        add_related_words(syn.member_meronyms())

        return related_words

    
    def expand_words(self, query: list[str], max_distance: int = 5) -> List[Tuple[str, float]]:

        output = set()
        processed = set()

        # this sounds high but we will use a cache to quickly get the embeddings
        MAX_NUMBER = 3000

        """
        n - NOUN
        v - VERB
        a - ADJECTIVE
        r - ADVERB

        We only want to expand nouns and verbs and penalize verbs more than nouns
        """
        pos_filter = ["n", "v"]
        
        pos_graph_weight = {
            # prefer nouns, but verbs are ok but include only half of them
            "n": 1,
            "v": 2
        }

        # bias slightly towards nouns
        pos_penalties = {
            "n": 1,
            "v": .95
        }

        # explore synsets for each word in words
        for i, word in enumerate(query):            
            # Get the root word
            root_word = self.wnl.lemmatize(word)
            
            queue = deque()
            queue.append((root_word, 0))


            # Perform a BFS
            while queue and len(output) < MAX_NUMBER:
                # pop left to process the closest words first for the true bfs experience
                w, depth = queue.popleft()

                if depth >= max_distance:
                    continue
                

                # Get the synsets for the word

                synsets = []
                for pos in pos_filter:
                    synsets.extend(wn.synsets(w, pos=pos))

                for synset in synsets:
                    new_weight = depth + pos_graph_weight[synset.pos()]
                    if new_weight > max_distance:
                        continue
                    
                    # Get the lemmas for the synset
                    if synset.lemmas():
                        lemmas = synset.lemmas()
                    else:
                        continue


                    # Filter out lemmas that are not in the vocab
                    lemmas = self.__related_synsets(synset)

                    for lemma in lemmas:
                        if lemma not in processed:
                            if lemma in stop_words or lemma not in self.vocab:
                                continue

                            node = (lemma, depth + new_weight)
                            output.add((lemma, pos_penalties[synset.pos()]))
                            queue.append(node)
                            processed.add(lemma)


        print(f"Finding best lemmas out of {len(output)} lemmas given query {query}")
        # find k best lemmas
        best = select_k_best_words(query, list(output), k=EXPAND_LIMIT)

        # do penalties
        return best
    
    def get_expanded_query(self, query: str) -> dict[str, set[str]]:
        # remove stopwords first
        removed_stopwords = [word for word in query.split() if word not in stop_words]

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
