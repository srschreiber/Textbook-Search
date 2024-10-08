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

nltk.download("wordnet")
nltk.download("wordnet_ic")

class WordExpansion:
    def __init__(self, sim_threshold: float = 0.2):
        self.wnl: WordNetLemmatizer = WordNetLemmatizer()
        self.brown_ic = wordnet_ic.ic("ic-brown.dat")
        self.cfg = Config()
        self.we = WordEmbedder()
        self.we.load()
        self.sim_threshold = sim_threshold
        
    
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

                        similarity = self.we.similarity(root_word, lemma)
                        if similarity < self.sim_threshold:
                            continue

                        # Calculate similarity if needed
                        node = (lemma, depth + 1)
                        output.append((lemma, similarity, depth + 1))
                        queue.append(node)
                        processed.add(lemma)

        # Sort the synonyms by the depth or similarity
        output = sorted(output, key=lambda x: x[1], reverse=True)
        return output
    
    def get_expanded_query(self, query: str) -> dict[str, set[str]]:
        # remove stopwords first
        removed_stopwords = [word for word in query.split() if word not in self.we.stop_words]

        expanded_query = defaultdict(set[str])
        for word in removed_stopwords:
            expanded = self.expand_word(word, max_distance=2)
            for w in expanded:
                expanded_query[word].add(w)

        return expanded_query

# Test it out with a word
if __name__ == "__main__":
    we = WordExpansion()

    query = "homotypic fusion with vesicles"

    print("ORIGINAL QUERY:", query)
    print("EXPANDED QUERY:", we.get_expanded_query(query))
