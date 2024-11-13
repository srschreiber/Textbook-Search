from lib.base_index import Index

class BM25Index(Index):
    def __init__(self, index_dir, output_dir, tokenizer):
        super().__init__(index_dir, output_dir, tokenizer)

