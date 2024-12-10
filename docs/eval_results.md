# Evaluation Results
The hybrid search was evaluated against a document from the Cranfield dataset using nDCG, using BM25 with Rocchio feedback as a baseline. The evaluation is not meant to be comprehensive, but is instead meant to highlight the key characteristics of this hybrid approach. You can find the Cranfield data under data/cranfield/cranfield.txt. For the FAISS index, bert-base-uncased was used as the embedding model.

## Evaluation Results over Single Cranfield Document
| Metric          | BM25 + Rocchio | Hybrid Search |
|-----------------|----------------|---------------|
| nDCG@10         | <span style="color:green">0.35813</span>        | <span style="color:red">0.26078 (-27.17%)</span>       |
| MAP@10          | <span style="color:green">0.24818</span>        | <span style="color:red">0.15812 (-36.28%)</span>       |

## Evaluation Results over 10 Queries with Strong Keywords
| Metric          | BM25 + Rocchio | Hybrid Search | 
|-----------------|----------------|---------------|
| nDCG@10         | <span style="color:green">0.43670</span>        | <span style="color:red">0.38894 (-10.93%)</span>       |
| MAP@10          | <span style="color:green">0.27950</span>        | <span style="color:red">0.22972 (-17.81%)</span>       |

## Evaluation Results over 10 Queries with Weak Keywords
| Metric          | BM25 + Rocchio | Hybrid Search |
|-----------------|----------------|---------------|
| nDCG@10         | <span style="color:red">0.13208</span>        | <span style="color:green">0.15552 (+17.74%)</span>       |
| MAP@10          | <span style="color:red">0.07531</span>        | <span style="color:green">0.076 (+0.92%)</span>         |

## Summary of Results
I noticed that over the specific Cranfield document, BM25 + Rocchio performed significantly better than the Hybrid search in nDCG@10 and MAP@10. I looked at the queries used for Cranfield and noticed that they used exceptionally strong keywords. 

For example: 
```
what similarity laws must be obeyed when constructing aeroelastic models of heated high speed aircraft .
```

This uses keywords that are well-represented in the source text. I then had a hypothesis: Queries that use few keywords captured in the text will benefit more from the Hybrid model than those that are very well defined.

I constructed two small sets of 10 queries. One set represented the original queries. The second set represented similar queries, but ones that show less understanding of the source material. For example, instead of the original query:

```
what similarity laws must be obeyed when constructing aeroelastic models of heated high speed aircraft .
```

the second set, that we call the small and less specific set, instead uses 

```
What is important when constructing models of really fast airplanes?
```

For nDCG@10 over this small dataset, the Hybrid model performed substantially better than the BM25 + Rocchio model. For MAP@10, the Hybrid model performed only marginally better, indicating that both approaches yielded a similar number of relevant documents, but the Hybrid approach yielded higher quality relevant documents.

In contrast, the small set of queries before alteration had BM25 + Rocchio model performing better across all metrics. 

This reinforces my theory that BM25 + Rocchio performs better with well-defined queries, while the Hybrid approach may be better suited for less well-defined queries.

## Limitations with Evaluation
- We evaluated over a single document from the Cranfield dataset
- For the less specific queries, we only evaluated over a small set of 10 queries
- Cranfield documents might be too long for FAISS to accurately capture, thus underestimating the performance of the Hybrid model. It would be better for us to break each document up into smaller chunks and re-evaluate, which I hypothesize would yield much better results in favor of the Hybrid model.