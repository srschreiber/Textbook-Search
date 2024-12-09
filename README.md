# Better E-Book Searching

## Links
- [Setup Instructions](docs/setup.md)
- [TextData Community](https://textdata.org/submissions/672398fc3d35dbf48754516c)

## Problem Statement
Modern e-books often rely on keyword-matching search to help readers locate topics within the text. While effective for experts or users with precise knowledge, this method falls short for users with only a general understanding. For instance, a first-year Molecular Biology student might search for "cell membrane" but not find relevant sections because the text refers to the "plasma membrane" instead.

## Proposed Solution
This project aims to enhance e-book search by combining a keyword-based approach with an embedding/bm25 hybrid approach to provide a more balanced search result. The goal is the best results are ones that have both strong keyword matching and semantic relavance.

The high-level approach is:

1. process the source text, such as the e-book
    1. parse the source text into sentences using spacy
    2. generate sliding windows of text, where each window is N sentences long with a step length of M. These windows correspond to documents
    3. create a lemmatized copy of the windows to be used by our BM25 engine
2. create a BM25 and FAISS index using lucene
3. rank the top L documents for a query using BM25 index (with Rocchio feedback)
4. create another set of rankings by reranking the results of step 3 using FAISS index search
5. score each document by taking a geometric mean between the BM25 rank and FAISS rank
6. sort the documents in ascending order based on their scores, and return the top K documents

### Why Use Geometric Mean Between the Two Sets of Rankings?
For scoring a document, I use a weighted geometric mean between the BM25 rank and the FAISS rank. The scoring function we use is:

![Geometric Mean Explanation](img/geom_mean.png)

Where
![Geom Mean Part 2](img/geom_mean_part2.png)

In this case, a smaller score would represent a more relevant document. I decided to use a geometric mean because it has the property of selecting documents that perform very well across either BM25 or FAISS, in comparison to a regular average which would score the document poorly if any of the ranks were exceptionally poor.

Another candidate I considered was the harmonic mean. In practice, this tended to be too sensitive to low ranks, selecting documents that had a decent score in one area but an exceptionally high score in another. The geometric mean provided more of a balanced metric, and generated better results in evaluation over a small dataset using nDCG.

## Evaluation
The hybrid search was evaluated against a document from the Cranfield dataset using nDCG, using BM25 with Rocchio feedback as a baseline. The evaluation is not meant to be comprehensive, but is instead meant to highlight the key characteristics of this hybrid approach. You can find the Cranfield data under data/cranfield/cranfield.txt.

### Evaluation Results over Single Cranfield Document
| Metric          | BM25 + Rocchio | Hybrid Search |
|-----------------|----------------|---------------|
| nDCG@10         | <span style="color:green">0.35813</span>        | <span style="color:red">0.26078 (-27.17%)</span>       |
| MAP@10          | <span style="color:green">0.24818</span>        | <span style="color:red">0.15812 (-36.28%)</span>       |

### Evaluation Results over 10 Queries with Strong Keywords
| Metric          | BM25 + Rocchio | Hybrid Search | 
|-----------------|----------------|---------------|
| nDCG@10         | <span style="color:green">0.43670</span>        | <span style="color:red">0.38894 (-10.93%)</span>       |
| MAP@10          | <span style="color:green">0.27950</span>        | <span style="color:red">0.22972 (-17.81%)</span>       |

### Evaluation Results over 10 Queries with Weak Keywords
| Metric          | BM25 + Rocchio | Hybrid Search |
|-----------------|----------------|---------------|
| nDCG@10         | <span style="color:red">0.13208</span>        | <span style="color:green">0.15552 (+17.74%)</span>       |
| MAP@10          | <span style="color:red">0.07531</span>        | <span style="color:green">0.076 (+0.92%)</span>         |

### Summary of Results
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

## Future Work

### More Thorough Evaluation Metrics
While the toy datasets yielded promising results, evaluation over much larger datasets is necessary to determine when the Hybrid approach is better. 

One example of a large dataset to potentially test on is TREC Robust. This dataset contains poorly or ambiguously defined queries, which should be an ideal use-case for our Hybrid model.

### Parameter Tuning
More information needs to be collected to determine the best way to weight the geometric mean between BM25 + Rocchio versus FAISS

### Learned Parameter Tuning
If the Hybrid approach is really better suited for more poorly defined queries, perhaps we can learn how "knowledgeable" a user is about a specific topic through data-mining and analyzing the query itself, and use that to set the geometric mean's weightings. If a user is extremely knowledgeable or provides a very good query, FAISS may not help nearly as much as BM25 for example.

