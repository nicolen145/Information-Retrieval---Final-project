# Information Retrieval – Final Project

## Overview
This project implements a search engine over the English Wikipedia corpus.
The system is based on an inverted index stored on Google Cloud Storage (GCS)
and exposes a RESTful API using Flask.

The inverted index was constructed in Assignment 3 and reused in this project
without modification.

---

## Repository Structure
```bash
├── search_frontend.py        # Flask REST API for the search engine
├── inverted_index_gcp.py     # GCP-based inverted index implementation
├── benchmark_queries.py      # Script for measuring query runtime
├── queries_train.json        # Training queries for evaluation
└── README.md
```



---

## Index Infrastructure
The system relies on a disk-based inverted index stored on Google Cloud Storage.
The index is accessed through the provided `InvertedIndex` abstraction, which supports:

- Compact binary encoding of posting lists
- On-demand loading of posting lists from GCS
- Efficient disk-based storage without loading the full index into memory

This design enables scalable retrieval over millions of documents.

---

## Search API Endpoints

### `/search`
The main search endpoint.
Performs TF-IDF and cosine similarity over document bodies, followed by
title-based reranking.

### `/search_body`
Returns search results using TF-IDF and cosine similarity over document bodies only.

### `/search_title`
Returns all documents whose titles contain at least one query term.
Results are ranked by the number of distinct query terms appearing in the title.

---

## Retrieval Model
- Tokenization with stopword removal (no stemming)
- TF-IDF weighting with logarithmic term frequency
- Cosine similarity between query and document vectors
- Optional coverage-based and title-based score boosting
- No query result caching (in accordance with project rules)

---

## Running the Search Engine

```bash
python3 search_frontend.py
```
The server runs on port 8080.

Example query:
```bash
http://<EXTERNAL_IP>:8080/search?query=information+retrieval
```


