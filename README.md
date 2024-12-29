# Ex4_RAGOverWikipedia_MultiMediaRetrieval-
# RAG Over Wikipedia - Multimedia Retrieval
This project implements a Retrieval-Augmented Generation (RAG) pipeline using Simple English Wikipedia. The system retrieves relevant information based on user queries and generates responses using a Local Language Model (LLM).

## Features
- Data Loading: Parses the Simple English Wikipedia dataset and organizes it hierarchically (Article → Paragraph → Sentence).
- Semantic Search: Uses sentence embeddings to find relevant sentences based on user queries.
- Context Expansion: Broadens the retrieved context by including surrounding paragraphs.
- Response Generation: Generates human-like answers to user queries using a lightweight LLM (Hugging Face's distilgpt2).
## Requirements
- Python 3.8 or higher
- Packages:
- transformers
- sentence-transformers
- numpy
- tqdm
- torch

## Installation
- Clone the repository
- Install the required packages

## Usage
- Load Data: The program uses the Simple English Wikipedia dataset in .jsonl.gz format. Ensure the dataset file is located in the data directory.
- Download the dataset if not available: http://sbert.net/datasets/simplewiki-2020-11-01.jsonl.gz -P data/
- Run the Main Program: scr/main.py
- The system will load the dataset, perform semantic search, expand contexts, and generate responses based on the query.
- Note: The time required to process the entire dataset depends on your machine's performance. Encoding all Wikipedia paragraphs may take approximately 4–5 hours. If you would like to test the program quickly, you can reduce the dataset to a smaller sample by adjusting the subset_size parameter in the load_data function within the main.py file.
