import os
from scr.data_loader import load_data
from scr.sentence_encoder import encode_paragraphs
from scr.semantic_search import semantic_search
from scr.context_expander import expand_context

# Filepath to the Simple English Wikipedia dataset
relative_filepath = "../data/simplewiki-2020-11-01.jsonl.gz"
filepath = os.path.abspath(relative_filepath)
print(f"Resolved file path: {filepath}")

# Step 1: Load a subset of the data for testing
print("Loading data...")
paragraphs = list(load_data(filepath, subset_size=100))  # Load only 100 paragraphs for testing
if not paragraphs:
    print("No paragraphs loaded. Exiting.")
    exit()
print(f"Loaded {len(paragraphs)} paragraphs from the dataset.")

# Step 2: Encode paragraphs into sentences with embeddings
print("Encoding paragraphs...")
encoded_paragraphs, model = encode_paragraphs(paragraphs, model_name="all-MiniLM-L6-v2")

# Step 3: Flatten encoded sentences for semantic search
encoded_sentences = [
    {
        'text': sentence['text'],
        'embedding': sentence['embedding'],
        'paragraph_text': paragraph['text'],
        'title': paragraph['title']
    }
    for paragraph in encoded_paragraphs
    for sentence in paragraph.get('sentences', [])
]

# Step 4: Perform semantic search
query = "What is the capital of France?"
print(f"Performing semantic search for query: {query}")
top_results = semantic_search(query, encoded_sentences, model, top_k=5)

# Step 5: Expand contexts based on search results
print("Expanding contexts...")
expanded_contexts = expand_context(top_results, paragraphs)

# Print results
print("\nTop Search Results:")
for result in top_results:
    print(f"Title: {result['title']}, Similarity: {result['similarity']:.2f}")
    print(f"Sentence: {result['text']}\n")

print("\nExpanded Contexts:")
for i, context in enumerate(expanded_contexts):
    print(f"Context {i + 1}: {context['text']}\n")
