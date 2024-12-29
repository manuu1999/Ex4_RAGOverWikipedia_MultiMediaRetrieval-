import os
from scr.data_loader import load_data
from scr.sentence_encoder import encode_paragraphs
from scr.semantic_search import semantic_search
from scr.context_expander import expand_context
from scr.llm_integration import generate_response

# Filepath to the dataset
relative_filepath = "../data/simplewiki-2020-11-01.jsonl.gz"
filepath = os.path.abspath(relative_filepath)

# Step 1: Load data
print("Loading data...")
paragraphs = list(load_data(filepath))  # Load a subset for testing
print(f"Loaded {len(paragraphs)} paragraphs.")

# Step 2: Encode paragraphs
print("Encoding paragraphs...")
encoded_paragraphs, model = encode_paragraphs(paragraphs, model_name="all-MiniLM-L6-v2")

# Step 3: Perform semantic search
query = "What is the capital of France?"
print(f"Performing semantic search for query: '{query}'")
encoded_sentences = [
    {'text': sentence['text'], 'embedding': sentence['embedding'], 'paragraph_text': paragraph['text'], 'title': paragraph['title']}
    for paragraph in encoded_paragraphs
    for sentence in paragraph.get('sentences', [])
    if 'text' in sentence and 'embedding' in sentence
]
top_results = semantic_search(query, encoded_sentences, model, top_k=3)

# Step 4: Expand context
print("Expanding contexts...")
expanded_contexts = expand_context(top_results, paragraphs)

# Step 5: Generate response with LLM
print("Generating response using LLM...")
context_combined = "\n\n".join([ctx['text'] for ctx in expanded_contexts])
llm_prompt = f"You are a question-answering bot. Based on the following context:\n\n{context_combined}\n\nAnswer the user's question: {query}"
response = generate_response(llm_prompt)

# Output the result
print("\nTop Search Results:")
for result in top_results:
    print(f"Title: {result['title']}, Similarity: {result['similarity']:.2f}")
    print(f"Sentence: {result['text']}\n")

print("\nExpanded Contexts:")
for i, context in enumerate(expanded_contexts):
    print(f"Context {i + 1}: {context['text']}\n")

print("\nGenerated Response:")
print(response)
