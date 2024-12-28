from sentence_transformers import util


def semantic_search(query, encoded_sentences, model, top_k=5):
    query_embedding = model.encode([query])  # Encode the query
    results = []

    # Iterate through sentences in encoded_sentences
    for sentence in encoded_sentences:
        if 'embedding' in sentence and 'text' in sentence:  # Validate sentence structure
            similarity = util.dot_score(query_embedding, [sentence['embedding']]).item()
            results.append({
                'text': sentence['text'],
                'similarity': similarity,
                'paragraph_text': sentence.get('paragraph_text', ''),
                'title': sentence.get('title', '')
            })

    results = sorted(results, key=lambda x: x['similarity'], reverse=True)[:top_k]  # Return top results
    if not results:
        print("No results found for the query.")
    return results
