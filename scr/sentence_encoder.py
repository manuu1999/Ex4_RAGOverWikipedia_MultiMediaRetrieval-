from sentence_transformers import SentenceTransformer


def encode_paragraphs(paragraphs, model_name='all-MiniLM-L6-v2', batch_size=1000):
    model = SentenceTransformer(model_name)
    encoded_paragraphs = []

    for i, batch_start in enumerate(range(0, len(paragraphs), batch_size)):
        batch = paragraphs[batch_start:batch_start + batch_size]
        for paragraph in batch:
            sentences = paragraph['text'].split('. ')  # Split paragraph into sentences
            embeddings = model.encode(sentences) if sentences else []  # Encode sentences only if non-empty
            paragraph['sentences'] = [
                {'text': s, 'embedding': e} for s, e in zip(sentences, embeddings)
            ]
        encoded_paragraphs.extend(batch)
        print(f"Encoded batch {i + 1}/{(len(paragraphs) + batch_size - 1) // batch_size}")

    # Debug: Print first 3 paragraphs to ensure encoding worked
    for paragraph in encoded_paragraphs[:3]:
        print("Encoded Paragraph Debug:", paragraph.get('sentences', []))

    return encoded_paragraphs, model
