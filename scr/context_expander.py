def expand_context(results, paragraphs, expand_by=1):
    expanded_contexts = []  # Initialize list for expanded contexts

    for result in results:
        title = result['title']
        target_paragraph_text = result['paragraph_text']
        related_paragraphs = [p['text'] for p in paragraphs if p['title'] == title]

        if target_paragraph_text in related_paragraphs:
            index = related_paragraphs.index(target_paragraph_text)
            start_index = max(0, index - expand_by)
            end_index = min(len(related_paragraphs), index + expand_by + 1)
            expanded_context = " ".join(related_paragraphs[start_index:end_index])
            expanded_contexts.append({'text': expanded_context, 'title': title})
        else:
            print(f"Target paragraph not found in article: {title}")

    return expanded_contexts
