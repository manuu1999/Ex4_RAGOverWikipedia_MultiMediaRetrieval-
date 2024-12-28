import gzip
import json


def load_data(filepath, subset_size=None):
    paragraphs = []  # Initialize an empty list to hold paragraphs
    try:
        with gzip.open(filepath, 'rt', encoding='utf8') as f:  # Open the gzipped JSONL file
            for line in f:  # Iterate through each line (each line represents an article)
                data = json.loads(line)  # Parse the line as JSON
                for i, paragraph in enumerate(data.get('paragraphs', [])):  # Safely access paragraphs
                    paragraphs.append({
                        'article_id': data.get('id', 'unknown'),  # Handle missing ID
                        'title': data.get('title', 'unknown'),  # Handle missing title
                        'paragraph_id': f"{data.get('id', 'unknown')}:{i}",  # Create unique paragraph ID
                        'text': paragraph  # Store the paragraph text
                    })
                if subset_size and len(paragraphs) >= subset_size:  # Stop if subset limit is reached
                    break
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return []  # Return empty list if the file doesn't exist
    except json.JSONDecodeError:
        print("Error decoding JSON. Please check the file format.")
        return []

    print(f"Loaded {len(paragraphs)} paragraphs from the dataset.")  # Print the number of paragraphs loaded
    return paragraphs
