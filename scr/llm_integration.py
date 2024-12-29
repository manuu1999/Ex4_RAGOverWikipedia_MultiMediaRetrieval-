from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the LLM and tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

# Ensure pad_token_id is set
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

def generate_response(prompt):
    """
    Generates a response using the LLM model.

    Args:
        prompt (str): The input prompt for the LLM.

    Returns:
        str: Generated response.
    """
    inputs = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=500)  # Truncate input if needed
    outputs = model.generate(
        inputs,
        max_new_tokens=150,  # Limit only the number of new tokens generated
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True  # Enable sampling for more diverse outputs
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response
