from transformers import AutoModelForCausalLM, AutoTokenizer
# tried first with open ai model gpt-4 but was not able to implement it without additional costs

# Load the pre-trained model and tokenizer (distilGPT2 for simplicity)
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Function to generate a response from the LLM
def generate_response(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt")  # Tokenize the prompt
    outputs = model.generate(inputs, max_length=100, num_return_sequences=1, do_sample=True)  # Generate a response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)  # Decode the output
    return response

# Example usage
prompt = "Write a haiku about artificial intelligence."
print("Prompt:", prompt)
response = generate_response(prompt)
print("Response:", response)
