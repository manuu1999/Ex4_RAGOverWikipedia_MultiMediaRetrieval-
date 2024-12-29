from transformers import AutoModelForCausalLM, AutoTokenizer

# Tried first with OpenAI model GPT-4 but was not able to implement it without additional costs
# Using distilgpt2 as a simple Hugging Face model

# Load the pre-trained model and tokenizer (distilGPT2 for simplicity)
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Ensure the tokenizer has a pad_token defined
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Function to generate a response from the LLM
def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=100,  # Limit response length
        num_return_sequences=1,
        do_sample=True  # Allow variability in the response
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Example usage
prompt = "What is the capital of France?"
print("Prompt:", prompt)
response = generate_response(prompt)
print("Response:", response)
