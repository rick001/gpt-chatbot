from transformers import GPTJForCausalLM, AutoTokenizer

# Load the model and tokenizer
model_name = "EleutherAI/gpt-j-6B"
model = GPTJForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs.input_ids, max_length=150, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Example usage
if __name__ == "__main__":
    user_input = "What is the capital of France?"
    response = generate_response(user_input)
    print(response)
