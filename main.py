from mlx_lm import generate, load

# Load the Granite 4.0-H Micro 4-bit quantized model
model, tokenizer = load("mlx-community/granite-4.0-h-micro-4bit")

prompt = "Hello"

if tokenizer.chat_template is not None:
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

response = generate(model, tokenizer, prompt=prompt, verbose=True)
print(response)
