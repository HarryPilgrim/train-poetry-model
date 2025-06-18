from transformers import pipeline

def generate_poem(prompt: str, model_path="./poetry_model"):
    generator = pipeline("text-generation", model=model_path, tokenizer=model_path)
    result = generator(prompt, max_length=150, num_return_sequences=1, do_sample=True, temperature=0.9)
    return result[0]['generated_text']