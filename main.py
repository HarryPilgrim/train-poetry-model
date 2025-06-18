from data_loader import extract_poems_from_json
from tokeniser_utils import load_and_tokenize
from train_model import train_poetry_model
from generate_poem import generate_poem
import torch
import torchvision
print(torch.__version__)
print(torchvision.__version__)
def main():
    # Step 1: Prepare data
    extract_poems_from_json("filtered_top_20_poets_data.json", "poems_dataset_TWO.txt")



    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
    # Step 2: Tokenize
    tokenized_dataset, tokenizer = load_and_tokenize("poems_dataset_TWO.txt")

    # Step 3: Train
    train_poetry_model(tokenized_dataset, tokenizer)

    # Step 4: Generate a poem
    prompt = "Write a poem about the last sunset on Earth"
    poem = generate_poem(prompt)
    print("\nGenerated Poem:\n", poem)

if __name__ == "__main__":
    main()