import torch
import torch.nn as nn

from tokenizer import CharTokenizer
from helpers import load_model, generate_text


def main():
    # Load tokenizer
    tokenizer = CharTokenizer()
    tokenizer.load("tokenizer.json")

    # Load model
    model = load_model("lstm_model.pt", vocab_size=tokenizer.vocab_size)

    # Ask user for prompt
    prompt = input("Enter a starting text: ")

    # Generate text
    output = generate_text(
        model=model,
        tokenizer=tokenizer,
        start_text=prompt,
        length=300,
        temperature=0.8
    )

    print("\nGenerated text:\n")
    print(output)


if __name__ == "__main__":
    main()
