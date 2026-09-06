from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. Choose a checkpoint
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"

# 2. Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

# 3. Multiple sequences
sequences = [
    "I've been waiting for a Hugging Face course my whole life.",
    "I really hate this movie."
]

# 4. Tokenize + padding + truncation + PyTorch tensors
tokens = tokenizer(
    sequences,
    padding=True,
    truncation=True,
    return_tensors="pt"
)

# 5. See what tokenizer produced
print("INPUT IDS:")
print(tokens["input_ids"])

print("\nATTENTION MASK:")
print(tokens["attention_mask"])

print("\nSHAPES:")
print("input_ids:", tokens["input_ids"].shape)
print("attention_mask:", tokens["attention_mask"].shape)

# 6. Send inputs to model
output = model(**tokens)

print("\nLOGITS:")
print(output.logits)