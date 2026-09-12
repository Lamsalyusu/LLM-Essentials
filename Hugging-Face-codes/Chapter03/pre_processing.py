from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
)

# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

raw_datasets = load_dataset("nyu-mll/glue", "mrpc")

print("\nDATASET:")
print(raw_datasets)

# --------------------------------------------------
# 2. Load tokenizer
# --------------------------------------------------

checkpoint = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# --------------------------------------------------
# 3. Look at one raw example
# --------------------------------------------------

print("\nRAW EXAMPLE:")
print(raw_datasets["train"][0])

# --------------------------------------------------
# 4. Tokenize a pair manually
# --------------------------------------------------

example = raw_datasets["train"][0]

inputs = tokenizer(
    example["sentence1"],
    example["sentence2"]
)

print("\nTOKENIZED PAIR:")
print(inputs)

# --------------------------------------------------
# 5. See the actual tokens
# --------------------------------------------------

tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"])

print("\nTOKENS:")
print(tokens)

# --------------------------------------------------
# 6. Define preprocessing function
# --------------------------------------------------

def tokenize_function(example):
    return tokenizer(
        example["sentence1"],
        example["sentence2"],
        truncation=True
    )

# --------------------------------------------------
# 7. Tokenize the entire dataset
# --------------------------------------------------

tokenized_datasets = raw_datasets.map(
    tokenize_function,
    batched=True
)

print("\nTOKENIZED DATASET:")
print(tokenized_datasets)

# --------------------------------------------------
# 8. Check one tokenized example
# --------------------------------------------------

print("\nTOKENIZED EXAMPLE:")
print(tokenized_datasets["train"][0])

# --------------------------------------------------
# 9. Dynamic padding
# --------------------------------------------------

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)

# --------------------------------------------------
# 10. Take 8 examples
# --------------------------------------------------

samples = tokenized_datasets["train"][:8]

# Remove columns containing raw strings
samples = {
    key: value
    for key, value in samples.items()
    if key not in ["idx", "sentence1", "sentence2"]
}

# --------------------------------------------------
# 11. Check original lengths
# --------------------------------------------------

print("\nORIGINAL LENGTHS:")
print([len(x) for x in samples["input_ids"]])

# --------------------------------------------------
# 12. Create batch using dynamic padding
# --------------------------------------------------

batch = data_collator(samples)

print("\nBATCH SHAPES:")

for key, value in batch.items():
    print(key, value.shape)

# --------------------------------------------------
# 13. Inspect actual batch
# --------------------------------------------------

print("\nINPUT IDS:")
print(batch["input_ids"])

print("\nATTENTION MASK:")
print(batch["attention_mask"])

print("\nLABELS:")
print(batch["labels"])