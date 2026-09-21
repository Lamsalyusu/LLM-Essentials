# L1.OAD DATASET
from datasets import load_dataset

raw_datasets = load_dataset(
    "code_search_net",
    "python"
)

# 2. TRAINING CORPUS GENERATOR
def get_training_corpus():
    dataset = raw_datasets["train"]

    for start_idx in range(0, len(dataset), 1000):
        samples = dataset[start_idx:start_idx + 1000]
        yield samples["whole_func_string"]


# 3. Load existing GPT-2 tokenizer
from transformers import AutoTokenizer

old_tokenizer = AutoTokenizer.from_pretrained("gpt2")
# 4. Train new tokenizer
training_corpus = get_training_corpus()

tokenizer = old_tokenizer.train_new_from_iterator(
    training_corpus,
    52000
)
# 5. Test it
example = '''def add_numbers(a, b):
    """Add the two numbers."""
    return a + b'''

tokens = tokenizer.tokenize(example)

print(tokens)
# 6. Save
tokenizer.save_pretrained(
    "code-search-net-tokenizer"
)
# 7. Load later
tokenizer = AutoTokenizer.from_pretrained(
    "code-search-net-tokenizer"
)