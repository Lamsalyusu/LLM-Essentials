# This section is about performing same fine tuining process as Trainer , but manually using PyTorch
# The over all flow is 
            # Dataset
            #    ↓
            # Tokenizer
            #    ↓
            # Tokenized Dataset
            #    ↓
            # DataLoader
            #    ↓
            # Model
            #    ↓
            # Loss
            #    ↓
            # Backward Pass
            #    ↓
            # Optimizer Step
            #    ↓
            # Learning-Rate Scheduler
            #    ↓
            # Evaluation


# The chapter also introduces Accelerate, which helps turn a single-device PyTorch training loop into a distributed training loop with minimal changes.
1
# ehy shall we use a manual training loop.
# previously we used :
# trainer.train()

# The Trainer API automatically handled many things:

# Preparing the dataset

# Creating batches

# Moving data to the correct device

# Forward pass

# Loss calculation

# Backpropagation

# Optimizer updates

# Learning-rate scheduling

# Evaluation

# Checkpointing

# Logging

# Distributed training support

# In this chapter, you implement those steps yourself.

# The goal is to understand:

# What exactly happens inside trainer.train()?

# A manual loop gives you maximum control, but requires more code and more responsibility.

# 2 Preprrocessing Recap
from datasets import load_dataset
from transformers import AutoTokenizer,DataCollatorWithPadding
raw_datasets = load_datasets("nyu-mll/glue","mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# The original course may show load_dataset("glue", "mrpc"). In your environment, use nyu-mll/glue if the shorter identifier gives an error.

# tokenizer function
def tokenize_function(example):
    return tokenizer(
        example["sentence1"],
        example["sentence2"],
        truncation=True,
    )

# apply it to every collaborator 
tokenized_datasets = raw_datasets.map(
    tokenize_function,
    batched=True,
)

# dynamic padding collator
data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)

# At this stage, each example contains fields such as:

# sentence1
# sentence2
# label
# idx
# input_ids
# token_type_ids
# attention_mask

# 3. Postprocessing the tokenized dataset

# When using Trainer, some dataset preparation is performed automatically.

# With a manual loop, you must do it yourself.

# The chapter performs three operations:

# Remove columns the model does not expect.

# Rename label to labels.

# Convert returned values into PyTorch tensors.

# 3.1 Remove unnecessary columns

tokenized_datasets = tokenized_datasets.remove_columns(
    ["sentence1", "sentence2", "idx"]
)

# Why remove them?

# Because the model expects numerical inputs such as:

# input_ids
# attention_mask
# token_type_ids
# labels

# It does not expect raw strings such as:

# sentence1
# sentence2

# If these string columns remain, the data collator or DataLoader may have trouble creating tensors.

# 3.2 Rename label to labels
tokenized_datasets = tokenized_datasets.rename_column(
    "label",
    "labels",
)

# This is important because Hugging Face models generally expect the argument name:

# labels=...

# For example:

outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    labels=labels,
)

# If the field remains named label, then this will not automatically match the model’s expected labels argument.

# 3.3 Set the dataset format to PyTorch
tokenized_datasets.set_format("torch")

# Before this, values may be returned as Python lists.
# After this, they are returned as PyTorch tensors.

# For example:

# Before:
# input_ids → Python list

# After:
# input_ids → torch.Tensor

# You can inspect the final columns:

print(tokenized_datasets["train"].column_names)

# Expected output:

[
    "attention_mask",
    "input_ids",
    "labels",
    "token_type_ids",
]