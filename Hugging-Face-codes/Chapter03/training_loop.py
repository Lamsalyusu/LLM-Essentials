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

apply it to every collaborator 
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