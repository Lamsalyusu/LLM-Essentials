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

# 4. Creating DataLoaders

# A DataLoader creates batches from a dataset.

# Import it:
from torch.utils.data import DataLoader

# Create the training DataLoader:
train_dataloader = DataLoader(
    tokenized_datasets["train"],
    shuffle=True,
    batch_size=8, 
    collate_fn=data_collator,
)

# Create the evaluation DataLoader:

eval_dataloader = DataLoader(
    tokenized_datasets["validation"],
    batch_size=8,
    collate_fn=data_collator,
)

# Meaning of each argument
tokenized_datasets["train"]

# The dataset used for training.

shuffle=True
# Randomizes the training examples at the beginning of each epoch.
# Why?
# Because always presenting examples in the same order may make training less effective or introduce unwanted ordering patterns.
# For evaluation, shuffling is usually unnecessary:

shuffle=False

# The default is already False, so the evaluation DataLoader does not need it.

batch_size=8
# The model processes 8 examples per batch.

collate_fn=data_collator
# The data collator:
# Collects individual examples into a batch.
# Dynamically pads sequences.

# Creates tensors with compatible shapes.
# Without the collator, sequences with different lengths may not form a rectangular tensor.

# 5 Inspecting one batch 
# Before training always verify that the batch looks correct 
for batch in train_dataloader:
    break

# this retrieves the first batch and stops the loop immediately 
# Now inspect its shapes:

print({
    key: value.shape
    for key, value in batch.items()
})

# Example output:

{
    "attention_mask": torch.Size([8, 65]),
    "input_ids": torch.Size([8, 65]),
    "labels": torch.Size([8]),
    "token_type_ids": torch.Size([8, 65]),
}

# The exact sequence length may differ because:

# Training data is shuffled.

# Dynamic padding pads to the longest sequence in that particular batch.

# 6. Loading the model

# Load the sequence-classification model:

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint,
    num_labels=2,
)

# Here:

checkpoint = "bert-base-uncased"

# The model is BERT with a classification head.

# For MRPC:

# label 0 → not equivalent
# label 1 → equivalent

# Therefore:

num_labels=2

# is appropriate.


# 7.Passing a batch through the model 

# Before writing the training loop, test one batch:

outputs = model(**batch)

# The **batch syntax expands the dictionary.

# This:

outputs = model(**batch)

# is equivalent to something conceptually like:

outputs = model(
    input_ids=batch["input_ids"],
    attention_mask=batch["attention_mask"],
    token_type_ids=batch["token_type_ids"],
    labels=batch["labels"],
)

# Print the loss and logits shape:

print(outputs.loss)
print(outputs.logits.shape)

# Example:
# tensor(0.5441, grad=<NllLossBackward>)
torch.Size([8, 2])

# Why does the model return a loss?

# Because the batch contains:

# labels

# When labels are provided, Hugging Face models calculate the appropriate loss automatically.

# For sequence classification, the model generally uses cross-entropy loss.

# Why are logits shaped [8, 2]?

# There are:

# 8 examples
# 2 class scores per example

# For example:

# Example 1 → [score_for_class_0, score_for_class_1]
# Example 2 → [score_for_class_0, score_for_class_1]
# ...

# The logits are raw scores, not probabilities.

# 8. Optimizers 
# The optimizers updates model parameters using their gradients 
# The chapter uses AdamW
from torch.optim import AdamW
optimizer = AdamW(
    model.parameters(),
    lr=5e-5,
)

# What does model.parameters() mean?
# It gives the optimizer access to the model’s trainable parameters.
# The optimizer uses their gradients to update them.


# 9. Adam vs AdamW

# The chapter explains that AdamW is similar to Adam but handles weight decay differently.

# Adam

# Adam combines:

# Momentum-like first-moment tracking.

# Second-moment tracking.

# Adaptive learning rates for each parameter.

# AdamW

# AdamW adds decoupled weight decay regularization.

# Weight decay discourages excessively large weights and can improve generalization.

# Example:

optimizer = AdamW(
    model.parameters(),
    lr=5e-5,
    weight_decay=0.01,
)

# The important distinction is:

# AdamW does not simply add weight decay into the gradient calculation in the same way as traditional Adam implementations. It decouples the weight-decay update from the adaptive gradient update.

# For Transformer fine-tuning, AdamW is a common default optimizer.