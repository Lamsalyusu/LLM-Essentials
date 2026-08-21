# PyTorch has many:
import torch 
import torch.nn as nn
nn.MSELoss()
nn.CrossEntropyLoss()
nn.BCELoss()
nn.BCEWithLogitsLoss()
nn.L1Loss()

# -----------------------------------------------------------------------------------

criterion = nn.CrossEntropyLoss()

# -----------------------------------------------------------------------------------

criterion = nn.MSELoss()

# -----------------------------------------------------------------------------------

# for LLMs we will use cross entropy loss constantly

# ===================================================================================
# ===================================================================================

# optimizers
torch.optim.Adam()
torch.optim.AdamW()
torch.optim.SGD()

# -----------------------------------------------------------------------------------
# for modern deep learning/LLMs
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr = 0.01
)

# ===================================================================================
# THE TRAINING LOOP 

model.train()

for X, y in train_loader:

    X = X.to(device)
    y = y.to(device)

# gradients lai suruma zero rakhne 
    optimizer.zero_grad()

# Forward Pass
    output = model(X)

# Loss calculation 
    loss = criterion(output, y)

# backpropagation
    loss.backward()

# optimization
    optimizer.step()

# ===================================================================================
# EVALUATION
model.train()

model.eval()

with torch.no_grad():
    pass


model.eval()

with torch.no_grad():

    output = model(X)

# ====================================================================================
# Gradient clipping
# Very relevant for deep networks and especially useful to understand for Transformers.

# helps to prevent exploding gradients
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0
)

# Usually:

loss.backward()
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    1.0
)

optimizer.step()

# ====================================================================================
# Learning-rate scheduler
# Instead of keeping:

lr = 0.001

# forever, you can change it during training.
# Example:

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=10,
    gamma=0.1
)

# Then after an epoch:

scheduler.step()

# For LLMs, you'll eventually encounter more sophisticated schedules such as:

                        # warmup
                        # cosine decay
                        # linear decay

# Those are more important than StepLR when you reach Transformers.


# ==================================================================================

