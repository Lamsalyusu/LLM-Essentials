import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader


# ============================================================
# 1. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ============================================================
# 2. DATASET
# ============================================================

class MyDataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


# ============================================================
# 3. CREATE DATA
# ============================================================

X_train = torch.randn(1000, 10)

y_train = torch.randint(
    0,
    3,
    (1000,)
)

X_val = torch.randn(200, 10)

y_val = torch.randint(
    0,
    3,
    (200,)
)


# ============================================================
# 4. DATASET + DATALOADER
# ============================================================

train_dataset = MyDataset(
    X_train,
    y_train
)

val_dataset = MyDataset(
    X_val,
    y_val
)


train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)


# ============================================================
# 5. MODEL
# ============================================================

class NeuralNetwork(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            # Input → hidden
            nn.Linear(10, 128),

            # Normalize activations
            nn.LayerNorm(128),

            # Activation
            nn.ReLU(),

            # Regularization
            nn.Dropout(0.2),

            # Hidden → hidden
            nn.Linear(128, 64),

            nn.ReLU(),

            # Hidden → output
            nn.Linear(64, 3)
        )

    def forward(self, x):

        return self.network(x)


# ============================================================
# 6. CREATE MODEL
# ============================================================

model = NeuralNetwork().to(device)


# ============================================================
# 7. LOSS
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# 8. OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01
)


# ============================================================
# 9. LEARNING RATE SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=10,
    gamma=0.1
)


# ============================================================
# 10. TRAINING
# ============================================================

epochs = 20


for epoch in range(epochs):

    # -----------------------------------------
    # Training mode
    # -----------------------------------------

    model.train()

    total_loss = 0
    correct = 0
    total = 0


    for X_batch, y_batch in train_loader:

        # Move data to device
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)


        # -------------------------------------
        # Clear previous gradients
        # -------------------------------------

        optimizer.zero_grad()


        # -------------------------------------
        # Forward pass
        # -------------------------------------

        outputs = model(X_batch)


        # -------------------------------------
        # Calculate loss
        # -------------------------------------

        loss = criterion(
            outputs,
            y_batch
        )


        # -------------------------------------
        # Backpropagation
        # -------------------------------------

        loss.backward()


        # -------------------------------------
        # Gradient clipping
        # -------------------------------------

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )


        # -------------------------------------
        # Update parameters
        # -------------------------------------

        optimizer.step()


        # -------------------------------------
        # Metrics
        # -------------------------------------

        total_loss += loss.item()

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        correct += (
            predictions == y_batch
        ).sum().item()

        total += y_batch.size(0)


    # -----------------------------------------
    # Scheduler
    # -----------------------------------------

    scheduler.step()


    train_accuracy = correct / total


    # ========================================================
    # 11. VALIDATION
    # ========================================================

    model.eval()

    val_loss = 0
    val_correct = 0
    val_total = 0


    with torch.no_grad():

        for X_batch, y_batch in val_loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)


            outputs = model(X_batch)


            loss = criterion(
                outputs,
                y_batch
            )


            val_loss += loss.item()


            predictions = torch.argmax(
                outputs,
                dim=1
            )


            val_correct += (
                predictions == y_batch
            ).sum().item()

            val_total += y_batch.size(0)


    val_accuracy = (
        val_correct / val_total
    )


    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"Train Loss: {total_loss:.4f} | "
        f"Train Acc: {train_accuracy:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.4f}"
    )


# ============================================================
# 12. SAVE MODEL
# ============================================================

torch.save(
    model.state_dict(),
    "model.pth"
)


# ============================================================
# 13. SAVE CHECKPOINT
# ============================================================

torch.save({

    "epoch": epoch,

    "model_state_dict":
        model.state_dict(),

    "optimizer_state_dict":
        optimizer.state_dict(),

    "loss": loss.item()

}, "checkpoint.pth")


# ============================================================
# 14. LOAD MODEL
# ============================================================

new_model = NeuralNetwork().to(device)

new_model.load_state_dict(
    torch.load(
        "model.pth",
        map_location=device
    )
)

new_model.eval()


# ============================================================
# 15. INFERENCE
# ============================================================

sample = torch.randn(
    1,
    10
).to(device)


with torch.no_grad():

    output = new_model(sample)

    prediction = torch.argmax(
        output,
        dim=1
    )

print("Prediction:", prediction.item())