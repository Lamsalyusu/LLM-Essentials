import torch
import torch.nn as nn


class MyModel(nn.Module):

    def __init__(self):
        super().__init__()

    #     # Define the layers
    #     self.layer1 = nn.Linear(2, 4)
    #     self.relu = nn.ReLU()
    #     self.layer2 = nn.Linear(4, 1)

    # def forward(self, x):

    #     # Define the data flow
    #     x = self.layer1(x)
    #     x = self.relu(x)
    #     x = self.layer2(x)

    #     return x
        self.network = nn.Sequential(
        nn.Linear(2,4),
        nn.ReLU(),
        nn.Linear(4,1)
        )
    def forward(self,x):
        return self.network(x)


# Create model
model = MyModel()

# Input: 3 samples, 2 features
x = torch.tensor([
    [1.0, 2.0],
    [3.0, 4.0],
    [5.0, 6.0]
])

# Forward pass
output = model(x)

print("Output:")
print(output)

print("\nModel parameters:")
for name, parameter in model.named_parameters():
    print(name, parameter.shape)