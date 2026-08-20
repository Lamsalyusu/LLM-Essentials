import torch

A = torch.tensor([
    [1, 2],
    [3, 4]
])

B = torch.tensor([
    [5, 6],
    [7, 8]
])

# C = A @ B

C = torch.matmul(A, B)
print(C)