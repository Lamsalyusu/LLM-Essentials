import torch 
import torch.nn as nn
embeddings =nn.Embedding(
    num_embeddings=10,
    embedding_dim = 4
)

tokens = torch.tensor([1,2,3,4])
output = embeddings(tokens)
print(output,output.shape)