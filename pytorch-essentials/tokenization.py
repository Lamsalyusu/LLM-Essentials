import torch 
import torch.nn as nn
embeddings =nn.Embedding(
    num_embeddings=10,
    embedding_dim = 4
)

tokens = torch.tensor([1,2,3,4])  
# this torch.tensor([1,2,3,4]) are token ids 0 -> I , 1 -> am like that
output = embeddings(tokens)
print(output,output.shape)

# nnn_embeddings = 10: means there are 10 possible token IDs
# 0,1,2,3,4,5,6,7,8,9 
# i,am, a, very, good, person ,in ,my, life, bro.

# embeddings_dim = 4 :  means each token gets a vector of 4 numbers 
# conceptually:
# Token 0 → [?, ?, ?, ?]
# Token 1 → [?, ?, ?, ?]
# Token 2 → [?, ?, ?, ?]
# ...
# Token 9 → [?, ?, ?, ?]

# means each token gets a vector of 4 number 


