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


class TinyModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.embedding = nn.Embedding(10, 4)

    def forward(self, x):

        return self.embedding(x)

model = TinyModel()

tokens = torch.tensor([1, 3, 5])

output = model(tokens)


                        # embedding
                        #    ↓
                        # Transformer
                        #    ↓
                        # prediction
                        #    ↓
                        # loss
                        #    ↓
                        # backward()
                        #    ↓
                        # embedding gradients
                        #    ↓
                        # optimizer.step()
                        #    ↓
                        # updated embeddings

# =====================================================================================
# B T C 
# I Love Playing Cricket 
# after tokenization -->  ["I", "Love","Playing","Cricket"]
# so we have 4 tokens

# -------------------------------------------------------------------------------------

# B --> Batch
# T --> Time / Sequence Length
# C = Channels / Embedding Dimensions
# think it as:
# How many sentences? we are processing at the same time
# How many tokens per sentence?
# How many numbers represent each token?

# -------------------------------------------------------------------------------------
#  "i love playing cricket" 
# 1 sentence (B), 4 token in this sentence(T), 8 dimensions for each of the words(C) --> (1,4,8)
#  if 3 sentence B = 3 , token must be equal if sentence 1 token 3 sentence 2 and 3 token must be same , each token embeddings -> 8 so --> (B ,T ,C)  --> (4,3,8)




