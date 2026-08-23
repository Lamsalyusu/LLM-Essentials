# why do we need multihead attentions ?

# consider we have a sentence called 
# "The boy played cricket because he loved the sport".

# here different relationship exists in the sentence 
# 1.One attention might learn  ---> "he" --> "boy"
# 2.Second attention might learn "sport --> "cricket"
# 3.Third might learn "played"--> "cricket"

# Main idea -> different attention heads can learn different relationships between tokens 
# Instead of:
#         One attention
#              ↓
#           Output

# we have:

#               Input
#                 │
#        ┌────────┼────────┐
#        ↓        ↓        ↓
#      Head 1   Head 2   Head 3   
#        │        │        │
#        └────────┼────────┘
#                 ↓
#            Concatenate
#                 ↓
#         Linear projection
#                 ↓
#              Output

# ======================================================================================
# One head vs multiple heads

# Suppose:

# C = 8

# and we choose:

# number of heads = 2

# We divide the embedding dimension:

# 8 / 2 = 4

# So:

# Head 1 → 4 dimensions
# Head 2 → 4 dimensions

# Together:

# 4 + 4 = 8

# This is why you'll commonly see:

# n_embd = 8
# n_head = 2
# head_size = n_embd // n_head

# ======================================================================================

# What happens inside each head?

# Remember your self-attention formula:

# Attention(Q,K,V)
# =
# softmax(QKᵀ / √dₖ)V
# Each head performs its own Q, K, V projections.

# =======================================================================================

#                  Input
#                    │
#         ┌──────────┴──────────┐
#         ↓                     ↓
#       Head 1                Head 2
#         │                     │
#       Q K V                 Q K V
#         │                     │
#    Attention              Attention
#         │                     │
#         ↓                     ↓
#       output                output
#         │                     │
#         └──────────┬──────────┘
#                    ↓
#              concatenate
#                    ↓
#             Linear layer
#                    ↓
#                 output

# =======================================================================================
# The tensor shapes

# This is where your understanding of B, T, C becomes extremely useful.

# Suppose:

# B = 2
# T = 4
# C = 8

# Input:

# (B, T, C)

# becomes:

# (2, 4, 8)

# We have:

# 2 sequences
# 4 tokens
# 8-dimensional embeddings

# Suppose:

# number of heads = 2

# Then:

# head_size = 8 / 2 = 4

# Each head works with:

# (B, T, head_size)

# which is:

# (2, 4, 4)

# =======================================================================================

# The important reshaping

# This is one of the most important PyTorch operations you'll see in Transformer code.

# Suppose:

# B, T, C = x.shape

# and:

# n_head = 2
# head_size = 4

# You might have:

# q = self.query(x)

# Initially:

# q.shape
# =
# (B, T, C)
# =
# (2, 4, 8)

# But we want:

# (B, n_head, T, head_size)

# So:

# (2, 4, 8)
#         ↓
# (2, 4, 2, 4)
#         ↓
# (2, 2, 4, 4)

# The final shape:

# (B, n_head, T, head_size)

# is:

# (2, 2, 4, 4)

# Read it as:

# 2 batches
# 2 attention heads
# 4 tokens
# 4 dimensions per head

# =======================================================================================

# Why do we transpose?

# You'll often see something like:

# q = q.view(B, T, n_head, head_size)
# q = q.transpose(1, 2)

# Before transpose:

# (B, T, n_head, head_size)

# After:

# (B, n_head, T, head_size)

# Why?

# Because we want PyTorch to organize the tensor as:

# batch
#  ↓
# heads
#  ↓
# tokens
#  ↓
# features

# This makes the attention matrix multiplication convenient.

# ==========================================================================================
# | Attention             | Q comes from  | K/V come from    | Typical use                 |
# | --------------------- | ------------- | ---------------- | --------------------------- |
# | Self-attention        | Same sequence | Same sequence    | Understanding relationships |
# | Causal self-attention | Same sequence | Same sequence    | GPT / text generation       |
# | Cross-attention       | One sequence  | Another sequence | Encoder-decoder models      |
# ==========================================================================================


        #           INPUT
        #             │
        #             ↓
        #      Token Embeddings
        #             │
        #             ↓
        #           X
        #             │
        #   ┌─────────┼─────────┐
        #   ↓         ↓         ↓
        #  Q          K         V
        #   │         │         │
        #   └────┬────┘         │
        #        ↓              │
        #       QKᵀ             │
        #        ↓              │
        #   Scale by √dₖ        │
        #        ↓              │
        #   Apply mask          │
        #        ↓              │
        #      Softmax          │
        #        ↓              │
        # Attention weights     │
        #        │              │
        #        └──────┬───────┘
        #               ↓
        #              × V
        #               ↓
        #         Attention output
        #               │
        #               ↓
        #       Multiple heads
        #               │
        #               ↓
        #         Concatenate
        #               │
        #               ↓
        #         Projection



import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads):

        super().__init__()

        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.head_dim = embed_dim // num_heads

        # Q, K, V projections
        self.query = nn.Linear(
            embed_dim,
            embed_dim
        )

        self.key = nn.Linear(
            embed_dim,
            embed_dim
        )

        self.value = nn.Linear(
            embed_dim,
            embed_dim
        )

        # Final projection
        self.projection = nn.Linear(
            embed_dim,
            embed_dim
        )


    def forward(self, x):

        B, T, C = x.shape


        # --------------------------------
        # Create Q, K, V
        # --------------------------------

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)


        # --------------------------------
        # Split into multiple heads
        # --------------------------------

        q = q.view(
            B,
            T,
            self.num_heads,
            self.head_dim
        )

        k = k.view(
            B,
            T,
            self.num_heads,
            self.head_dim
        )

        v = v.view(
            B,
            T,
            self.num_heads,
            self.head_dim
        )


        # --------------------------------
        # Move heads before sequence
        # --------------------------------

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)


        # --------------------------------
        # Attention scores
        # --------------------------------

        scores = (
            q @ k.transpose(-2, -1)
        )


        # --------------------------------
        # Scale
        # --------------------------------

        scores = scores / (
            self.head_dim ** 0.5
        )


        # --------------------------------
        # Softmax
        # --------------------------------

        attention_weights = F.softmax(
            scores,
            dim=-1
        )


        # --------------------------------
        # Weighted sum of values
        # --------------------------------

        output = (
            attention_weights @ v
        )


        # --------------------------------
        # Put heads back together
        # --------------------------------

        output = output.transpose(1, 2)

        output = output.contiguous().view(
            B,
            T,
            C
        )


        # --------------------------------
        # Final projection
        # --------------------------------

        output = self.projection(output)

        return output


x = torch.randn(
    2,
    4,
    8
)

attention = MultiHeadAttention(
    embed_dim=8,
    num_heads=2
)

output = attention(x)

print(x.shape)
print(output.shape)