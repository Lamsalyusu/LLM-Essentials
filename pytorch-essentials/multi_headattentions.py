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