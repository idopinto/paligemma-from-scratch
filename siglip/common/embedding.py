import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(num_embeddings, embedding_dim), requires_grad=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight[x]  # fancy indexing


# Test if Embedding is equivalent to nn.Embedding
if __name__ == "__main__":
    num_embeddings = 10
    embedding_dim = 3
    torch.manual_seed(0)

    custom_emb = Embedding(num_embeddings, embedding_dim)
    torch.manual_seed(0)
    nn_emb = nn.Embedding(num_embeddings, embedding_dim)

    # Copy weights so both are identical for fair test
    nn_emb.weight.data.copy_(custom_emb.weight.data)

    # Test with random indices
    indices = torch.randint(0, num_embeddings, (4,))
    out_custom = custom_emb(indices)
    out_nn = nn_emb(indices)

    print("Custom Embedding output:\n", out_custom)
    print("nn.Embedding output:\n", out_nn)
    print("Are the outputs equal?", torch.allclose(out_custom, out_nn))
