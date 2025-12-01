---
draft: true
title: Attention Mechanism
---

```python
import torch
torch.manual_seed(83160)
vocab_size = 512
embedding_dim = 81  # dimensionality of model (d_model)
num_heads = 3  # number of attention heads
d_k = int(embedding_dim / num_heads)  # dimensionality of the key(and query) vectors, d_key == d_query
embedding_matrix = torch.randn(vocab_size, embedding_dim, requires_grad=True)  # the lookup table of tokens
input_seq_tokens_vec = torch.tensor(
    [47, 53, 29, 23, 17, 19, 31]
)  # input vector (e.g. one sentence in natural language)
X = embedding_matrix[input_seq_tokens_vec]  # input matrix shaped (input_seq_len, d_model)

W_Q, W_K, W_V = (
    torch.randn(embedding_dim, embedding_dim, requires_grad=True),
    torch.randn(embedding_dim, embedding_dim, requires_grad=True),
    torch.randn(embedding_dim, embedding_dim, requires_grad=True),
)
b_Q, b_K, b_V = (
    torch.randn(embedding_dim, requires_grad=True),
    torch.randn(embedding_dim, requires_grad=True),
    torch.randn(embedding_dim, requires_grad=True),
)

Q = X @ W_Q + b_Q
K = X @ W_K + b_K
V = X @ W_V + b_V

print("Single-head shapes: ", Q.shape, K.shape, V.shape)


# Reshape Q,K,V Tensors for Multi-head attention
input_seq_len = X.shape[0]
Q = Q.view(input_seq_len, num_heads, d_k).transpose(
    0, 1
)  # shape: (num_heads, input_seq_len, d_k= embedding_dim/num_heads)
K = K.view(input_seq_len, num_heads, d_k).transpose(
    0, 1
)  # shape: (num_heads, input_seq_len, d_k= embedding_dim/num_heads)
V = V.view(input_seq_len, num_heads, d_k).transpose(
    0, 1
)  # shape: (num_heads, input_seq_len, d_k= embedding_dim/num_heads)
print(f"{num_heads}-head shapes: ", Q.shape, K.shape, V.shape)
# now `Q,K,V` became 3D tensors with shape: (num_heads, input_seq_len, embedding_dim/num_heads)


scores = Q @ K.transpose(-1, -2) / (d_k**0.5)
weights = torch.softmax(scores, dim=-1)
attention_output = weights @ V

print(attention_output.shape)

```
