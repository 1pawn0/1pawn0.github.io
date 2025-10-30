# [**torch.nn.functional.scaled_dot_product_attention**](https://docs.pytorch.org/docs/main/generated/torch.nn.functional.scaled_dot_product_attention.html)

```python
torch.nn.functional.scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False
) -> Tensor
```

#### Parameters:

###### `query` tensor shape:

**`(batch_size,..., number_of_heads_of_query, target_sequence_length, embedding_dimension_of_the_query_and_key)`**

###### `key` tensor shape:

**`(batch_size,..., number_of_heads_of_key_and_value, source_sequence_length, embedding_dimension_of_the_query_and_key)`**

###### `value` tensor shape:

**`(batch_size,..., number_of_heads_of_value, source_sequence_length, embedding_dimension_of_the_value)`**

###### `Attention` output tensor shape:

**`(batch_size,..., number_of_heads_of_query, target_sequence_length, embedding_dimension_of_the_value)`**

###### Shape legend:

$N$: Batch size
$S$: Source sequence length
$L$: Target sequence length
$E$: Embedding dimension of the query and key
$E_v$: Embedding dimension of the value
$H_q$: Number of heads of query
$H$: Number of heads of key and value
