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

###### `Query`(`Q`), `Key`(`K`), and `Value`(`V`) concepts in attention mechanisms

`Query`(`Q`): Represents the element for which attention is being calculated. It can be thought of as "what I am looking for" or the specific piece of information that needs context. For example, if processing a word in a sentence, its Query vector would represent what information that word needs from other words in the sentence.
`Key`(`K`): Represents the information that each element in the input sequence offers for comparison. It can be thought of as "what information I have to offer." The Key vectors of all other elements in the sequence are compared against the Query vector to determine their relevance or similarity.
`Value`(`V`): Represents the actual content or meaning of each element in the input sequence. It can be thought of as "the actual content I carry." Once the attention weights are calculated based on the Query-Key similarity, these weights are used to take a weighted sum of the Value vectors, effectively combining the relevant information from other elements into a contextualized representation of the Query element.

In summary:

> `Query`: seeks information.
> `Key`: offers information for comparison.
> `Value`: provides the actual content to be retrieved and combined.
