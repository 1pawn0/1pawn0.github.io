# Text Generation Inference with LLMs

Source: [https://huggingface.co/learn/llm-course/chapter1/8?fw=pt](https://huggingface.co/learn/llm-course/chapter1/8?fw=pt)

### Basic Definitions

###### Inference

The process of using a trained LLM to generate human-like text from a given input prompt.

###### Attention Mechanism

It gives LLMs their ability to understand context and generate coherent responses. When predicting the next word, not every word in a sentence carries equal weight. This ability to **focus on relevant information** is what we call attention.

###### Context Length

The **maximum** number of **tokens** (words or parts of words) that the LLM can **process at once**. Think of it as the size of the model’s **working memory**.

### The Two-Phase Inference Process

#### I. The Prefill Phase

- It’s where all the initial ingredients are processed and made ready.

1. **Tokenization**: Converting the input text into tokens (think of these as the basic building blocks the model understands)
2. **Embedding Conversion**: Transforming these tokens into numerical representations that capture their meaning
3. **Initial Processing**: Running these embeddings through the model’s neural networks to create a rich understanding of the context

> The prefill phase needs to process **all** input tokens at once. Think of it as reading and understanding an **entire paragraph** before starting to write a response. This phase is **compute-intensive**.

#### II. The Decode Phase

- This is where the actual text generation happens. The model generates one token at a time in what we call an *autoregressive* process (where each new token depends on all previous tokens).

All of these four steps happen for ***each new token***:

1. **Attention Computation**: Looking back at all previous tokens to understand context
2. **Probability Calculation**: Determining the likelihood of each possible next token
3. **Token Selection**: Choosing the next token based on these probabilities
4. **Continuation Check**: Deciding whether to continue or stop generation

> In the decode phase, the model needs to keep track of **all** previously generated tokens and their relationships. This phase is **memory-intensive**.

<iframe src="https://agents-course-decoding-visualizer.hf.space" frameborder="0" width="1000" height="600"></iframe>
