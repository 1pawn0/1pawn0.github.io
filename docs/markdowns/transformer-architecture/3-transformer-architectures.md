# Transformer Architectures

Source:
[https://huggingface.co/learn/llm-course/chapter1/6?fw=pt](https://huggingface.co/learn/llm-course/chapter1/6?fw=pt)

### The three architectures of transformer models

- encoder-only
- decoder-only
- encoder-decoder (sequence-to-sequence).

#### Encoder Models

- At each stage, the attention layers can access ==all== the words in the initial sentence.
- These models are often characterized as having ==bi-directional attention==, and are often called ==auto-encoding== models.
- Pretrained by ==masking== random words in a given sentence and tasking the model with finding or reconstructing the initial sentence.

##### Use cases of encoder models:

Encoder models used for tasks requiring an ==understanding of the full sentence==, such as:

- sentence classification
- named entity recognition (and more generally word classification)
- extractive question answering

###### Representatives of encoder models:

- BERT
- DistilBERT
- ModernBERT

#### Decoder models

- At each stage, ==for a given word== the attention layers can only access the words positioned ==before== it in the sentence.
- They are often called ==auto-regressive== models.
- Pretrained by ==predicting== the ==next== word in the sentence.
- Used for ==text generation==

###### Representatives of this family of models include:

- Hugging Face SmolLM Series
- Meta’s Llama Series
- Google’s Gemma Series
- DeepSeek’s V3

### Modern Large Language Models (LLMs)

Most modern Large Language Models (LLMs) use the ==decoder-only== architecture.

Modern LLMs are typically trained in two phases:

1. **Pretraining**: The model learns to predict the next token on vast amounts of text data
2. **Instruction tuning**: The model is fine-tuned to follow instructions and generate helpful responses

#### Capabilities of modern LLMs

| **Capability**     | **Description**                                  | **Example**                                     |
|--------------------|--------------------------------------------------|-------------------------------------------------|
| Text generation    | Creating coherent and contextually relevant text | Writing essays, stories, or emails              |
| Summarization      | Condensing long documents into shorter versions  | Creating executive summaries of reports         |
| Translation        | Converting text between languages                | Translating English to Spanish                  |
| Question answering | Providing answers to factual questions           | “What is the capital of France?”                |
| Code generation    | Writing or completing code snippets              | Creating a function based on a description      |
| Reasoning          | Working through problems step by step            | Solving math problems or logical puzzles        |
| Few-shot learning  | Learning from a few examples in the prompt       | Classifying text after seeing just 2-3 examples |

### Sequence-to-sequence (encoder-decoder) models

- At each stage, the attention layers of the encoder can access all the words in the initial sentence, whereas the attention layers of the decoder can only access the words positioned before a given word in the input.
- The pretraining of these models can take different forms, but it often involves reconstructing a sentence for which the input has been somehow corrupted (for instance by masking random words). The pretraining of the T5 model consists of replacing random spans of text (that can contain several words) with a single mask special token, and the task is then to predict the text that this mask token replaces.

##### Use cases of Sequence-to-sequence (encoder-decoder) models:

Sequence-to-sequence models used for ==generating== new sentences ==depending on a given input==, such as:

- Summarization
- Translation
- Generative question answering

###### Representatives of this family of models include:

- BART
- mBART
- Marian
- T5

#### Practical applications of Sequence-to-sequence (encoder-decoder) models:

Sequence-to-sequence models excel at tasks that require transforming one form of text into another while preserving meaning.

| **Application**         | **Description**                                  | **Example Model** |
|-------------------------|--------------------------------------------------|:-----------------:|
| Machine translation     | Converting text between languages                |    Marian, T5     |
| Text summarization      | Creating concise summaries of longer texts       |     BART, T5      |
| Data-to-text generation | Converting structured data into natural language |        T5         |
| Grammar correction      | Fixing grammatical errors in text                |        T5         |
| Question answering      | Generating answers based on context              |     BART, T5      |

#### Choosing the right architecture

The answers to these questions will guide you toward the right architecture:

1. What kind of ==understanding== does your task need? (Bidirectional or unidirectional)
2. Are you ==generating== new text or ==analyzing== existing text?
3. Do you need to ==transform== one sequence into another?

| **Task**                               | **Suggested Architecture** | **Examples**  |
|----------------------------------------|:--------------------------:|:-------------:|
| Text classification (sentiment, topic) |          Encoder           | BERT, RoBERTa |
| Text generation (creative writing)     |          Decoder           |  GPT, LLaMA   |
| Translation                            |      Encoder-Decoder       |   T5, BART    |
| Summarization                          |      Encoder-Decoder       |   BART, T5    |
| Named entity recognition               |          Encoder           | BERT, RoBERTa |
| Question answering (extractive)        |          Encoder           | BERT, RoBERTa |
| Question answering (generative)        | Encoder-Decoder or Decoder |    T5, GPT    |
| Conversational AI                      |          Decoder           |  GPT, LLaMA   |

### Attention mechanisms

Most transformer models use full attention in the sense that the attention matrix is square. It can be a big computational bottleneck when you have long texts. Longformer and reformer are models that try to be more efficient and use a sparse version of the attention matrix to speed up training.

Standard attention mechanisms have a computational complexity of O(n²), where n is the sequence length. This becomes problematic for very long sequences.

#### LSH attention

Reformer uses LSH attention...

- [ ] TODO

#### Local attention

Longformer uses local attention...
![local_attention_mask.png](img/local_attention_mask.png)

- [ ] TODO

#### Axial positional encodings

- [ ] TODO