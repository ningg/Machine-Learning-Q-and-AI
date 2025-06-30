# Machine Learning Q and AI {#machine-learning-q-and-ai .post-title style="text-align: left;"}

## 30 Essential Questions and Answers on Machine Learning and AI {#essential-questions-and-answers-on-machine-learning-and-ai .post-subtitle}

By Sebastian Raschka. [Free to read](#table-of-contents). Published by
[No Starch Press](https://nostarch.com/machine-learning-q-and-ai).\
Copyright Â© 2024-2025 by Sebastian Raschka.

![Machine Learning and Q and
AI](../images/2023-ml-ai-beyond.jpg){.right-image-shadow-30}

> Machine learning and AI are moving at a rapid pace. Researchers and
> practitioners are constantly struggling to keep up with the breadth of
> concepts and techniques. This book provides bite-sized bits of
> knowledge for your journey from machine learning beginner to expert,
> covering topics from various machine learning areas. Even experienced
> machine learning researchers and practitioners will encounter
> something new that they can add to their arsenal of techniques.

\

# Chapter 17: Encoder- and Decoder-Style Transformers [](#chapter-17-encoder--and-decoder-style-transformers)

[]{#ch17 label="ch17"}

**What are the differences between encoder- and decoder-based language
transformers?**

Both encoder- and decoder-style architectures use the same
self-attention layers to encode word tokens. The main difference is that
encoders are designed to learn embeddings that can be used for various
predictive modeling tasks such as classification. In contrast, decoders
are designed to generate new texts, for example, to answer user queries.

This chapter starts by describing the original transformer architecture
consisting of an encoder that processes input text and a decoder that
produces translations. The subsequent sections then describe how models
like BERT and RoBERTa utilize only the encoder to understand context and
how the GPT architectures emphasize decoder-only mechanisms for text
generation.

## The Original Transformer [](#the-original-transformer)

The original transformer architecture introduced in
ChapterÂ [\[ch16\]](../ch16){reference="ch16" reference-type="ref"} was
developed for English-to-French and English-to-German language
translation. It utilized both an encoder and a decoder, as illustrated
in
FigureÂ [\[fig:ch17-fig01\]](#fig:ch17-fig01){reference="fig:ch17-fig01"
reference-type="ref"}.

::: figurewide
![image](../images/ch17-fig01.png){style="width:5.625in"}
:::

InFigureÂ [\[fig:ch17-fig01\]](#fig:ch17-fig01){reference="fig:ch17-fig01"
reference-type="ref"},the input text (that is, the sentences of the text
to betranslated) is first tokenized into individual word tokens, which
are then encoded via an embedding layer before they enter the encoder
part (see ChapterÂ [\[ch01\]](../ch01){reference="ch01"
reference-type="ref"} for more on embeddings). After a positional
encoding vector is added to each embedded word,the embeddings go through
a multi-head self-attention layer. This layer is followed by an addition
step, indicated by a plus sign (+) in
FigureÂ [\[fig:ch17-fig01\]](#fig:ch17-fig01){reference="fig:ch17-fig01"
reference-type="ref"}, which performs a layer normalization and adds the
original embeddings via a skip connection,also known as a *residual* or
*shortcut* connection. Following this is a LayerNormblock, short for
*layernormalization*, which normalizes the activations of the previous
layer to improve the stability of the neural networkâ€™s training. The
addition of the original embeddings and the layer normalization steps
are often summarized as the *Add&Normstep*. Finally, after entering the
fully connected networkâ€"a small,multilayer perceptron consisting of
two fully connected layers with a nonlinear activation function in
betweenâ€"the outputs are again added and normalized before they are
passed to a multi-head self-attention layer of the decoder.

The decoder in
FigureÂ [\[fig:ch17-fig01\]](#fig:ch17-fig01){reference="fig:ch17-fig01"
reference-type="ref"} has a similar overall structure to the encoder.
The key difference is that the inputs and outputs are different: the
encoder receives the input text to be translated, while the decoder
generates the translated text.

### Encoders [](#encoders)

The encoder part in the original transformer, as illustrated in
FigureÂ [\[fig:ch17-fig01\]](#fig:ch17-fig01){reference="fig:ch17-fig01"
reference-type="ref"}, is responsible for understanding and extracting
the relevant information from the input text. It then outputs a
continuous representation (embedding) of the input text, which is passed
to the decoder. Finally, the decoder generates the translated text
(target language) based on the continuous representation received from
the encoder.

Over the years, various encoder-only architectures have been developed
based on the encoder module of the original transformer model outlined
earlier. One notable example is BERT, which stands for bidirectional
encoder representations from transformers.

As noted in ChapterÂ [\[ch14\]](../ch14){reference="ch14"
reference-type="ref"}, BERT is an encoder-only architecture based on the
transformerâ€™s encoder module. The BERT model is pretrained on a large
text corpus using masked language modeling and next-sentence prediction
tasks. FigureÂ [1.1](#fig:ch17-fig02){reference="fig:ch17-fig02"
reference-type="ref"} illustrates the masked language modeling
pretraining objective used in BERT-style transformers.

![BERT randomly masks 15 percent of the input tokens during
pretraining.](../images/ch17-fig02.png){#fig:ch17-fig02}

As FigureÂ [1.1](#fig:ch17-fig02){reference="fig:ch17-fig02"
reference-type="ref"} demonstrates, the main idea behind masked language
modeling is to mask (or replace) random word tokens in the input
sequence and then train the model to predict the original masked tokens
based on the surrounding context.

Inadditiontothemaskedlanguagemodelingpretrainingtaskillustrated in
FigureÂ [1.1](#fig:ch17-fig02){reference="fig:ch17-fig02"
reference-type="ref"}, the next-sentence prediction task asks the model
to predict whether the original documentâ€™s sentence order of two
randomly shuffled sentences is correct. For example, say that two
sentences, in random order, are separated by the \[SEP\] token (*SEP* is
short for *separate*). The brackets are a part of the tokenâ€™s notation
and are used to make it clear that this is a special token as opposed to
a regular word in the text. BERT-style transformers also use a \[CLS\]
token. The \[CLS\] token serves as a placeholder token for the model,
prompting the model to return a *True* or *False* label indicating
whether the sentences are in the correct order:

- â€œ\[CLS\] Toast is a simple yet delicious food. \[SEP\] Itâ€™s often
  served with butter, jam, or honey.â€?

- â€œ\[CLS\] Itâ€™s often served with butter, jam, or honey. \[SEP\]
  Toast is a simple yet delicious food.â€?

The masked language and next-sentence pretraining objectives allow BERT
to learn rich contextual representations of the input texts, which can
then be fine-tuned for various downstream tasks like sentiment analysis,
question answering, and named entity recognition. Itâ€™s worth noting
that this pretraining is a form of self-supervised learning (see
ChapterÂ [\[ch02\]](../ch02){reference="ch02" reference-type="ref"} for
more details on this type of learning).

RoBERTa, which stands for robustly optimized BERT approach, is an
improved version of BERT. It maintains the same overall architecture as
BERT but employs several training and optimization improvements, such as
larger batch sizes, more training data, and eliminating the
next-sentence prediction task. These changes have resulted in RoBERTa
achieving better performance on various natural language understanding
tasks than BERT.

### Decoders [](#decoders)

Coming back to the original transformer architecture outlined in
FigureÂ [\[fig:ch17-fig01\]](#fig:ch17-fig01){reference="fig:ch17-fig01"
reference-type="ref"}, the multi-head self-attention mechanism in the
decoder is similar to the one in the encoder, but it is masked to
prevent the model from attending to future positions, ensuring that the
predictions for position *i* can depend only on the known outputs at
positions less than *i*. As illustrated in
FigureÂ [\[fig:ch17-fig03\]](#fig:ch17-fig03){reference="fig:ch17-fig03"
reference-type="ref"}, the decoder generates the output word by word.

::: figurewide
![image](../images/ch17-fig03.png){style="width:5.625in"}
:::

This masking (shown explicitly in
FigureÂ [\[fig:ch17-fig03\]](#fig:ch17-fig03){reference="fig:ch17-fig03"
reference-type="ref"}, although it occurs internally in the decoderâ€™s
multi-head self-attention mechanism) is essential to maintaining the
transformer modelâ€™s autoregressive property during training and
inference. This autoregressive property ensures that the model generates
output tokens one at a time and uses previously generated tokens as
context for generating the next word token.

Over the years, researchers have built upon the original encoder-decoder
transformer architecture and developed several decoder-only models that
have proven highly effective in various natural language
processingÂ tasks.Â The most notable models include the GPT family,
which we briefly discussed in
ChapterÂ [\[ch14\]](../ch14){reference="ch14" reference-type="ref"} and
in various other chapters throughout the book. *GPT* stands for
*generative pretrained transformer*. The GPT series comprises
decoder-only models pretrained on large-scale unsupervised text data and
fine-tuned for specific tasks such as text classification, sentiment
analysis, question answering, and summarization. The GPT models,
including at the time of writing GPT-2, GPT-3, and GPT-4, have shown
remarkable performance in various benchmarks and are currently the most
popular architecture for natural language processing.

One of the most notable aspects of GPT models is their emergent
properties. Emergent properties are the abilities and skills that a
model develops due to its next-word prediction pretraining. Even though
these models were taught only to predict the next word, the pretrained
models are capable of text summarization, translation, question
answering, classification, and more. Furthermore, these models can
perform new tasks without updating the model parameters via in-context
learning, which weâ€™ll discuss in more detail in
ChapterÂ [\[ch18\]](../ch18){reference="ch18" reference-type="ref"}.

## Encoder-Decoder Hybrids [](#encoder-decoder-hybrids)

Next to the traditional encoder and decoder architectures, there have
been advancements in the development of new encoder-decoder models that
lev- Â erage the strengths of both components. These models often
incorporate novel techniques, pretraining objectives, or architectural
modifications to enhance their performance in various natural language
processing tasks. Some notable examples of these new encoder-decoder
models include BART and T5.

Encoder-decoder models are typically used for natural language
processing tasks that involve understanding input sequences and
generating output sequences, often with different lengths and
structures. They are particularly good at tasks where there is a complex
mapping between the input and output sequences and where it is crucial
to capture the relationships between the elements in both sequences.
Some common use cases for encoder-decoder models include text
translation and summarization.

## Terminology [](#terminology)

All of these methodsâ€"encoder-only, decoder-only, and encoder-decoder
modelsâ€"are sequence-to-sequence models (often abbreviated as
*seq2seq*). While we refer to BERT-style methods as â€œencoder-only,â€?
the description may be misleading since these methods also *decode* the
embeddings into output tokens or text during pretraining. In other
words, both encoder-only and decoder-only architectures perform
decoding.

However, the encoder-only architectures, in contrast to decoder-only and
encoder-decoder architectures, donâ€™t decode in an autoregressive
fashion. *Autoregressive decoding* refers to generating output sequences
one token at a time, conditioning each token on the previously generated
tokens. Encoder-only models do not generate coherent output sequences in
this manner. Instead, they focus on understanding the input text and
producing task-specific outputs, such as labels or token predictions.

## Contemporary Transformer Models [](#contemporary-transformer-models)

In brief, encoder-style models are popular for learning embeddings used
in classification tasks, encoder-decoder models are used in generative
tasks where the output heavily relies on the input (for example,
translation and summarization), and decoder-only models are used for
other types of generative tasks, including Q&A. Since the first
transformer architecture emerged, hundreds of encoder-only,
decoder-only, and encoder-decoder hybrids have been developed, as
diagrammed in FigureÂ [1.2](#fig:ch17-fig04){reference="fig:ch17-fig04"
reference-type="ref"}.

![Some of the most popular large language transformers organized by\
architecture type and
developer](../images/ch17-fig04.png){#fig:ch17-fig04
style="width:99.0%"}

While encoder-only models have gradually become less popular,
decoder-only models like GPT have exploded in popularity, thanks to
breakthroughs in text generation via GPT-3, ChatGPT, and GPT-4. However,
encoder-only models are still very useful for training predictive models
based on text embeddings as opposed to generating texts.

### Exercises [](#exercises)

17-1. As discussed in this chapter, BERT-style encoder models are
pretrained using masked language modeling and next-sentence prediction
pretraining objectives. How could we adopt such a pretrained model for a
classification task (for example, predicting whether a text has a
positive or negative sentiment)?

17-2. Can we fine-tune a decoder-only model like GPT for classification?

## References [](#references)

- The Bahdanau attention mechanism for RNNs: Dzmitry Bahdanau, Kyunghyun
  Cho, and Yoshua Bengio, â€œNeural Machine Translation by Jointly
  Learning to Align and Translateâ€? (2014),
  <https://arxiv.org/abs/1409.0473>.

- The original BERT paper, which popularized encoder-style transformers
  with a masked word and a next-sentence prediction pretraining
  objective: Jacob Devlin et al., â€œBERT: Pre-training of Deep
  Bidirectional Transformers for Language Understandingâ€? (2018),
  <https://arxiv.org/abs/1810.04805>.

- RoBERTaimprovesuponBERTbyoptimizingtrainingprocedures,usinglargertrainingdatasets,andremovingthenext-sentencepred-
  Â ictiontask:YinhanLiuetal.,â€œRoBERTa:ARobustlyOptimizedBERTPretrainingApproachâ€?(2019),<https://arxiv.org/abs/1907.11692>.

- The BART encoder-decoder architecture: Mike Lewis et al., â€œBART:
  Denoising Sequence-to-Sequence Pre-training for Natural Language
  Generation, Translation, and Comprehensionâ€? (2018),
  <https://arxiv.org/abs/1910.13461>.

- The T5 encoder-decoder architecture: Colin Raffel et al., â€œExploring
  the Limits of Transfer Learning with a Unified Text-to-Text
  Transformerâ€? (2019), <https://arxiv.org/abs/1910.10683>.

- The paper proposing the first GPT architecture: Alec Radford et al.,
  â€œImproving Language Understanding by Generative Pre-Trainingâ€?
  (2018),
  <https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf>.

- The GPT-2 model: Alec Radford et al., â€œLanguage Models Are
  Unsupervised Multitask Learnersâ€? (2019),
  <https://www.semanticscholar.org/paper/Language-Models-are-Unsupervised-Multitask-Learners-Radford-Wu/9405cc0d6169988371b2755e573cc28650d14dfe>.

- The GPT-3 model: Tom B. Brown et al., â€œLanguage Models Are Few-Shot
  Learnersâ€? (2020), <https://arxiv.org/abs/2005.14165>.

\

------------------------------------------------------------------------

