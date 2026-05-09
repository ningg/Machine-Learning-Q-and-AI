



# Chapter 8: The Success of Transformers
> 本章从注意力、预训练、规模与并行性等方面归纳 Transformer 取得成功的关键因素。
[](#chapter-8-the-success-of-transformers)



**What are the main factors that have contributed to the success of
transformers?**

促成 Transformer 广泛成功的要素主要有哪些？

In recent years, transformers have emerged as the most successful neural
network architecture, particularly for various natural language
processing tasks. In fact, transformers are now on the cusp of becoming
state of the art for computer vision tasks as well. The success of
transformers can be attributed to several key factors, including their
`attention mechanisms`, ability to be `parallelized easily`, 
`unsupervised pretraining`, and `high parameter counts`.

近年来 Transformer 成为最成功的神经网络架构之一，尤其在众多 NLP 任务上如此；在计算机视觉领域也接近最前沿。其成功可归因于 `attention mechanisms`（注意力机制）、易于 `parallelized easily`（并行化）、`unsupervised pretraining`（无监督预训练）以及 `high parameter counts`（高参数量）等因素。

> Tips: 
> 
> - 注意力机制，使得模型可以关注到输入序列中的重要部分，从而提高模型性能。
> - 模型并行化，从而提高训练速度。
> - 自监督预训练，使得模型可以利用大量无标签数据，从而提高模型性能。
> - 高参数数量，使得模型可以学习到更复杂的特征，从而提高模型性能。

## The Attention Mechanism
> 本节区分视觉启发式注意力与 Transformer 自注意力，并解释动态权重相对固定层权重的特点。
[](#the-attention-mechanism)

The self-attention mechanism found in transformers is one of the key
design components that make transformer-based LLMs so successful.
However, transformers are not the first architecture to utilize
attention mechanisms.

Transformer 中的自注意力是驱动大规模语言模型成功的核心构件之一，但它并非首个使用注意力的架构。

Attention mechanisms were first developed in the context of image
recognition back in 2010, before being adopted to aid the translation of
long sentences in recurrent neural networks.
(Chapter [\[ch16\]](./ch16/_books_ml-q-and-ai-ch16.md)
compares the attention mechanisms found in recurrent neural networks and
transformers in greater detail.)

注意力机制最早出现在 2010 年前后的图像识别研究中，随后被用于帮助循环神经网络翻译长句；上文括号中所述第十六章对两类注意力有更细致的对照。

The aforementioned attention mechanism is inspired by human vision,
focusing on specific parts of an image (foveal glimpses) at a time to
process information hierarchically and sequentially. In contrast, the
fundamental mechanism underlying transformers is a self-attention
mechanism used for sequence-to-sequence tasks, such as machine
translation and text generation. It allows each token in a sequence to
attend to all other tokens, thus providing context-aware representations
of each token.

前述注意力借用人眼视觉：一次聚焦于图像局部（foveal glimpses）并以层级、顺序方式加工信息。Transformer 的基本单元则是面向序列到序列任务的自注意力：序列中每个 token 可关注其余 token，从而获得上下文相关的表示。

> Tips: 人类视觉系统，是分层处理的，先关注到特定部分（`foveal glimpses` 中央凹注视），然后逐步关注到更多部分(以`分层`和`顺序`的方式处理信息)。
> 
> - 人类视觉：像看东西一样，先看重点部分，再逐步看更多细节
> - Transformer自注意力：每个词都能"看到"句子中的所有其他词，理解上下文关系



What makes attention mechanisms so unique and useful? For the following
illustration, suppose we are using an encoder network on a fixed-length
representation of the input sequence or image -- this can be a fully
connected, convolutional, or attention-based encoder.

注意力机制的独特用处何在？下文示意中，假设编码器对固定长度的序列或图像表征进行处理——可以是全连接、卷积或注意力型编码器。

> Tips:  FIXME, 注意力权重是动态的？？？ 因为跟元素相对位置有关？元素就是 token？
> 
> - 在Transformer中，编码器使用**自注意力机制**，计算每个输入token相对于序列中其他token的重要性，从而让模型关注输入序列中的相关部分。
> - 概念上，注意力机制允许，Transformer关注序列或图像的不同部分。
> - 表面上，这听起来非常类似于`全连接层`，其中每个输入元素与下一个层中的输入元素的权重连接。
> - 在注意力机制中，计算注意力权重涉及将每个输入元素与所有其他元素进行比较。
> - 通过这种方法获得的注意力权重是`动态`的，并且依赖于输入。
> - 相比之下，卷积或全连接层的权重在训练后是`固定的`，如Figure [8.1](#fig-ch08-fig01)所示。



In a transformer, the encoder uses self-attention mechanisms to compute
the importance of each input token relative to other tokens in the
sequence, allowing the model to focus on relevant parts of the input
sequence. Conceptually, attention mechanisms allow the transformers to
attend to different parts of a sequence or image. On the surface, this
sounds very similar to a fully connected layer where each input element
is connected via a weight with the input element in the next layer. In
attention mechanisms, the computation of the attention weights involves
comparing each input element to all others. The attention weights
obtained by this approach are dynamic and input dependent. In contrast,
the weights of a convolutional or fully connected layer are fixed after
training, as illustrated in
Figure [8.1](#fig-ch08-fig01).

在 Transformer 里，编码器用自注意力衡量各 token 相对重要性，使模型聚焦输入的相关片段；概念上它允许模型“看向”序列或图像的不同区域。表面上这与全连接层类似，但注意力权重通过两两比较输入元素得到，因而动态依赖输入；卷积或全连接权重在训练完成后则固定不变，见图 [8.1](#fig-ch08-fig01)。

<a id="fig-ch08-fig01"></a>

<div align="center">
  <img src="./images/ch08-fig01.png" alt="The conceptual difference between model weights in fully connected layers (top) and attention scores (bottom)" width="78%" />
  <div><b>Figure 8.1</b></div>
</div>

As the top part of
Figure [8.1](#fig-ch08-fig01) shows, once trained, the weights of fully
connected layers remain fixed regardless of the input. In contrast, as
shown at the bottom, self-attention weights change depending on the
inputs, even after a transformer is trained.

如图 [8.1](#fig-ch08-fig01) 上部所示，训练完成后全连接权重不随输入改变；下部则表明即便模型已训练完毕，自注意力权重仍会随输入变化。

> Tips: 
> 
> - 注意力机制，允许神经网络**选择性**地对**不同输入特征**的重要性进行**加权**，从而让模型专注于给定任务的输入的`最相关部分`。
> - 这提供了对每个词或图像token的**上下文理解**，允许更细致的解释，这是使Transformer如此成功的一个方面。  

Attention mechanisms allow a neural network to selectively weigh the
importance of different input features, so the model can focus on the
mostrelevant parts of the input for a given task. This provides a
contextual understanding of each word or image token, allowing for more
nuanced interpretations, which is one of the aspects that can make
transformers work so well.

注意力机制让网络按重要性加权不同输入特征，从而使模型专注于任务相关的 mostrelevant（原文连写如此）区域；这为每个词或图像 token 提供上下文语义与更细粒度解释，也是 Transformer 奏效的原因之一。

## Pretraining via Self-Supervised Learning
> 本节强调大规模无标注数据上的自监督预训练如何为下游任务提供通用语言表征。
[](#pretraining-via-self-supervised-learning)

> Tips: 
> 
> - 自监督预训练，是Transformer成功的一个重要因素。
> - 在自监督预训练中，Transformer模型被训练来预测句子中的缺失词或文档中的下一个句子。
> - 通过学习预测这些缺失词或下一个句子，模型被迫学习语言的通用表示，可以针对各种下游任务进行微调。


Pretraining transformers via self-supervised learning on large,
unlabeled datasets is another key factor in the success of transformers.
During pretraining, the transformer model is trained to predict missing
words in a sentence or the next sentence in a document, for example. By
learning to predict these missing words or the next sentence, the model
is forced to learn general representations of language that can be
fine-tuned for a wide range of downstream tasks.

在海量无标注数据上做自监督预训练是 Transformer 成功的又一关键：例如预测句中缺失词或文档的下一句；通过这些代理任务，模型被迫学习可迁移到多种下游任务的通用语言表征。

While unsupervised pretraining has been highly effective for natural
language processing tasks, its effectiveness for computer vision tasks
is still an active area of research. (Refer to
Chapter [\[ch02\]](./ch02/_books_ml-q-and-ai-ch02.md) for
a more detailed discussion of self-supervised learning.)

无监督预训练在 NLP 上极为有效，但在计算机视觉上的优劣仍是活跃研究方向；上文括号指引第二章查看自监督学习的详细讨论。

## Large Numbers of Parameters
> 本节讨论超大参数量、缩放律以及数据规模与模型规模协同扩展的意义。
[](#large-numbers-of-parameters)


One noteworthy characteristic of transformers is their large model
sizes. For example, the popular 2020 GPT-3 model consists of 175 billion
trainable parameters, while other transformers, such as switch
transformers, have trillions of parameters.

Transformer 的显著特征之一是大规模：例如 2020 年前后知名的 GPT-3 约有 1750 亿可训练参数，Switch Transformer 等则可到万亿量级。

The scale and number of trainable parameters of transformers are
essential factors in their modeling performance, particularly for
large-scale natural language processing tasks. 
For instance, `linear scaling laws` suggest that the training loss decreases proportionally
with an increase in model size, so a doubling of the model size can
halve the training loss.

参数规模对建模能力至关重要，尤其在大规模 NLP 上；例如 `linear scaling laws`（线性缩放律）暗示训练损失随模型增大近似按比例下降，模型体量翻倍或可使训练损失减半。

This, in turn, can lead to better performance on the downstream target
task. However, it is essential to scale the model size and the number of
training tokens equally. This means the number of training tokens should
be doubled for every doubling of model size.

这有助于下游任务表现，但模型体量与训练 token 数需要协同扩展：模型规模每翻倍，训练 token 也应大致翻倍。

Since labeled data is limited, utilizing large amounts of data during
unsupervised pretraining is vital.

标注数据有限时，在无监督预训练阶段尽可能利用更多文本尤为关键。

To summarize, large model sizes and large datasets are critical factors
in transformers' success. Additionally, using self-supervised
learning, the ability to pretrain transformers is closely tied to using
large model sizes and large datasets. This combination has been critical
in enabling the success of transformers in a wide range of natural
language processing tasks.

综上，大模型与大语料是 Transformer 成功的基石；自监督预训练能力又与二者紧密耦合，这一组合支撑了其在众多 NLP 任务中的突破。

> Tips: 总而言之，Transformer的成功，很大程度上归功于其`大模型`和`大数据`的使用。
>
> - 线性缩放定律：训练损失与模型大小成正比，因此增加模型大小可以减少训练损失。
> - 训练tokens数量：训练tokens数量应该与模型大小成正比，因此增加模型大小应该增加训练tokens数量。


## Easy Parallelization
> 本节说明固定长度 token 序列与成对注意力计算为何便于在多核或多机上并行。
[](#easy-parallelization)

Training `large models` on `large datasets` requires `vast computational resources`,
and it's key that the computations can be parallelized to utilize
these resources.

要用大数据训练大模型需要巨量算力，因此计算能否并行化以吃满硬件至关重要。

Fortunately, transformers are easy to parallelize since they take a fixed-length
sequence of word or image tokens as input. For instance, the
self-attention mechanism used in most transformer architectures involves
computing the weighted sum between a pair of input elements.
Furthermore, these pair-wise token comparisons can be computed
independently, as illustrated in
Figure [8.2](#fig-ch08-fig02), making the self-attention mechanism relatively
easy to parallelize across different GPU cores.

幸运的是 Transformer 以固定长度的词或图像 token 序列为输入，较易并行；多数架构中的自注意力需计算输入元素两两之间的加权和，而这些成对比较彼此独立（见图 [8.2](#fig-ch08-fig02)），因而在多 GPU 核心上相对容易并行。

<a id="fig-ch08-fig02"></a>

<div align="center">
  <img src="./images/ch08-fig02.png" alt="A simplified self-attention mechanism without weight parameters" width="78%" />
  <div><b>Figure 8.2</b></div>
</div>

In addition, the individual weight matrices used in the self-attention
mechanism (not shown in Figure [8.2](#fig-ch08-fig02)) can be distributed across different machines for
distributed and parallel computing.

此外，自注意力中的各权重矩阵（图 [8.2](#fig-ch08-fig02) 未画出）也可分布到多台机器以实现分布式并行计算。

## Exercises
> 习题引导读者辨析自注意力“易并行”与整体计算昂贵之间的矛盾，并思考其与特征选择的关系。
[](#exercises)

8-1. As discussed in this chapter, self-attention is easily
parallelizable, yet transformers are considered computationally
expensive due to self-attention. How can we explain this contradiction?

习题 8-1：本章称自注意力易于并行，但 Transformer 又因自注意力而计算昂贵，如何理解这一表面矛盾？

8-2. Since self-attention scores represent importance weights for the
various input elements, can we consider self-attention to be a form of
feature selection?

习题 8-2：若自注意力得分表示各输入元素的重要性权重，能否把它视作一种特征选择？

## References
> 本节列出视觉注意力、Transformer 原始论文、稀疏巨型 Transformer、缩放律与算力最优训练等相关文献与博文链接。
[](#references)

- An example of an attention mechanism in the context of image rec-
   ognition: Hugo Larochelle and Geoffrey Hinton, "Learning to
  Combine Foveal Glimpses with a Third-Order Boltzmann Machine"?
  (2010), <https://dl.acm.org/doi/10.5555/2997189.2997328>.

Larochelle 与 Hinton 在图像语境下使用注意力机制的示例（2010）。

- The paper introducing the self-attention mechanism with the original
  transformer architecture: Ashish Vaswani et al., "Attention Is All
  You Need"? (2017), <https://arxiv.org/abs/1706.03762>.

Vaswani 等在原始 Transformer 论文中引入自注意力机制（2017）。

- Transformers can have trillions of parameters: William Fedus, Barret
  Zoph, and Noam Shazeer, "Switch Transformers: Scaling to Trillion
  Parameter Models with Simple and Efficient Sparsity"? (2021),
  <https://arxiv.org/abs/2101.03961>.

Fedus、Zoph、Shazeer 关于可达万亿参数规模的 Switch Transformer（2021）。

- Linear scaling laws suggest that training loss decreases
  proportionally with an increase in model size: Jared Kaplan et al.,
  "Scaling Laws for Neural Language Models"? (2020),
  <https://arxiv.org/abs/2001.08361>.

Kaplan 等关于神经语言模型缩放律的论文（2020）。

- Research suggests that in transformer-based language models, the
  training tokens should be doubled for every doubling of model size:
  Jordan Hoffmann et al., "Training Compute-Optimal Large Language
  Models"? (2022), <https://arxiv.org/abs/2203.15556>.

Hoffmann 等指出 Transformer 语言模型在规模翻倍时应同步加倍训练 token 的研究（2022）。

- Formoreabouttheweightsusedinself-attentionandcross-attention
  mechanisms, check out my blog post: "Understanding and Coding the
  Self-Attention Mechanism of Large Language Models from Scratch"? at
  <https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html>.

作者关于从零理解并实现自注意力权重的博文链接。


------------------------------------------------------------------------

