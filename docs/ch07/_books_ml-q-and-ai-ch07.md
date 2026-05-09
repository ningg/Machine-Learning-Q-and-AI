







# Chapter 7: Multi-GPU Training Paradigms
> 本章梳理多卡训练的主要范式及其利弊，并给出在不同规模模型下的选用建议。
[](#chapter-7-multi-gpu-training-paradigms)



**What are the different multi-GPU training paradigms, and what are
their respective advantages and disadvantages?**

多 GPU 训练有哪些典型范式，各自优缺点又是什么？

Multi-GPU training paradigms can be categorized into two groups:
dividing data for parallel processing with multiple GPUs and dividing
the model among multiple GPUs to handle memory constraints when the
model size surpasses that of a single GPU. Data parallelism falls into
the first category, while model parallelism and tensor parallelism fall
into the second category. Techniques like pipeline parallelism borrow
ideas from both categories. In addition, current software
implementations such as DeepSpeed, Colossal AI, and others blend
multiple approaches into a hybrid technique.

多 GPU 范式大致可分两类：其一是在多块 GPU 间划分数据做并行；其二是在单卡放不下时将模型切分到多卡。数据并行属第一类；模型并行与张量并行属第二类；流水线并行则融合二者思想；DeepSpeed、Colossal AI 等框架还会混合多种策略。

This chapter introduces several training paradigms and provides advice
on which to use in practice.

本章将介绍若干训练范式并讨论实践中如何取舍。


This chapter primarily uses the term `GPUs` to describe the
hardware utilized for parallel processing. However, the same concepts
and techniques discussed can be applied to other specialized hardware
devices, such as tensor processing units (TPUs) or other accelerators,
depending on the specific architecture and requirements of the system.

本章主要以 `GPUs` 指代并行硬件，但所述概念同样适用于 TPU 等其他加速器，具体取决于系统架构与需求。


## The Training Paradigms
> 以下各小节依次讨论模型并行、数据并行、张量并行与序列并行等范式。
[](#the-training-paradigms)

The following sections discuss the model parallelism, data parallelism,
tensor parallelism, and sequence parallelism multi-GPU training
paradigms.

下面分别阐述模型并行、数据并行、张量并行与序列并行这几类多 GPU 训练范式。

### Model Parallelism
> 本节解释模型（层间）并行如何把大块模型分层摆放及其瓶颈。
[](#model-parallelism)

*Model parallelism*, or *inter-op parallelism*, is a technique in which
different sections of a large model are placed on different GPUs and are
computed sequentially, with intermediate results passed between the
devices. This allows for the training and execution of models that might
not fit entirely on a single device, but it can require intricate
coordination to manage the dependencies between different parts of the
model.

*模型并行*（亦称 *inter-op parallelism*）把大模型的不同部分放到不同 GPU 上顺序计算并在设备间传递中间结果，从而运行单卡放不下的模型，但需要精细协调各部分依赖。

Model parallelism is perhaps the most intuitive form of parallelization
across devices. For example, for a simple neural network that consists
of only two layers""a hidden layer and an output layer""we can keep
one layer on one GPU and the other layer on another GPU. Of course, this
can scale to an arbitrary number of layers and GPUs.

模型并行或许是跨设备并行里最直观的一种：例如仅有隐藏层与输出层两层的小网络，可各放一块 GPU，并可推广到任意多层多卡。

This is a good strategy for dealing with limited GPU memory where the
complete network does not fit into one GPU. However, there are more
efficient ways of using multiple GPUs, such as tensor parallelism,
because the chain-like structure (layer 1 on GPU 1 $\rightarrow$
layer 2 on GPU 2 $\rightarrow$ ...) in model parallelism
introduces a bottleneck. In other words, a major disadvantage of model
parallelism is that the GPUs have to wait for each other. They cannot
efficiently work in parallel, as they depend on one other's outputs.

当显存不足、整网无法装入单卡时这是可行策略；但更高效的用法是张量并行等，因为模型并行中层链（GPU1→GPU2→…）会形成瓶颈——各卡必须互相等待，难以充分并行。

### Data Parallelism
> 本节描述默认的多卡数据并行流程及其“每卡一份完整模型”的限制。
[](#data-parallelism)

*Data parallelism* has been the default mode for multi-GPU training for
several years. Here, we divide a minibatch into smaller microbatches.
Each GPU then processes a microbatch separately to compute the loss and
loss gradients for the model weights. After the individual devices
process the microbatches, the gradients are combined to compute the
weight update for the next round.

*数据并行* 多年来是多 GPU 训练的默认模式：将小批量拆成微批量，各卡独立算损失与梯度再聚合梯度更新权重。

An advantage of data parallelism over model parallelism is that the GPUs
can run in parallel. Each GPU processes a portion of the training
minibatch, that is, a microbatch. However, a caveat is that each GPU
requires a full copy of the model. This is obviously not feasible if we
have large models that don't fit into the GPU's VRAM.

相对模型并行，数据并行能让各卡真正并行处理同一 minibatch 的不同微批量；但每卡都要驻留完整模型，模型超大无法装入显存时即不可行。

### Tensor Parallelism
> 本节说明张量并行如何在单次运算内切分矩阵以实现更高并行度。
[](#tensor-parallelism)

*Tensor parallelism*, or *intra-op parallelism*, is a more efficient
form of model parallelism. Here, the weight and activation matrices are
spread across the devices instead of distributing whole layers across
devices: the individual matrices are split, so we split an individual
matrix multiplication across GPUs.

*张量并行*（*intra-op parallelism*）是更高效的模型并行：按矩阵而非整层切分，把一次矩阵乘分摊到多卡。

We can implement tensor parallelism using basic principles of linear
algebra; we can split a matrix multiplication across two GPUs in a row-
or column-wise fashion, as illustrated in
Figure [7.1](#fig-ch07-fig01) for two GPUs. (This concept can be extended to an
arbitrary number of GPUs.)

可借助线性代数把矩阵乘按行或列拆到两块 GPU 上，见图 [7.1](#fig-ch07-fig01)，并可推广到更多设备。

<a id="fig-ch07-fig01"></a>

<div align="center">
  <img src="./images/ch07-fig01.png" alt="image" width="78%" />
  <div><b>Figure 7.1</b></div>
</div>

Like model parallelism, tensor parallelism allows us to work around
memory limitations. At the same time, it also lets us execute operations
in parallel, similar to data parallelism.

与模型并行类似，张量并行可绕过显存上限，同时又能像数据并行那样并行执行算子。

A small weakness of tensor parallelism is that it can result in high
communication overhead between the multiple GPUs across which the
matrices are split or sharded. For instance, tensor parallelism requires
frequent synchronization of the model parameters across devices, which
can slow down the overall training process.

其弱点在于跨卡通信频繁、开销大，参数同步可能拖慢整体训练。

Figure [7.2](#fig-ch07-fig02) compares model, data, and tensor parallelism.

图 [7.2](#fig-ch07-fig02) 对照了三种并行方式。

<a id="fig-ch07-fig02"></a>

<div align="center">
  <img src="./images/ch07-fig02.png" alt="A comparison of model, data, and tensor parallelism" width="78%" />
  <div><b>Figure 7.2</b></div>
</div>

In model parallelism, we put different layers onto different GPUs to
work around GPU memory limitations. In data parallelism, we split a
batch across GPUs to train copies of the model in parallel, averaging
gradients for the weight update afterward. In tensor parallelism, we
split matrices (inputs and weights) across different GPUs for parallel
processing when models are too large to fit into GPU memory.

模型并行把不同层放到不同卡以节省显存；数据并行把批次分到各卡并行训练完整模型副本再平均梯度；张量并行则在模型过大时把输入与权重矩阵切到多卡上并行计算。

### Pipeline Parallelism
> 本节介绍流水线并行如何缓解模型并行的空闲时间及其设计与通信代价。
[](#pipeline-parallelism)

In *pipeline parallelism*, activations are passed during the forward
pass, as in model parallelism. The twist is that the gradients of the
input tensor are passed backward to prevent the devices from being idle.
In a sense, pipeline parallelism is a sophisticated hybrid version of
data and model parallelism.

*流水线并行* 前向仍像模型并行那样传递激活；区别在于反向传递输入张量的梯度以减少设备空转，可视为数据并行与模型并行的混合升级版。

We can think of pipeline parallelism as a form of model parallelism that
tries to minimize the sequential computation bottleneck, enhancing the
parallelism between the individual layers sitting on different devices.
However, pipeline parallelism also borrows ideas from data parallelism,
such as splitting minibatches further into microbatches.

它试图减轻模型并行的顺序瓶颈、提高跨设备层的并行度，同时也借鉴数据并行中再划微批的思想。

Pipeline parallelism is definitely an improvement over model
parallelism, though it is not perfect and there will be idle bubbles. A
further disadvantage of pipeline parallelism is that it may require
significant effort to design and implement the pipeline stages and
associated communication patterns. Additionally, the performance gains
it generates may not be as substantial as those from other
parallelization techniques, such as pure data parallelism, especially
for small models or in cases where the communication overhead is high.

流水线并行相对纯模型并行明显更好，但仍会出现流水线气泡；设计与实现各阶段及通信模式成本高，且在小模型或通信昂贵场景下收益未必胜过纯数据并行。

For modern architectures that are too large to fit into GPU memory, it
is more common nowadays to use a blend of data parallelism and tensor
parallelism techniques instead of pipeline parallelism.

对当今显存放不下的超大架构，更常见的做法是数据并行与张量并行组合，而非单独依赖流水线并行。

### Sequence Parallelism
> 本节聚焦长序列 Transformer 训练中序列切分并行及其局限。
[](#sequence-parallelism)

*Sequence parallelism* aims to address computational bottlenecks when
working with long sequences using transformer-based LLMs. More
specifically, one shortcoming of transformers is that the self-attention
mechanism (the original scaled-dot product attention) scales
quadratically with the input sequence length. There are, of course, more
efficient alternatives to the original attention mechanism that scale
linearly.

*序列并行* 面向基于 Transformer 的长序列场景；原始缩放点积自注意力随序列长度平方增长，虽有线性复杂度的替代注意力，但普及度不同。

However, these efficient self-attention mechanisms are less popular, and
most people still prefer the original scaled-dot product attention
mechanism as of this writing. Sequence parallelism, illustrated in
Figure [7.3](#fig-ch07-fig03), splits the input sequence into smaller chunks to
be distributed across GPUs, which aims to reduce computation memory
constraints of self-attention mechanisms.

写作本书时主流仍多用原始缩放点积注意力；序列并行（图 [7.3](#fig-ch07-fig03)）把输入序列切段分布到多卡，以缓解自注意力的算存压力。

<a id="fig-ch07-fig03"></a>

<div align="center">
  <img src="./images/ch07-fig03.png" alt="Sequence parallelism divides long inputs among GPUs." width="78%" />
  <div><b>Figure 7.3</b></div>
</div>

How does sequence parallelism relate to the multi-GPU techniques
discussed earlier? Sequence parallelism deals specifically with
sequential data, tensor parallelism deals with the model's internal
structure, and data parallelism deals with how the training data is
divided. Theoretically, since each of these parallelism strategies
addresses a different aspect of the computational challenge, they can
thus be combined in various ways to optimize the training or inference
process. Sequence parallelism is not as well studied as other
parallelization techniques, however.

序列并行针对序列数据维度的拆分，张量并行针对模型内部结构，数据并行针对样本划分；理论上三者关注点不同可组合优化训练或推理，但序列并行研究相对较少。

While sequence parallelism appears useful in practice, it also
introduces additional communication overheads similar to the
aforementioned parallelism techniques. Like data parallelism, it
requires us to duplicate the model and make sure it fits into the device
memory. Another of its disadvantages (depending on the implementation)
for multi-GPU training of transformers is that breaking up the input
sequence into smaller subsequences can decrease the model's accuracy
(mainly when the model is applied to longer sequences).

实践中序列并行有用，但也会引入额外通信；与数据并行类似仍需复制模型并确保单卡可装载。某些实现下把长序列切碎还可能损伤精度（尤其在模型本就面向更长上下文时）。

## Recommendations
> 本节依据模型是否能装入单卡等因素给出多范式组合的实践建议。
[](#recommendations)

Practical recommendations depend on the context. If we train small
models that fit onto a single GPU, then data parallelism strategies may
be the most efficient. Performance gains from pipeline parallelism may
not be as significant as those from other parallelization techniques,
such as data parallelism, especially for small models or in cases where
the communication overhead is high.

是否选用何种范式取决于场景：小模型可装入单卡时数据并行往往最高效；流水线并行在小模型或通信昂贵时收益有限。

If models are too large to fit into the memory of a single GPU, we need
to explore model or tensor parallelism. Tensor parallelism is naturally
moreefficient; the GPUs can work in parallel since there is no
sequential dependency as in model parallelism.

若单卡装不下，则需模型并行或张量并行；张量并行天然 moreefficient（原文连写如此），各卡可无模型并行那样的顺序依赖而并行工作。

Modern multi-GPU strategies also typically combine data parallelism and
tensor parallelism.

当下主流策略也多将数据并行与张量并行结合使用。

## Exercises
> 习题探讨 Adam 下的显存问题以及在 CPU 上做数据并行的合理性。
[](#exercises)

7-1. Suppose we are implementing our own version of tensor parallelism,
which works great when we train our model with a standard stochastic
gradient descent optimizer. However, when we try the Adam optimizer by
Diederik P. Kingma and Jimmy Ba, we encounter an out-of-memory device.
What problem might explain this issue?

习题 7-1：自研张量并行配合 vanilla SGD 训练正常，但换成 Kingma 与 Ba 的 Adam 就出现设备 OOM，可能原因是什么？

7-2. Suppose we don't have access to a GPU and are considering using
data parallelism on the CPU. Is this a good idea?

习题 7-2：若没有 GPU，仅在 CPU 上做数据并行是否值得推荐？

## References
> 本节列出 Adam、DeepSpeed、ColossalAI、流水线与序列并行及 Transformer 效率综述等参考文献。
[](#references)

- The original paper on the Adam optimizer: Diederik P. Kingma and Jimmy
  Ba, "Adam: A Method for Stochastic Optimization"? (2014),
  <https://arxiv.org/abs/1412.6980>.

Kingma 与 Ba 提出 Adam 优化器的原始论文（2014）。

- FormoreonDeepSpeedandColossal-AIformulti-GPUtraining:
  <https://github.com/microsoft/DeepSpeed> and
  <https://github.com/hpcaitech/ColossalAI>.

DeepSpeed 与 ColossalAI 的多 GPU 训练资料仓库链接。

- Pipeline parallelism tutorials and research by the DeepSpeed team:
  <https://www.deepspeed.ai/tutorials/pipeline> and Yanping Huang et
  al., "GPipe: Efficient Training of Giant Neural Networks Using
  Pipeline Parallelism"? (2018), <https://arxiv.org/abs/1811.06965>.

DeepSpeed 流水线教程及 Huang 等 GPipe 论文（2018）。

- The paper proposing sequence parallelism for transformer-based
  language models: Shenggui Li et al., "Sequence Parallelism: Long
  Sequence Training from \[a\] System\[s\] Perspective"? (2022),
  <https://arxiv.org/abs/2105.13120>.

Li 等面向 Transformer 语言模型提出序列并行的论文（2022）。

- The scaled-dot product attention mechanism was proposed with the
  original transformer architecture: Ashish Vaswani et al., "Attention
  Is All You Need"? (2017), <https://arxiv.org/abs/1706.03762>.

Vaswani 等在原始 Transformer 论文中提出的缩放点积注意力（2017）。

- A survey covering alternatives to the original self-attention
  mechanism that scale linearly: Yi Tay et al., "Efficient
  Transformers: A Survey"? (2020), <https://arxiv.org/abs/2009.06732>.

Tay 等关于高效（含线性复杂度）Transformer 变体的综述（2020）。

- A survey covering additional techniques to improve the training
  efficiency of transformers: Bohan Zhuang et al., "A Survey on
  Efficient Training of Transformers"? (2023),
  <https://arxiv.org/abs/2302.01107>.

Zhuang 等关于提升 Transformer 训练效率的综述（2023）。

- Modern multi-GPU strategies typically combine data parallelism and
  tensor parallelism. Popular examples include DeepSpeed stages 2 and 3,
  described in this tutorial on the zero redundancy optimizer:
  <https://www.deepspeed.ai/tutorials/zero/>.

当代多 GPU 方案常结合数据与张量并行；DeepSpeed ZeRO 阶段 2、3 等的教程链接。


------------------------------------------------------------------------

