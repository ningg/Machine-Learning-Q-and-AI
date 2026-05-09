







# Chapter 22: Speeding Up Inference
> 本章在**不改动模型架构、不牺牲精度**的前提下，梳理推理加速的通用手段：并行、向量化、循环分块、算子融合与量化，并说明各自要点与边界。
[](#chapter-22-speeding-up-inference)



**What are techniques to speed up model inference through optimization
without changing the model architecture or sacrificing accuracy?**

在不改变模型结构、也不以牺牲精度为代价的前提下，有哪些优化手段可以加快**推理（inference）**？

In machine learning and AI, *model inference* refers to making
predictions or generating outputs using a trained model. The main
general techniques for improving model performance during inference
include parallelization, vectorization, loop tiling, operator fusion,
and quantization, which are discussed in detail in the following
sections.

在机器学习与 AI 中，*模型推理*指用已训练模型做预测或生成输出。推理阶段常见的通用加速思路包括并行化、向量化、循环分块（loop tiling）、算子融合与量化等，下文逐节展开。

> Tips:
> 
> - 优化模型**推理速度**，有多种方法，包括：`并行化`、`向量化`、`循环分块`、`算子融合`、`量化`等。
> - 这些方法，将在后续章节中详细讨论。

## Parallelization
> 本节说明如何通过批量处理同时利用硬件并行，并以图示对比串行与批量推理的瓶颈差异。
[](#parallelization)

One common way to achieve better parallelization during inference is to
run the model on a batch of samples rather than on a single sample at a
time. This is sometimes also referred to as *batched inference* and
assumes that we are receiving multiple input samples or user inputs
simultaneously or within a short time window, as illustrated in
Figure [22.1](#fig-ch22-fig01).

推理阶段提升并行度的一种常见做法，是对**一批样本**而非单条依次前向传播；这也常称为*批量推理（batched inference）*，前提是在同一时间或很短时间窗内会汇聚多条请求，见图 [22.1](#fig-ch22-fig01)。

> Tips: **并行化**，也被称为`批量推理`，同时或短时间窗口内，接收到多个输入样本或用户输入，模型同时处理。

<a id="fig-ch22-fig01"></a>

<div align="center">
  <img src="./images/ch22-fig01.png" alt="Sequential inference and batched inference" width="78%" />
  <div><b>Figure 22.1</b></div>
</div>

Figure [22.1](#fig-ch22-fig01) shows sequential inference processing one item at
a time, which creates a bottleneck if there are several samples waiting
to be classified. In batched inference, the model processes all four
samples at the same time.

图 [22.1](#fig-ch22-fig01) 表明：逐条串行推理会在多条样本排队时形成瓶颈；而批量推理可让模型一次性处理图中的四条输入。

## Vectorization
> 本节解释向量化的含义、与 BLAS 的关系，并说明在 TensorFlow、PyTorch 等框架中张量运算如何自动受益。
[](#vectorization)

*Vectorization* refers to performing operations on entire data
structures, such as arrays (tensors) or matrices, in a single step
rather than using iterative constructs like `for` loops. Using vectorization, multiple operations from
the loop are performed simultaneously using single instruction, multiple
data (SIMD) processing, which is available on most modern CPUs.

*向量化*指对数组（张量）、矩阵等整体一次完成运算，而不是用 `for` 循环逐元素迭代；底层往往借助 CPU 上的 SIMD（单指令多数据）把循环中的多次操作并行完成。

> Tips: **向量化**，也被称为`单指令多数据`，在现代 CPU 上，可以同时处理多个数据。

This approach takes advantage of the low-level optimizations in many
computing systems and often results in significant speedups. For
example, it might rely on BLAS.

这种做法能吃到不少系统级低层优化，往往带来可观加速，例如可依赖 BLAS。

*BLAS* (which is short for *Basic Linear Algebra Subprograms*) is a
specification that prescribes a set of low-level routines for performing
common linear algebra operations such as vector addition, scalar
multiplication, dot products, matrix multiplication, and others. Many
array and deep learning libraries like NumPy and PyTorch use BLAS under
the hood.

*BLAS*（Basic Linear Algebra Subprograms）规定了一套底层例程，用于向量加、标量乘、点积、矩阵乘等常见线性代数运算；NumPy、PyTorch 等库在底层常会调用 BLAS。

To illustrate vectorization with an example, suppose we wanted to
compute the dot product between two vectors. The non-vectorized way of
doing this would be to use a `for` loop, iterating over each element of the array one
by one. However, this can be quite slow, especially for large arrays.
With vectorization, you can perform the dot product operation on the
entire array at once, as shown in
Figure [22.2](#fig-ch22-fig02).

举例：两向量点积若用 `for` 逐元素累加，大数组上会很慢；向量化则可对整体一次性完成点积运算，见图 [22.2](#fig-ch22-fig02)。

<a id="fig-ch22-fig02"></a>

<div align="center">
  <img src="./images/ch22-fig02.png" alt="A classic loop versus a vectorized dot product computation in Python" width="65%" />
  <div><b>Figure 22.2</b></div>
</div>

In the context of linear algebra or deep learning frameworks like
TensorFlow and PyTorch, vectorization is typically done automatically.
This is because these frameworks are designed to work with
multidimensional arrays (also known as *tensors*), and their operations
are inherently vectorized. This means that when you perform functions
using these frameworks, you automatically leverage the power of
vectorization, resulting in faster and more efficient computations.

在线性代数或 TensorFlow、PyTorch 等深度学习框架里，向量化通常由框架自动完成：它们面向多维数组（*张量*）设计，算子本身即按向量化实现，因而调用高层 API 时往往已自动获得更快、更省的开销。

## Loop Tiling
> 本节介绍循环分块如何改善缓存局部性，说明二维示例与在 Python 等高层语言中通常依赖底层库代劳的原因。
[](#loop-tiling)

*Loop tiling* (also often referred to as *loop nest optimization*) is an
advanced optimization technique to enhance data locality by breaking
down a loop's iteration space into smaller chunks or "tiles."? This
ensures that once data is loaded into cache, all possible computations
are performed on it before the cache is cleared.

*循环分块（loop tiling）*（也常称*循环嵌套优化*）通过把迭代空间划成较小的“块”，提升**数据局部性**：数据一旦进缓存，就尽量在同一块上把能算的都算完，再被淘汰。

> Tips: **循环分块**，也被称为`循环嵌套优化`，将循环的迭代空间分成小块，确保数据加载到缓存后，所有可能的计算都在缓存中完成，然后缓存被清除。

Figure [22.3](#fig-ch22-fig03) illustrates the concept of loop tiling for
accessing elements in a two-dimensional array. In a regular
`for` loop, we iterate over columns and rows one element at a time, whereas in loop tiling, we
subdivide the array into smaller tiles.

图 [22.3](#fig-ch22-fig03) 用二维数组访问示意循环分块：常规双重循环逐元素扫；分块则把数组切成小 tile 依次处理。

<a id="fig-ch22-fig03"></a>

<div align="center">
  <img src="./images/ch22-fig03.png" alt="Loop tiling in a two-dimensional array" width="78%" />
  <div><b>Figure 22.3</b></div>
</div>

Note that in languages such as Python, we don't usually perform loop
tiling, because Python and many other high-level languages do not allow
control over cache memory like lower-level languages such as C and C++
do. These kinds of optimizations are often handled by underlying
libraries like NumPy and PyTorch when performing operations on large
arrays.

注意：在 Python 等语言里，应用层通常不会手写循环分块，因为对缓存的控制不如 C/C++ 那样直接；这类优化往往由 NumPy、PyTorch 等底层实现在大张量运算中完成。

> Tips: 在 Python 等高级语言中，通常不进行循环分块，因为这些语言不提供对缓存内存的控制，如 C 和 C++ 等底层语言。这些优化通常由底层库（如 NumPy 和 PyTorch）在处理大型数组时自动处理。

## Operator Fusion
> 本节说明算子融合如何减少循环与访存开销，并与循环分块、重参数化等概念区分与联系。
[](#operator-fusion)

*Operator fusion*, sometimes called *loop fusion*, is an optimization
technique that combines multiple loops into a single loop. This is
illustrated in Figure [22.4](#fig-ch22-fig04),
the product of an array of numbers are fused into a single loop.

*算子融合（operator fusion）*有时也称*循环融合*：把多段循环合成一段，图 [22.4](#fig-ch22-fig04) 示意将数组求积的多重循环并为单一循环。

> Tips: **算子融合**，也被称为`循环融合`，将多个循环合并成一个循环。

<a id="fig-ch22-fig04"></a>

<div align="center">
  <img src="./images/ch22-fig04.png" alt="Fusing two loops (left) into one (right)" width="78%" />
  <div><b>Figure 22.4</b></div>
</div>

`Operator fusion` can improve the performance of a model by reducing the
overhead of loop control, decreasing memory access times by improving
cache performance, and possibly enabling further optimizations through
vectorization. You might think this behavior of `vectorization` would be
incompatible with `loop tiling`, in which we break a
`for` loop into multiple loops.

算子融合可减少循环控制开销、改善缓存行为从而降访存成本，并可能为进一步向量化创造条件。乍看之下，向量化与把 `for` 拆成多段的循环分块似乎相冲突。

> Tips: **算子融合**，可以提高模型性能，通过减少循环控制的开销，提高缓存性能，并可能通过向量化进一步优化。


However, these techniques are actually complementary, used for different
optimizations, and applicable in different situations. `Operator fusion`
is about reducing the total number of loop iterations and improving data
locality when the entire data fits into cache. `Loop tiling` is about
improving cache utilization when dealing with larger multidimensional
arrays that do not fit into cache.

但二者实为互补：融合侧重减少总迭代次数、在数据能落入缓存时改善局部性；分块则面对塞不进缓存的大张量，侧重提高缓存利用率。



Related to operator fusion is the concept of *reparameterization*, which
can often also be used to simplify multiple operations into one. Popular
examples include training a network with multibranch architectures that
are reparameterized into single-stream architectures during inference.
This reparameterization approach differs from traditional operator
fusion in that it does not merge multiple operations into a single
operation. Instead, it rearranges the operations in the network to
create a more efficient architecture for inference. In the so-called
RepVGG architecture, for example, each branch during training consists
of a series of convolutions. Once training is complete, the model is
reparameterized into a single sequence of convolutions.

与融合相关的是*重参数化（reparameterization）*：可把多步运算在推理时等价改写得更紧凑。典型例子是多分支结构训练、推理时并成单路；这与把多个算子合成“一个算子”的传统融合不同，而是**重排**计算图以得到更利于推理的结构。RepVGG 中，训练期每分支是一串卷积，训练结束后可重参为单序列卷积。

> Tips: **重参数化**，也被称为`重参数化优化`，将多个操作合并成一个操作。

## Quantization
> 本节说明量化如何降算力与存储需求、后训练量化与量化感知训练的区别，并交代为何严格“不伤精度”时量化与本章主题略有张力，以及蒸馏、剪枝等为何超出本章范围。
[](#quantization)

*Quantization* reduces the computational and storage requirements of
machine learning models, particularly deep neural networks. This
technique involves converting the floating-point numbers (technically
discrete but representing continuous values within a specific range) for
implementing weights and biases in a trained neural network to more
discrete, lower-precision representations such as integers. Using
less precision reduces the model size and makes it quicker to execute,
which can lead to significant improvements in speed and hardware
efficiency during inference.

*量化*降低模型（尤其深度网络）的计算与存储成本：把表示权重与偏置的浮点（技术上离散但近似连续区间）映射到更低比特、更离散的表示（如整数）。精度位宽下降可缩小模型、加快执行，从而提升推理速度与硬件效率。

> Tips: **量化**，也被称为`量化优化`，将浮点数转换为整数，减少模型大小和计算量，提高推理速度。

In the realm of deep learning, it has become increasingly common to
quantize trained models down to 8-bit and 4-bit integers. These
techniques are especially prevalent in the deployment of large language
models.

在深度学习部署中，把已训练模型压到 8 bit、4 bit 整数已很常见，大语言模型场景尤为突出。

There are two main categories of quantization. In **post-training quantization**, 
the model is first trained normally with full-precision
weights, which are then quantized after training. 
**Quantization-aware training**, on the other hand, introduces the quantization step during
the training process. This allows the model to learn to compensate for
the effects of quantization, which can help maintain the model's
accuracy.

量化主要有两类：**后训练量化（post-training quantization）**先照常全精度训练，训完再量化；**量化感知训练（quantization-aware training）**则在训练过程中模拟量化，让模型学会补偿量化误差，有助于保住精度。

> Tips: 量化，一般分为 2 大类：`后训练量化`、`量化感知训练`。
> 
> - 后训练量化，在训练完成后，对模型进行量化。
> - 量化感知训练，在训练过程中，引入量化步骤，让模型学习量化带来的影响。

However, it's important to note that quantization can occasionally
lead to a reduction in model accuracy. Since this chapter focuses on
techniques to speed up model inference *without* sacrificing accuracy,
quantization is not as good a fit for this chapter as the previous
categories.

需注意量化有时会带来精度损失；本章强调**不伤准确率**的加速，因而量化不如前几类那样“严丝合缝”地贴合本章设问。

> Tips: 量化，可能会导致模型精度下降，因此，本章再不讨论量化。


Other techniques to improve inference speeds include knowledge
distillation and pruning, discussed in
Chapter [\[ch06\]](./ch06/_books_ml-q-and-ai-ch06.md).
However, these techniques affect the model architecture, resulting in
smaller models, so they are out of scope for this chapter's question.

提升推理速度还可借助知识蒸馏、剪枝等（见第 6 章），但它们会改变模型结构、得到更小的网络，故不在本章“不改架构”的限定之内。

> Tips: 其他提升推理速度的策略，包括：知识蒸馏、剪枝等，之前章节已经讨论过;但是，这些策略会影响模型架构，导致模型变小，因此，也不在本章讨论范围内。

## Exercises
> 本节习题引导思考多 GPU 推理的实际瓶颈，以及向量化与循环分块各自最理想的适用情景。
[](#exercises)

22-1. Chapter [\[ch07\]](./ch07/_books_ml-q-and-ai-ch07.md) covered several multi-GPU training paradigms to
speed up modeltraining.UsingmultipleGPUscan,intheory,alsospeedupmodel
inference. However, in reality, this approach is often not the most
efficient or most practical option. Why is that?

22-1. 第 [ch07](./ch07/_books_ml-q-and-ai-ch07.md) 章介绍过多 GPU 训练范式以加速训练；理论上多 GPU 也能加速推理，但实际中却常不是最高效或最务实的选择，原因是什么？

22-2. Vectorization and loop tiling are two strategies for optimizing
operations that involve accessing array elements. What would be the
ideal situation in which to use each?

22-2. 向量化和循环分块都用于优化涉及数组访问的运算；各自最理想的使用情境分别是什么？

## References
> 本节列出 BLAS、循环分块经典论文、RepVGG 以及 LLM 8 bit / 4 bit 量化代表作等链接。
[](#references)

- The official BLAS website: <https://www.netlib.org/blas/>.

- BLAS 官方网站：<https://www.netlib.org/blas/>。

- The paper that proposed loop tiling: Michael Wolfe, "More Iteration
  Space Tiling"? (1989),
  <https://dl.acm.org/doi/abs/10.1145/76263.76337>.

- 提出循环分块的论文：Michael Wolfe，《More Iteration Space Tiling?》(1989)，<https://dl.acm.org/doi/abs/10.1145/76263.76337>。

- RepVGG CNN architecture merging operations in inference mode: Xiaohan
  Ding et al., "RepVGG: Making VGG-style ConvNets Great Again"?
  (2021), <https://arxiv.org/abs/2101.03697>.

- 推理阶段合并运算的 RepVGG：Ding 等，《RepVGG: Making VGG-style ConvNets Great Again?》(2021)，<https://arxiv.org/abs/2101.03697>。

- A new method for quantizing the weights in large language mod-  els
  downto8-bitintegerrepresentations:TimDettmersetal., "LLM.int8():
  8-bit Matrix Multiplication for Transformers at Scale"? (2022),
  <https://arxiv.org/abs/2208.07339>.

- 大语言模型权重量化到 8 bit 整数：Dettmers 等，《LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale?》(2022)，<https://arxiv.org/abs/2208.07339>。

- A new method for quantizing the weights in LLMs farther down to 4-bit
  integers: Elias Frantar et al., "GPTQ: Accurate Post-Training
  Quantization for Generative Pre-trained Transformers"? (2022),
  <https://arxiv.org/abs/2210.17323>.

- 将 LLM 权重进一步压到 4 bit 整数：Frantar 等，《GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers?》(2022)，<https://arxiv.org/abs/2210.17323>。


------------------------------------------------------------------------

