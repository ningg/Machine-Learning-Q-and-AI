







# Chapter 22: Speeding Up Inference
[](#chapter-22-speeding-up-inference)



**What are techniques to speed up model inference through optimization
without changing the model architecture or sacrificing accuracy?**

In machine learning and AI, *model inference* refers to making
predictions or generating outputs using a trained model. The main
general techniques for improving model performance during inference
include parallelization, vectorization, loop tiling, operator fusion,
and quantization, which are discussed in detail in the following
sections.

> Tips:
> 
> - 优化模型**推理速度**，有多种方法，包括：`并行化`、`向量化`、`循环分块`、`算子融合`、`量化`等。
> - 这些方法，将在后续章节中详细讨论。

## Parallelization
[](#parallelization)

One common way to achieve better parallelization during inference is to
run the model on a batch of samples rather than on a single sample at a
time. This is sometimes also referred to as *batched inference* and
assumes that we are receiving multiple input samples or user inputs
simultaneously or within a short time window, as illustrated in
Figure [22.1](#fig-ch22-fig01).

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

## Vectorization
[](#vectorization)

*Vectorization* refers to performing operations on entire data
structures, such as arrays (tensors) or matrices, in a single step
rather than using iterative constructs like `for` loops. Using vectorization, multiple operations from
the loop are performed simultaneously using single instruction, multiple
data (SIMD) processing, which is available on most modern CPUs.

> Tips: **向量化**，也被称为`单指令多数据`，在现代 CPU 上，可以同时处理多个数据。

This approach takes advantage of the low-level optimizations in many
computing systems and often results in significant speedups. For
example, it might rely on BLAS.

*BLAS* (which is short for *Basic Linear Algebra Subprograms*) is a
specification that prescribes a set of low-level routines for performing
common linear algebra operations such as vector addition, scalar
multiplication, dot products, matrix multiplication, and others. Many
array and deep learning libraries like NumPy and PyTorch use BLAS under
the hood.

To illustrate vectorization with an example, suppose we wanted to
compute the dot product between two vectors. The non-vectorized way of
doing this would be to use a `for` loop, iterating over each element of the array one
by one. However, this can be quite slow, especially for large arrays.
With vectorization, you can perform the dot product operation on the
entire array at once, as shown in
Figure [22.2](#fig-ch22-fig02).

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

## Loop Tiling
[](#loop-tiling)

*Loop tiling* (also often referred to as *loop nest optimization*) is an
advanced optimization technique to enhance data locality by breaking
down a loop's iteration space into smaller chunks or "tiles."? This
ensures that once data is loaded into cache, all possible computations
are performed on it before the cache is cleared.

> Tips: **循环分块**，也被称为`循环嵌套优化`，将循环的迭代空间分成小块，确保数据加载到缓存后，所有可能的计算都在缓存中完成，然后缓存被清除。

Figure [22.3](#fig-ch22-fig03) illustrates the concept of loop tiling for
accessing elements in a two-dimensional array. In a regular
`for` loop, we iterate over columns and rows one element at a time, whereas in loop tiling, we
subdivide the array into smaller tiles.

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

> Tips: 在 Python 等高级语言中，通常不进行循环分块，因为这些语言不提供对缓存内存的控制，如 C 和 C++ 等底层语言。这些优化通常由底层库（如 NumPy 和 PyTorch）在处理大型数组时自动处理。

## Operator Fusion
[](#operator-fusion)

*Operator fusion*, sometimes called *loop fusion*, is an optimization
technique that combines multiple loops into a single loop. This is
illustrated in Figure [22.4](#fig-ch22-fig04),
the product of an array of numbers are fused into a single loop.

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

> Tips: **算子融合**，可以提高模型性能，通过减少循环控制的开销，提高缓存性能，并可能通过向量化进一步优化。


However, these techniques are actually complementary, used for different
optimizations, and applicable in different situations. `Operator fusion`
is about reducing the total number of loop iterations and improving data
locality when the entire data fits into cache. `Loop tiling` is about
improving cache utilization when dealing with larger multidimensional
arrays that do not fit into cache.



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

> Tips: **重参数化**，也被称为`重参数化优化`，将多个操作合并成一个操作。

## Quantization
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

> Tips: **量化**，也被称为`量化优化`，将浮点数转换为整数，减少模型大小和计算量，提高推理速度。

In the realm of deep learning, it has become increasingly common to
quantize trained models down to 8-bit and 4-bit integers. These
techniques are especially prevalent in the deployment of large language
models.

There are two main categories of quantization. In **post-training quantization**, 
the model is first trained normally with full-precision
weights, which are then quantized after training. 
**Quantization-aware training**, on the other hand, introduces the quantization step during
the training process. This allows the model to learn to compensate for
the effects of quantization, which can help maintain the model's
accuracy.

> Tips: 量化，一般分为 2 大类：`后训练量化`、`量化感知训练`。
> 
> - 后训练量化，在训练完成后，对模型进行量化。
> - 量化感知训练，在训练过程中，引入量化步骤，让模型学习量化带来的影响。

However, it's important to note that quantization can occasionally
lead to a reduction in model accuracy. Since this chapter focuses on
techniques to speed up model inference *without* sacrificing accuracy,
quantization is not as good a fit for this chapter as the previous
categories.

> Tips: 量化，可能会导致模型精度下降，因此，本章再不讨论量化。


Other techniques to improve inference speeds include knowledge
distillation and pruning, discussed in
Chapter [\[ch06\]](./ch06/_books_ml-q-and-ai-ch06.md).
However, these techniques affect the model architecture, resulting in
smaller models, so they are out of scope for this chapter's question.

> Tips: 其他提升推理速度的策略，包括：知识蒸馏、剪枝等，之前章节已经讨论过;但是，这些策略会影响模型架构，导致模型变小，因此，也不在本章讨论范围内。

## Exercises
[](#exercises)

22-1. Chapter [\[ch07\]](./ch07/_books_ml-q-and-ai-ch07.md) covered several multi-GPU training paradigms to
speed up modeltraining.UsingmultipleGPUscan,intheory,alsospeedupmodel
inference. However, in reality, this approach is often not the most
efficient or most practical option. Why is that?

22-2. Vectorization and loop tiling are two strategies for optimizing
operations that involve accessing array elements. What would be the
ideal situation in which to use each?

## References
[](#references)

- The official BLAS website: <https://www.netlib.org/blas/>.

- The paper that proposed loop tiling: Michael Wolfe, "More Iteration
  Space Tiling"? (1989),
  <https://dl.acm.org/doi/abs/10.1145/76263.76337>.

- RepVGG CNN architecture merging operations in inference mode: Xiaohan
  Ding et al., "RepVGG: Making VGG-style ConvNets Great Again"?
  (2021), <https://arxiv.org/abs/2101.03697>.

- A new method for quantizing the weights in large language mod-  els
  downto8-bitintegerrepresentations:TimDettmersetal., "LLM.int8():
  8-bit Matrix Multiplication for Transformers at Scale"? (2022),
  <https://arxiv.org/abs/2208.07339>.

- A new method for quantizing the weights in LLMs farther down to 4-bit
  integers: Elias Frantar et al., "GPTQ: Accurate Post-Training
  Quantization for Generative Pre-trained Transformers"? (2022),
  <https://arxiv.org/abs/2210.17323>.


------------------------------------------------------------------------

