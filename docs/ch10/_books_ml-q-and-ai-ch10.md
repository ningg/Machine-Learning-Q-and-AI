



# Chapter 10: Sources of Randomness
> 本章归类并讨论深度网络训练与推理中导致不可复现结果的各种随机性来源。
[](#chapter-10-sources-of-randomness)



**What are the common sources of randomness when training deep neural
networks that can cause non-reproducible behavior during training and
inference?**

训练深度神经网络时，哪些常见的随机性来源会导致训练与推理阶段的不可复现行为？

When training or using machine learning models such as deep neural
networks, several sources of randomness can lead to different results
every time we train or run these models, even though we use the same
overall settings. Some of these effects are accidental and some are
intended. The following sections categorize and discuss these various
sources of randomness.

训练或使用深度神经网络等模型时，即使整体设置相同，多种随机性仍可能使每次训练或运行的结果不同；其中有些是意外，有些是刻意的。以下几节将这些随机性分类并逐一讨论。

> Tips: 
> 
> - 在训练和使用机器学习模型时，如深度神经网络，**随机性**会导致每次`训练`或`运行`模型时得到不同的结果，即使我们使用相同的配置。
> - 这些随机性可能是`偶然`的，也可能是`故意`的。


Optional hands-on examples for most of these categories are provided in
the *supplementary/q10-random-sources* subfolder at
<https://github.com/rasbt/MachineLearning-QandAI-book>.

多数类别的动手示例见图书配套仓库中的 *supplementary/q10-random-sources* 子文件夹：<https://github.com/rasbt/MachineLearning-QandAI-book>。

## Model Weight Initialization
> 随机权重初始化、非凸损失下的不同局部极小，以及用随机种子固定初始化。
[](#model-weight-initialization)

All common deep neural network frameworks, including TensorFlow and
PyTorch, randomly initialize the weights and bias units at each layer by
default. This means that the final model will be different every time we
start the training. The reason these trained models will differ when we
start with different random weights is the nonconvex nature of the loss,
as illustrated in
Figure [10.1](#fig-ch10-fig01). As the figure shows, the loss will converge to different local minima depending on where the initial starting weights are located.

TensorFlow、PyTorch 等框架默认对各层权重与偏置随机初始化，因此每次开训会得到不同模型。从不同随机权重出发会得到不同训练结果的原因在于损失的非凸性（见图 [10.1](#fig-ch10-fig01)）：初值位置不同，损失会收敛到不同局部极小。

> Tips: 初始化权重，会导致不同的局部最优解。

<a id="fig-ch10-fig01"></a>

<div align="center">
  <img src="./images/ch10-fig01.png" alt="Different starting weights can lead to different final weights." width="78%" />
  <div><b>Figure 10.1</b></div>
</div>

In practice, it is therefore recommended to run the training (if the
computational resources permit) at least a handful of times; unlucky
initial weights can sometimes cause the model not to converge or to
converge to a local minimum corresponding to poorer predictive accuracy.

实践中若算力允许，建议至少重复训练数次；糟糕的初值有时会导致不收敛或收敛到预测更差的局部极小。

> Tips: 实践中，建议`至少`运行`训练几次`，以避免不幸运的初始权重，导致模型不收敛或收敛到较差的局部最优解。

However, we can make the random weight initialization deterministic by
seeding the random generator. For instance, if we set the seed to a
specific value like 123, the weights will still initialize with small
random values. Nonetheless, the neural network will consistently
initialize with the **same small random weights**, enabling accurate
reproduction of results.

但通过给随机数生成器设种子可使初始化确定：例如固定种子为 123，权重仍为小幅随机值，但每次都会得到**同一组小幅随机初值**，从而可复现实验。

> Tips: 通过设置`随机种子`，可以使得`初始化`权重参数是`确定的`。

## Dataset Sampling and Shuffling
> 数据集划分、k 折等与随机抽样相关的变异性。
[](#dataset-sampling-and-shuffling)

When we train and evaluate machine learning models, we usually start by
dividing a dataset into training and test sets. This requires random
sampling since we have to decide which examples we put into a training
set and which examples we put into a test set.

训练与评估模型时通常先把数据划分为训练集与测试集，这需要随机抽样来决定哪些样本进训练、哪些进测试。

In practice, we often use model evaluation techniques such as *k*-fold
cross-validation or holdout validation. In holdout validation, we split
the training set into `training`, `validation`, and `test` datasets, which are
also sampling procedures influenced by randomness. Similarly, unless we
use a fixed random seed, we get a different model each time we partition
the dataset or tune or evaluate the model using *k*-fold
cross-validation since the training partitions will differ.

实践中还常用 *k* 折交叉验证或留出法；留出法下再把数据分为`训练`、`验证`、`测试`，同样受随机划分影响。若不固定种子，每次划分或用 *k* 折调参/评估时，训练子集不同，模型也会不同。

> Tips: 
> 
> - 在训练和评估机器学习模型时，我们通常将数据集分为`训练集`、`验证集`和`测试集`。
> - 这需要`随机采样`，因为我们必须决定将哪些样本放入训练集，哪些放入验证集，哪些放入测试集。
> - 除非我们使用固定的随机种子，否则每次划分数据集或使用`k`折交叉验证时，我们都会得到不同的模型。


## Nondeterministic Algorithms
> 网络结构中的随机成分（如 dropout）及如何配合种子与推理模式获得确定性。
[](#nondeterministic-algorithms)

We may include random components and algorithms depending on the
architecture and hyperparameter choices. A popular example of this is
**dropout**.

按架构与超参选择，模型中可能包含随机成分，**dropout** 是常见例子。

`Dropout` works by randomly setting a fraction of a layer's units to
zero during training, which helps the model learn more robust and
generalized representations. This "**dropping out**" is typically
applied at each training iteration with a probability *p*, a
hyperparameter that controls the fraction of units dropped out. Typical
values for *p* are in the range of 0.2 to 0.8.

`Dropout` 在训练时以概率 *p* 随机将一层中一部分单元置零，促使表示更稳健、泛化更好；*p* 控制丢弃比例，典型取值约在 0.2–0.8。

To illustrate this concept,
Figure [10.2](#fig-ch10-fig02) shows a small neural network where dropout
randomly drops a subset of the hidden layer nodes in each forward pass
during training.

图 [10.2](#fig-ch10-fig02) 示意小网络在训练每次前向时 dropout 如何随机屏蔽部分隐层节点。

<a id="fig-ch10-fig02"></a>

<div align="center">
  <img src="./images/ch10-fig02.png" alt="In dropout, hidden nodes are intermittently and randomly disabled during each forward pass in training." width="78%" />
  <div><b>Figure 10.2</b></div>
</div>

To create reproducible training runs, we must seed the random generator 
before training with dropout (analogous to seeding the random
generator before initializing the model weights). During inference, we
need to disable dropout to guarantee deterministic results. Each deep
learning framework has a specific setting for that purpose -- a PyTorch
example is included in the *supplementary/q10-random-sources* subfolder
at <https://github.com/rasbt/MachineLearning-QandAI-book>.

要复现含 dropout 的训练，须在训练前设种子（与权重初始化前设种子类似）。推理时应关闭 dropout 以保证确定性。各框架有对应设置；PyTorch 示例见 *supplementary/q10-random-sources*：<https://github.com/rasbt/MachineLearning-QandAI-book>。

> Tips: 
> 
> - 为了创建可复现的训练运行，我们必须在训练前设置随机种子（类似于在初始化模型权重之前设置随机种子）。
> - 在推理时，我们需要禁用`dropout`，以保证结果的确定性。
> - 每个深度学习框架都有特定的设置，以实现这一点。

## Different Runtime Algorithms
> 同名运算的不同 GPU 实现（如卷积）、数值微差累积与确定性算法选项。
[](#different-runtime-algorithms)

The most intuitive or simplest implementation of an algorithm or method
is not always the best one to use in practice. For example, when
training deep neural networks, we often use efficient alternatives and
approximations to gain speed and resource advantages during training and
inference.

理论上最直观的实现未必是实践最优；深度网络训练常用更快或更省资源的近似实现。

A popular example is the convolution operation used in convolutional
neural networks. There are several possible ways to implement the
convolution operation:

卷积是典型例子，实现途径有多种：

The `classic direct convolution` The common implementation of discrete
convolution via an element-wise product between the input and the
window, followed by summing the result to get a single number. (See
Chapter [\[ch12\]](./ch12/_books_ml-q-and-ai-ch12.md) for
a discussion of the convolution operation.)

`经典直接卷积`：输入与窗口逐元相乘再累加成标量。（卷积运算讨论见第 [\[ch12\]](./ch12/_books_ml-q-and-ai-ch12.md) 章。）

`FFT-based` convolution Uses `fast Fourier transform` (FFT) to convert the
convolution into an element-wise multiplication in the frequency domain.

`基于 FFT 的卷积`：用快速傅里叶变换在频域做逐元乘法实现卷积。

`Winograd-based` convolution An efficient algorithm for small filter sizes
(like $3\times3$ that
reduces the number of multiplications required for the convolution.

`基于 Winograd 的卷积`：适用于小核（如 $3\times3$），减少乘法次数。

Different convolution algorithms have different trade-offs in terms of
memory usage, computational complexity, and speed. By default, libraries
such as the **CUDA Deep Neural Network library** (`cuDNN`), which are used in
PyTorch and TensorFlow, can choose different algorithms for performing
convolution operations when running deep neural networks on GPUs.
However, the deterministic algorithm choice has to be explicitly
enabled. In PyTorch, for example, this can be done by setting


不同算法在内存、复杂度与速度上权衡各异。PyTorch/TensorFlow 调用的 **cuDNN** 等在 GPU 上默认可能切换卷积算法；要使用确定性算法需显式开启，例如 PyTorch 中设置

``` 
torch.use_deterministic_algorithms(True)
```

While these approximations yield similar results, subtle numerical
differences can accumulate during training and cause the training to
converge to slightly different local minima.

近似结果看似相近，但细微数值差异在训练中会累积，使收敛到的局部极小略有不同。

> Tips:  算法自身也会带来随机性，特别是不同优化算法实现，本身得到的就是近似效果。
> 
> - **不同的卷积算法**在`内存`使用、`计算复杂度`和`速度`方面有不同的权衡。
> - 默认情况下，PyTorch和TensorFlow等库中的`CUDA Deep Neural Network library`（`cuDNN`）可以选择不同的算法来执行卷积操作。
> - 但是，确定性算法的选择必须显式启用。
> - 在PyTorch中，可以通过设置`torch.use_deterministic_algorithms(True)`来启用确定性算法。

## Hardware and Drivers
> 精度、硬件与库优化导致的数值差异。
[](#hardware-and-drivers)

Training deep neural networks on different hardware can also produce
different results due to small numeric differences, even when the same
algorithms are used and the same operations are executed. These
differences may sometimes be due to different numeric precision for
floating-point operations. However, small numeric differences may also
arise due to hardware and software optimization, even at the same
precision.

在不同硬件上训练即使用相同算法与运算，也可能因微小数值差异而结果不同：可能源于浮点精度不同，也可能在相同精度下仍因硬件与软件优化产生偏差。

> Tips: 硬件和驱动也会带来随机性，特别是不同硬件平台，不同优化库，不同优化算法实现，本身得到的就是近似效果。
> 
> - 不同的`数值精度`，会导致不同的结果。
> - 不同的`硬件`和`软件`优化，会导致不同的结果。

For instance, different hardware platforms may have specialized
optimizations or libraries that can slightly alter the behavior of deep
learning algorithms. To give one example of how different GPUs can
produce different modeling results, the following is a quotation from
the official NVIDIA documentation: "Across different architectures, no
cuDNN routines guarantee bit-wise reproducibility. For example, there is
no guarantee of bit-wise reproducibility when comparing the same routine
run on NVIDIA  $Volta^{TM}$ and NVIDIA  $Turing^{TM}$ \[. . .\] and NVIDIA Ampere
architecture."?

例如不同平台有专门优化或库，会轻微改变数值行为。NVIDIA 官方文档说明：跨架构时 cuDNN 例程不保证按位可复现；例如在 Volta、Turing 与 Ampere 等架构上运行同一例程，无法保证按位一致。

> Tips: 
> 
> - 不同的硬件平台可能具有专门的优化或库，可以稍微改变深度学习算法的性能。
> - 例如，不同的GPU可以产生不同的建模结果。


## Randomness and Generative AI
> 生成式模型在推理中“有意为之”的随机性；扩散噪声与 LLM 采样策略。
[](#randomness-and-generative-ai)

Besides the various sources of randomness mentioned earlier, certain
models may also exhibit random behavior during inference that we can
think of as "**randomness by design**."? For instance, generative image
and language models may create different results for identical prompts
to produce a diverse sample of results. For image models, this is often
so that users can select the most accurate and aesthetically pleasing
image. For language models, this is often to vary the responses, for
example, in chat agents, to avoid repetition.

除前述来源外，部分模型在推理阶段会表现出可称为“**设计上的随机性**”的行为：例如生成式图像/语言模型对同一提示给出不同输出以丰富样本；图像侧便于用户挑选；语言侧（如聊天）可减少重复。

> Tips: 
> 
> - 除了前面提到的各种随机性来源，某些模型在**推理时**也可能表现出**随机行为**，我们可以将其视为“**设计上的随机性**”。
> - 例如，生成式图像和语言模型可能会对相同的提示产生不同的结果，以产生多样化的结果样本。
> - 对于图像模型，这通常是为了让用户选择最准确和最吸引人的图像。
> - 对于语言模型，这通常是为了避免重复，例如在聊天代理中。


The intended randomness in generative image models during inference is
often due to sampling different noise values at each step of the reverse
process. In diffusion models, a noise schedule defines the noise
variance added at each step of the diffusion process.

生成式图像模型推理中的有意随机常来自反向过程各步对噪声的不同采样；扩散模型用噪声调度规定每步所加噪声方差。

> Tips: 
> 
> - 在生成式图像模型中，推理时的随机性，通常是由于在反向过程中对不同的`噪声值`进行采样。
> - 在扩散模型中，`噪声调度`定义了在扩散过程中添加的`噪声方差`。


Autoregressive LLMs like GPT tend to create different outputs for the
same input prompt (GPT will be discussed at greater length in
Chapters [\[ch14\]](./ch14/_books_ml-q-and-ai-ch14.md)
and [\[ch17\]](./ch17/_books_ml-q-and-ai-ch17.md)). The
ChatGPT user interface even has a Regenerate Response button for that
purpose. The ability to generate different results is due to the
sampling strategies these models employ. Techniques such as top-*k*
sampling, nucleus sampling, and temperature scaling influence the
model's output by controlling the degree of randomness. This is a
feature, not a bug, since it allows for diverse responses and prevents
the model from producing overly deterministic or repetitive outputs.
(See Chapter [\[ch09\]](./ch09/_books_ml-q-and-ai-ch09.md)
for a more in-depth overview of generative AI and deep learning models;
see Chapter [\[ch17\]](./ch17/_books_ml-q-and-ai-ch17.md)
for more detail on autoregressive LLMs.)

自回归 LLM（如 GPT）对同一提示常产生不同输出（GPT 详见第 [\[ch14\]](./ch14/_books_ml-q-and-ai-ch14.md)、[\[ch17\]](./ch17/_books_ml-q-and-ai-ch17.md) 章）；ChatGPT 界面亦有“重新生成”按钮。差异来自 top-*k*、核采样、温度等采样策略对随机程度的控制——这是特性而非缺陷，可带来多样性并避免刻板重复。（生成式模型总览见第 [\[ch09\]](./ch09/_books_ml-q-and-ai-ch09.md) 章，自回归 LLM 细节见第 [\[ch17\]](./ch17/_books_ml-q-and-ai-ch17.md) 章。）

> Tips: 
> 
> - 自回归语言模型（如GPT）倾向于对相同的输入提示，产生不同的输出。
> - 这是因为这些模型，采用了不同的`采样策略`。
> - 例如，`top-*k*采样`、`核采样`和`温度缩放`等技术，通过控制随机性程度，影响模型的输出。


*Top-[k]{.upright} sampling*, illustrated in
Figure [10.3](#fig-ch10-fig03), works by sampling tokens from the top *k* most
probable candidates at each step of the next-word generation process.

*Top-[k]{.upright} 采样*（图 [10.3](#fig-ch10-fig03)）在每一步从概率最高的 *k* 个词元中采样下一词。

<a id="fig-ch10-fig03"></a>

<div align="center">
  <img src="./images/ch10-fig03.png" alt="Top-k sampling" width="78%" />
  <div><b>Figure 10.3</b></div>
</div>

Given an input prompt, the language model produces a probability
distribution over the entire vocabulary (the candidate words) for the
next token. Each token in the vocabulary is assigned a probability based
on the model's understanding of the context. The selected top-*k*
tokens are then renormalized so that the probabilities sum to 1.
Finally, a token is sampled from the renormalized top-*k* probability
distribution and is appended to the input prompt. This process is
repeated for the desired length of the generated text or until a stop
condition is met.

给定提示，模型为下一词元给出全词表上的概率分布；按上下文为每个候选词赋概率。取 top-*k* 后重归一化使和为 1，再从中采样一词并拼回提示；重复直至达到目标长度或停止条件。

*Nucleus sampling* (also known as *top-p sampling*),
illustrated in Figure [10.4](#fig-ch10-fig04), is an alternative to top-*k* sampling.

*核采样*（*top-p 采样*，图 [10.4](#fig-ch10-fig04)）是 top-*k* 的替代方案。

<a id="fig-ch10-fig04"></a>

<div align="center">
  <img src="./images/ch10-fig04.png" alt="Nucleus sampling" width="78%" />
  <div><b>Figure 10.4</b></div>
</div>

Similar to top-*k* sampling, the goal of nucleus sampling is to balance
diversity and coherence in the output. However, nucleus and top-*k*
sampling differ in how to select the candidate tokens for sampling at
each step of the generation process. Top-*k* sampling selects the *k*
most probable tokens from the probability distribution produced by the
language model, regardless of their probabilities. The value of *k*
remains fixed throughout the generation process. Nucleus sampling, on
the other hand, selects tokens based on a probability threshold *p*, as
shown in Figure [10.4](#fig-ch10-fig04). It then accumulates the most probable tokens in
descending order until their cumulative probability meets or exceeds the
threshold *p*. In contrast to top-*k* sampling, the size of the
candidate set (nucleus) can vary at each step.

与 top-*k* 类似，核采样也在多样性与连贯性间折中，但候选集构造不同：top-*k* 固定取概率最高的 *k* 个，*k* 全程不变；核采样按阈值 *p*（图 [10.4](#fig-ch10-fig04)）从高概率往下累加直至累计概率 ≥ *p*，故每步候选集大小可变。

> Tips: 
> 
> - 与top-*k*采样类似，核采样的目标是平衡输出中的多样性和连贯性。
> - 然而，核采样和top-*k*采样在选择每个生成步骤中的候选标记时有所不同。
> - **top-*k*采样**从语言模型产生的概率分布中选择`概率最高`的*k*个标记，而**核采样**则根据`累计的概率阈值`*p*选择标记。

## Exercises
> 习题：top-*k*/核采样的确定性；推理时何时希望 dropout 仍随机。
[](#exercises)

10-1. Suppose we train a neural network with top-*k* or nucleus sampling
where *k* and *p* are hyperparameter choices. Can we make the model
behave deterministically during inference without changing the code?

10-1. 若在训练中使用 top-*k* 或核采样且 *k*、*p* 为超参，不改代码能否在推理时使模型行为确定？

10-2. In what scenarios might random dropout behavior during inference
be desired?

10-2. 何种场景下会希望在推理阶段仍保留 dropout 的随机行为？

## References
> 延伸阅读：模型评估、dropout 原论文、FFT/Winograd 卷积与 PyTorch/cuDNN 确定性设置。
[](#references)

- For more about different data sampling and model evaluation
  techniques, see my article: "Model Evaluation, Model Selection, and
  Algorithm Selection in Machine Learning"? (2018),
  <https://arxiv.org/abs/1811.12808>.

- 关于数据抽样与模型评估等：见文章 "Model Evaluation, Model Selection, and Algorithm Selection in Machine Learning"（2018），<https://arxiv.org/abs/1811.12808>。

- The paper that originally proposed the dropout technique: Nitish
  Srivastavaetal.,"Dropout:ASimpleWaytoPreventNeuralNet-  works from
  Overfitting"? (2014),
  [*https://jmlr.org/papers/v15/sriva*](https://jmlr.org/papers/v15/srivastava14a.html)
  [*stava14a.html*](https://jmlr.org/papers/v15/srivastava14a.html).

- 提出 dropout 的论文：Nitish Srivastava 等，"Dropout: A Simple Way to Prevent Neural Networks from Overfitting"（2014），[*https://jmlr.org/papers/v15/sriva*](https://jmlr.org/papers/v15/srivastava14a.html) [*stava14a.html*](https://jmlr.org/papers/v15/srivastava14a.html)。

- A detailed paper on FFT-based convolution: Lu Chi, Borui Jiang, and
  Yadong Mu, "Fast Fourier Convolution"? (2020),
  <https://dl.acm.org/doi/abs/10.5555/3495724.3496100>.

- FFT 卷积详解：Lu Chi、Borui Jiang、Yadong Mu，《Fast Fourier Convolution》（2020），<https://dl.acm.org/doi/abs/10.5555/3495724.3496100>。

- Details on Winograd-based convolution: Syed Asad Alam et al.,
  "Winograd Convolution for Deep Neural Networks: Efficient Point
  Selection"? (2022), <https://arxiv.org/abs/2201.10369>.

- Winograd 卷积：Syed Asad Alam 等，《Winograd Convolution for Deep Neural Networks: Efficient Point Selection》（2022），<https://arxiv.org/abs/2201.10369>。

- More information about the deterministic algorithm settings in
  PyTorch:
  <https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html>.

- PyTorch 确定性算法：<https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html>。

- For details on the deterministic behavior of NVIDIA graphics cards,
  see the "Reproducibility"? section of the official NVIDIA
  documentation:
  <https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html#reproducibility>.

- NVIDIA 显卡/cuDNN 可复现性参见官方文档 “Reproducibility”：<https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html#reproducibility>。


------------------------------------------------------------------------

