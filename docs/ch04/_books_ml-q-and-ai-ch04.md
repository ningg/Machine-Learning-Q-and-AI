

# Chapter 4: The Lottery Ticket Hypothesis
> 本章解释彩票假设（随机初始化大网中存在可单独训练达到同精度的小子网）、迭代幅度剪枝流程，并讨论其在训练/推理上的潜力与现实限制及练习。
[](#chapter-4-the-lottery-ticket-hypothesis)



**What is the lottery ticket hypothesis, and, if it holds true, how is
it useful in practice?**

什么是彩票假设？若成立，在实践中能带来什么用处？

The lottery ticket hypothesis is a concept in neural network training
that posits that within a randomly initialized neural network, there
exists a `subnetwork` (or `winning ticket`?) that can, when trained
separately, achieve the same accuracy on a test set as the full network
after being trained for the same number of steps. This idea was first
proposed by Jonathan Frankle and Michael Carbin in 2018.

彩票假设认为：在随机初始化的大网络里存在一个 `subnetwork`（「中奖彩票」），单独训练同样步数即可达到与整网相当的测试精度；该思想由 Frankle 与 Carbin 于 2018 年提出。

> Tips: 彩票假设 `lottery ticket hypothesis`，是神经网络训练中，一个重要的概念。它指出，在随机初始化的神经网络中，存在一个`子网络`（或`彩票`），当单独训练时，可以达到与完整网络相同的准确率。

This chapter illustrates the lottery hypothesis step by step, then goes
over *weight pruning*, one of the key techniques to create smaller
subnetworks as part of the lottery hypothesis methodology. Lastly, it
discusses the practical implications and limitations of the hypothesis.

本章逐步演示该假设，再介绍作为核心手段的 *weight pruning*（权重剪枝），最后讨论其现实意义与局限。

> Tips: 本章将展示`彩票假设`的训练过程，然后介绍`权重剪枝`，这是`彩票假设`方法论中，创建较小子网络的关键技术。最后，讨论`彩票假设`的实际应用和局限性。

## The Lottery Ticket Training Procedure
> 本节按 Figure 4.1 的四步说明：训练大网至收敛、结构化/非结构化剪枝、迭代幅度剪枝、重置为原始初始化再训剪枝子网等，直至得到 winning ticket。
[](#the-lottery-ticket-training-procedure)

Figure [4.1](#fig-ch04-fig01) illustrates the training procedure for the lottery
ticket hypothesis in four steps, which we'll discuss one by one to
help clarify the concept.

Figure [4.1](#fig-ch04-fig01) 用四步说明彩票假设训练流程，下文逐步拆解。

<a id="fig-ch04-fig01"></a>

<div align="center">
  <img src="./images/ch04-fig01.png" alt="The lottery hypothesis training procedure" width="78%" />
  <div><b>Figure 4.1</b></div>
</div>

In Figure [4.1](#fig-ch04-fig01), we start with a large `neural network` that we
train until convergence , meaning we put in our best efforts to make it
perform as well as possible on a target dataset (for example, minimizing
training loss and maximizing classification accuracy). This large neural
network is initialized as usual using small random weights.

第 1 步：用常规小随机初始化训练大 `neural network` 至收敛（在目标任务上尽量降低损失、提高精度）。

Next, as shown in
Figure [4.1](#fig-ch04-fig01), we `prune` the neural network's `weight` parameters
, removing them from the network. We can do this by setting the weights
to zero to create sparse weight matrices. Here, we can either prune
individual weights, known as *unstructured* pruning, or prune larger
"chunks"? from the network, such as entire convolutional filter
channels. This is known as *structured* pruning.

第 2 步：对权重 `prune`，可置零得到稀疏矩阵；可逐权重 *unstructured* 剪枝，也可整通道等 *structured* 剪枝。

> Tips: 剪枝时，有两种方式，一种是`unstructured pruning`，一种是`structured pruning`。他们的差异是，`unstructured pruning`是逐个剪枝，而`structured pruning`是剪枝整个`卷积核`。

The original lottery hypothesis approach follows a concept known as
*iterative magnitude pruning*, where the weights with the lowest
magnitudes are removed in an iterative fashion. (We will revisit this
concept in Chapter [\[ch06\]](./ch06/_books_ml-q-and-ai-ch06.md) when discussing techniques to reduce overfitting.)

原始方法采用 *iterative magnitude pruning*：反复剪掉幅值最小的权重（第 6 章还会从过拟合角度再谈）。

> Tips: 迭代剪枝 `iterative magnitude pruning`。

After the pruning step, we reset the weights to the original small
random values used in step 1 in
Figure [4.1](#fig-ch04-fig01) and train the pruned network . It's worth
emphasizing that we do not reinitialize the pruned network with any
small random weights (as is typical for iterative magnitude pruning),
and instead we reuse the weights from step 1.

剪枝后把**保留下来的权重**重置为第 1 步训练开始时的同一组小随机初值（而非对剪枝后网络全新随机初始化），再训练剪枝子网。

> Tips: 剪枝后，我们重置权重为原始小随机值，并训练剪枝后的网络。??? 没理解 FIXME

We then repeat the pruning steps 2 through 4 until we reach the desired
network size. For example, in the original lottery ticket hypothesis
paper, the authors successfully reduced the network to 10 percent of its
original size without sacrificing classification accuracy. As a nice
bonus, the pruned (sparse) network, referred to as the *winning ticket*,
even demonstrated improved generalization performance compared to the
original (large and dense) network.

随后重复 2–4 步直至达到目标规模；原文可将网络缩至约 10% 参数量而不损精度，且稀疏 *winning ticket* 有时泛化更好。

## Practical Implications and Limitations
> 本节讨论：若总能找到小网等价大网，对训练成本与推理部署的意义；以及目前仍需先训大网、初始化敏感等现实障碍。
[](#practical-implications-and-limitations)

If it's possible to identify smaller subnetworks that have the same
predictive performance as their up-to-10-times-larger counterparts, this
can have significant implications for both neural training and
inference. Given the ever-growing size of modern neural network
architectures, this can help cut training costs and infrastructure.

若能稳定找到与十倍规模大网同等性能的小子网，将显著影响训练与推理成本，对日益膨胀的模型架构尤为重要。

> Tips: 如果可以识别出与完整网络具有相同预测性能的较小子网络，这对于神经网络的`训练`和`推理`都有显著的影响，可以显著`降低训练成本`和`基础设施成本`。

Sound too good to be true? Maybe. If winning tickets can be identified
efficiently, this would be very useful in practice. However, at the time
of writing, there is no way to find the winning tickets without training
the original network. Including the pruning steps would make this even
more expensive than a regular training procedure. Moreover, after the
publication of the original paper, researchers found that the original
weight initialization may not work to find winning tickets for
larger-scale networks, and additional experimentation with the initial
weights of the pruned networks is required.

但现实是：若不先训练完整大网就难以找到中奖彩票，加上迭代剪枝往往比普通训练更贵；后续研究还发现原始初始化对大模型未必有效，需对剪枝子网初值做更多探索。

The good news is that winning tickets do exist. Even if it's currently
not possible to identify them without training their larger neural
network counterparts, they can be used for more efficient inference
after training.

好消息是 winning ticket 确实存在；即便发现过程仍依赖大网训练，训练完成后仍可用于更高效推理。

## Exercises
> 本节练习：子网效果不佳时的下一步；以及 ReLU 的 max(0,x) 形式与彩票假设训练的关系。
[](#exercises)

4-1. Suppose we're trying out the lottery ticket hypothesis approach
and find that the performance of the subnetwork is not very good
(compared to the original network). What next steps might we try?

4-1. 若按彩票假设得到的子网性能明显弱于原网，可尝试哪些下一步？

4-2. The simplicity and efficiency of the rectified linear unit (ReLU)
activation function have made it one of the most popular activation
functions in neural network training, particularly in deep learning,
where it helps to mitigate problems like the vanishing gradient. The
ReLU activation function is defined by the mathematical expression
max(0, *x*). This means that if the input *x* is positive, the function
returns *x*, but if the input is negative or 0, the function returns 0.
How is the lottery ticket hypothesis related to training a neural
network with ReLU activation functions?

4-2. ReLU 定义为 max(0,*x*)，广泛用于缓解梯度消失。它与用 ReLU 训练的网络及彩票假设之间有何联系？

## References
> 本节列出彩票假设原文、结构化剪枝、线性模式连通性后续工作及随机权重子网络等相关文献链接。
[](#references)

- The paper proposing the lottery ticket hypothesis: Jonathan Frankle
  and Michael Carbin, "The Lottery Ticket Hypothesis: Finding Sparse,
  Trainable Neural Networks"? (2018),
  <https://arxiv.org/abs/1803.03635>.

- 彩票假设原文 (2018)：<https://arxiv.org/abs/1803.03635>。

- The paper proposing structured pruning for removing larger parts, such
  as entire convolutional filters, from a network: Hao Li et al.,
  "Pruning Filters for Efficient ConvNets"? (2016),
  <https://arxiv.org/abs/1608.08710>.

- 结构化剪枝（整卷积核）(2016)：<https://arxiv.org/abs/1608.08710>。

- Follow-up work on the lottery hypothesis, showing that the original
  weight initialization may not work to find winning tickets for
  larger-scale networks, and additional experimentation with the initial
  weights of the pruned networks is required: Jonathan Frankle et al.,
  "Linear Mode Connectivity and the Lottery Ticket Hypothesis"?
  (2019), <https://arxiv.org/abs/1912.05671>.

- 线性模式连通性与大网初值问题 (2019)：<https://arxiv.org/abs/1912.05671>。

- An improved lottery ticket hypothesis algorithm that finds smaller
  networks that match the performance of a larger network exactly: Vivek
  Ramanujan et al., "What's Hidden in a Randomly Weighted Neural
  Network?"? (2020), <https://arxiv.org/abs/1911.13299>.

- 随机初始化网络中隐藏小网 (2020)：<https://arxiv.org/abs/1911.13299>。


------------------------------------------------------------------------

