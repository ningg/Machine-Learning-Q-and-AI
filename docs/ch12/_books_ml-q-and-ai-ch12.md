



# Chapter 12: Fully Connected and Convolutional Layers
> 在什么条件下卷积可实现与全连接相同的计算：核尺寸等于感受野或核尺寸为 1。
[](#chapter-12-fully-connected-and-convolutional-layers)



**Under which circumstances can we replace fully connected layers with
convolutional layers to perform the same computation?**

在什么情况下可以用卷积层替代全连接层而保持同一计算？

Replacing fully connected layers with convolutional layers can offer
advantages in terms of hardware optimization, such as by utilizing
specialized hardware accelerators for convolution operations. This can
be particularly relevant for edge devices.

用卷积替代全连接可借助卷积专用硬件加速器做优化，对边缘设备尤有意义。

> Tips: `卷积层` 替代 `全连接层`，有下面收益
> - 卷积操作可以`硬件加速`
> - 这在`边缘设备`上非常关键。

There are exactly two scenarios in which fully connected layers and
convolutional layers are equivalent: when the size of the convolutional
filter is equal to the size of the receptive field and when the size of
the convolutional filter is 1. As an illustration of these two
scenarios, consider a fully connected layer with two input and four
output units, as shown in
Figure [12.1](#fig-ch12-fig01).

全连接与卷积完全等价恰有两种情形：卷积核尺寸等于感受野尺寸，或核尺寸为 1。下文用两层入、四层出的全连接为例（图 [12.1](#fig-ch12-fig01)）。

> Tips: `全连接层`和`卷积层`在两种情况下是`等价`的： 
>   - 当`卷积核`的大小等于`感受野`的大小。
>   - 当`卷积核`的大小为`1` 。
> 
> `receptive field` 感受野，在 CNN 和 RNN 中，有不同的含义。
> - 在卷积神经网络 (CNN) 中，感受野是指网络中某个特定层的神经元在输入图像上映射的区域大小（像素范围）。它是`空间`维度上的概念。
> - 在 循环神经网络 (RNN) 中，感受野是`时间`维度上的概念。它衡量的是当前状态在时间轴上向后能追溯到多远的输入信息。


<a id="fig-ch12-fig01"></a>

<div align="center">
  <img src="./images/ch12-fig01.png" alt="Four inputs and two outputs connected via eight weight parameters" width="78%" />
  <div><b>Figure 12.1</b></div>
</div>

The fully connected layer in this figure consists of eight weights and
two bias units. We can compute the output nodes via the following dot
products:

图中该全连接层有 8 个权重与 2 个偏置，输出由下述点积得到：

Node 1

$$w_{1, 1} \times x_1 + w_{1, 2} \times x_2 + w_{1, 3} \times x_3 + w_{1, 4} \times x_4 + b_1$$

Node 2

$$w_{2, 1} \times x_1 + w_{2, 2} \times x_2 + w_{2, 3} \times x_3 + w_{2, 4} \times x_4 + b_2$$

The following two sections illustrate scenarios in which convolutional
layers can be defined to produce exactly the same computation as the
fully connected layer described.

下面两节给出可定义卷积层使其与上述全连接计算完全一致的两种情形。

## When the Kernel and Input Sizes Are Equal
> 核与输入同尺寸时不滑动窗口，逐通道逐元乘再求和即点积。
[](#when-the-kernel-and-input-sizes-are-equal)

Let's start with the first scenario, where the size of the
convolutional filter is equal to the size of the receptive field. Recall
from Chapter [\[ch11\]](./ch11/_books_ml-q-and-ai-ch11.md)
how we compute a number of parameters in a convolutional kernel with one
input channel and multiple output channels. We have a kernel size of
$2×2$, one input channel, and two output channels. The input
size is also $2×2$, a reshaped version of the four inputs
depicted in Figure [12.2](#fig-ch12-fig02).

第一种情形：核尺寸等于感受野尺寸。回想第 [\[ch11\]](./ch11/_books_ml-q-and-ai-ch11.md) 章中单输入通道、多输出时卷积核参数的计算。这里核为 $2×2$，单输入通道、双输出通道，输入亦为 $2×2$，可由图 [12.1](#fig-ch12-fig01) 的四个输入reshape 而成，见图 [12.2](#fig-ch12-fig02)。

<a id="fig-ch12-fig02"></a>

<div align="center">
  <img src="./images/ch12-fig02.png" alt="A convolutional layer with a 2x2 kernel that equals the input size and two output channels" width="78%" />
  <div><b>Figure 12.2</b></div>
</div>


If the convolutional kernel dimensions equal the input size, as depicted
in Figure [12.2](#fig-ch12-fig02), there is no sliding window mechanism in the
convolutional layer. For the first output channel, we have the following
set of weights:

若如图 [12.2](#fig-ch12-fig02) 所示核尺寸与输入相同，则无滑动窗口。第一输出通道权重为：

$$W_1 = \begin{bmatrix} w_{1, 1} & w_{1, 2} \\ w_{1, 3} & w_{1, 4} \end{bmatrix}$$

For the second output channel, we have the following set of weights:

第二输出通道：

$$W_2 = \begin{bmatrix} w_{2, 1} & w_{2, 2} \\ w_{2, 3} & w_{2, 4} \end{bmatrix}$$

If the inputs are organized as

输入排成矩阵

$$x = \begin{bmatrix} x_1 & x_2 \\ x_3 & x_4 \end{bmatrix}$$

we calculate the first output channel as $o_1 = \sum_i (W_1 \times x_i) + b_1$, where the convolutional operator \* is equal to
an element-wise multiplication. In other words, we perform an
element-wise multiplication between two matrices, $W_1$ and **x**, and
then compute the output as the sum over these elements; this equals the
dot product in the fully connected layer. Lastly, we add the bias unit.
The computation for the second output channel works analogously: $o_2 = \sum_i (W_2 \times x_i) + b_2$.

则第一输出通道 $o_1 = \sum_i (W_1 \times x_i) + b_1$，此处卷积算子 \* 等于逐元相乘后对全体元素求和，即与全连接的向量点积一致，再加偏置；第二输出通道同理：$o_2 = \sum_i (W_2 \times x_i) + b_2$。

As a bonus, the supplementary materials for this book include PyTorch
code to show this equivalence with a hands-on example in the
`supplementary/q12-fc-cnn-equivalence` subfolder at
<https://github.com/rasbt/MachineLearning-QandAI-book.

图书配套仓库 `supplementary/q12-fc-cnn-equivalence` 中含 PyTorch 示例演示该等价性：<https://github.com/rasbt/MachineLearning-QandAI-book>。

## When the Kernel Size Is 1
> 把特征维当作 1×1 空间上的“多通道”，用 1×1 卷积实现全连接式混合。
[](#when-the-kernel-size-is-1)

The second scenario assumes that we reshape the input into an input
"image"? with $1×1$ dimensions where the number of "color
channels"? equals the number of input features, as depicted in
Figure [12.3](#fig-ch12-fig03).

第二种情形：将输入 reshape 为 $1×1$ 的“图像”，通道数等于特征数（图 [12.3](#fig-ch12-fig03)）。

<a id="fig-ch12-fig03"></a>

<div align="center">
  <img src="./images/ch12-fig03.png" alt="The number of output nodes equals the number of channels if the kernel size is equal to the input size." width="78%" />
  <div><b>Figure 12.3</b></div>
</div>

Each kernel consists of a stack of weights equal to the number of input
channels. For instance, for the first output layer, the weights are

每个核在通道方向上堆叠的权重个数等于输入通道数。例如第一输出：

$$W_1 = \begin{bmatrix} w^{(1)}_1 & w^{(2)}_1 & w^{(3)}_1 & w^{(4)}_1 \end{bmatrix}$$

while the weights for the second channel are:

第二输出：

$$W_2 = \begin{bmatrix} w^{(1)}_2 & w^{(2)}_2 & w^{(3)}_2 & w^{(4)}_2 \end{bmatrix}$$

To get a better intuitive understanding of this computation, check out
the illustrations in Chapter [\[ch11\]](./ch11/_books_ml-q-and-ai-ch11.md), which describe how to compute the parameters in a
convolutional layer.

更直观理解可参考第 [\[ch11\]](./ch11/_books_ml-q-and-ai-ch11.md) 章中对卷积层参数计算的图示说明。

## Recommendations
> 等价性本身在通用 CPU 上不必然提速；但与专用卷积加速器结合时有意义。
[](#recommendations)

The fact that fully connected layers can be implemented as equivalent
convolutional layers does not have immediate performance or other
advantages on standard computers. However, replacing fully connected
layers with convolutional layers can offer advantages in combination
with developing specialized hardware accelerators for convolution
operations.

在普通计算机上，把全连接改写成等价卷积并不自动带来性能等好处；但若配合卷积专用硬件加速器，则可能获益。

Moreover, understanding the scenarios where fully connected layers are
equivalent to convolutional layers aids in understanding the mechanics
of these layers. It also lets us implement convolutional neural networks
without any use of fully connected layers, if desired, to simplify code
implementations.

此外，弄清等价情形有助于理解两种层的机理；若愿意，亦可实现完全不含全连接的 CNN 以简化代码。

> Tips: 进一步，理解`卷积层`和`全连接层`的等价性，有助于理解这些层的机制。
> 
> 此外，如果需要，我们可以实现卷积神经网络，而不使用任何全连接层，以简化代码实现。

## Exercises
> 习题：步长增大如何破坏等价；padding 是否影响等价性。
[](#exercises)

12-1. How would increasing the stride affect the equivalence discussed
in this chapter?

12-1. 增大步长会如何影响本章讨论的等价关系？

12-2. Does padding affect the equivalence between fully connected layers
and convolutional layers?

12-2. padding 是否影响全连接与卷积层之间的等价性？


------------------------------------------------------------------------

