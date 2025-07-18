







# Chapter 12: Fully Connected and Convolutional Layers
[](#chapter-12-fully-connected-and-convolutional-layers)



**Under which circumstances can we replace fully connected layers with
convolutional layers to perform the same computation?**

Replacing fully connected layers with convolutional layers can offer
advantages in terms of hardware optimization, such as by utilizing
specialized hardware accelerators for convolution operations. This can
be particularly relevant for edge devices.

> Tips: `卷积层` 替代 `全连接层`，有下面收益
> - 卷积操作可以`硬件加速`
> - 这在`边缘设备`上非常关键。

There are exactly two scenarios in which fully connected layers and
convolutional layers are equivalent: when the size of the convolutional
filter is equal to the size of the receptive field and when the size of
the convolutional filter is 1. As an illustration of these two
scenarios, consider a fully connected layer with two input and four
output units, as shown in
Figure [1.1](#fig-ch12-fig01).

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
</div>

The fully connected layer in this figure consists of eight weights and
two bias units. We can compute the output nodes via the following dot
products:

Node 1

$$w_{1, 1} \times x_1 + w_{1, 2} \times x_2 + w_{1, 3} \times x_3 + w_{1, 4} \times x_4 + b_1$$

Node 2

$$w_{2, 1} \times x_1 + w_{2, 2} \times x_2 + w_{2, 3} \times x_3 + w_{2, 4} \times x_4 + b_2$$

The following two sections illustrate scenarios in which convolutional
layers can be defined to produce exactly the same computation as the
fully connected layer described.

## When the Kernel and Input Sizes Are Equal
[](#when-the-kernel-and-input-sizes-are-equal)

Let's start with the first scenario, where the size of the
convolutional filter is equal to the size of the receptive field. Recall
from Chapter [\[ch11\]](./ch11/_books_ml-q-and-ai-ch11.md)
how we compute a number of parameters in a convolutional kernel with one
input channel and multiple output channels. We have a kernel size of
$2×2$, one input channel, and two output channels. The input
size is also $2×2$, a reshaped version of the four inputs
depicted in Figure [1.2](#fig-ch12-fig02).

<a id="fig-ch12-fig02"></a>

<div align="center">
  <img src="./images/ch12-fig02.png" alt="A convolutional layer with a 2x2 kernel that equals the input size and two output channels" width="78%" />
</div>


If the convolutional kernel dimensions equal the input size, as depicted
in Figure [1.2](#fig-ch12-fig02), there is no sliding window mechanism in the
convolutional layer. For the first output channel, we have the following
set of weights:

$$W_1 = \begin{bmatrix} w_{1, 1} & w_{1, 2} \\ w_{1, 3} & w_{1, 4} \end{bmatrix}$$

For the second output channel, we have the following set of weights:

$$W_2 = \begin{bmatrix} w_{2, 1} & w_{2, 2} \\ w_{2, 3} & w_{2, 4} \end{bmatrix}$$

If the inputs are organized as

$$x = \begin{bmatrix} x_1 & x_2 \\ x_3 & x_4 \end{bmatrix}$$

we calculate the first output channel as $o_1 = \sum_i (W_1 \times x_i) + b_1$, where the convolutional operator \* is equal to
an element-wise multiplication. In other words, we perform an
element-wise multiplication between two matrices, $W_1$ and **x**, and
then compute the output as the sum over these elements; this equals the
dot product in the fully connected layer. Lastly, we add the bias unit.
The computation for the second output channel works analogously: $o_2 = \sum_i (W_2 \times x_i) + b_2$.

As a bonus, the supplementary materials for this book include PyTorch
code to show this equivalence with a hands-on example in the
`supplementary/q12-fc-cnn-equivalence` subfolder at
<https://github.com/rasbt/MachineLearning-QandAI-book>.

## When the Kernel Size Is 1
[](#when-the-kernel-size-is-1)

The second scenario assumes that we reshape the input into an input
"image"? with $1×1$ dimensions where the number of "color
channels"? equals the number of input features, as depicted in
Figure [1.3](#fig-ch12-fig03).

<a id="fig-ch12-fig03"></a>

<div align="center">
  <img src="./images/ch12-fig03.png" alt="The number of output nodes equals the number of channels if the kernel size is equal to the input size." width="78%" />
</div>

Each kernel consists of a stack of weights equal to the number of input
channels. For instance, for the first output layer, the weights are

$$W_1 = \begin{bmatrix} w^{(1)}_1 & w^{(2)}_1 & w^{(3)}_1 & w^{(4)}_1 \end{bmatrix}$$

while the weights for the second channel are:

$$W_2 = \begin{bmatrix} w^{(1)}_2 & w^{(2)}_2 & w^{(3)}_2 & w^{(4)}_2 \end{bmatrix}$$

To get a better intuitive understanding of this computation, check out
the illustrations in Chapter [\[ch11\]](./ch11/_books_ml-q-and-ai-ch11.md), which describe how to compute the parameters in a
convolutional layer.

## Recommendations
[](#recommendations)

The fact that fully connected layers can be implemented as equivalent
convolutional layers does not have immediate performance or other
advantages on standard computers. However, replacing fully connected
layers with convolutional layers can offer advantages in combination
with developing specialized hardware accelerators for convolution
operations.

Moreover, understanding the scenarios where fully connected layers are
equivalent to convolutional layers aids in understanding the mechanics
of these layers. It also lets us implement convolutional neural networks
without any use of fully connected layers, if desired, to simplify code
implementations.

> Tips: 进一步，理解`卷积层`和`全连接层`的等价性，有助于理解这些层的机制。
> 
> 此外，如果需要，我们可以实现卷积神经网络，而不使用任何全连接层，以简化代码实现。

## Exercises
[](#exercises)

12-1. How would increasing the stride affect the equivalence discussed
in this chapter?

12-2. Does padding affect the equivalence between fully connected layers
and convolutional layers?


------------------------------------------------------------------------

