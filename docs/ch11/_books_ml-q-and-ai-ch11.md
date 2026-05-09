



# Chapter 11: Calculating the Number of Parameters
> 如何从卷积与全连接层逐层清点参数并求和；参数规模对复杂度、数据量与内存的意义。
[](#chapter-11-calculating-the-number-of-parameters)


**How do we compute the number of parameters in a convolutional
neural network, and why is this information useful?**

如何计算卷积神经网络中的参数个数，这些信息有何用处？

Knowing the number of parameters in a model helps gauge the model's
size, which affects storage and memory requirements. The following
sections will explain how to compute the convolutional and fully
connected layer parameter counts.

知道参数个数有助于衡量模型体量，从而影响存储与显存需求。以下几节说明如何清点卷积层与全连接层的参数。

> Tips: 模型参数的数量，是衡量模型大小（存储空间大笑）的重要指标，用于估算所需的存储空间。重点关注 `卷积层`和`全连接层`。

## How to Find Parameter Counts
> 通过一个具体 CNN 架构从左到右逐层累加可训练权重与偏置。
[](#how-to-find-parameter-counts)

Suppose we are working with a convolutional network that has two `convolutional layers` 
with kernel size 5 and kernel size 3, respectively.

设想一个 CNN，含有两个 `卷积层`，核大小分别为 5 与 3。

* The first convolutional layer has 3 input channels and 5 output
channels, 
* and the second one has 5 input channels and 12 output
channels. 
* The stride of these `convolutional layers` is 1. 

第一层有 3 个输入通道、5 个输出通道；第二层有 5 个输入通道、12 个输出通道；两层步长均为 1。

Furthermore, the network has two `pooling layers`, 

此外网络有两个 `池化层`，

* one with a kernel size of 3 and a stride of 2, 
* and another with a kernel size of 5 and a stride of 2. 

其一池化核为 3、步长为 2；其二池化核为 5、步长为 2。

It also has two `fully connected hidden layers` with 192 and 128 hidden units
each, where the output layer is a` classification layer` for 10 classes.

另有两个 `全连接隐层`，单元数分别为 192 与 128，输出层为 10 类的 `分类层`。


The architecture of this network is illustrated in
Figure [11.1](#fig-ch11-fig01).

网络结构见图 [11.1](#fig-ch11-fig01)。

<a id="fig-ch11-fig01"></a>

<div align="center">
  <img src="./images/ch11-fig01.png" alt="image" width="78%" />
  <div><b>Figure 11.1</b></div>
</div>


What is the number of trainable parameters in this convolutional
network? We can approach this problem from left to right, computing the
number of parameters for each layer and then summing up these counts to
obtain the total number of parameters. Each layer's number of
trainable parameters consists of weights and bias units.

该网络有多少可训练参数？可自左向右逐层计算再求和；每层的可训练参数包括权重与偏置。

### Convolutional Layers
> 卷积核尺寸、输入/输出通道数与偏置如何决定卷积层参数量。
[](#convolutional-layers)

In a `convolutional layer`, the number of weights depends on the
kernel's width and height and the number of input and output channels.
The number of bias units depends on the number of output channels only.
To illustrate the computation step by step, suppose we have a kernel
width and height of 5, one input channel, and one output channel, as
illustrated in Figure [11.2](#fig-ch11-fig02).

`卷积层`的权重数取决于核高宽与输入、输出通道数；偏置个数仅由输出通道数决定。为逐步说明，设核为 5×5、单输入单输出通道，见图 [11.2](#fig-ch11-fig02)。

> Tips: 卷积层的参数数量，取决于`卷积核`(kernel)的宽度、高度、输入通道数和输出通道数。

<a id="fig-ch11-fig02"></a>

<div align="center">
  <img src="./images/ch11-fig02.png" alt="A convolutional layer with one input channel and one output channel" width="78%" />
  <div><b>Figure 11.2</b></div>
</div>

In this case, we have 26 parameters, since we have $5 \times 5 = 25$ weights via the kernel plus the bias unit. 
The computation to determine an output value or pixel $z$ 
is $z = b + \sum_j w_j x_j$, where $x_j$ represents an input pixel, $w_j$ represents
a weight parameter of the kernel, and $b$ is the bias unit.

此情形共 26 个参数：核上 $5 \times 5 = 25$ 个权重加 1 个偏置。输出 $z = b + \sum_j w_j x_j$，其中 $x_j$ 为输入像素，$w_j$ 为核权重，$b$ 为偏置。

> Tips: 一个**卷积核** `kernel` 的参数量，`weights` = 宽度 x 高度。

Now, suppose we have three input channels, as illustrated in
Figure [11.3](#fig-ch11-fig03).

现设三输入通道，见图 [11.3](#fig-ch11-fig03)。

<a id="fig-ch11-fig03"></a>

<div align="center">
  <img src="./images/ch11-fig03.png" alt="A convolutional layer with three input channels and one output channel" width="78%" />
  <div><b>Figure 11.3</b></div>
</div>

In that case, we compute the output value by performing the
aforementioned operation, $\sum_j w_j x_j$, for each input
channel and then add the bias unit. For three input channels, this would
involve three different kernels with three sets of weights:

此时对每个输入通道各自做 $\sum_j w_j x_j$ 再相加并加偏置；三通道即三套权重：

$$z = \sum_j w^{(1)}_{j} x_j + \sum_j w^{(2)}_{j} x_j + \sum_j w^{(3)}_{j} x_j + b$$

Since we have three sets of weights
($w^{(1)}, w^{(2)}, and w^{(3)}$ for $j = [1, 25]$), we have 
$3 \times 25 + 1 = 76$ parameters in this convolutional layer.

三组权重 ($w^{(1)}, w^{(2)}, w^{(3)}$，各 $j=1..25$) 故有 $3 \times 25 + 1 = 76$ 个参数。

> Tips: 每个**输入通道**，对应的**卷积核** `kernel`，都是**独立的参数**.

We use one kernel for each output channel, where each kernel is unique
to a given output channel. Thus, if we extend the number of output
channels from one to five, as shown in
Figure [11.4](#fig-ch11-fig04), we extend the number of parameters by a factor of 5. 

每一输出通道用各自一套核。若输出通道由 1 扩到 5（图 [11.4](#fig-ch11-fig04)），参数量约为原来的 5 倍。

In other words, if the kernel for one output channel has 76
parameters, the 5 kernels required for the five output channels will
have $5 \times 76 = 380$ parameters.

若单输出通道需 76 个参数，五通道共 $5 \times 76 = 380$ 个参数。

<a id="fig-ch11-fig04"></a>

<div align="center">
  <img src="./images/ch11-fig04.png" alt="A convolutional layer with three input channels and five output channels" width="78%" />
  <div><b>Figure 11.4</b></div>
</div>

Returning to the neural network architecture illustrated in
Figure [11.1](#fig-ch11-fig01) at the beginning of this section, we compute the
number of parameters in the convolutional layers based on the kernel
size and number of input and output channels. For example, the first
convolutional layer has three input channels, five output channels, and
a kernel size of 5. Thus, its number of parameters is $5 \times (5 \times 5 \times 3) + 5 = 380$. 

回到本节开头的图 [11.1](#fig-ch11-fig01)：第一层 3 入 5 出、核 5，参数为 $5 \times (5 \times 5 \times 3) + 5 = 380$。

The second convolutional
layer, with five input channels, 12 output channels, and a kernel size
of 3, has $12 \times (3 \times 3 \times 5) + 12 = 552$ parameters. 

第二层 5 入 12 出、核 3，参数 $12 \times (3 \times 3 \times 5) + 12 = 552$。

Since the `pooling layers` do not have any trainable parameters, 
we can count $380 + 552 = 932$ for the convolutional part of this architecture.

`池化层`无可训练参数，卷积部分共 $380 + 552 = 932$。

Next, let's see how we can compute the number of parameters of fully
connected layers.

接着看全连接层参数如何计算。

### Fully Connected Layers
> 全连接层：权重为输入维度乘输出维度，再加输出侧偏置。
[](#fully-connected-layers)

Counting the number of parameters in a `fully connected layer` is
relatively straightforward. A fully connected node connects each input
node to each output node, so the number of weights is the number of
inputs times the number of outputs plus the bias units added to the
output. For example, if we have a fully connected layer with five inputs
and three outputs, as shown
in Figure [11.5](#fig-ch11-fig05), we have $5 \times 3 = 15$ weights and three bias units, that is,
18 parameters total.

`全连接层`参数很直接：每个输出与所有输入相连，权重数为输入数×输出数，再加每个输出的偏置。例如 5 入 3 出（图 [11.5](#fig-ch11-fig05)）：$5 \times 3 = 15$ 个权重、3 个偏置，共 18。

> Tips: 一个**全连接层** `fully connected layer` 的参数量，`weights` = 输入节点数 x 输出节点数。

<a id="fig-ch11-fig05"></a>

<div align="center">
  <img src="./images/ch11-fig05.png" alt="A fully connected layer with five inputs and three outputs" width="78%" />
  <div><b>Figure 11.5</b></div>
</div>

Returning once more to the neural network architecture illustrated in
Figure [11.1](#fig-ch11-fig01), we can now calculate the parameters in the fully
connected layers as follows: $192 \times 128 + 128 = 24,704$ in the
first fully connected layer and $128 \times 10 + 10 = 1,290$ in the
second fully connected layer, the output layer. 

再回到图 [11.1](#fig-ch11-fig01)：第一层全连接 $192 \times 128 + 128 = 24,704$；第二层（输出）$128 \times 10 + 10 = 1,290$。

Hence, we have $24,704 + 1,290 = 25,994$ in the fully connected part of this network. 

故全连接部分共 $24,704 + 1,290 = 25,994$。

After adding the 932 parameters from the convolutional layers and the 25,994
parameters from the fully connected layers, we can conclude that this
network's total number of parameters is $26,926$.

卷积部分 932 与全连接 25,994 相加，总参数量为 $26,926$。

As a bonus, interested readers can find PyTorch code to compute the
number of parameters programmatically in the
*supplementary/q11-conv-size* subfolder at
<https://github.com/rasbt/MachineLearning-QandAI-book>.

补充：读者可在仓库 *supplementary/q11-conv-size* 找到用 PyTorch 编程统计参数的示例：<https://github.com/rasbt/MachineLearning-QandAI-book>。

## Practical Applications
> 参数规模与模型复杂度、训练数据需求及硬件可承载量的关系。
[](#practical-applications)

Why do we care about the number of parameters at all? First, we can use
this number to estimate a model's complexity. As a rule of thumb, the
more parameters there are, the more training data we'll need to train
the model well.

为何要关心参数数量？其一可用它估算模型复杂度：粗略地说，参数越多，通常需要更多训练数据才能学好。

> Tips: 模型`参数的数量`，是衡量模型`复杂度`的重要指标，用于估算所需的`训练数据量`。

The number of parameters also lets us estimate the size of the neural
network, which in turn helps us estimate whether the network can fit
into GPU memory. Although the memory requirement during training often
exceeds the model size due to the additional memory required for
carrying out matrix multiplications and storing gradients, model size
gives us a ballpark sense of whether training the model on a given
hardware setup is feasible.

其二可估计网络体量，从而判断能否装入 GPU 显存。训练时实际占用常大于纯模型体积（前向乘积、梯度存储等），但模型大小仍能给出在既定硬件上是否大致可训的粗判。

> Tips: 模型参数的数量，是衡量模型`大小`的重要指标，用于估算模型是否能`fit`到`GPU`的`内存`中。


## Exercises
> 习题：SGD 与 Adam 需存哪些额外状态；三处 BatchNorm 增加多少参数。
[](#exercises)

11-1. Suppose we want to optimize the neural network using a plain
stochastic gradient descent (SGD) optimizer or the popular Adam
optimizer. What are the respective numbers of parameters that need to be
stored for SGD and Adam?

11-1. 若用普通 SGD 或 Adam 优化该网络，分别需要为优化器额外存储多少组与参数同形状的变量？

11-2. Suppose we're adding three batch normalization (BatchNorm)
layers: one after the first convolutional layer, one after the second
convolutional layer, and another one after the first fully connected
layer (we typically do not want to add BatchNorm layers to the output
layer). How many additional parameters do these three BatchNorm layers
add to the model?

11-2. 若在第一个卷积后、第二个卷积后以及第一个全连接后各加一个 BatchNorm（通常不在输出层加），这三个 BatchNorm 共为模型增加多少参数？


------------------------------------------------------------------------

