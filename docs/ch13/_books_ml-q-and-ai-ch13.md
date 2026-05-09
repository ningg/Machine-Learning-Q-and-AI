







# Chapter 13: Large Training Sets for Vision Transformers
> ViT 相比 CNN 归纳偏置更少，常与大规模预训练结合；本章归纳 CNN 的空间/局部先验并解释 ViT 的数据需求与表现。
[](#chapter-13-large-training-sets-for-vision-transformers)



**Why do vision transformers (ViTs) generally require larger training
sets than convolutional neural networks (CNNs)?**

为什么视觉 Transformer（ViT）通常比卷积神经网络（CNN）需要更大的训练集？

Each machine learning algorithm and model encodes a particular set of
assumptions or prior knowledge, commonly referred to as *inductive
biases*, in its design. Some inductive biases are workarounds to make
algorithms computationally more feasible, other inductive biases are
based on domain knowledge, and some inductive biases are both.

每种机器学习算法与模型都会在设计中编码一组假设或先验知识，常称为归纳偏置（inductive biases）。其中一些是为了让算法在计算上可行，一些基于领域知识，还有一些两者兼有。

CNNs and ViTs can be used for the same tasks, including image
classification, object detection, and image segmentation. CNNs are
mainly composed of convolutional layers, while ViTs consist primarily of
multi-head attention blocks (discussed in
Chapter [\[ch08\]](./ch08/_books_ml-q-and-ai-ch08.md) in
the context of transformers for natural language inputs).

CNN 与 ViT 可用于相同任务，包括图像分类、目标检测与图像分割。CNN 主要由卷积层构成，ViT 则主要由多头注意力块组成（在自然语言输入上的 Transformer 背景见第 [\[ch08\]](./ch08/_books_ml-q-and-ai-ch08.md) 章）。

CNNs have more inductive biases that are hardcoded as part of the
algorithmic design, so they generally require less training data than
ViTs. In a sense, ViTs are given more degrees of freedom and can or must
learn certain inductive biases from the data (assuming that these biases
are conducive to optimizing the training objective). However, everything
that needs to be learned requires more training examples.

CNN 在算法设计中硬编码了更多归纳偏置，因此通常比 ViT 需要更少训练数据。某种意义上 ViT 自由度更大，能从数据中学习某些归纳偏置（只要这些偏置有利于优化训练目标）；但凡是需要从数据里学的，都需要更多样例。

The following sections explain the main inductive biases encountered in
CNNs and how ViTs work well without them.

以下几节说明 CNN 中的主要归纳偏置，以及 ViT 在没有这些偏置时如何仍能良好工作。

## Inductive Biases in CNNs
> CNN 通过局部连接、权值共享、层次处理等先验获得平移等变/近似不变性；这些是本章与 ViT 对比的核心。
[](#inductive-biases-in-cnns)

The following are the primary inductive biases that largely define how
CNNs function:

下列归纳偏置在很大程度上定义了 CNN 的工作方式：

Local connectivity In CNNs, each unit in a hidden layer is connected to
only a subset of neurons in the previous layer. We can justify this
restriction by assuming that neighboring pixels are more relevant to
each other than pixels that are farther apart. As an intuitive example,
consider how this assumption applies to the context of recognizing edges
or contours in an image.

局部连接：在 CNN 中，隐层每个单元只与前一层的部分神经元相连。可基于「相邻像素比远处像素更相关」来辩护；直观上这适用于识别图像中的边缘或轮廓。

Weight sharing Via the convolutional layers, we use the same small set
of weights (the kernels or filters) throughout the whole image. This
reflects the assumption that the same filters are useful for detecting
the same patterns in different parts of the image.

权值共享：通过卷积层，整幅图像共用同一小组权重（核或滤波器），体现「同一组滤波器可用于在图像不同位置检测同一种模式」。

Hierarchical processing CNNs consist of multiple convolutional layers to
extract features from the input image. As the network progresses from
the input to the output layers, low-level features are successively
combined to form increasingly complex features, ultimately leading to
the recognition of more complex objects and shapes. Furthermore, the
convolutional filters in these layers learn to detect specific patterns
and features at different levels of abstraction.

层次处理：CNN 用多层卷积从输入图像提取特征；从输入到输出，低层特征逐层组合为更复杂特征，最终识别更复杂的物体与形状；各层滤波器还在不同抽象层次学习检测特定模式与特征。

Spatial invariance CNNs exhibit the mathematical property of spatial
invariance, meaning the output of a model remains consistent even if the
input signal is shifted to a different location within the spatial
domain. This characteristic arises from the combination of local
connectivity, weight sharing, and the hierarchical architecture
mentioned earlier.

空间不变性：CNN 具有空间不变性这一数学性质，即输入在空间域平移时输出仍可保持一致（在一定意义下）；它来自前述局部连接、权值共享与层次结构的结合。

The combination of local connectivity, weight sharing, and hierarchical
processing in a CNN leads to spatial invariance, allowing the model to
recognize the same pattern or feature regardless of its location in the
input image.

局部连接、权值共享与层次处理共同带来空间不变性，使模型能在输入图像任意位置识别同一模式或特征。

*Translation invariance* is a specific case of spatial invariance in
which the output remains the same after a shift or translation of the
input signal in the spatial domain. In this context, the emphasis is
solely on moving an object to a different location within an image
without any rotations or alterations of its other attributes.

平移不变性是空间不变性的特例：在空间域平移输入后输出不变；此处强调仅改变物体在图像中的位置，而不旋转或改变其他属性。

In reality, convolutional layers and networks are not truly
translation-invariant; rather, they achieve a certain level of
translation equivariance. What is the difference between translation
invariance and equivariance? *Translation invariance* means that the
output does not change with an input shift, while *translation
equivariance* implies that the output shifts with the input in a
corresponding manner. In other words, if we shift the input object to
the right, the results will correspondingly shift to the right, as
illustrated in Figure [13.1](#fig-ch13-fig01).

实际上卷积层与网络并非严格平移不变，而是达到一定程度的平移等变（translation equivariance）。区别是：平移不变指输入平移后输出不变；平移等变指输出随输入作对应平移——例如物体右移，响应也右移，见图 [13.1](#fig-ch13-fig01)。

<a id="fig-ch13-fig01"></a>

<div align="center">
  <img src="./images/ch13-fig01.png" alt="Equivariance under different image translations" width="78%" />
  <div><b>Figure 13.1</b></div>
</div>

As Figure [13.1](#fig-ch13-fig01) shows, under translation invariance, we get the
same output pattern regardless of the order in which we apply the
operations: transformation followed by translation or translation
followed by transformation.

如图 [13.1](#fig-ch13-fig01) 所示，在平移不变下，先变换再平移与先平移再变换得到相同输出模式。

As mentioned earlier, CNNs achieve translation equivariance through a
combination of their local connectivity, weight sharing, and
hierarchical processing properties.
Figure [13.2](#fig-ch13-fig02) depicts a convolutional operation to illustrate
the local connectivity and weight-sharing priors. This figure
demonstrates the concept of translation equivariance in CNNs, in which a
convolutional filter captures the input signal (the two dark blocks)
irrespective of where it is located in the input.

如前所述，CNN 通过局部连接、权值共享与层次处理获得平移等变。图 [13.2](#fig-ch13-fig02) 用卷积操作示意局部连接与权值共享先验，并表明卷积核可捕捉输入信号（两个深色块）无论其位于输入何处。

<a id="fig-ch13-fig02"></a>

<div align="center">
  <img src="./images/ch13-fig02.png" alt="Convolutional filters and translation equivariance" width="78%" />
  <div><b>Figure 13.2</b></div>
</div>

Figure [13.2](#fig-ch13-fig02) shows a $3 \times 3$ input image that
consists of two nonzero pixel values in the upper-left corner (top
portion of the figure) or upper-right corner (bottom portion of the
figure). If we apply a $2 \times 2$ convolutional filter to these
two input image scenarios, we can see that the output feature maps
contain the same extracted pattern, which is on either the left (top of
the figure) or the right (bottom of the figure), demonstrating the
translation equivariance of the convolutional operation.

图 [13.2](#fig-ch13-fig02) 展示 $3 \times 3$ 输入，非零像素在左上（图上方）或右上（图下方）。对这两种情形施加 $2 \times 2$ 卷积核可见输出特征图提取到相同模式，分别位于左或右，体现卷积的平移等变。

For comparison, a fully connected network such as a multilayer
perceptron lacks this spatial invariance or equivariance. To illustrate
this point, picture a multilayer perceptron with one hidden layer. Each
pixel in the input image is connected with each value in the resulting
output. If we shift the input by one or more pixels, a different set of
weights will be activated, as illustrated in
Figure [13.3](#fig-ch13-fig03).

相比之下，多层感知机等全连接网络不具备这种空间不变性或等变性：单层隐层时每个输入像素与每个输出相连，输入平移一格或多格会激活不同权重，见图 [13.3](#fig-ch13-fig03)。

<a id="fig-ch13-fig03"></a>

<div align="center">
  <img src="./images/ch13-fig03.png" alt="Location-specific weights in fully connected layers" width="78%" />
  <div><b>Figure 13.3</b></div>
</div>

Like fully connected networks, ViT architecture (and transformer
architecture in general) lacks the inductive bias for spatial invariance
or equi-  variance. For instance, the model produces different outputs
if we place the same object in two different spatial locations within an
image. This is not ideal, as the semantic meaning of an object (the
concept that an object represents or conveys) remains the same based on
its location. Consequently, it must learn these invariances directly
from the data. To facilitate learning useful patterns present in CNNs
requires pretraining over a larger dataset.

与全连接网络类似，ViT（及一般 Transformer）缺乏空间不变或等变的归纳偏置：同一物体放在图像中两处可能得到不同输出，而语义上位置不应改变含义，故这些不变性须从数据中学；要达到 CNN 中那种有用模式往往要在更大数据集上预训练。

A common workaround for adding positional information in ViTs is to use
relative positional embeddings (also known as *relative positional
encodings*) that consider the relative distance between two tokens in
the input sequence. However, while relative embeddings encode
information that helps transformers keep track of the relative location
of tokens, the transformer still needs to learn from the data whether
and how far spatial information is relevant for the task at hand.

在 ViT 中加入位置信息的常用做法是相对位置嵌入（relative positional encodings），刻画两 token 的相对距离；但相对嵌入虽有助于记录相对次序，模型仍须从数据学习空间信息对当前任务是否、以及在多大程度上重要。

## ViTs Can Outperform CNNs
> 参数规模与预训练数据量到位时 ViT 可超越 CNN；视觉领域常用大规模有监督预训练（如 ImageNet）而非仅靠语言模型式的无监督预训练。
[](#vits-can-outperform-cnns)

The hardcoded assumptions via the inductive biases discussed in previous
sections reduce the number of parameters in CNNs substantially compared
to fully connected layers. On the other hand, ViTs tend to have larger
numbers of parameters than CNNs, which require more training data.
(Refer to Chapter [\[ch11\]](./ch11/_books_ml-q-and-ai-ch11.md) for a refresher on how to precisely calculate the
number of parameters in fully connected and convolutional layers.)

前述硬编码假设使 CNN 相对全连接层参数更少；ViT 参数量往往更大，因而需要更多训练数据（全连接与卷积层参数计数可复习第 [\[ch11\]](./ch11/_books_ml-q-and-ai-ch11.md) 章）。

ViTs may underperform compared to popular CNN architectures without
extensivep retraining, but they can perform very well with a
sufficiently large pretraining dataset. In contrast to language
transformers, where unsupervised pretraining (such as
self-supervisedlearning, disussed in
Chapter [\[ch02\]](./ch02/_books_ml-q-and-ai-ch02.md) ) is
a preferred choice, vision transformers are often pretrained using
large, labeled datasets like ImageNet, which provides millions of
labeled images for training, and regular supervised learning.

若无大规模预训练，ViT 可能逊于常见 CNN；预训练数据足够大时则可很强。与语言 Transformer 偏好无监督/自监督预训练（见第 [\[ch02\]](./ch02/_books_ml-q-and-ai-ch02.md) 章）不同，ViT 常在 ImageNet 等大规模有标注数据上做常规有监督预训练。

An example of ViTs surpassing the predictive performance of CNNs, given
enough data, can be observed from initial research on the ViT
architecture, as shown in the paper "An Image Is Worth 16x16 Words:
Transformers for Image Recognition at Scale."? This study compared
ResNet, a type of convolutional network, with the original ViT design
using different dataset sizes for pretraining. The findings also showed
that the ViT model excelled over the convolutional approach only after
being pretrained on a minimum of 100 million images.

在数据足够时 ViT 超过 CNN 预测性能的例子可见于原始 ViT 论文 "An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale."：该工作比较 ResNet 与 ViT 在不同预训练规模下的表现，并显示 ViT 仅在对至少约一亿张图像预训练后才明显优于卷积路线。

## Inductive Biases in ViTs
> ViT 将图像切块（patchify）并在块间建立全局关系；其特征更均匀、偏全局与低频形状，与 CNN 的纹理/高频偏置形成对照。
[](#inductive-biases-in-vits)

ViTs also possess some inductive biases. For example, vision
transformers *patchify* the input image to process each input patch
individually. Here, each patch can attend to all other patches so that
the model learns relationships between far-apart patches in the input
image, as illustrated in
Figure [13.4](#fig-ch13-fig04).

ViT 也有归纳偏置：例如将输入图像切块（patchify）并对每块单独处理；每块可关注所有其他块，从而学习图像中相距较远的块之间的关系，见图 [13.4](#fig-ch13-fig04)。

<a id="fig-ch13-fig04"></a>

<div align="center">
  <img src="./images/ch13-fig04.png" alt="How a vision transformer operates on image patches" width="78%" />
  <div><b>Figure 13.4</b></div>
</div>

The patchify inductive bias allows ViTs to scale to larger image sizes
without increasing the number of parameters in the model, which can be
computationally expensive. By processing smaller patches individually,
ViTs can efficiently capture spatial relationships between image regions
while benefiting from the global context captured by the self-attention
mechanism.

切块先验使 ViT 可扩展到大分辨率而不同比暴涨参数量（大参数量很贵）；逐块处理结合自注意力的全局上下文，能高效刻画区域间空间关系。

This raises another question: how and what do ViTs learn from the
training data? ViTs learn more uniform feature representations across
all layers, with self-attention mechanisms enabling early aggregation of
global information. In addition, the residual connections in ViTs
strongly propagate features from lower to higher layers, in contrast to
the more hierarchical structure of CNNs.

进一步：ViT 从训练数据中学到什么？其各层表示更均匀，自注意力使全局信息较早汇聚；残差连接又强有力地自下而上传播特征，与 CNN 更分明的层次结构不同。

ViTs tend to focus more on global than local relationships because their
self-attention mechanism allows the model to consider long-range
dependencies between different parts of the input image. Consequently,
the self-attention layers in ViTs are often considered low-pass filters
that focus more on shapes and curvature.

ViT 更偏重全局关系，因自注意力可看输入远距离依赖；因而其自注意层常被视为更偏低频、关注形状与曲率。

In contrast, the convolutional layers in CNNs are often considered
high-pass filters that focus more on texture. However, keep in mind that
convolutional layers can act as both high-pass and low-pass filters,
depending on the learned filters at each layer. High-pass filters detect
an image's edges, fine details, and texture, while low-pass filters
capture more global, smooth features and shapes. CNNs achieve this by
applying convolutional kernels of varying sizes and learning different
filters at each layer.

相比之下 CNN 卷积层常被视为更偏高频与纹理；但具体每层可兼具高通与低通，取决于学到的滤波器——高通抓边缘、细节与纹理，低通抓更全局、平滑的形状；CNN 通过不同核尺寸与各层不同滤波器实现。

## Recommendations
> 实务建议：数据与算力足够可首选或混用 ViT；EfficientNetV2 等 CNN 仍省数据省显存；混合卷积与注意力的架构正在收敛「两全」。
[](#recommendations)

ViTs have recently begun outperforming CNNs if enough data is available
for pretraining. However, this doesn't make CNNs obsolete, as methods
such as the popular EfficientNetV2 CNN architecture are less memory and
data hungry.

近来只要预训练数据足够 ViT 已开始超过 CNN，但 CNN 并未过时：例如 EfficientNetV2 等仍更省显存、更省数据。

Moreover, recent ViT architectures don't rely solely on large
datasets, parameter numbers, and self-attention. Instead, they have
taken inspiration from CNNs and added soft convolutional inductive
biases or even complete convolutional layers to get the best of both
worlds.

此外最新 ViT 不只堆数据、参数与自注意力，还借鉴 CNN，引入软性卷积式归纳偏置甚至完整卷积层，以兼取两者之长。

In short, vision transformer architectures without convolutional layers
generally have fewer spatial and locality inductive biases than
convolutional neuralnetworks. Consequently, vision transformers need to
learn data-related concepts such as local relationships among pixels.
Thus, vision transformers require more training data to achieve good
predictive performance and produce acceptable visual representations in
generative modeling contexts.

简言之，无卷积层的 ViT 比 CNN 更少空间与局部性先验，须从数据学习像素间局部关系等概念，因而要达到良好预测或在生成建模中得到可接受的视觉表示通常需要更多训练数据。

### Exercises
> 习题：块大小与计算量及精度的权衡。
[](#exercises)

13-1. Consider the patchification of the input images shown in
Figure [13.4](#fig-ch13-fig04). The size of the resulting patches controls a
computational and predictive performance trade-off. The optimal patch
size depends on the application and desired trade-off between
computational cost and model performance. Do smaller patches typically
result in higher or lower computational costs?

习题 13-1：考虑图 [13.4](#fig-ch13-fig04) 的切块：块尺寸控制计算与预测性能的权衡，最优块尺寸依应用与成本—性能取舍而定。更小的块通常带来更高还是更低的计算成本？

13-2. Following up on the previous question, do smaller patches
typically lead to a higher or lower prediction accuracy?

习题 13-2：接续上一问：更小的块通常带来更高还是更低的预测准确率？

## References
> 本节列出 ViT 原点论文、相对位置编码、ViT 与 CNN 表征对比、EfficientNetV2，以及融入卷积的 ViT（ConViT、CvT）等文献链接。
[](#references)

- The paper proposing the original vision transformer model: Alexey
  Dosovitskiy et al., "An Image Is Worth 16x16 Words: Transformers for
  Image Recognition at Scale"? (2020),
  <https://arxiv.org/abs/2010.11929>.

Dosovitskiy 等提出原始 ViT 的论文（2020），链接见上。

- A workaround for adding positional information in ViTs is to use
  relative positional embeddings: Peter Shaw, Jakob Uszkoreit, and
  Ashish Vaswani, "Self-Attention with Relative Position
  Representations"? (2018), <https://arxiv.org/abs/1803.02155>.

Shaw、Uszkoreit、Vaswani 关于自注意与相对位置表示的论文（2018）。

- Residual connections in ViTs strongly propagate features from lower to
  higher layers, in contrast to the more hierarchical structure of CNNs:
  Maithra Raghu et al., "Do Vision Transformers See Like Convolutional
  Neural Networks?"? (2021), <https://arxiv.org/abs/2108.08810>.

Raghu 等讨论 ViT 残差连接自下而上强传播特征、与 CNN 层次性的对比（2021）。

- AdetailedresearcharticlecoveringtheEfficientNetV2CNNarchitecture:MingxingTanandQuocV.Le,"EfficientNetV2:
  SmallerMo-
   delsandFasterTraining"?(2021),<https://arxiv.org/abs/2104.00298>.

Tan、Le 关于 EfficientNetV2 CNN 的论文（2021）。

- A ViT architecture that also incorporates convolutional layers:
  StÃ©phane d'Ascoli et al., "ConViT: Improving Vision Transform-
   ers with Soft Convolutional Inductive Biases"? (2021),
  <https://arxiv.org/abs/2103.10697).

d'Ascoli 等 ConViT：以软性卷积归纳偏置改进 ViT（2021）。

- Another example of a ViT using convolutional layers: Haiping Wu
  et al., "CvT: Introducing Convolutions to Vision Transformers"?
  (2021), <https://arxiv.org/abs/2103.15808>.

Wu 等 CvT：将卷积引入 ViT（2021）。


------------------------------------------------------------------------

