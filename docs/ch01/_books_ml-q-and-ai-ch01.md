





# Chapter 1: Embeddings, Latent Space, and Representations 
> 本章围绕嵌入向量、潜空间与表示三者的联系与区别展开，说明它们在机器学习中表示与编码信息的方式，并配有练习与参考文献。
[](#chapter-1-embeddings-latent-space-and-representations)



**In deep learning, we often use the terms *embedding vectors*,
*representations*, and *latent space*. What do these concepts have in
common, and how do they differ?**

在深度学习中，我们常说 *embedding vectors*（嵌入向量）、*representations*（表示）和 *latent space*（潜空间）。这些概念有什么共同点、又有什么不同？

While these three terms are often used interchangeably, we can make
subtle distinctions between them:

这三个术语常被混用，但仍可作出细致区分：

- `Embedding vectors` are representations of input data where similar
  items are close to each other.

- `Latent vectors` are intermediate representations of input data.

- `Representations` are encoded versions of the original input.

- `Embedding vectors`（嵌入向量）是对输入数据的表示，相似项在空间中彼此靠近。
- `Latent vectors`（潜向量）是输入的中间表示。
- `Representations`（表示）则是对原始输入的编码形式。

The following sections explore the relationship between embeddings,
latent vectors, and representations and how each functions to encode
information in machine learning contexts.

下文将探讨嵌入、潜向量与表示之间的关系，以及它们在机器学习语境中各自如何承担编码信息的角色。

## Embeddings
> 本节定义嵌入向量、说明其与 one-hot 及神经网络层输出的关系，并讨论相似性、结构保持与数学上的单射映射。
[](#embeddings)

`Embedding vectors`, or *embeddings* for short, encode relatively
high-dimensional data into relatively low-dimensional vectors.

嵌入向量（简称 *embeddings*）把相对高维的数据编码到相对低维的向量中。

> Tips: 嵌入向量，简称 `嵌入`，是输入数据的一种表示形式，相似的输入、对应的嵌入向量`彼此接近`；通常，将高维数据，转换为低维嵌入向量。

We can apply embedding methods to create a continuous dense (non-sparse)
vector from a (sparse) `one-hot` encoding. 

我们可以用嵌入方法，从（稀疏的）`one-hot` 编码得到连续、稠密（非稀疏）的向量。

*One-hot encoding* is a method
used to represent categorical data as binary vectors, where each
category is mapped to a vector containing 1 in the position
corresponding to the category's index, and 0 in all other positions.

*One-hot encoding*（独热编码）用二进制向量表示分类数据：每个类别在对应索引位置为 1，其余位置为 0。

This ensures that the categorical values are represented in a way that
certain machine learning algorithms can process. For example, if we have
a categorical variable Color with three categories, Red, Green, and
Blue, the one-hot encoding would represent Red as \[1, 0, 0\], Green as
\[0, 1, 0\], and Blue as \[0, 0, 1\]. These one-hot encoded categorical
variables can then be mapped into continuous embedding vectors by
utilizing the learned weight matrix of an embedding layer or module.

这样分类值就能被某些机器学习算法处理。例如变量 Color 有红、绿、蓝三类，则红为 \[1, 0, 0\]，绿为 \[0, 1, 0\]，蓝为 \[0, 0, 1\]。这些 one-hot 向量再通过嵌入层或模块学到的权重矩阵映射为连续嵌入向量。

> 独热编码：`one-hot` 编码，是一种将分类数据转换为`二进制向量`的方法，其中每个类别映射到包含 1 的向量，在对应类别的索引位置为1，其他位置为0。

We can also use embedding methods for dense data such as images. For
example, the last layers of a convolutional neural network may yield
embedding vectors, as illustrated in
Figure [1.1](#fig-ch01-fig01) .

嵌入同样可用于图像等稠密数据；例如卷积神经网络的靠后层可产生嵌入向量，见 Figure [1.1](#fig-ch01-fig01)。

<a id="fig-ch01-fig01"></a>

<div style="text-align:center">
  <img src="./images/ch01-fig01.png" alt="An input embedding (left) and an embedding from a neural network(right)" style="width:70%;">
  <div><b>Figure 1.1</b></div>
</div>

To be technically correct, all intermediate layer outputs of a neural
network could yield embedding vectors. Depending on the training
objective, the output layer may also produce useful embedding vectors.
For the sake of simplicity, the convolutional neural network in
Figure [1.1](#fig-ch01-fig01)

严格来说，神经网络任意中间层的输出都可视为嵌入向量；视训练目标而定，输出层也可能给出有用的嵌入。为简明起见，Figure [1.1](#fig-ch01-fig01) 中的卷积网络

Embeddings can have higher or lower numbers of dimensions than the
original input. For instance, using embeddings methods for extreme
expression, we can encode data into two-dimensional dense and continuous
representations for visualization purposes and clustering analysis, as
illustrated in Figure [1.2](#fig-ch01-fig02).

嵌入的维数可以高于或低于原始输入；例如为可视化与聚类，可把数据压到二维稠密连续表示，见 Figure [1.2](#fig-ch01-fig02)。

<a id="fig-ch01-fig02"></a>

<div style="text-align:center">
  <img src="./images/ch01-fig02.png" alt="fig-ch01-fig02 Mapping words (left) and images (right) to a two-dimensional feature space" style="width:70%;">
  <div><b>Figure 1.2</b></div>
</div>

**A fundamental property** of embeddings is that they encode *distance* or
*similarity*. This means that embeddings capture the semantics of the
data such that similar inputs are close in the embeddings space.

嵌入的一个**根本性质**是编码*距离*或*相似性*：相似输入在嵌入空间中彼此靠近，从而承载数据的语义。

>Tips: **嵌入向量**，具有一个重要的**性质**，即编码`距离相近`或`相似性`。这意味着嵌入向量能够捕捉数据的语义，使得相似的输入在嵌入空间中彼此接近。这也称为`结构保持` structure-preserving 特性。

For readers interested in a more formal explanation using mathematical
terminology, an embedding is an `injective` and `structure-preserving` map
between an input space *X* and the embedding space *Y*. This implies
that similar inputs will be located at points in close proximity within
the embedding space, which can be seen as the "structure-preserving" 
characteristic of the embedding.

若用更正式的数学语言，嵌入是从输入空间 *X* 到嵌入空间 *Y* 的一个 `injective`（单射）且 `structure-preserving`（结构保持）的映射：相似输入在嵌入空间中位置接近，即所谓「结构保持」。

>Tips: 嵌入向量，是输入空间 *X* 和嵌入空间 *Y* 之间的一个**单向**和**结构保持**映射。这意味着相似的输入，在嵌入空间中彼此接近，这就是`结构保持`特性。

## Latent Space 
> 本节说明潜空间与嵌入空间的关系、潜特征与自编码器瓶颈，以及自编码器目标如何促使相似输入在潜空间中靠近。
[](#latent-space)

*Latent space* is typically used synonymously with *embedding space*,
the space into which embedding vectors are mapped.

*Latent space*（潜空间）通常与 *embedding space*（嵌入空间）同义，即嵌入向量被映射到的空间。

>Tips: 隐空间 \ 潜空间，通常与**嵌入空间**同义，是`嵌入向量`映射到的空间。

Similar items can appear close in the latent space; however, this is not
a strict requirement. More loosely, we can think of the latent space as
any feature space that contains features, often compressed versions of
the original input features. These latent space features can be learned
by a neural network, such as an `autoencoder` that reconstructs input
images, as shown in
Figure [1.3](#fig-ch01-fig03).

相似样本在潜空间中可以靠近，但并非硬性要求。更宽泛地，潜空间可理解为包含（常为压缩后的）输入特征的任何特征空间；这些潜特征可由网络学习，例如用 `autoencoder` 重建输入图像，见 Figure [1.3](#fig-ch01-fig03)。

<a id="fig-ch01-fig03"></a>

<div style="text-align:center">
  <img src="./images/ch01-fig03.png" alt="fig-ch01-fig03 An autoencoder reconstructing the input image" style="width:70%;">
  <div><b>Figure 1.3</b></div>
</div>

The **bottleneck** in
Figure [1.3](#fig-ch01-fig03) represents a small, intermediate neural network
layer that encodes or maps the input image into a lower-dimensional
representation. We can think of the target space of this mapping as a
latent space. The training objective of the autoencoder is to
reconstruct the input image, that is, to minimize the distance between
the input and output images. In order to optimize the training
objective, the autoencoder may learn to place the encoded features of
similar inputs (for example, pictures of cats) close to each other in
the latent space, thus creating useful embedding vectors where similar
inputs are close in the embedding (latent) space.

Figure [1.3](#fig-ch01-fig03) 中的**瓶颈**层是较小的中间层，把输入图像编码/映射到低维表示，其目标空间可视为潜空间。自编码器的训练目标是最小化输入与重建输出之间的距离；为优化该目标，网络可能学会把相似输入（例如猫图）的编码在潜空间中拉近，从而在嵌入（潜）空间中形成「相似近邻」的有用嵌入。

## Representation 
> 本节界定 representation 与嵌入、潜向量的关系，并说明简单流程（如 one-hot）也可产生表示及其在后续分析中的作用。
[](#representation)

A *representation* is an encoded, typically intermediate form of an
input. For instance, an embedding vector or vector in the latent space
is a representation of the input, as previously discussed. However,
representations can also be produced by simpler procedures. For example,
one-hot encoded vectors are considered representations of an input.

*Representation*（表示）是输入的编码形式，常为中间形式；嵌入向量或潜空间中的向量都是表示。表示也可由更简单流程产生，例如 one-hot 向量也是一种输入表示。

The key idea is that the representation captures some essential features
or characteristics of the original data to make it useful for further
analysis or processing.

关键在于表示要抓住原始数据的某些本质特征或特性，以便后续分析或处理。

>Tips: 表示/表征 representation，是输入的一种编码形式，通常是中间形式。关键点是，它能够捕捉输入的一些`本质特征`或`特性`，可用于后续分析。

## Exercises 
> 本节为两道练习题：其一讨论 AlexNet 式网络中哪些层适合作为 embedding；其二列举非嵌入类型的输入表示。
[](#exercises)

1-1. Suppose we're training a convolutional network with five
convolutional layers followed by three fully connected (FC) layers,
similar to AlexNet (<https://en.wikipedia.org/wiki/AlexNet>), as
illustrated in
Figure [1.4](#fig-ch01-fig04).

1-1. 假设我们训练一个五层卷积 + 三层全连接（FC）的网络，结构类似 AlexNet（<https://en.wikipedia.org/wiki/AlexNet>），如 Figure [1.4](#fig-ch01-fig04) 所示。

<a id="fig-ch01-fig04"></a>

<div style="text-align:center">
  <img src="./images/ch01-fig04.png" alt="fig-ch01-fig04" style="width:70%;">
  <div><b>Figure 1.4</b></div>
</div>

We can think of these fully connected layers as two hidden layers and an
output layer in a multilayer perceptron. Which of the neural network
layers can be utilized to produce useful embeddings? Interested readers
can find more details about the AlexNet architecture and implementation
in the original publication by Alex Krizhevsky, Ilya Sutskever, and
Geoffrey Hinton.

可把三层 FC 看作多层感知机中的两个隐层加一个输出层。哪些层适合用来提取有用的 embedding？细节见 Alex Krizhevsky、Ilya Sutskever与 Geoffrey Hinton 的原始论文。


<a id="fig-ch01-fig03"></a>

<div style="text-align:center">
  <img src="./images/ch01-as-01.png" alt="Answer 1" style="width:70%;">
  <div><b>Answer 1-1</b></div>
</div>

**AlexNet 的完整流程**：

* **输入图像 → 卷积层堆叠 (Conv1–Conv5) → 展平 → 全连接层 (FC6–FC8)**
* 标红的 **Conv5、FC6、FC7** 是常用来提取有用 **embedding 特征** 的层。
  * Conv5 → 检测 / 分割（保留空间位置信息）
  * FC6 → 迁移学习（整体结构特征）
  * FC7 → 图像检索 / 聚类（高级语义表示）


>**Answer:**
>
>> asking: in an AlexNet-style convolutional neural network (five conv layers + three >FC layers), **which layers can provide useful embeddings** that capture semantic >information about the input.
>
>Let’s break it down:
>
>1. **Convolutional Layers (early to mid layers)**
>
>* **Early conv layers (conv1–conv2):**
>  These capture **low-level features** like edges, corners, textures, and simple >shapes.
>  → Not very useful as "semantic embeddings," but good for transfer to tasks >requiring fine textures (e.g., medical images, style).
>
>* **Mid-depth conv layers (conv3–conv5):**
>  These capture **mid- to high-level features** (object parts, motifs, hierarchical >compositions).
>  → Often very useful as **general embeddings** for vision tasks like detection, >segmentation, or retrieval.
>  (In fact, features from conv5 of AlexNet were widely used in early transfer >learning.)
>
>2. **Fully Connected Layers**
>
>* **First two FC layers (fc6, fc7):**
>  These act like **dense "semantic bottlenecks."**
>
>  * `fc6` embeddings capture global object structure.
>  * `fc7` embeddings are particularly popular for transfer learning and >representation learning because they’re **highly discriminative yet not tied to the >final class labels.**
>    → Both are excellent choices for embeddings.
>
>* **Last FC layer (fc8, the classifier):**
>  This outputs class scores (e.g., 1000-dim for ImageNet).
>  → Too task-specific (logits for specific labels). Not generally useful as >embeddings unless you remove the softmax and use the logits for re-scoring.
>
>3. **Best Practices**
>
>* If you want **general embeddings** (for clustering, retrieval, transfer learning):
>  * Use `conv5` (after flatten + pooling) or `fc6`/`fc7`.
>  * `fc7` is most common, since it captures high-level semantics.
>* If you want **task-specific representations:**
>  * Use `fc8` logits.
>* If you want **visual features for downstream CV models (e.g., detection, >segmentation):**
>  * Use convolutional feature maps (conv3–conv5), since they preserve spatial >structure.
>
> 4. **Summary:**
>
>Useful embeddings can be taken from **the last convolutional layer (conv5) or the >penultimate fully connected layers (fc6, fc7)**. The final output layer (fc8) is >usually not used for embeddings since it’s tied to a specific classification task.



1-2. Name some types of input representations that are not embeddings.

1-2. 请举出一些属于输入表示、但不是 embedding 的类型。

## References 
> 本节列出本章涉及的 AlexNet 原始论文链接，供延伸阅读。
[](#references)

- The original paper describing the AlexNet architecture and
  implementation: Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton,
  "ImageNet Classification with Deep Convolutional Neural Networks"?
  (2012),
  <https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks>.

- AlexNet 架构与实现的原始论文：Alex Krizhevsky、Ilya Sutskever、Geoffrey Hinton，《ImageNet Classification with Deep Convolutional Neural Networks》(2012)，<https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks>。


------------------------------------------------------------------------

