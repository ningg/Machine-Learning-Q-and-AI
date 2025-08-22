







# Chapter 2: Self-Supervised Learning
[](#chapter-2-self-supervised-learning)



**What is self-supervised learning, when is it useful, and what are the
main approaches to implementing it?**

*Self-supervised learning* is a pretraining procedure that lets neural
networks leverage large, unlabeled datasets in a supervised fashion.
This chapter compares self-supervised learning to transfer learning, a
related method for pretraining neural networks, and discusses the
practical applications of self-supervised learning. Finally, it outlines
the main categories of self-supervised learning.

> Tips: 自监督学习，是一种预训练方法，让神经网络利用`无标签`的大数据集，进行`监督学习`。
> 其实，使用`无标签`数据，自动构造了`伪标签`（例如：遮挡部分内容、预测缺失内容），也是一种`监督学习`。



## Self-Supervised Learning vs. Transfer Learning
[](#self-supervised-learning-vs-transfer-learning) 

Self-supervised learning is related to `transfer learning`, a technique in
which a model pretrained on one task is reused as the starting point for
a model on a second task. For example, suppose we are interested in
training an image classifier to classify bird species. In transfer
learning, we would pretrain a convolutional neural network on the
ImageNet dataset, a large, labeled image dataset with many different
categories, including various objects and animals. After pretraining on
the general ImageNet dataset, we would take that pretrained model and
train it on the smaller, more specific target dataset that contains the
bird species of interest. (Often, we just have to change the
class-specific output layer, but we can otherwise adopt the pretrained
network as is.)

Figure [2.1](#fig-ch02-fig01) illustrates the process of transfer learning.

<a id="fig-ch02-fig01"></a>

<div align="center">
  <img src="./images/ch02-fig01.png" alt="Pretraining with conventional transfer learning" width="78%" />
  <div><b>Figure 2.1</b></div>
</div>

> Tips: 自监督学习，与 迁移学习 是相关的。
> - 相同点在于，都是使用`预训练`的模型，然后进行`微调`。
> - 差异在于，迁移学习是使用`有标签`的数据，而自监督学习使用`无标签`的数据。

Self-supervised learning is an alternative approach to transfer learning
in which the model is pretrained not on labeled data but on *unlabeled*
data. We consider an unlabeled dataset for which we do not have label
information, and then we find a way to obtain labels from the
dataset's structure to formulate a prediction task for the neural
network, as illustrated in
Figure [2.2](#fig-ch02-fig02). These self-supervised training tasks are also
called *pretext tasks*.

<a id="fig-ch02-fig02"></a>

<div align="center">
  <img src="./images/ch02-fig02.png" alt="Pretraining with self-supervised learning" width="78%" />
  <div><b>Figure 2.2</b></div>
</div>

The main difference between transfer learning and self-supervised
learning lies in how we obtain the labels during step 1 in
Figures [2.1](#fig-ch02-fig01)
and [2.2](#fig-ch02-fig02). In transfer learning, we assume that the labels
are provided along with the dataset; they are typically created
by human labelers. In self-supervised learning, the labels can be
directly derived from the training examples.

> Tips: 自监督学习中，数据集的标签，可以`直接`从训练样本中`推导`出来。

A self-supervised learning task could be a missing-word prediction in a
natural language processing context. For example, given the sentence
"It is beautiful and sunny outside,"? we can mask out the word
*sunny*, feed the network the input "It is beautiful and \[MASK\]
outside,"? and have the network predict the missing word in the
"\[MASK\]"? location. Similarly, we could remove image patches in a
computer vision context and have the neural network fill in the blanks.
These are just two examples of self-supervised learning tasks; many more
methods and paradigms for this type of learning exist.

In sum, we can think of self-supervised learning on the pretext task as
*representation learning*. We can take the pretrained model to fine-tune
it on the target task (also known as the *downstream* task).

## Leveraging Unlabeled Data
[](#leveraging-unlabeled-data)

Large neural network architectures require large amounts of labeled data
to perform and generalize well. However, for many problem areas, we
don't have access to large labeled datasets. With self-supervised
learning, we can leverage unlabeled data. Hence, self-supervised
learning is likely to be useful when working with large neural networks
and with a limited quantity of labeled training data.

Transformer-based architectures that form the basis of LLMs and vision
transformers are known to require self-supervised learning for
pretraining to perform well.

For small neural network models such as multilayer perceptrons with two
or three layers, self-supervised learning is typically considered
neither useful nor necessary.

> Tips: 对于`小型`的神经网络模型，如具有两到三层的`多层感知器`，**自监督学习** 在这种情况下 *不实用* 也 *不必要* 。

Self-supervised learning likewise isn't useful in traditional machine
learning with nonparametric models such as tree-based random forests or
gradient boosting. Conventional tree-based methods do not have a fixed
parameter structure (in contrast to the weight matrices, for example).
Thus, conventional tree-based methods are not capable of transfer
learning and are incompatible with self-supervised learning.

> Tips: 对于`非参数模型`，如基于树的**随机森林**或**梯度提升**，`自监督学习`通常不适用。
> 
> 传统的基于树的方法没有固定的参数结构（与权重矩阵相比），因此传统的基于树的方法**无法进行迁移学习**，也不兼容自监督学习。
> 
> FIXME 没理解???

## Self-Prediction and Contrastive Self-Supervised Learning
[](#self-prediction-and-contrastive-self-supervised-learning)

There are two main categories of self-supervised learning:
`self-prediction` and `contrastive self-supervised` learning. In
*self-prediction*, illustrated in
Figure [2.3](#fig-ch02-fig03), we typically change or hide parts of the input
and train the model to reconstruct the original inputs, such as by
using a perturbation mask that obfuscates certain pixels in an image.

<a id="fig-ch02-fig03"></a>

<div align="center">
  <img src="./images/ch02-fig03.png" alt="Self-prediction after applying a perturbation mask" width="52%" />
  <div><b>Figure 2.3</b></div>
</div>

A classic example is a denoising autoencoder that learns to remove noise
from an input image. Alternatively, consider a masked autoencoder that
reconstructs the missing parts of an image, as shown in
Figure [2.4](#fig-ch02-fig04).

<a id="fig-ch02-fig04"></a>

<div align="center">
  <img src="./images/ch02-fig04.png" alt="A masked autoencoder reconstructing a masked image" width="78%" />
  <div><b>Figure 2.4</b></div>
</div>

Missing (`masked`) input self-prediction methods are also commonly used in
natural language processing contexts. Many generative LLMs, such as GPT,
are trained on a next-word prediction pretext task (GPT will be
discussed at greater length in
Chapters [\[ch14\]](./ch14/_books_ml-q-and-ai-ch14.md)
and [\[ch17\]](./ch17/_books_ml-q-and-ai-ch17.md). Here,
we feed the network text fragments, where it has to predict the next
word in the sequence (as we'll discuss further in
Chapter [\[ch17\]](./ch17/_books_ml-q-and-ai-ch17.md)).

In *contrastive self-supervised learning*, we train the neural network
to learn an embedding space where similar inputs are close to each other
and dissimilar inputs are far apart. In other words, we train the
network to produce embeddings that minimize the distance between similar
training inputs and maximize the distance between dissimilar training
examples.

Let's discuss contrastive learning using concrete example inputs.
Suppose we have a dataset consisting of random animal images. First, we
draw a random image of a cat (the network does not know the label,
because we assume that the dataset is unlabeled). We then augment,
corrupt, or perturb this cat image, such as by adding a random noise
layer and cropping it differently, as shown in
Figure [2.5](#fig-ch02-fig05).

<a id="fig-ch02-fig05"></a>

<div align="center">
  <img src="./images/ch02-fig05.png" alt="Image pairs encountered in contrastive learning" width="78%" />
  <div><b>Figure 2.5</b></div>
</div>

The perturbed cat image in this figure still shows the same cat, so we
want the network to produce a similar embedding vector. We also consider
a random image drawn from the training set (for example, an elephant,
but again, the network doesn't know the label).

For the cat-elephant pair, we want the network to produce dissimilar
embeddings. This way, we implicitly force the network to capture the
image's core content while being somewhat agnostic to small
differences and noise. For example, the simplest form of a contrastive
loss is the $L_2$-norm (Euclidean distance) between the embeddings
produced by model $M(\cdot)$. Let's say we update the model
weights to decrease the distance $||M(cat) - M(cat')||_2$ and increase the distance
$||M(cat) - M(elephant)||_2$.

Figure [2.6](#fig-ch02-fig06) summarizes the central concept behind 
`contrastive learning` for the perturbed image scenario. The model is shown twice,
which is known as a *siamese network* setup. Essentially, the same model
is utilized in two instances: first, to generate the embedding for the
original training example, and second, to produce the embedding for the
perturbed version of the sample.

<a id="fig-ch02-fig06"></a>

<div align="center">
  <img src="./images/ch02-fig06.png" alt="Contrastive learning" width="78%" />
  <div><b>Figure 2.6</b></div>
</div>

This example outlines the main idea behind contrastive learning, but
many subvariants exist. Broadly, we can categorize these into *sample*
contrastive and *dimension* contrastive methods. The elephant-cat
example in Figure [2.6](#fig-ch02-fig06) illustrates a sample contrastive method, where we
focus on learning embeddings to minimize and maximize distances between
training pairs. In *dimension*-contrastive approaches, on the other
hand, we focus on making only certain variables in the embedding
representations of similar training pairs appear close to each other
while maximizing the distance of others.

> Tips: 对比学习，可以分为`样本对比`和`维度对比`两种方法。
> 
> - 样本对比，关注于学习嵌入，以最小化/最大化`训练对`之间的距离。
> - 维度对比，关注于使相似`训练对`中的**某些变量**接近，同时最大化其他变量的距离。


## Exercises
[](#exercises)

2-1. How could we apply self-supervised learning to video data?

2-2. Can self-supervised learning be used for tabular data represented
as rows and columns? If so, how could we approach this?

## Talk

>> **Talk 1：自监督学习，使用了无标签数据，进行预训练，本质也是一种有监督训练，理解对吗？**
> 
> 你的理解非常接近，但里面有一点细微差别，我帮你梳理下：
> 
> ##### 1. **传统监督学习**
> 
> * 有人工标注的标签 $y$。
> * 目标函数直接基于 $(x, y)$ 优化（例如图像分类：输入图片，标签是「猫/狗」）。
> 
> 
> ##### 2. **自监督学习 (Self-Supervised Learning, SSL)**
> 
> * **使用无标签数据**（比如大量文本、图片、语音）。
> * 但它会 **自动生成“伪标签”**（pretext task），来构造监督信号。
>   例子：
> 
>   * NLP 里 **掩码预测 (masked language modeling)**：给模型输入 “我去 \[MASK] 吃饭”，伪标签就> 是被遮住的词。
>   * CV 里 **对比学习**：从同一张图生成两个不同增强视图，伪标签就是“这俩是同一个对象”。
> 
> 换句话说，**自监督 ≠ 无监督**。
> 它其实 **是监督学习的一种特例**，只不过监督信号不是人手标注，而是 **从数据本身挖掘出来的监督信息**。
> 
> 
> ##### 3. **本质理解**
> 
> * **是监督学习**：因为它还是通过“预测某个目标”来训练，损失函数需要“标签”（哪怕是自生成的）。
> * **区别在于标签的来源**：
> 
>   * 监督学习 → 外部人工提供。
>   * 自监督学习 → 数据内部挖掘。
> 
> ##### 4. **结论** 
>
> 所以你的说法「自监督学习，本质也是一种有监督训练」是正确的，只是要更精确地说：
> 
> * 它是一种 **“利用数据自身生成监督信号的监督学习”**。
> 


## References
[](#references)

- For more on the ImageNet dataset:
  <https://en.wikipedia.org/wiki/ImageNet>.

- An example of a contrastive self-supervised learning method: Ting Chen
  et al., "A Simple Framework for Contrastive Learning of Visual
  Representations"? (2020), <https://arxiv.org/abs/2002.05709>.

- An example of a dimension-contrastive method: Adrien Bardes, Jean
  Ponce, and Yann LeCun, "VICRegL: Self-Supervised Learning of Local
  Visual Features"? (2022), <https://arxiv.org/abs/2210.01571>.

- If you plan to employ self-supervised learning in practice: Randall
  Balestriero et al., "A Cookbook of Self-Supervised Learning"?
  (2023), <https://arxiv.org/abs/2304.12210>.

- A paper proposing a method of transfer learning and self-supervised
  learning for relatively small multilayer perceptrons on tabular
  datasets: Dara Bahri et al., "SCARF: Self-Supervised Contrastive
  Learning Using Random Feature Corruption"? (2021),
  <https://arxiv.org/abs/2106.15147>.

- A second paper proposing such a method: Roman Levin et al.,
  "Transfer Learning with Deep Tabular Models"? (2022),
  [*https://arxiv.org/abs/*](https://arxiv.org/abs/2206.15306)
  [*2206.15306*](https://arxiv.org/abs/2206.15306).


------------------------------------------------------------------------

