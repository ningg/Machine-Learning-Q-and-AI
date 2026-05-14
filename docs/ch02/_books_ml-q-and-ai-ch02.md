







# Chapter 2: Self-Supervised Learning
> 本章定义自监督学习及其与迁移学习的关系，说明如何利用无标签数据，并介绍自预测与对比式自监督两大范式及练习、延伸阅读。
[](#chapter-2-self-supervised-learning)



**What is self-supervised learning, when is it useful, and what are the
main approaches to implementing it?**

什么是自监督学习、何时有用、以及实现它的主要途径有哪些？

*Self-supervised learning* is a pretraining procedure that lets neural
networks leverage large, unlabeled datasets in a supervised fashion.
This chapter compares self-supervised learning to transfer learning, a
related method for pretraining neural networks, and discusses the
practical applications of self-supervised learning. Finally, it outlines
the main categories of self-supervised learning.

*自监督学习*是一种预训练流程，使神经网络能以「监督」的方式利用大规模无标签数据。本章将自监督学习与迁移学习对照，讨论其实用场景，并概括自监督学习的主要类别。

> Tips: 自监督学习，是一种预训练方法，让神经网络利用`无标签`的大数据集，进行`监督学习`。
> 其实，使用`无标签`数据，自动构造了`伪标签`（例如：遮挡部分内容、预测缺失内容），也是一种`监督学习`。



## Self-Supervised Learning vs. Transfer Learning
> 本节对比迁移学习（有标签 ImageNet 预训练再微调）与自监督学习（从无标签数据构造伪标签的 pretext task），并说明二者在「标签来源」上的根本差异。
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

自监督学习与 `transfer learning`（迁移学习）相关：后者把在任务 A 上预训练的模型作为任务 B 的起点。例如先在大型有标签数据集 ImageNet 上预训练卷积网络，再在小规模鸟类数据集上微调（常只需替换最后的分类头，其余权重可沿用）。

Figure [2.1](#fig-ch02-fig01) illustrates the process of transfer learning.

Figure [2.1](#fig-ch02-fig01) 示意传统迁移学习的流程。

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

自监督学习则改为在无标签数据上预训练：利用数据结构「自造」标签，把预测任务交给网络，见 Figure [2.2](#fig-ch02-fig02)；这类任务也称 *pretext tasks*（前置/代理任务）。

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

迁移学习与自监督学习的关键差别在于 Figure 2.1 与 2.2 中第 1 步的标签从何而来：前者依赖人工标注；后者标签可直接由样本本身推导。

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

例如在 NLP 中可遮住词让模型预测；在视觉中可挖去图像块让网络补全。这里仅举两例，自监督任务的形式还有很多。

In sum, we can think of self-supervised learning on the pretext task as
*representation learning*. We can take the pretrained model to fine-tune
it on the target task (also known as the *downstream* task).

总之，可把在代理任务上的自监督预训练视为 *representation learning*（表示学习）；随后在目标（*downstream*）任务上微调即可。

## Leveraging Unlabeled Data
> 本节说明自监督在大模型与 ViT 等架构中的必要性，以及在小 MLP 或树模型上通常不适用、也不兼容迁移的原因。
[](#leveraging-unlabeled-data)

Large neural network architectures require large amounts of labeled data
to perform and generalize well. However, for many problem areas, we
don't have access to large labeled datasets. With self-supervised
learning, we can leverage unlabeled data. Hence, self-supervised
learning is likely to be useful when working with large neural networks
and with a limited quantity of labeled training data.

大网络往往需要大量有标签数据才能泛化好；许多领域却缺乏标注。自监督让我们利用无标签数据，因而在「大网络 + 有标签数据少」的场景特别有价值。

Transformer-based architectures that form the basis of LLMs and vision
transformers are known to require self-supervised learning for
pretraining to perform well.

LLM 与视觉 Transformer 等架构的预训练普遍依赖自监督才能发挥性能。

For small neural network models such as multilayer perceptrons with two
or three layers, self-supervised learning is typically considered
neither useful nor necessary.

对仅两三层的小 MLP，自监督通常既不必要也不划算。

> Tips: 对于`小型`的神经网络模型，如具有两到三层的`多层感知器`，**自监督学习** 在这种情况下 *不实用* 也 *不必要* 。

Self-supervised learning likewise isn't useful in traditional machine
learning with nonparametric models such as tree-based random forests or
gradient boosting. Conventional tree-based methods do not have a fixed
parameter structure (in contrast to the weight matrices, for example).
Thus, conventional tree-based methods are not capable of transfer
learning and are incompatible with self-supervised learning.

在随机森林、梯度提升等非参数树模型上，自监督同样不适用：树方法没有固定参数矩阵式的表示，难以做迁移，也与自监督流程不兼容。

> Tips: 对于`非参数模型`，如基于树的**随机森林**或**梯度提升**，`自监督学习`通常不适用。
> 
> 传统的基于树的方法没有固定的参数结构（与权重矩阵相比），因此传统的基于树的方法**无法进行迁移学习**，也不兼容自监督学习。
> 
> FIXME 没理解???

<details>
<summary>详细解释</summary>
<div>

你提到的这个观点非常关键，它点出了**树模型**与**神经网络**在底层机制上的根本差异。下面我来拆解一下为什么“非参数模型”（以树模型为代表）通常不适用自监督学习和迁移学习。

---

##### 1. 先理解“非参数模型”在这里的含义

- **参数模型**（如神经网络）有固定数量的可训练参数（权重和偏置）。这些参数构成一个连续的、可微的空间，可以通过梯度下降来优化。
- **非参数模型**（如决策树、随机森林、GBDT）的“复杂度”会随数据量增长而增长（树会变深、叶子会变多）。它们没有全局共享的、连续的参数向量。

但更关键的区别是：**树模型学到的是一组基于特征值的硬性决策规则**（例如“如果年龄 < 30 且收入 > 5 万，则预测为 1”）。

---

##### 2. 自监督学习为什么依赖“参数结构”？

自监督学习（如 BERT、SimCLR）的核心流程：
1. 设计一个**前置任务**（例如掩盖部分输入、预测旋转角度）。
2. 用大量无标签数据训练一个神经网络，让网络学习到**通用的特征表示**（通常是一个高维向量）。
3. 将这个预训练好的网络（或它的前几层）迁移到下游任务，做微调。

**为什么树模型不行？**
- **没有连续的表示空间**：神经网络输出的向量可以看作对输入的一种“嵌入”。树模型输出的是一个离散的叶子节点索引或一个标量预测值，没有中间层可以拿出来作为“通用特征”。
- **不可微**：自监督学习的损失函数需要通过梯度反向传播来更新参数。树模型的分裂规则是不可微的（需要贪心搜索分裂点），无法用梯度优化前置任务。
- **硬性规则不具泛化性**：树模型学到的规则完全依赖原始特征的量纲和分布。如果前置任务（比如预测图像是否被旋转）用像素值训练出一棵树，这棵树对“旋转”的判断规则无法迁移到“物体分类”任务——因为树的每个节点都是对具体特征值的硬性阈值，不像神经网络那样可以组合成抽象概念。

---

##### 3. 迁移学习为什么不兼容树模型？

迁移学习需要**知识跨任务或跨领域重用**。例如：
- 用 ImageNet 训练一个 CNN，取它的前几层作为边缘检测器，再应用到医学图像分类。
- 用语言模型预训练，将 Transformer 层作为通用句法/语义提取器。

**树模型无法做到这一点，原因如下：**

1. **没有可分离的特征提取器**  
   神经网络的低层学习到的是一般性的模式（边缘、纹理、词性），这些模式可以移植。而树模型是把所有特征混杂在一起做分裂，你无法单独取出“第一棵树的第一个分裂”作为通用模块——它们严重耦合了具体任务的标签分布。

2. **特征空间必须完全一致**  
   迁移学习常会遇到输入维度变化（比如预训练是 224x224 图像，下游是 128x128）。神经网络可以通过调整第一层卷积核大小来适应。树模型则严格依赖训练时的特征名称、顺序和取值范围。一旦特征数量或含义变化，整棵树就失效了。

3. **输出形式受限**  
   树模型通常直接输出类别或数值，不产生中间表示。如果你想把“预测图像是否包含猫”的森林迁移到“预测图像是否包含狗”，你无法重用任何规则——因为每棵树的每个叶子都专门针对“猫”的概率做出调整。

---

##### 4. 是否存在特例或改进工作？

- **随机森林的“表示学习”变体**：有些工作将树模型嵌入到神经网络中（如深度森林、NODE——Neural Oblivious Decision Ensembles），通过使树结构可微来支持自监督学习。但这时候已经不再是传统的“非参数树”，而是转化为一种有参数的近似结构。
- **将树模型的叶子输出作为特征**：你可以用预训练好的随机森林对数据做编码（例如输出每个样本落入的叶子索引），再用这些索引作为下游模型的输入。但这属于**特征工程**，不是端到端的迁移学习，且前置任务的监督信号无法更新树本身。

**结论**：传统树模型的设计哲学是“独立、贪心、不可微”，而自监督学习和迁移学习依赖“连续、可微、共享表示”。两者在底层范式上不匹配，因此通常认为树模型不适用于这些场景。

---
---

</div>
</details>

## Self-Prediction and Contrastive Self-Supervised Learning
> 本节区分自预测（掩码重建、去噪等）与对比式自监督（拉近正样本、拉远负样本），并介绍 Siamese 设定及样本对比与维度对比两类变体。
[](#self-prediction-and-contrastive-self-supervised-learning)

There are two main categories of self-supervised learning:
`self-prediction` and `contrastive self-supervised` learning. In
*self-prediction*, illustrated in
Figure [2.3](#fig-ch02-fig03), we typically change or hide parts of the input
and train the model to reconstruct the original inputs, such as by
using a perturbation mask that obfuscates certain pixels in an image.

**自监督学习**大致分两类：`self-prediction` 与 `contrastive self-supervised`。`self-prediction`(**自预测**)常遮挡或扰动输入的一部分，再训练模型重建原输入，见 Figure [2.3](#fig-ch02-fig03)。

<a id="fig-ch02-fig03"></a>

<div align="center">
  <img src="./images/ch02-fig03.png" alt="Self-prediction after applying a perturbation mask" width="52%" />
  <div><b>Figure 2.3</b></div>
</div>

A classic example is a denoising autoencoder that learns to remove noise
from an input image. Alternatively, consider a masked autoencoder that
reconstructs the missing parts of an image, as shown in
Figure [2.4](#fig-ch02-fig04).

经典例子包括去噪自编码器，以及 Figure [2.4](#fig-ch02-fig04) 所示的掩码自编码器（MAE）重建缺失区域。

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

NLP 中也广泛使用掩码或缺失输入式的自预测；许多生成式 LLM（如 GPT）以下一词预测为代理任务（详见第 14、17 章）。

In *contrastive self-supervised learning*, we train the neural network
to learn an embedding space where similar inputs are close to each other
and dissimilar inputs are far apart. In other words, we train the
network to produce embeddings that minimize the distance between similar
training inputs and maximize the distance between dissimilar training
examples.

在 *contrastive self-supervised learning* 中，我们训练网络使相似输入的嵌入彼此靠近、不相似输入彼此远离。

Let's discuss contrastive learning using concrete example inputs.
Suppose we have a dataset consisting of random animal images. First, we
draw a random image of a cat (the network does not know the label,
because we assume that the dataset is unlabeled). We then augment,
corrupt, or perturb this cat image, such as by adding a random noise
layer and cropping it differently, as shown in
Figure [2.5](#fig-ch02-fig05).

下面用猫图举例：从无标签动物图中随机抽一张猫，对其加噪、裁剪等得到扰动版本，见 Figure [2.5](#fig-ch02-fig05)。

<a id="fig-ch02-fig05"></a>

<div align="center">
  <img src="./images/ch02-fig05.png" alt="Image pairs encountered in contrastive learning" width="78%" />
  <div><b>Figure 2.5</b></div>
</div>

The perturbed cat image in this figure still shows the same cat, so we
want the network to produce a similar embedding vector. We also consider
a random image drawn from the training set (for example, an elephant,
but again, the network doesn't know the label).

扰动后的猫仍应视为同一只猫，故希望嵌入相近；再抽一张例如大象的图作为负样本，网络不知道类别标签。

For the cat-elephant pair, we want the network to produce dissimilar
embeddings. This way, we implicitly force the network to capture the
image's core content while being somewhat agnostic to small
differences and noise. For example, the simplest form of a contrastive
loss is the $L_2$-norm (Euclidean distance) between the embeddings
produced by model $M(\cdot)$. Let's say we update the model
weights to decrease the distance $||M(cat) - M(cat')||_2$ and increase the distance
$||M(cat) - M(elephant)||_2$.

对猫–象对则希望嵌入远离，从而迫使网络抓住语义主体而对小幅差异不敏感。最简单的对比损失可用嵌入间 $L_2$ 距离表示：缩小 $||M(cat) - M(cat')||_2$、增大 $||M(cat) - M(elephant)||_2$。

Figure [2.6](#fig-ch02-fig06) summarizes the central concept behind 
`contrastive learning` for the perturbed image scenario. The model is shown twice,
which is known as a *siamese network* setup. Essentially, the same model
is utilized in two instances: first, to generate the embedding for the
original training example, and second, to produce the embedding for the
perturbed version of the sample.

Figure [2.6](#fig-ch02-fig06) 概括扰动图像场景下的对比学习：同一模型接两份输入，称为 *siamese network*（孪生网络）设定。

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

对比学习还有许多子变体，可粗分为 *sample contrastive*（样本对距离）与 *dimension contrastive*（只让嵌入的某些维度接近、其余维度推远）。

> Tips: 对比学习，可以分为`样本对比`和`维度对比`两种方法。
> 
> - 样本对比，关注于学习嵌入，以最小化/最大化`训练对`之间的距离。
> - 维度对比，关注于使相似`训练对`中的**某些变量**接近，同时最大化其他变量的距离。


## Exercises
> 本节提出两道思考题：自监督如何用于视频；以及表格数据上是否可行、如何设计。
[](#exercises)

2-1. How could we apply self-supervised learning to video data?

2-1. 如何把自监督学习用到视频数据上？

2-2. Can self-supervised learning be used for tabular data represented
as rows and columns? If so, how could we approach this?

2-2. 行列表格数据能否做自监督？若可以，可如何设计？

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
> 本节列出 ImageNet、SimCLR、VICRegL、自监督 cookbook 及表格数据相关论文链接。
[](#references)

- For more on the ImageNet dataset:
  <https://en.wikipedia.org/wiki/ImageNet>.

- 关于 ImageNet 数据集：<https://en.wikipedia.org/wiki/ImageNet>。

- An example of a contrastive self-supervised learning method: Ting Chen
  et al., "A Simple Framework for Contrastive Learning of Visual
  Representations"? (2020), <https://arxiv.org/abs/2002.05709>.

- 对比式自监督示例：Ting Chen 等，SimCLR (2020)，<https://arxiv.org/abs/2002.05709>。

- An example of a dimension-contrastive method: Adrien Bardes, Jean
  Ponce, and Yann LeCun, "VICRegL: Self-Supervised Learning of Local
  Visual Features"? (2022), <https://arxiv.org/abs/2210.01571>.

- 维度对比示例：VICRegL (2022)，<https://arxiv.org/abs/2210.01571>。

- If you plan to employ self-supervised learning in practice: Randall
  Balestriero et al., "A Cookbook of Self-Supervised Learning"?
  (2023), <https://arxiv.org/abs/2304.12210>.

- 实践指南：自监督 Cookbook (2023)，<https://arxiv.org/abs/2304.12210>。

- A paper proposing a method of transfer learning and self-supervised
  learning for relatively small multilayer perceptrons on tabular
  datasets: Dara Bahri et al., "SCARF: Self-Supervised Contrastive
  Learning Using Random Feature Corruption"? (2021),
  <https://arxiv.org/abs/2106.15147>.

- 表格数据上小 MLP 的自监督：SCARF (2021)，<https://arxiv.org/abs/2106.15147>。

- A second paper proposing such a method: Roman Levin et al.,
  "Transfer Learning with Deep Tabular Models"? (2022),
  [*https://arxiv.org/abs/*](https://arxiv.org/abs/2206.15306)
  [*2206.15306*](https://arxiv.org/abs/2206.15306).

- 另一篇深度表格迁移：Levin 等 (2022)，<https://arxiv.org/abs/2206.15306>。


------------------------------------------------------------------------

