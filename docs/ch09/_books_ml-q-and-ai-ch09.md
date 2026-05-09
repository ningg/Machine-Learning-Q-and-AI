


# Chapter 9: Generative AI Models
> 本章界定生成式建模，并按类别综述深度生成模型及其主要优缺点。
[](#chapter-9-generative-ai-models)



**What are the popular categories of deep generative models in deep
learning (also called *generative AI*), and what are their respective
downsides?**

深度学习中有哪些主流的深度生成模型类别（也常称*生成式 AI*），各自的主要缺点又是什么？

Many different types of deep generative models have been applied to
generating different types of media: images, videos, text, and audio.
Beyond these types of media, models can also be repurposed to generate
domain-specific data, such as organic molecules and protein structures.
This chapter will first define generative modeling and then outline each
type of generative model and discuss its strengths and weaknesses.

已有多种深度生成模型被用来生成不同类型的媒介：图像、视频、文本与音频。除这些媒介外，模型也可改用于生成领域特定数据（如有机分子与蛋白质结构）。本章先定义生成式建模，再逐类概述各类生成模型并讨论其优势与劣势。

## Generative vs. Discriminative Modeling
> 对比生成式与判别式建模：联合分布与条件分布、朴素贝叶斯与逻辑回归的经典对照。
[](#generative-vs-discriminative-modeling)

In traditional machine learning, there are two primary approaches to
modeling the relationship between input data (*x*) and output labels
(*y*): `generative models` and `discriminative models`. 

在经典机器学习中，刻画输入数据（*x*）与输出标签（*y*）之间的关系主要有两类思路：`生成式模型`与`判别式模型`。

- `Generative models` aim to capture the underlying probability distribution of the input data
  *p*(*x*) or the joint distribution *p*(*x*, *y*) between inputs and
  labels. 

- `生成式模型`力求刻画输入数据的边缘分布 *p*(*x*) 或输入与标签的联合分布 *p*(*x*, *y*)。

- In contrast, `discriminative models` focus on modeling the
conditional distribution *p*(*y* | *x*) of the labels given the
inputs.

- 相对地，`判别式模型`关注在给定输入条件下标签的条件分布 *p*(*y* | *x*)。

A classic example that highlights the differences between these approaches is to compare the `naive Bayes classifier` and the `logistic regression classifier`.

一个经典对照是比较`朴素贝叶斯分类器`与`逻辑回归分类器`。

- Both classifiers estimate the class label probabilities *p*(*y* | *x*) and can be used for classification tasks. 

- 两者都估计类别标签概率 *p*(*y* | *x*)，并可做分类。

- However, `logistic regression` is considered a discriminative model because it directly models the conditional probability distribution *p*(*y* | *x*) of the class labels given the input features without making assumptions about the underlying joint distribution of inputs and labels. 

- 但`逻辑回归`被视为判别式模型，因其直接建模给定特征时的条件分布 *p*(*y* | *x*)，而不假设输入与标签的联合分布形式。

- `Naive Bayes`, on the other hand, is considered a generative model because it models the joint probability distribution *p*(*x*, *y*) of the input features *x* and the output labels *y*. By learning the joint distribution, a generative model like naive Bayes captures the underlying data generation process, which enables it to generate new samples from the distribution if needed.

- `朴素贝叶斯`则是生成式模型：它建模特征 *x* 与标签 *y* 的联合分布 *p*(*x*, *y*)；通过学习联合分布，该类生成式模型刻画了数据生成过程，从而在需要时能从分布中采样新样本。

> Tips: 
> 
> - 贝叶斯分类器，假设输入和输出之间存在**联合概率分布**；可以生成新的样本，因为它是生成模型；
> - 逻辑回归分类器，假设输入和输出之间存在**条件概率分布**；不能生成新的样本；


## Types of Deep Generative Models
> 概览“深度”生成模型常见类型及下文将讨论的各类方法。
[](#types-of-deep-generative-models)

When we speak of *deep* generative models or deep generative AI, we
often loosen this definition to include all types of models capable of
producing realistic-looking data (typically text, images, videos, and
sound). The remainder of this chapter briefly discusses the different
types of deep generative models used to generate such data.

说到*深度*生成模型或深度生成式 AI，人们常把定义放宽：凡能生成观感逼真的数据（多为文本、图像、视频与声音）的模型都可算入其中。本章余下部分将简要讨论用于生成此类数据的各类深度生成模型。

### Energy-Based Models
> 基于能量的模型（EBM）：能量函数、深度玻尔兹曼机及其在生成中的局限性。
[](#energy-based-models)

*Energy-based models (EBMs)* are a class of generative models that learn
an energy function, which assigns a scalar value (energy) to each data
point. Lower energy values correspond to more likely data points. The
model is trained to minimize the energy of real data points while
increasing the energy of generated data points.

*基于能量的模型（EBM）*学习一个能量函数，为每个数据点赋予标量能量；能量越低的数据点越可能。训练时降低真实样本的能量并抬高生成样本的能量。

Examples of EBMs include `deep Boltzmann machines (DBMs)`.

EBM 的例子包括`深度玻尔兹曼机（DBM）`。

One of the early breakthroughs in deep learning, DBMs provide a means to learn complex representations of data. 
You can think of them as a form of unsupervised pretraining, resulting in models that can then be fine-tuned for various tasks.

DBM 曾是深度学习早期的突破之一：可学习数据的复杂表征，可视作一种无监督预训练，得到的模型随后可对各种任务微调。

Somewhat similar to naive Bayes and logistic regression, DBMs and
multilayer perceptrons (MLPs) can be thought of as generative and
discriminative counterparts, with DBMs focusing on capturing the data
generation process and MLPs focusing on modeling the decision boundary
between classes or mapping inputs to outputs.

与朴素贝叶斯和逻辑回归的类比类似，DBM 与多层感知机（MLP）可看作生成式与判别式的一对：前者侧重刻画数据生成过程，后者侧重刻画类间决策边界或将输入映射到输出。

A `DBM` consists of multiple layers of hidden nodes, as shown in Figure [9.1](#fig-ch09-fig01). As the figure illustrates, along with the hidden
layers, there's usually a visible layer that corresponds to the
observable data. This visible layer serves as the input layer where the
actual data or features are fed into the network. In addition to using a
different learning algorithm than MLPs (contrastive divergence instead
of backpropagation), DBMs consist of binary nodes (neurons) instead of
continuous ones.

`DBM` 含多层隐藏节点，见图 [9.1](#fig-ch09-fig01)。如图所示，除隐藏层外通常还有对应可观测数据的可见层；该层充当输入层，真实数据或特征由此馈入网络。相较 MLP，DBM 使用不同的学习算法（对比散度而非反向传播），且节点为二进制而非连续值。

<a id="fig-ch09-fig01"></a>

<div align="center">
  <img src="./images/ch09-fig01.png" alt="A four-layer deep Boltzmann machine with three stacks of hidden nodes" width="78%" />
  <div><b>Figure 9.1</b></div>
</div>

Suppose we are interested in generating images. A DBM can learn the
joint probability distribution over the pixel values in a simple image
dataset like MNIST. To generate new images, the DBM then samples from
this distribution by performing a process called *Gibbs sampling*. Here,
the visible layer of the DBM represents the input image. To generate a
new image, the DBM starts by initializing the visible layer with random
values or, alternatively, uses an existing image as a seed. Then, after
completing several Gibbs sampling iterations, the final state of the
visible layer represents the generated image.

若要生成图像，DBM 可在 MNIST 这类简单数据集上学习像素联合分布；生成时通过 *Gibbs 采样*从该分布采样。可见层表示输入图像：可随机初始化可见层或以已有图像为种子，经若干步 Gibbs 迭代后，可见层的终态即为生成图像。

DBMs played an important historical role as one of the first deep
generative models, but they are no longer very popular for generating
data. They are expensive and more complicated to train, and they have
lower expressivity compared to the newer models described in the
following sections, which generally results in lower-quality generated
samples.

DBM 作为早期深度生成模型有重要历史地位，但如今已较少用于生成数据：训练昂贵且复杂，表达力也弱于下文较新模型，通常导致生成样本质量偏低。

### Variational Autoencoders
> 变分自编码器（VAE）：编码器/解码器、ELBO 式损失（重构 + KL）及常见短板。
[](#variational-autoencoders)

*Variational autoencoders (VAEs)* are built upon the principles of
variational inference and autoencoder architectures. *Variational
inference* is a method for approximating complex probability
distributions by optimizing a simpler, tractable distribution to be as
close as possible to the true distribution. *Autoencoders* are
unsupervised neural networks that learn to compress input data into a
low-dimensional representation (encoding) and subsequently reconstruct
the original data from the compressed representation (decoding) by
minimizing the reconstruction error.

*变分自编码器（VAE）*建立在变分推断与自编码结构之上。*变分推断*用更易处理的分布去逼近复杂真实分布；*自编码器*是无监督网络，将输入压缩为低维编码再解码重构，并通过最小化重构误差学习。

The VAE model consists of two main submodules: an encoder network and a
decoder network. The encoder network takes, for example, an input image
and maps it to a latent space by learning a probability distribution
over the latent variables. This distribution is typically modeled as a
Gaussian with parameters (mean and variance) that are functions of
the inputimage. The decoder network then takes a sample from the learned
latent distribution and reconstructs the input image from this sample.
The goal of the VAE is to learn a compact and expressive latent
representation that captures the essential structure of the input data
while being able to generate new images by sampling from the latent
space. (See Chapter [\[ch01\]](./ch01/_books_ml-q-and-ai-ch01.md) for more details on latent representations.)

VAE 含编码器与解码器：编码器（如对输入图像）将数据映射到潜变量上的分布（常为高斯，其均值与方差为输入的函数）。解码器从该分布采样并重构输入。目标是学得紧凑而有表达力的潜空间，既抓住数据结构，又能从中采样生成新图像（潜变量更多细节见第 [\[ch01\]](./ch01/_books_ml-q-and-ai-ch01.md) 章）。

Figure [9.2](#fig-ch09-fig02) illustrates the encoder and decoder submodules of
an auto-encoder, where $x'$ represents the reconstructed input
*x*. In a standard variational autoencoder, the latent vector is sampled
from a distribution that approximates a standard Gaussian distribution.

图 [9.2](#fig-ch09-fig02) 示意自编码器的编解码子模块；$x'$ 表示重构的输入 *x*。标准 VAE 中潜向量从逼近标准高斯的分布中采样。

<a id="fig-ch09-fig02"></a>

<div align="center">
  <img src="./images/ch09-fig02.png" alt="An autoencoder" width="78%" />
  <div><b>Figure 9.2</b></div>
</div>

Training a VAE involves optimizing the model's parameters to minimize
a loss function composed of two terms: a reconstruction loss and a
Kullback -- Leibler-divergence (KL-divergence) regularization term. The
reconstruction loss ensures that the decoded samples closely resemble
the input images, while the KL-divergence term acts as a surrogate loss
that encourages the learned latent distribution to be close to a
predefined prior distribution (usually a standard Gaussian). To generate
new images, we then sample points from the latent space's prior
(standard Gaussian) distribution and pass them through the decoder
network, which generates new, diverse images that look similar to the
training data.

训练 VAE 需最小化由重构项与 KL 散度正则项构成的损失：重构项使解码结果贴近输入，KL 项促使学到的潜分布接近预设先验（常为标准高斯）。生成新图像时，从先验（标准高斯）采样并经解码器前向，可得到与训练数据相似而又多样的图像。

Disadvantages of VAEs include their complicated loss function consisting
of separate terms, as well as their often low expressiveness. The latter
can result in blurrier images compared to other models, such as
generative adversarial networks.

VAE 的缺点包括损失由多分项构成、较为复杂，以及表达力常偏弱——相对生成对抗网络等模型，图像往往更模糊。

### Generative Adversarial Networks
> 生成对抗网络（GAN）：生成器与判别器的对抗训练、不稳定与模式坍缩等问题。
[](#generative-adversarial-networks)

*Generative adversarial networks (GANs)* are models consisting of
interacting subnetworks designed to generate new data samples that are
similar to a given set of input data. While both GANs and VAEs are
latent variable models that generate data by sampling from a learned
latent space, their architectures and learning mechanisms are
fundamentally different.

*生成对抗网络（GAN）*由相互作用的子网络组成，用以生成与训练集相似的新样本。GAN 与 VAE 虽都从潜变量空间采样生成，但其结构与学习机理根本不同。

GANs consist of two neural networks, a generator and a discriminator,
that are trained simultaneously in an adversarial manner. The generator
takes a random noise vector from the latent space as input and generates
a synthetic data sample (such as an image). The discriminator's task
is to distinguish between real samples from the training data and fake
samples generated by the generator, as illustrated in
Figure [9.3](#fig-ch09-fig03).

GAN 包含生成器与判别器两个网络，同时对抗训练：生成器以潜空间中的随机噪声为输入合成样本（如图）；判别器则区分训练集中的真实样本与生成器输出的假样本（见图 [9.3](#fig-ch09-fig03)）。

<a id="fig-ch09-fig03"></a>

<div align="center">
  <img src="./images/ch09-fig03.png" alt="A generative adversarial network" width="78%" />
  <div><b>Figure 9.3</b></div>
</div>

The generator in a GAN somewhat resembles the decoder of a VAE in terms
of its functionality. During inference, both GAN generators and VAE
decoders take random noise vectors sampled from a known distribution
(for example, a standard Gaussian) and transform them into synthetic
data samples, such as images.

就功能而言，GAN 的生成器与 VAE 的解码器有相似之处：推断时二者都把来自已知分布（如标准高斯）的随机噪声向量变换为图像等合成数据。

One significant disadvantage of GANs is their unstable training due to
the adversarial nature of the loss function and learning process.
Balancing the learning rates of the generator and discriminator can be
difficult and can often result in oscillations, mode collapse, or
non-convergence. The second main disadvantage of GANs is the low
diversity of their generated outputs, often due to mode collapse. Here,
the generator is able to fool the discriminator successfully with a
small set of samples, which are representative of only a small subset of
the original training data.

GAN 的主要缺点是训练不稳：对抗目标使生成器与判别器的学习率难以平衡，易出现震荡、模式坍缩或不收敛。第二缺点是输出多样性不足，常源于模式坍缩——生成器用少量即能骗过判别器的样本，只覆盖训练分布的一小部分。

### Flow-Based Models
> 流模型（正则化流）：可逆变换、NICE 思路、似然与其他方法的取舍。
[](#flow-based-models)

The core concept of *flow-based models*, also known as *normalizing
flows*, is inspired by long-standing methods in statistics. The primary
goal is to transform a simple probability distribution (like a Gaussian)
into a more complex one using invertible transformations.

*流模型*亦称*正则化流*，思想源于统计学：用可逆变换将简单分布（如高斯）变形为复杂分布。

Although the concept of `normalizing flows` has been apart of the statistics field
for a long time, the implementation of early flow-based deep learning
models, particularly for image generation, is a relatively recent
development. One of the pioneering models in this area was the
*non-linear independent components estimation (NICE)* approach. NICE
begins with a simple probability distribution, often something
straightforward like a normal distribution. You can think of this as a
kind of "random noise,"? or data with no particular shape or
structure. NICE then applies a series of transformations to this simple
distribution. Each transformation is designed to make the datalook more
like the final target (for instance, the distribution of real-world
images). These transformations are "invertible,"? meaning we can
always reverse them back to the original simple distribution. After
several successive transformations, the simple distribution has morphed
into a complex distribution that closely matches the distribution of the
target data (such as images). We can now generate new data that looks
like the target data by picking random points from this complex
distribution.

`正则化流`在统计中由来已久，但早期用于图像生成的流式深度学习实现相对较新。先驱之一是 *NICE（非线性独立分量估计）*：从简单分布（如正态，可类比无结构的“随机噪声”）出发，施加一系列变换，使分布逐步更接近目标（如真实图像分布）；每一步可逆，可回到原始简单分布。经多次变换后，简单分布被塑造成贴近目标数据的复杂分布，再从该分布采样即可生成看起来像目标的数据。

Figure [9.4](#fig-ch09-fig04) illustrates the concept of a flow-based model,
which maps the complex input distribution to a simpler distribution and
back.

图 [9.4](#fig-ch09-fig04) 示意流模型如何将复杂输入分布映射到简单分布并可逆还原。

<a id="fig-ch09-fig04"></a>

<div align="center">
  <img src="./images/ch09-fig04.png" alt="A flow-based model" width="78%" />
  <div><b>Figure 9.4</b></div>
</div>

At first glance, the illustration is very similar to the VAE
illustration in Figure [9.2](#fig-ch09-fig02). However, while VAEs use neural network encoders
like convolutional neural networks, the flow-based model uses simpler
decoupling layers, such as simple linear transformations. Additionally,
while the decoder in a VAE is independent of the encoder, the
data-transforming functions in the flow-based model are mathematically
inverted to obtain the outputs.

乍看之下该图与 VAE（图 [9.2](#fig-ch09-fig02)）很像；但 VAE 编码器常为 CNN 等复杂网络，流模型则用更简单的解耦层（如线性变换）。另外 VAE 的解码器与编码器分离，而流模型的数据变换函数在数学上成对互为逆以实现前向与逆推。

Unlike VAEs and GANs, flow-based models provide exact likelihoods, which
gives us insights into how well the generated samples fit the training
data distribution. This can be useful in anomaly detection or density
estimation, for example. However, the quality of flow-based models for
generating image data is usually lower than GANs. Flow-based models also
often require more memory and computational resources than GANs or VAEs
since they must store and compute inverses of transformations.

与 VAE、GAN 不同，流模型可给出精确似然，有助判断生成样本与训练分布的贴合度，可用于异常检测或密度估计等。但若论图像生成质量通常不如 GAN；且需存储与计算变换的逆，内存与算力开销常高于 GAN 或 VAE。

### Autoregressive Models
> 自回归模型：逐步预测下一 token/像素、链式分解、优缺点（慢、长程依赖等）。
[](#autoregressive-models)

*Autoregressive models* are designed to predict the next value based on
current (and past) values. LLMs for text generation, like ChatGPT
(discussed further in Chapter [\[ch17\]](./ch17/_books_ml-q-and-ai-ch17.md)), are one popular example of this type of model.

*自回归模型*根据当前（及过往）取值预测下一个值；文本生成的 LLM（如 ChatGPT，第 [\[ch17\]](./ch17/_books_ml-q-and-ai-ch17.md) 章详述）是典型例子。

Similar to generating one word at a time, in the context of image generation, autoregressive models like `PixelCNN` try to predict one pixel
at a time, given the pixels they have seen so far. Such a model might
predict pixels from top left to bottom right, in a raster scan order, or
in any other defined order.

类比逐词生成，在图像领域 `PixelCNN` 等模型在给定已见像素条件下一次预测一个新像素：可按光栅扫描等任意预定顺序从左到右上至下地进行。

To illustrate how autoregressive models generate an image one pixel at a
time, suppose we have an image of size *H* × *W* (where *H* is
the height and *W* is the width), ignoring the color channel for
simplicity's sake. This image consists of *N* pixels, where $i = 1, \ldots, N$. The probability of observing a particular image in the
dataset is then $P(Image) = P(i_1, i_2, \ldots, i_N)$.
Basedon the chain rule of probability in statistics, we can decompose
this joint probability into conditional probabilities:

说明自回归图像生成：设图像尺寸为 *H*×*W*（暂忽略颜色通道），共 *N* 个像素，$i = 1, \ldots, N$。某张图像的概率为 $P(Image) = P(i_1, i_2, \ldots, i_N)$。由概率论的链式法则，可将联合分解为一系列条件概率：

$$
\begin{aligned}
P( { Image })&=P\left(i_1, i_2, \ldots, i_N\right) \\
&=P\left(i_1\right) \cdot P\left(i_2 \mid i_1\right) \cdot P\left(i_3 \mid i_1, i_2\right) \ldots P\left(i_N \mid i_1 \ldots i_{N-1}\right)
\end{aligned}
$$

Here, $P(i_1)$ is the probability of the first pixel, $P(i_2 | i_1)$ is the probability of the second pixel given the first pixel, $P(i_3 | i_1, i_2)$ is the probability of the third pixel given the first and second pixels, and so on.

其中 $P(i_1)$ 是第一个像素的概率，$P(i_2 | i_1)$ 是在第一个已知时第二个像素的概率，依此类推。

In the context of image generation, an autoregressive model essentially
tries to predict one pixel at a time, as described earlier, given the
pixels it has seen so far.

在图像生成中，自回归模型本质上就是按前述方式，在已见像素的条件下逐步预测下一像素。

Figure [9.5](#fig-ch09-fig05) illustrates this process, where pixels $i_1, \ldots, i_{53}$ represent the context and pixel $i_{54}$ is the next pixel to be generated.

图 [9.5](#fig-ch09-fig05) 中 $i_1, \ldots, i_{53}$ 为上下文，$i_{54}$ 为待生成的下一像素。

<a id="fig-ch09-fig05"></a>

<div align="center">
  <img src="./images/ch09-fig05.png" alt="Autoregressive pixel generation" width="78%" />
  <div><b>Figure 9.5</b></div>
</div>

The advantage of autoregressive models is that the next-pixel (or word)
prediction is relatively straightforward and interpretable. In addition,
auto-  regressive models can compute the likelihood of data exactly,
similar to flow-based models, which can be useful for tasks like anomaly
detection. Furthermore, autoregressive models are easier to train than
GANs as they don't suffer from issues like mode collapse and other
training instabilities.

其优点在于下一步（像素或词）预测直观易懂；似然可与流模型一样精确计算，利于异常检测等；且无 GAN 的模式坍缩等问题，通常更易训练。

However, autoregressive models can be slow at generating new samples.
This is because they have to generate data one step at a time (for
example, pixel by pixel for images), which can be computationally
expensive. Autoregressive models may also struggle to capture
long-range dependencies because each output is conditioned only on
previously generated outputs.

缺点是采样慢：必须逐步生成（如图逐像素），计算成本高；且每步只看已生成部分，长程依赖较难刻画。

In terms of overall image quality, autoregressive models are therefore
usually worse than GANs but are easier to train.

就整体图像质量而言，自回归模型通常弱于 GAN，但训练更平易。

### Diffusion Models
> 扩散模型：前向加噪与反向去噪，与流模型的区别及采样速度取舍。
[](#diffusion-models)

As discussed in the previous section, flow-based models transform a
simple distribution (such as a standard normal distribution) into a
complex one (the target distribution) by applying a sequence of
invertible and differentiable transformations (flows). Like flow-based
models, *diffusion models* alsoapply a series of transformations.
However, the underlying concept is fundamentally different.

如上一节所述，流模型用可逆可微的一系列“流”把简单分布变到复杂目标分布；*扩散模型*也施加多步变换，但原理根本不同。

Diffusion models transform the input data distribution into a simple
noise distribution over a series of steps using stochastic differential
equations. Diffusion is a stochastic process in which noise is
progressively added to the data until it resembles a simpler
distribution, like Gaussian noise. To generate new samples, the process
is then reversed, starting from noise and progressively removing it.

扩散模型通过随机微分方程等，在若干步内把数据分布逐步变成简单噪声分布：扩散即向数据逐步加噪直至接近高斯噪声；生成则反向从噪声出发逐步去噪。

Figure [9.6](#fig-ch09-fig06) outlines the process of adding and removing
Gaussian noise from an input image *x*. During inference, the reverse
diffusion process is used to generate a new image *x*, starting with the
noise tensor *z~n~* sampled from a Gaussian distribution.

图 [9.6](#fig-ch09-fig06) 概括对输入图像 *x* 加减高斯噪声的过程；推断时用反向扩散从采样得到的高斯噪声张量 *z~n~* 生成新图像 *x*。

<a id="fig-ch09-fig06"></a>

<div align="center">
  <img src="./images/ch09-fig06.png" alt="The diffusion process" width="78%" />
  <div><b>Figure 9.6</b></div>
</div>

While both diffusion models and flow-based models are generative models
aiming to learn complex data distributions, they approach the problem
from different angles. Flow-based models use deterministic invertible
transformations, while diffusion models use the aforementioned
stochastic diffusion process.

二者都是学习复杂分布的生成模型，但角度不同：流模型用确定性可逆变换，扩散模型用前述随机扩散过程。

Recent projects have established state-of-the-art performance in
generating high-quality images with realistic details and textures.
Diffusion models are also easier to train than GANs. The downside of
diffusion models, however, is that they are slower to sample from since
they require running a series of sequential steps, similar to flow-based
models and autoregressive models. This can make diffusion models less
practical for some applications requiring fast sampling.

近年扩散模型在高细节、高质感图像生成上可达顶尖水平，且通常比 GAN 更易训练。缺点在于采样需顺序多步运行，速度与流模型、自回归模型类似偏慢，在需要快速生成的场景不够实用。

### Consistency Models
> 一致性模型：单步去噪映射、ODE 轨迹及与扩散在质量与速度上的比较。
[](#consistency-models)

*Consistency models* train a neural network to
map a noisy image to a clean one. The network is trained on a
dataset of pairs of noisy and clean images and learns to identify
patterns in the clean images that are modified by noise. Once the
network is trained, it can be used to generate reconstructed images from
noisy images in one step.

*一致性模型*训练网络把含噪图像映射到干净图像：在 noisy/clean 成对数据上学习噪声如何改变干净图中的结构；训练后可一步从噪声图重建或生成。

Consistency model training employs an *ordinary differential equation (ODE)*
trajectory, a path that a noisy image follows as it is gradually
denoised. The ODE trajectory is defined by a set of differential
equations that describe how the noise in the image changes over time, as
illustrated in Figure [9.7](#fig-ch09-fig07).

训练借助 *常微分方程（ODE）* 轨迹描述图像随时间逐步去噪的路径；微分方程刻画噪声如何演变，见图 [9.7](#fig-ch09-fig07)。

<a id="fig-ch09-fig07"></a>

<div align="center">
  <img src="./images/ch09-fig07.png" alt="Trajectories of a consistency model for image denoising" width="78%" />
  <div><b>Figure 9.7</b></div>
</div>

As Figure [9.7](#fig-ch09-fig07) demonstrates, we can think of consistency models
as models that learn to map any point from a probability flow ODE, which
smoothly converts data to noise, to the input.

如图 [9.7](#fig-ch09-fig07) 所示，可把一致性模型理解为：学习将概率流 ODE 上任意一点（该 ODE 平滑地把数据变成噪声）映射回数据（输入）端。

At the time of writing, consistency models are the most recent type of
generative AI model. Based on the original paper proposing this method,
consistency models rival diffusion models in terms of image quality.
Consistency models are also faster than diffusion models because they do
not require an iterative process to generate images; instead, they
generate images in a single step.

写作时一致性模型是较新的一类生成式方法；原论文称其在图像质量上可与扩散模型媲美。因无需迭代多步、可单步生成图像，推断快于扩散模型。

However, while consistency models allow for faster inference, they are
still expensive to train because they require a large dataset of pairs
of noisy and clean images.

但训练成本仍高：需要大量含噪与干净图像的成对数据。

## Recommendations
> 各模型适用场景小结：历史意义、似然估计与当下图像生成主流选择。
[](#recommendations)

Deep Boltzmann machines are interesting from a historical perspective
since they were one of the pioneering models to effectively demonstrate
the concept of unsupervised learning. Flow-based and autoregressive
models may be useful when you need to estimate exact likelihoods.
However, other models are usually the first choice when it comes to
generating high-quality images.

深玻尔兹曼机在历史上率先有效展示无监督学习，有其价值。若需精确似然，流模型与自回归模型更有用；但追求高质量图像时，人们通常首选其他架构。

In particular, VAEs and GANs have competed for years to generate the
best high-fidelity images. However, in 2022, diffusion models began to
take over image generation almost entirely. Consistency models are a
promising alternative to diffusion models, but it remains to be seen
whether they become more widely adopted to generate state-of-the-art
results. The trade-off here is that sampling from diffusion models is
generally slower since it involves a sequence of noise-removal steps
that must be run in order, similar to autoregressive models. This can
make diffusion models less practical for some applications requiring
fast sampling.

VAE 与 GAN 曾长期竞逐高保真图像；2022 年后扩散模型几乎主导图像生成。一致性模型是有前景的替代，但能否广泛用于 SOTA 尚待观察。折中在于：扩散采样需顺序多步去噪，类似自回归，故在要快速采样的应用中不够方便。

## Exercises
> 习题：评价生成图像质量；如何用一致性模型生成新图像。
[](#exercises)

9-1. How would we evaluate the quality of the images generated by a
generative AI model?

9-1. 如何评价生成式 AI 模型所得图像的质量？

9-2. Given this chapter's description of consistency models, how would
we use them to generate new images?

9-2. 根据本章对一致性模型的描述，应如何用其生成新图像？

## References
> 参考文献：VAE、GAN、NICE、PixelCNN、Stable Diffusion 与一致性模型等原始资料与实现链接。
[](#references)

- The original paper proposing variational autoencoders: Diederik P.
  Kingma and Max Welling, "Auto-Encoding Variational Bayes"? (2013),
  <https://arxiv.org/abs/1312.6114>.

- 提出变分自编码器的原始论文：Diederik P. Kingma 与 Max Welling，《Auto-Encoding Variational Bayes》（2013），<https://arxiv.org/abs/1312.6114>。

- The paper introducing generative adversarial networks: Ian J.
  Goodfellow et al., "Generative Adversarial Networks"? (2014),
  <https://arxiv.org/abs/1406.2661>.

- 引入生成对抗网络的论文：Ian J. Goodfellow 等，《Generative Adversarial Networks》（2014），<https://arxiv.org/abs/1406.2661>。

- The paper introducing NICE: Laurent Dinh, David Krueger, and Yoshua
  Bengio, "NICE: Non-linear Independent Components Estimation"?
  (2014), <https://arxiv.org/abs/1410.8516>.

- 介绍 NICE 的论文：Laurent Dinh、David Krueger 与 Yoshua Bengio，《NICE: Non-linear Independent Components Estimation》（2014），<https://arxiv.org/abs/1410.8516>。

- The paper proposing the autoregressive PixelCNN model: Aaron van den
  Oord et al., "Conditional Image Generation with PixelCNN Decoders"?
  (2016), <https://arxiv.org/abs/1606.05328>.

- 提出自回归 PixelCNN 的论文：Aaron van den Oord 等，《Conditional Image Generation with PixelCNN Decoders》（2016），<https://arxiv.org/abs/1606.05328>。

- The paper introducing the popular Stable Diffusion latent diffusion
  model: Robin Rombach et al., "High-Resolution Image Synthesis with
  Latent Diffusion Models"? (2021), <https://arxiv.org/abs/2112.10752>.

- 介绍广泛使用的 Stable Diffusion（潜空间扩散）的论文：Robin Rombach 等，《High-Resolution Image Synthesis with Latent Diffusion Models》（2021），<https://arxiv.org/abs/2112.10752>。

- The Stable Diffusion code implementation:
  [*https://github.com/Comp*](https://github.com/CompVis/stable-diffusion)
  [*Vis/stable-diffusion*](https://github.com/CompVis/stable-diffusion).

- Stable Diffusion 代码实现：[*https://github.com/Comp*](https://github.com/CompVis/stable-diffusion) [*Vis/stable-diffusion*](https://github.com/CompVis/stable-diffusion)。

- The paper originally proposing consistency models: Yang Song et al.,
  "Consistency Models"? (2023), <https://arxiv.org/abs/2303.01469>.

- 最初提出一致性模型的论文：Yang Song 等，《Consistency Models》（2023），<https://arxiv.org/abs/2303.01469>。


------------------------------------------------------------------------
