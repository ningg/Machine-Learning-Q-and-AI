







# Chapter 5: Reducing Overfitting with Data
> 本章围绕如何通过获取更多或改造训练数据来减轻神经网络监督学习中的过拟合现象展开说明。
[](#chapter-5-reducing-overfitting-with-data)



**Suppose we train a neural network classifier in a supervised fashion
and notice that it suffers from overfitting. What are some of the common
ways to reduce overfitting in neural networks through the use of altered
or additional data?**

假设我们用监督方式训练神经网络分类器时发现出现过拟合，那么在仅靠改动或补充数据的前提下，有哪些常用手段可以减轻这种现象？

*Overfitting*, a common problem in machine learning, occurs when a model
fits the training data too closely, learning its noise and outliers
rather than the underlying pattern. As a result, the model performs well
on the training data but poorly on unseen or test data. While it is
ideal to prevent overfitting, it's often not possible to completely
eliminate it. Instead, we aim to reduce or minimize overfitting as much
as possible.

过拟合是机器学习中的常见问题：模型与训练数据贴得过紧，把噪声和异常也学进去，而不是学到背后的规律，因而在训练集上表现好、在未见过或测试数据上表现差。尽管理想情况是彻底防止过拟合，但往往无法完全消除，只能尽量减轻。

The most successful techniques for reducing overfitting revolve around
collecting more high-quality labeled data. However, if collecting more
labeled data is not feasible, we can `augment` the existing data or
leverage unlabeled data for `pretraining`.

减轻过拟合最成功的做法仍是收集更多高质量标注数据；若做不到，则可以对现有数据做 `augment`（增强），或利用无标注数据做 `pretraining`（预训练）。

> Tips: **减少**`过拟合`，最有效的技术是收集更多`高质量的标签数据`；此外，还可以使用`数据增强`和`预训练`等技术。

## Common Methods
> 本节将数据集相关的经典抗过拟合手段归为获取更多数据、数据增强与预训练几类并加以概述。
[](#common-methods)

This chapter summarizes the most prominent examples of dataset-related
techniques that have stood the test of time, grouping them into the
following categories: `collecting more data`, `data augmentation`, and
`pretraining`.

本章归纳经受住时间考验的数据集层面技术，并把它们归为：`collecting more data`（收集更多数据）、`data augmentation`（数据增强）和 `pretraining`（预训练）。

### Collecting More Data
> 本节说明为何扩充训练数据能改善泛化，以及如何通过学习曲线判断模型是否还能从更多数据中获益。
[](#collecting-more-data)

One of the best ways to reduce overfitting is to collect more
(good-quality) data. We can plot learning curves to find out whether a
given model would benefit from more data. To construct a learning curve,
we train the model to different training set sizes (10 percent, 20
percent, and so on) and evaluate the trained model on the same
fixed-size validation or test set. As shown in
Figure [5.1](#fig-ch05-fig01), the validation accuracy increases as the training
set sizes increase. This indicates that we can improve the model's
performance by collecting more data.

减轻过拟合的途径之一是收集更多（高质量）数据。可通过绘制学习曲线判断模型是否还能从更多数据中获益：在不同训练集比例（10%、20% 等）下训练同一模型，并在固定大小的验证集或测试集上评估；如图 [5.1](#fig-ch05-fig01) 所示，验证准确率随训练规模增大而上升，说明收集更多数据有望提升性能。

<a id="fig-ch05-fig01"></a>

<div align="center">
  <img src="./images/ch05-fig01.png" alt="The learning curve plot of a model fit to different training\set sizes" width="78%" />
  <div><b>Figure 5.1</b></div>
</div>

The gap between training and validation performance indicates the degree
of overfitting--the more extensive the gap, the more overfitting
occurs. Conversely, the slope indicating an improvement in the
validation performance suggests the model is `underfitting` and can
benefit from more data. Typically, **additional data** can decrease both
`underfitting` and `overfitting`.

训练与验证表现的差距反映过拟合程度：差距越大过拟合越严重；反之，验证性能仍有改善空间则说明模型处于 `underfitting`（欠拟合），还能从更多数据中获益。一般而言，**additional data**（额外数据）可同时缓解 `underfitting` 与 `overfitting`。

### Data Augmentation
> 本节介绍数据增强的含义、图像领域常见做法，以及与合成数据生成的关系。
[](#data-augmentation)

`Data augmentation` refers to generating new data records or features
based on existing data. It allows for the expansion of a dataset without
additional data collection.

`Data augmentation`（数据增强）指在现有数据基础上生成新的样本或特征，从而在不另行采集的情况下扩大数据集规模。

> Tips: **数据增强** `data augmentation`，是一种常用的技术，用于增加数据集的大小和多样性。它通过`生成新的数据`，来扩展数据集，而不需要额外的数据收集。

`Data augmentation` allows us to create different versions of the original
input data, which can improve the model's generalization performance.
Why? Augmented data can help the model improve its ability to
generalize, since it makes it harder to memorize spurious information
via training examples or features--or, in the case of image data, exact
pixel values for specific pixel locations.

数据增强能基于原始输入构造多种变体，从而提升泛化；因为增强后的数据让模型更难靠死记硬背训练样本、特征或（对图像而言）特定位置的像素值来记住虚假规律。

> Tips: 数据增强 `data augmentation`，可以**提高模型的泛化性能**。为什么？因为增强后的数据，使模型忽略`虚假信息`，典型场景：在`图像数据`中，会弱化`特定像素`的`像素值`。

Figure [5.2](#fig-ch05-fig02) highlights common image data augmentation
techniques, including `increasing brightness`, `flipping`, and `cropping`.

图 [5.2](#fig-ch05-fig02) 展示了常见的图像数据增强手段，包括 `increasing brightness`（提高亮度）、`flipping`（翻转）和 `cropping`（裁剪）。

<a id="fig-ch05-fig02"></a>

<div align="center">
  <img src="./images/ch05-fig02.png" alt="A selection of different image data augmentation techniques" width="78%" />
  <div><b>Figure 5.2</b></div>
</div>

Data augmentation is usually standard for image data (see
Figure [5.2](#fig-ch05-fig02)) and text data (discussed further in
Chapter [\[ch15\]](./ch15/_books_ml-q-and-ai-ch15.md),
but data augmentation methods for tabular data also exist.

对图像数据（见图 [5.2](#fig-ch05-fig02)）和文本数据（第十五章将进一步论述）而言，数据增强常常是标配；表格数据也存在相应增强方法。


Instead of collecting more data or augmenting existing data, it is also
possible to generate new, `synthetic data`. While more common for image
data and text, generating synthetic data is also possible for tabular
datasets.

除采集更多数据或增强现有数据外，还可生成新的 `synthetic data`（合成数据）；这在图像与文本上更常见，但表格数据集同样可以合成。

> Tips:数据增强 `data augmentation`，是图像数据和文本数据的标准技术。 除了`数据增强`，还可以`生成合成数据`。

### Pretraining
> 本节说明如何利用大规模无标注数据的自监督预训练，以及迁移学习与少样本学习在缓解小数据集过拟合中的作用。
[](#pretraining)

As discussed in Chapter [\[ch02\]](./ch02/_books_ml-q-and-ai-ch02.md), self-supervised learning lets us leverage large,
unlabeled datasets to pretrain neural networks. This can also help
reduce overfitting on the smaller target datasets.

如第二章所述，自监督学习可利用大规模无标注数据预训练神经网络，从而在较小的目标任务数据上减轻过拟合。

As an alternative to `self-supervised learning`, `traditional transfer learning` 
on large labeled datasets is also an option. Transfer learning
is most effective if the labeled dataset is closely related to the
target domain. For instance, if we train a model to classify bird
species, we can pretrain a network on a large, general animal
classification dataset. However, if such a large animal classification
dataset is unavailable, we can also pretrain the model on the relatively
broad ImageNet dataset.

除 `self-supervised learning`（自监督学习）外，也可在大规模**有标注**数据上做 `traditional transfer learning`（传统迁移学习）；源域与目标域越接近效果越好。例如训练鸟类分类器时，可先在大规模通用动物分类数据上预训练；若没有这类数据，也可用覆盖面更广的 ImageNet 预训练。

A dataset may be extremely small and unsuitable for supervised
learning--for example, if it contains only a handful of labeled
examples per class. If our classifier needs to operate in a context
where the collection of additional labeled data is not feasible, we may
also consider `few-shot learning`.

某些数据集可能极小，以至于不适合监督学习——例如每类只有屈指可数的标注样本；若分类器必须在难以继续获取标注的场景下运行，也可考虑 `few-shot learning`（少样本学习）。

## Other Methods
> 本节列举若干其他与数据或训练设置相关的抗过拟合思路，并预告下一章将从模型角度继续讨论。
[](#other-methods)

The previous sections covered the main approaches to using and modifying
datasets to reduce overfitting. However, this is not an exhaustive list.
Other common techniques include:

前文涵盖了利用与改造数据集缓解过拟合的主要路径，但并非穷举。其他常见手段包括：

- Feature engineering and normalization

特征工程与特征归一化。

- The inclusion of adversarial examples and label or feature noise

引入对抗样本以及标签或特征噪声。

- Label smoothing

标签平滑（label smoothing）。

- Smaller batch sizes

更小的批量大小。

- Data augmentation techniques such as Mixup, Cutout, and CutMix

Mixup、Cutout、CutMix 等数据增强类方法。

> Tips: 减弱`过度拟合`，还可以使用下述技术：
> 
> - `特征工程`和`归一化`：改进特征选择和标准化数据
> - `对抗样本`和`标签或特征噪声`：添加对抗样本或噪声来增强模型的鲁棒性
> - `标签平滑`：软化标签，避免模型对训练标签过于自信
> - `更小的批量大小`：使用较小的batch size来增加训练的随机性
> - `数据增强`技术，如`Mixup`、`Cutout`和`CutMix`
>     - `Mixup`：混合不同样本的数据
>     - `Cutout`：随机遮挡图像的部分区域
>     - `CutMix`：将一张图像的一部分替换为另一张图像的对应部分


The next chapter covers additional techniques to reduce overfitting from
a model perspective, and it concludes by discussing which regularization
techniques we should consider in practice.

下一章将从模型与训练角度补充更多抗过拟合技术，并讨论实践中应如何选择正则化策略。

## Exercises
> 本章习题围绕迁移学习可行性以及不当增强导致精度下降的原因展开思考。
[](#exercises)

5-1. Suppose we train an XGBoost model to classify images based on
manually extracted features obtained from collaborators. The dataset of
labeled training examples is relatively small, but fortunately, our
collaborators also have a labeled training set from an older project on
a related domain. We're considering implementing a transfer learning
approach to train the XGBoost model. Is this a feasible option? If so,
how could we do it? (Assume we are allowed to use only XGBoost and not
another classification algorithm or model.)

习题 5-1：假设我们用合作方提供的手工特征训练 XGBoost 做图像分类，当前标注训练集较小，但对方在相近领域还有一个旧项目的标注集；我们想在仅用 XGBoost、不能用其他算法的前提下做迁移学习，这是否可行？若可行应如何做？

5-2. Suppose we're working on the image classification problem of
implementing MNIST-based handwritten digit recognition. We've added a
decent amount of data augmentation to try to reduce overfitting.
Unfortunately, we find that the classification accuracy is much worse
than it was before the augmentation. What are some potential reasons for
this?

习题 5-2：在 MNIST 手写数字识别任务中，我们加入了较多数据增强以期减轻过拟合，却发现准确率反而明显下降，可能有哪些原因？

## References
> 本节列出表格数据增强、合成数据、预处理、噪声标签、批量与学习率、对抗训练、标签平滑与 Mixup 等相关文献与资料链接。
[](#references)

- Apaperondataaugmentationfortabulardata:DerekSnow, "DeltaPy: A
  Framework for Tabular Data Augmentation in Py-  thon"? (2020),
  <https://github.com/firmai/deltapy>.

Snow 等人关于表格数据增强框架 DeltaPy 的介绍（2020），仓库见链接。

- The paper proposing the GReaT method for generating synthetic tabular
  data using an auto-regressive generative large language model: Vadim
  Borisov et al., "Language Models Are Realistic Tabular Data
  Generators"? (2022), <https://arxiv.org/abs/2210.06280>.

Borisov 等提出用自回归生成式大语言模型合成表格数据的 GReaT 方法（2022）。

- ThepaperproposingtheTabDDPMmethodforgeneratingsynthetictabulardatausingadiffusionmodel:AkimKotelnikovetal.,"TabDDPM:
  Modelling Tabular Data with Diffusion Models"? (2022),
  <https://arxiv.org/abs/2209.15421>.

Kotelnikov 等提出用扩散模型生成合成表格数据的 TabDDPM（2022）。

- Scikit-learn's user guide offers a section on preprocessing data,
  featuring techniques like feature scaling and normalization that can
  enhance your model's performance:
  <https://scikit-learn.org/stable/modules/preprocessing.html>.

Scikit-learn 用户指南中的数据预处理章节，涵盖特征缩放与归一化等有助于模型性能的技术。

- A survey on methods for robustly training deep models with noisy
  labels that explores techniques to mitigate the impact of incorrect or
  misleading target values: Bo Han et al., "A Survey of Label-noise
  Representation Learning: Past, Present and Future"? (2020),
  <https://arxiv.org/abs/2011.04406>.

Han 等关于在噪声标签下稳健训练深度模型的综述（2020）。

- Theoretical and empirical evidence to support the idea that control-
   ling the ratio of batch size to learning rate in stochastic gradient
  descent is crucial for good modeling performance in deep neural
  networks: Fengxiang He, Tongliang Liu, and Dacheng Tao, "Control
  Batch Size and Learning Rate to Generalize Well: Theoretical and
  Empirical Evidence"? (2019),
  <https://dl.acm.org/doi/abs/10.5555/3454287.3454390>.

He、Liu、Tao 关于随机梯度下降中批量大小与学习率比例对深度网络泛化至关重要的理论与实证研究（2019）。

- Inclusion of adversarial examples, which are input samples designed to
  mislead the model, can improve prediction performance by making the
  model more robust: Cihang Xie et al., "Adversarial Examples Improve
  Image Recognition"? (2019), <https://arxiv.org/abs/1911.09665>.

Xie 等表明引入旨在误导模型的对抗样本可通过增强鲁棒性改善预测性能（2019）。

- Label smoothing is a regularization technique that mitigates the im-
   pact of potentially incorrect labels in the dataset by replacing
  hard 0 and 1 classification targets with softened values: Rafael
  MÃ¼ller, Simon Kornblith, and Geoffrey Hinton, "When Does Label
  Smoothing Help?"? (2019), <https://arxiv.org/abs/1906.02629>.

MÃ¼ller、Kornblith、Hinton 提出标签平滑：用软化后的目标替代硬 0/1 标签以减轻错误标注影响（2019）。

- Mixup, a popular method that trains neural networks on blended data
  pairs to improve generalization and robustness: Hongyi Zhang et al.,
  "Mixup: Beyond Empirical Risk Minimization"? (2018),
  <https://arxiv.org/abs/1710.09412>.

Zhang 等提出的 Mixup：在成对混合样本上训练网络以提升泛化与鲁棒性（2018）。


------------------------------------------------------------------------

