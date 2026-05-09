







# Chapter 30: Limited Labeled Data
> 本章系统梳理监督学习中标注数据稀缺时的主要路径（增标、自助式扩充、迁移与自监督、主动学习、小样本与元学习、弱/半监督、自训练、多任务与多模态、归纳偏置等），并给出选用建议、练习与文献。
[](#chapter-30-limited-labeled-data)



**Suppose we plot a learning curve (as shown in
Figure [5.1](./ch05/_books_ml-q-and-ai-ch05.md#fig-ch05-fig01) on page , for example) and find the machine
learning model overfits and could benefit from more training data. What
are some different approaches for dealing with limited labeled data in
supervised machine learning settings?**

若学习曲线（例如 Figure [5.1](./ch05/_books_ml-q-and-ai-ch05.md#fig-ch05-fig01)）显示模型过拟合、仍可从更多数据中受益，在监督学习设定下应对「标注有限」有哪些不同思路？

> 学习曲线（Learning Curve）是机器学习中用于评估模型性能随训练数据量变化趋势的图表。它通常用于诊断模型是否存在过拟合或欠拟合问题。

In lieu of collecting more data, there are several methods related to
regular supervised learning that we can use to improve model performance
in limited labeled data regimes.

在无法立即收集更多数据时，仍有一类贴近常规监督学习的方法，用于在标注稀缺时提升模型表现。

> 除了收集更多数据，还有几种方法，用于改进标**签数据有限**时的模型性能。


## Improving Model Performance with Limited Labeled Data
> 本节作为总领，引出后续各子节：在训练数据有限时，多种学习范式如何提供帮助。
[](#improving-model-performance-with-limited-labeled-data)

The following sections explore various machine learning paradigms that
help in scenarios where training data is limited.

下文将分节讨论在训练数据有限时有所帮助的各类机器学习范式。

### Labeling More Data
> 本节讨论通过收集更多标注样本直接提升效果，以及其在成本、算力与可及性上的现实制约。
[](#labeling-more-data)

Collecting additional training examples is often the best way to improve
the performance of a model (a learning curve is a good diagnostic for
this). However, this is often not feasible in practice, because
acquiring high-quality data can be costly, computational resources and
storage might be insufficient, or the data may be hard to access.

收集更多带标签样本往往是提升模型性能的首选（学习曲线对此是很好的诊断工具）。但实践中常受限于高质量数据昂贵、算力与存储不足，或数据难以获取。

> 收集更多训练数据，通常是提高模型性能的最佳方法（学习曲线是诊断此问题的一个很好的指标）。
> 
> 然而，这在实践中通常不可行，因为获取高质量数据可能很昂贵，计算资源和存储空间可能不足，或者数据可能难以获取。

### Bootstrapping the Data
> 本节联系第 5、21 章，说明用增强或合成样本「扩充」数据以及提升数据质量的路径。
[](#bootstrapping-the-data)

Similar to the techniques for reducing overfitting discussed in
Chapter [\[ch05\]](./ch05/_books_ml-q-and-ai-ch05.md), it
can be helpful to "bootstrap" the data by generating modified
(augmented) or artificial (synthetic) training examples to boost the
performance of the predictive model. Of course, improving the quality of
data can also lead to the improved predictive performance of a model, as
discussed in Chapter [\[ch21\]](./ch21/_books_ml-q-and-ai-ch21.md).

与第 5 章中讨论的缓解过拟合技术类似，可通过生成经修改（增强）或人工（合成）的训练样本来「扩充」数据，以提升预测模型表现。当然，提升数据质量同样能改善预测性能，见第 21 章。

> 与第 5 章讨论的减少过拟合的技术类似，可以通过生成修改（增强）或人工（合成）训练示例来“引导”数据，以提高预测模型的性能。
> 
> 当然，提高数据质量也可以提高模型的预测性能，如第 21 章所述。

### Transfer Learning
> 本节说明在大规模通用数据上预训练再在目标任务上微调的基本流程，并对比深度网络与树模型在可更新参数上的差异。
[](#transfer-learning)

Transfer learning describes training a model on a general dataset (for example,
ImageNet) and then fine-tuning the pretrained target dataset (for
example, a dataset consisting of different bird species), as outlined in
Figure [30.1](#fig-ch30-fig01).

迁移学习指先在大规模通用数据（如 ImageNet）上训练，再在目标任务数据（如某鸟类细粒度数据集）上微调预训练模型；流程见 Figure [30.1](#fig-ch30-fig01)。

> 迁移学习描述了在通用数据集（例如 ImageNet）上训练模型，然后对预训练的目标数据集（例如包含不同鸟类物种的数据集）进行微调，如图 1.1 所示。

<a id="fig-ch30-fig01"></a>

<div align="center">
  <img src="./images/ch30-fig01.png" alt="The process of transfer learning" width="52%" />
  <div><b>Figure 30.1</b></div>
</div>

Transfer learning is usually done in the context of deep learning, where
model weights can be updated. This is in contrast to tree-based methods,
since most decision tree algorithms are nonparametric models that do not
support iterative training or parameter updates.

迁移学习通常出现在深度学习中，以便迭代更新权重。相对地，多数树模型为非参数、难以像神经网络那样不断迭代更新参数。

> 迁移学习，通常用于深度学习场景，其中可以更新模型权重。这与基于树的方法形成对比，因为大多数决策树算法是非参数模型，不支持迭代训练或参数更新。

### Self-Supervised Learning
> 本节说明自监督如何从数据本身构造辅助任务，及其与「无监督预训练」称谓的关系，并给出语言与视觉上的典型例子及章节引用。
[](#self-supervised-learning)

Similar to transfer learning, in self-supervised learning, the model is
pretrained on a different task before being fine-tuned to a target task
for which only limited data exists. However, self-supervised learning
usually relies on label information that can be directly and
automatically extracted from unlabeled data. Hence, self-supervised
learning is also often called **unsupervised pretraining**.

与迁移学习类似，自监督学习先在另一任务上预训练，再对仅有少量标注的目标任务微调。不同之处在于辅助标签通常可直接从无标注数据中自动构造，因此也常被称为**无监督预训练**。

> 与迁移学习类似，在自监督学习中，模型在不同的任务上进行预训练，然后针对目标任务进行微调，而目标任务只有有限的数据。
> 
> 然而，自监督学习通常依赖于可以直接从无标签数据中自动提取的标签信息。因此，自监督学习也经常被称为**无监督预训练**。

Common examples of self-supervised learning include the *next word*
(used in GPT, for example) or *masked word* (used in BERT, for example)
pretraining tasks in language modeling, covered in more detail in
Chapter [\[ch17\]](./ch17/_books_ml-q-and-ai-ch17.md).
Another intuitive example from computer vision includes *inpainting*:
predicting the missing part of an image that was randomly removed,
illustrated in Figure [30.2](#fig-ch30-fig02).

自监督常见例包括语言模型中的 *next word*（如 GPT）或 *masked word*（如 BERT）预训练任务，详见第 17 章；在视觉上，*inpainting*（预测被随机挖去的图像区域）亦很典型，见 Figure [30.2](#fig-ch30-fig02)。

<a id="fig-ch30-fig02"></a>

<div align="center">
  <img src="./images/ch30-fig02.png" alt="Inpainting for self-supervised learning" width="52%" />
  <div><b>Figure 30.2</b></div>
</div>

For more detail on self-supervised learning, see
Chapter [\[ch02\]](./ch02/_books_ml-q-and-ai-ch02.md).

更多自监督学习背景见第 2 章。

### Active Learning
> 本节说明主动学习如何通过优先级策略挑选待标注点，并借助 oracle 在有限标注预算下提升模型。
[](#active-learning)

In active learning, illustrated in
Figure [30.3](#fig-ch30-fig03), we typically involve manual labelers or users for
feedback during the learning process. However, instead of labeling the
entire dataset up front, active learning includes a prioritization
scheme for suggesting unlabeled data points for labeling to maximize the
machine learning model's performance.

在主动学习（Figure [30.3](#fig-ch30-fig03)）中，训练过程会引入标注者或用户反馈；但并非一次性标全量数据，而是按优先级挑选最值得标注的未标注点，以最大化模型性能。

> 在主动学习中，如图 1.3 所示，我们通常涉及手动标签器或用户在训练过程中提供反馈。
> 
> 然而，与提前标记整个数据集不同，主动学习包括一个优先级方案，用于建议未标记的数据点进行标记，以最大化机器学习模型的性能。

<a id="fig-ch30-fig03"></a>

<div align="center">
  <img src="./images/ch30-fig03.png" alt="In active learning, a model queries an oracle for labels." width="52%" />
  <div><b>Figure 30.3</b></div>
</div>

The term **active learning** refers to the fact that the model actively
selects data for labeling. For example, the simplest form of active
learning selects data points with high prediction uncertainty for
labeling by a human annotator (also referred to as an *oracle*).

**主动学习**强调模型主动挑选待标注数据。例如最简策略是优先标注模型预测不确定度高的样本，由人工标注者（亦称 *oracle*）完成标注。

### Few-Shot Learning
> 本节概述 few-shot、zero-shot 及基于提示的大模型示例，并指向第 3 章延伸阅读。
[](#few-shot-learning)

In a `few-shot` learning scenario, we often deal with extremely small
datasets that include only a handful of examples per class. In research
contexts, 1-shot(one example per class) and 5-shot (five examples per
class) learning scenarios are very common. An extreme case of few-shot
learning is `zero-shot` learning, where no labels are provided. Popular
examples of zero-shot learning include GPT-3 and related language
models, where the user has to provide all the necessary information via
the input prompt, as illustrated in
Figure [30.4](#fig-ch30-fig04).

在 `few-shot` 设置下，每类往往只有极少样本；研究中常见 1-shot、5-shot。`zero-shot` 更进一步，不提供目标任务标注，仅靠提示词与模型先验；GPT-3 等即是一例，见 Figure [30.4](#fig-ch30-fig04)。

> 在少样本学习场景中，我们通常处理包含每个类别只有少量示例的极端情况。
> 
> 在研究上下文中，1-shot（每个类别一个示例）和5-shot（每个类别五个示例）学习场景非常常见。
> 
> 小样本学习的极端情况，是**零样本学习**，其中没有提供标签。
> 
> 零样本学习的流行示例包括 GPT-3 和相关语言模型，其中用户必须通过输入提示提供所有必要信息，如图 1.4 所示。

<a id="fig-ch30-fig04"></a>

Zero-shot classification with ChatGPT

For more detail on few-shot learning, see
Chapter [\[ch03\]](./ch03/_books_ml-q-and-ai-ch03.md).

少样本学习的更多细节见第 3 章。

### Meta-Learning
> 本节区分「学会学习」式的元学习与从数据集提取 meta-features 的另一分支，并说明其与少样本学习及特征表示的关系。
[](#meta-learning)

Meta-learning involves developing methods that determine how machine
learning algorithms can best learn from data. We can therefore think of
meta-learning as "learning to learn."? The machine learning community
has developed several approaches for meta-learning. Within the machine
learning community, the term *meta-learning* doesn't just represent
multiple subcategories and approaches; it is also occasionally employed
to describe related yet distinct processes, leading to nuances in its
interpretation and application.

元学习（meta-learning）研究如何设计方法，使机器学习算法更有效地从数据中学习，可概括为「学会学习」。社区内该术语涵盖多类子方向，有时也用于相近却不尽相同的流程，因而解读上存在细微差别。

Meta-learning is one of the main subcategories of few-shot learning.
Here, the focus is on learning a good feature extraction module, which
converts support and query images into vector representations. These
vector representations are optimized for determining the predicted class
of the query example via comparisons with the training examples in the
support set. (This form of meta-learning is illustrated in
Chapter [\[ch03\]](./ch03/_books_ml-q-and-ai-ch03.md) on
page .) Another branch of meta-learning unrelated to the few-shot
learning approach is focused on extracting metadata (also called
*meta-features*) from datasets for supervised learning tasks, as
illustrated in Figure [30.5](#fig-ch30-fig05). The meta-features are descriptions of the dataset
itself. For example, these can include the number of features and
statistics of the different features (kurtosis, range, mean, and so on).

元学习是少样本学习的主要子类之一：此处重点学习良好的特征提取模块，把 support/query 图像映射为向量，并通过与支持集样本比较来预测 query 类别（见第 3 章）。另一支与少样本路线无关的元学习则关注从数据集中抽取 *meta-features* 以辅助监督学习任务，见 Figure [30.5](#fig-ch30-fig05)；meta-features 描述数据集自身，例如特征个数与各特征的峰度、范围、均值等统计量。

<a id="fig-ch30-fig05"></a>

<div align="center">
  <img src="./images/ch30-fig05.png" alt="The meta-learning process involving the extraction of metadata" width="78%" />
  <div><b>Figure 30.5</b></div>
</div>

The extracted meta-features provide information for selecting a machine
learning algorithm for the dataset at hand. Using this approach, we can
narrow down the algorithm and hyperparameter search spaces, which helps
reduce overfitting when the dataset is small.

抽取的 meta-features 可为当前数据集选择合适算法提供依据，从而缩小算法与超参搜索空间，在小数据上也有助于抑制过拟合。

### Weakly Supervised Learning
> 本节说明弱监督如何借助外部、往往含噪的标注函数为无标签数据生成标签，并引出 PU-learning 子类。
[](#weakly-supervised-learning)

Weakly supervised learning, illustrated in
Figure [30.6](#fig-ch30-fig06), involves using an external label source to
generate labels for an unlabeled dataset. Often, the labels created by a
weakly supervised labeling function are more noisy or inaccurate than
those produced by a human or domain expert, hence the term *weakly*
supervised. We can develop or adopt a rule-based classifier to create
the labels in weakly supervised learning; these rules usually cover only
a subset of the unlabeled dataset.

弱监督学习（Figure [30.6](#fig-ch30-fig06)）用外部标签源为无标注数据生成标签；此类标签往往比人工或专家标注更噪或不完全，故称为 *weakly* supervised。可用规则分类器覆盖无标签数据的子集来产生标签。

<a id="fig-ch30-fig06"></a>

<div align="center">
  <img src="./images/ch30-fig06.png" alt="Weakly supervised learning uses external labeling functions to train machine learning models." width="78%" />
  <div><b>Figure 30.6</b></div>
</div>

Let'sreturntotheexampleofemailspamclassificationfromChapter [\[ch23\]](./ch23/_books_ml-q-and-ai-ch23.md) to illustrate a rule-based approach for data
labeling. In weak supervision, we could design a rule-based classifier
based on the keyword *SALE* in the email subject header line to identify
a subset of spam emails. Note that while we may use this rule to label
certain emails as spam positive, we should not apply this rule to label
emails without *SALE* as non-spam. Instead, we should either leave those
unlabeled or apply a different rule to them.

回到第 23 章的垃圾邮件例子：可用主题含 *SALE* 的规则把一部分邮件标为垃圾邮件；但不能据此把不含 *SALE* 的邮件一律标为非垃圾，而应留空或换用其他规则。

There is a subcategory of weakly supervised learning referred to as
PU-learning. In *PU-learning*, which is short for *positive-unlabeled
learning*, we label and learn only from positive examples.

弱监督的一个子类是 PU-learning（*positive-unlabeled learning*）：仅在正例与未标注数据上学习与标注。

### Semi-Supervised Learning
> 本节对比半监督与弱监督在「如何造标签」上的差异，并说明二者可组合使用的情形。
[](#semi-supervised-learning)

Semi-supervised learning is closely related to weakly supervised
learning: it also involves creating labels for unlabeled instances in
the dataset. The main difference between these two methods lies in *how*
we create the labels. In weak supervision, we create labels using an
external labeling function that is often noisy, inaccurate, or covers
only a subset of the data. In semi-supervision, we do not use an
external label function; instead, we leverage the structure of the data
itself. We can, for example, label additional data points based on the
density of neighboring labeled data points, as illustrated in
Figure [30.7](#fig-ch30-fig07).

半监督与弱监督都涉及为未标注样本造标签，差别在于*如何*造：弱监督依赖外部、常含噪且未必全覆盖的标注函数；半监督则利用数据自身结构（例如依据已标注邻域的密度）为更多点赋伪标签，见 Figure [30.7](#fig-ch30-fig07)。

<a id="fig-ch30-fig07"></a>

<div align="center">
  <img src="./images/ch30-fig07.png" alt="Semi-supervised learning" width="78%" />
  <div><b>Figure 30.7</b></div>
</div>

While we can apply weak supervision to an entirely unlabeled dataset,
semi-supervised learning requires at least a portion of the data to be
labeled. In practice, it is possible first to apply weak supervision to
label a subset of the data and then to use semi-supervised learning to
label instances that were not captured by the labeling functions.

弱监督可施于完全无标签数据；半监督则至少需要一部分已标注样本。实践中可先弱监督标出一部分，再对规则未覆盖的样本用半监督补标。

Thanks to their close relationship, semi-supervised learning is
sometimes referred to as a subcategory of weakly supervised learning,
and vice versa.

二者关系密切，文献中有时把半监督视为弱监督的子类，反之亦然。

### Self-Training
> 本节说明自训练如何用（伪）模型为数据打伪标签，及其与弱监督、半监督的联系，并关联知识蒸馏。
[](#self-training)

Self-training falls somewhere between semi-supervised learning and
weakly supervised learning. For this technique, we train a model to
label the dataset or adopt an existing model to do the same. This model
is also referred to as a *pseudo-labeler*.

自训练介于半监督与弱监督之间：先训练或选用一个模型为数据集打伪标签，该模型亦称 *pseudo-labeler*。

Self-training does not guarantee accurate labels and is thus related to
weakly supervised learning. Moreover, while we use or adopt a machine
learning model for this pseudo-labeling, self-training is also related
to semi-supervised learning.

伪标签未必准确，因而与弱监督相关；同时又借助机器学习模型生成标签，故也与半监督相近。

An example of self-training is knowledge distillation, discussed in
Chapter [\[ch06\]](./ch06/_books_ml-q-and-ai-ch06.md).

自训练的一例是第 6 章讨论的知识蒸馏。

### Multi-Task Learning
> 本节介绍多任务学习中的辅助任务、硬/软参数共享及其作为归纳偏置的作用。
[](#multi-task-learning)

Multi-task learning trains neural networks on multiple, ideally related
tasks. For example, if we are training a classifier to detect spam
emails, spam classification is the main task. In multi-task learning, we
can add one or more related tasks for the model to solve, referred to as
*auxiliary tasks*. For the spam email example, an auxiliary task could
be classifying the email's topic or language.

多任务学习在多个（最好相关的）任务上联合训练网络。例如垃圾邮件检测为主任务时，可把邮件主题或语种分类等作为 *auxiliary tasks*（辅助任务）。

Typically, multi-task learning is implemented via multiple loss
functions that have to be optimized simultaneously, with one loss
function for each task. The auxiliary tasks serve as an inductive bias,
guiding the model to prioritize hypotheses that can explain multiple
tasks. This approach often results in models that perform better on
unseen data. There are two subcategories of multi-task learning:
multi-task learning with hard parameter sharing and multi-task learning
with soft parameter sharing.
Figure [30.8](#fig-ch30-fig08) illustrates the difference between these two
methods.

实现上通常为每个任务各设一个损失并同时优化；辅助任务提供归纳偏置，使模型偏好能同时解释多任务的假设，常改善未见数据上的表现。多任务又分硬参数共享与软参数共享，见 Figure [30.8](#fig-ch30-fig08)。

<a id="fig-ch30-fig08"></a>

<div align="center">
  <img src="./images/ch30-fig08.png" alt="Multi-task learning: hard vs soft parameter sharing" width="78%" />
  <div><b>Figure 30.8</b></div>
</div>

In *hard* parameter sharing, as shown in
Figure [30.8](#fig-ch30-fig08), only the output layers are task specific, while
all the tasks share the same hidden layers and neural network backbone
architecture. In contrast, *soft* parameter sharing uses separate neural
networks for each task, but regularization techniques such as distance
minimization between parameter layers are applied to encourage
similarity among the networks.

*Hard* 参数共享中各任务共享隐藏层与骨干，仅输出头任务相关；*soft* 参数共享则为每任务各用一套网络，并通过层间距离正则等鼓励参数相似。

### Multimodal Learning
> 本节区分多任务与多模态输入，介绍匹配损失与联合编码器/单模块（如 VideoBERT）等做法，并讨论直接优化下游目标的情形。
[](#multimodal-learning)

While multi-task learning involves training a model with multiple tasks
and loss functions, multimodal learning focuses on incorporating
multiple types of input data.

多任务强调多损失、多任务；多模态则强调融合多种输入模态。

Common examples of multimodal learning are architectures that take both
image and text data as input (though multimodal learning is not
restricted to only two modalities and can be used for any number of
input modalities). Depending on the task, we may employ a matching loss
that forces the embedding vectors between related images and text to be
similar, as shown in
Figure [30.9](#fig-ch30-fig09). (See
Chapter [\[ch01\]](./ch01/_books_ml-q-and-ai-ch01.md) for
more on embedding vectors.)

常见做法是同时输入图像与文本（也可多于两种模态）；可用匹配损失拉近相关图文嵌入，见 Figure [30.9](#fig-ch30-fig09)（嵌入概念见第 1 章）。

<a id="fig-ch30-fig09"></a>

<div align="center">
  <img src="./images/ch30-fig09.png" alt="Multimodal learning with a matching loss" width="78%" />
  <div><b>Figure 30.9</b></div>
</div>

Figure [30.9](#fig-ch30-fig09) shows image and text encoders as separate
components. The image encoder can be a convolutional backbone or a
vision transformer, and the language encoder can be a recurrent neural
network or language transformer. However, it's common nowadays to use
a single transformer-based module that can simultaneously process image
and text data. For example, the VideoBERT model has a joint module that
processes both video and text for action classification and video
captioning.

图中将图像编码器与文本编码器分开；图像侧可用 CNN 或 ViT，语言侧可用 RNN 或语言 Transformer。如今更常见单一 Transformer 同时处理图文；VideoBERT 即用联合模块同时处理视频与文本以做动作分类与视频描述。

Optimizing a matching loss, as shown in
Figure [30.9](#fig-ch30-fig09), can be useful for learning embeddings that can be
applied to various tasks, such as image classification or summarization.
However, it is also possible to directly optimize the target loss, like
classification or regression, as
Figure [30.10](#fig-ch30-fig10) illustrates.

优化匹配损失可学到通用嵌入，用于分类、摘要等多种下游；也可如图 Figure [30.10](#fig-ch30-fig10) 所示，直接优化分类或回归等监督目标。

<a id="fig-ch30-fig10"></a>

<div align="center">
  <img src="./images/ch30-fig10.png" alt="Multimodal learning for optimizing a supervised learning objective" width="78%" />
  <div><b>Figure 30.10</b></div>
</div>

Figure [30.10](#fig-ch30-fig10) shows data being collected from two different
sensors. One could be a thermometer and the other could be a video
camera. The signal encoders convert the information into embeddings
(sharing the same number of dimensions), which are then concatenated to
form the input representation for the model.

Figure [30.10](#fig-ch30-fig10) 展示来自不同传感器（如温度计与摄像头）的数据：各模态经编码器映射到同维嵌入后拼接，作为模型输入表示。

Intuitively, models that combine data from different modalities
generally perform better than unimodal models because they can leverage
more information. Moreover, recent research suggests that the key to the
sucess of multimodal learning is the improved quality of the latent
space representation.

直觉上，融合多模态通常优于单模态，因为可利用更多信息；近期工作认为多模态成功的关键在于潜空间表示质量的提升。

### Inductive Biases
> 本节说明更强归纳偏置（如 CNN 相对 ViT）如何降低数据需求，并指向第 13 章。
[](#inductive-biases)

Choosing models with stronger inductive biases can help lower data
requirements by making assumptions about the structure of the data. For
example, due to their inductive biases, convolutional networks require
less data than vision transformers, as discussed in
Chapter [\[ch13\]](./ch13/_books_ml-q-and-ai-ch13.md).

选择归纳偏置更强的模型，可通过对数据结构的先验假设降低样本需求；例如 CNN 通常比 ViT 更省数据，见第 13 章。

## Recommendations
> 本节用决策图（Figure 30.11）概括在多种技术之间如何按情境取舍，并提醒与过拟合相关章节的衔接。
[](#recommendations)

Of all these techniques for reducing data requirements, how should we
decide which ones to use in a given situation?

在诸多降低数据需求的技术中，应如何按情境选择？

Techniques like collecting more data, data augmentation, and feature
engineering are compatible with all the methods discussed in this
chapter. Multi-task learning and multimodal inputs can also be used with
the learning strategies outlined here. If the model suffers from
overfitting, we should also include techniques discussed in
Chapters [\[ch05\]](./ch05/_books_ml-q-and-ai-ch05.md) and
[\[ch06\]](./ch06/_books_ml-q-and-ai-ch06.md).

收集更多数据、数据增强与特征工程可与本章各范式组合；多任务与多模态输入亦可并用。若仍存在过拟合，还应结合第 5、6 章的技术。

But how can we choose between active learning, few-shot learning,
transfer learning, self-supervised learning, semi-supervised learning,
and weakly supervised learning? Deciding which supervised learning
technique(s) to try is highly context dependent. You can use the diagram
in Figure [30.11](#fig-ch30-fig11) as a guide to choosing the best method for your
particular project.

在主动学习、少样本、迁移、自监督、半监督与弱监督之间如何选，高度依赖上下文；可用 Figure [30.11](#fig-ch30-fig11) 的流程图作项目内导航。

<a id="fig-ch30-fig11"></a>

<div align="center">
  <img src="./images/ch30-fig11.png" alt="Recommendations for choosing a supervised learning technique" width="78%" />
  <div><b>Figure 30.11</b></div>
</div>

Note that the dark boxes in
Figure [30.11](#fig-ch30-fig11) are not terminal nodes but arc back to the second
box, "Evaluate model performance"?; additional arrows were omitted to
avoid visual clutter.

注意 Figure [30.11](#fig-ch30-fig11) 中深色框并非终端，而是回连到「Evaluate model performance」等节点；为简洁省略了部分箭头。

## Exercises
> 本节为两道综合题：制造缺陷检测场景下自监督/迁移的用法，以及主动学习中在深度网络过自信时替代置信度的思路。
[](#exercises)

30-1. Suppose we are given the task of constructing a machine learning
model that utilizes images to detect manufacturing defects on the outer
shells of tablet devices similar to iPads. We have access to millions of
images of various computing devices, including smartphones, tablets, and
computers, which are not labeled; thousands of labeled pictures of
smartphones depicting various types of damage; and hundreds of labeled
images specifically related to the target task of detecting
manufacturing defects on tablet devices. How could we approach this
problem using self-supervised learning or transfer learning?

30-1. 假设要用图像检测类 iPad 平板外壳制造缺陷：有数百万未标注的各型设备图、数千张带标注的手机损伤图、以及数百张与平板缺陷直接相关的标注图。如何用自监督学习或迁移学习来建模？

30-2. In active learning, selecting difficult examples for human
inspection and labeling is often based on confidence scores. Neural
networks can provide such scores by using the logistic sigmoid or
softmax function in the output layer to calculate class-membership
probabilities. However, it is widely recognized that deep neural
networks exhibit overconfidence on out-of-distribution data, rendering
their use in active learning ineffective. What are some other methods to
obtain confidence scores using deep neural networks for active learning?

30-2. 主动学习常按置信度挑选难例；深度网络可用 sigmoid/softmax 给出类概率，但在分布外数据上往往过自信，使该策略失效。还有哪些为深度网络获取「置信度」以用于主动学习的方法？

## References
> 本节列出增量决策树、多任务学习、VideoBERT、多模态表示理论、主动学习综述与过自信相关论文链接。
[](#references)

- While decision trees for incremental learning are not commonly
  implemented, algorithms for training decision trees in an itera-
   tive fashion do exist:
  [*https://en.wikipedia.org/wiki/Incremental*](https://en.wikipedia.org/wiki/Incremental_decision_tree)
  [*\_decision_tree*](https://en.wikipedia.org/wiki/Incremental_decision_tree).

- 增量训练决策树的算法条目：<https://en.wikipedia.org/wiki/Incremental_decision_tree>。

- Models trained with multi-task learning often outperform models
  trained on a single task: Rich Caruana, "Multitask Learning"?
  (1997), <https://doi.org/10.1023%2FA%3A1007379606734>.

- 多任务学习经典论文 Caruana (1997)：<https://doi.org/10.1023%2FA%3A1007379606734>。

- A single transformer-based module that can simultaneously process
  image and text data: Chen Sun et al., "VideoBERT: A Joint Model for
  Video and Language Representation Learning"? (2019),
  <https://arxiv.org/abs/1904.01766>.

- VideoBERT 联合视频与文本 (2019)：<https://arxiv.org/abs/1904.01766>。

- The aforementioned research suggesting the key to the success of
  multimodal learning is the improved quality of the latent space
  representation: Yu Huang et al., "What Makes Multi-Modal Learning
  Better Than Single (Provably)"? (2021),
  <https://arxiv.org/abs/2106.04538>.

- 多模态优于单模态的表示视角 Huang 等 (2021)：<https://arxiv.org/abs/2106.04538>。

- For more information on active learning: Zhen et al., "A Comparative
  Survey of Deep Active Learning"? (2022),
  <https://arxiv.org/abs/2203.13450>.

- 深度主动学习综述 (2022)：<https://arxiv.org/abs/2203.13450>。

- For a more detailed discussion on how out-of-distribution data can
  lead to overconfidence in deep neural networks: Anh Nguyen, Jason
  Yosinski, and Jeff Clune, "Deep Neural Networks Are Easily Fooled:
  High Confidence Predictions for Unrecognizable Images"? (2014),
  <https://arxiv.org/abs/1412.1897>.

- 深度网络对不可识别图像给出高置信度 (2014)：<https://arxiv.org/abs/1412.1897>。


------------------------------------------------------------------------

