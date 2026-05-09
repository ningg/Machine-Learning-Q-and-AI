







# Chapter 3: Few-Shot Learning
> 本章介绍小样本学习（few-shot）与常规监督在数据组织上的差异，解释 N-way K-shot、支持集与 episode，并简述元学习与基于近邻的嵌入思路及练习。
[](#chapter-3-few-shot-learning)



**What is few-shot learning? How does it differ from the conventional
training procedure for supervised learning?**

什么是小样本学习？它与常规监督学习训练流程有何不同？

*Few-shot learning* is a type of supervised learning for small training
sets with a very small example-to-class ratio. In regular supervised
learning, we train models by iterating over a training set where the
model always sees a `fixed set` of classes. In few-shot learning, we are
working on a `support set` from which we create multiple training tasks to
assemble training episodes, where each training task consists of
different classes.

*Few-shot learning* 是在「每类样本极少」设定下的监督学习：常规监督在固定类别集上迭代整个训练集；小样本则从 `support set` 采样多组任务组成 episode，每组任务的类别集合可以不同。

> Tips: 
> 
> - 小样本学习，关注于学习模型，以适应新的任务。
> - 在传统的监督学习中，我们通过迭代训练集来训练模型，模型总是看到固定的类集。
> - 在小样本学习中，我们从一个`支持集`开始，创建`多个训练任务`来组装训练集，每个训练任务包含不同的分类。



## Datasets and Terminology
> 本节对比常规训练/测试集与小样本中的 support、query、base 类与 episode，并用图示说明训练与测试阶段类别不重叠的设定及常见元学习策略。
[](#datasets-and-terminology)

In supervised learning, we fit a model on a `training dataset` and
evaluate it on a` test dataset`. The training set typically contains a
relatively large number of examples per class. For example, in a
supervised learning context, the Iris dataset, which has 50 examples per
class, is considered a tiny dataset. For deep learning models, on the
other hand, even a dataset like MNIST that has 5,000 training examples
per class is considered very small.

常规监督在 `training dataset` 上拟合、在 `test dataset` 上评估；每类样本数通常较多（Iris 每类 50 已算很小；MNIST 每类约 5000 对深度学习仍偏小）。

In `few-shot learning`, the number of examples per class is much smaller.
When specifying the few-shot learning task, we typically use the term
N-*way* K-*shot*, where

* *N* stands for the number of classes 
* and *K* stands for the number of examples per class. 

The most common values are *K* = 1 or *K* = 5. For instance, in a 5-way 1-shot problem, there are
five classes with only one example each.
Figure [3.1](#fig-ch03-fig01) depicts a 3-way 1-shot setting to illustrate the
concept with a smaller example.

在 `few-shot learning` 中每类样本极少；常用 **N-way K-shot** 描述：*N* 为类别数，*K* 为每类支撑样本数，常见 *K*=1 或 5。Figure [3.1](#fig-ch03-fig01) 用 3-way 1-shot 示意。

<a id="fig-ch03-fig01"></a>

<div align="center">
  <img src="./images/ch03-fig01.png" alt="Training tasks in few-shot learning" width="78%" />
  <div><b>Figure 3.1</b></div>
</div>

Rather than fitting the model to the training dataset, we can think of
`few-shot learning` as "**learning to learn.**"? In contrast to supervised
learning, `few-shot learning` uses not a training dataset but a
so-called `support set`, from which we sample training tasks that mimic
the use-case scenario during prediction. With each training task comes a
query image to be classified. The model is trained on several training
tasks from the support set; this is called an `episode`.

也可把 few-shot 理解为「**学会学习**」：不用传统大训练集，而从 `support set` 反复采样任务；每个任务带待分类的 query；在 support 上串起多轮任务称为一个 `episode`。

> Tips: 小样本学习，可以看作是`学习如何学习`。
> 
> 与传统的监督学习不同，小样本学习不使用训练集，而是使用所谓的`支持集`，从中采样训练任务，以模仿预测时的使用场景。
> 
> 每个训练任务都有一个查询图像需要分类。
> 
> 模型在支持集的多个训练任务上进行训练；这称为`一个训练轮次`。
>
> FIXME??? 不理解



Next, during testing, the model receives a new task with classes
different from those seen during training. The classes encountered in
training are also called `base classes`, and the support set during
training is also often called the `base set`. Again, the task is to
classify the query images. Test tasks are similar to training tasks,
except that none of the classes during testing overlap with those
encountered during training, as illustrated in
Figure [3.2](#fig-ch03-fig02).

测试时模型见到的新任务类别应与训练时不同；训练阶段见到的类常称 `base classes`，对应 support 也称 `base set`。测试任务形式类似，但类别与训练不重叠，见 Figure [3.2](#fig-ch03-fig02)。

<a id="fig-ch03-fig02"></a>

<div align="center">
  <img src="./images/ch03-fig02.png" alt="Classes seen during training and testing" width="78%" />
  <div><b>Figure 3.2</b></div>
</div>

As Figure [3.2](#fig-ch03-fig02) shows, the support and query sets contain
different images from the same class during training. The same is true
during testing. However, notice that the classes in the support and
query sets differ from the support and query sets encountered during
training.

如图，训练时 support 与 query 虽图像不同但属同一批训练类别；测试时亦然，但测试见到的类别集与训练阶段完全不同。

There are many different types of few-shot learning. In the most common,
*meta-learning*, training is essentially about updating the model's
parameters such that it can *adapt* well to a new task. On a high level,
one few-shot learning strategy is to learn a model that produces
embeddings where we can find the target class via a nearest-neighbor
search among the images in the support set.
Figure [3.3](#fig-ch03-fig03) illustrates this approach.

小样本流派众多；常见的 *meta-learning* 通过更新参数使模型能快速 *adapt* 新任务。一种高层策略是学习嵌入，再在 support 上做最近邻以判定 query 类别，见 Figure [3.3](#fig-ch03-fig03)。

<a id="fig-ch03-fig03"></a>

<div align="center">
  <img src="./images/ch03-fig03.png" alt="Learning embeddings that are suitable for classification" width="78%" />
  <div><b>Figure 3.3</b></div>
</div>

The model learns how to produce good embeddings from the support set to
classify the query image based on finding the most similar embedding
vector. 

模型学会从 support 生成利于分类的嵌入，通过最相似嵌入向量判定 query。

## Exercises
> 本节练习：如何把 MNIST 划分成 one-shot 设定；以及 few-shot 的实际应用场景。

3-1. MNIST (<https://en.wikipedia.org/wiki/MNIST_database>) is a classic
and popular machine learning dataset consisting of 50,000 handwritten
digits from 10 classes corresponding to the digits 0 to 9. How can we
partition the MNIST dataset for a one-shot classification context?

3-1. MNIST（<https://en.wikipedia.org/wiki/MNIST_database>）含 10 类手写数字各约 5000 训练样本。若要构造 one-shot 分类实验，应如何划分数据？

3-2. What are some real-world applications or use cases for few-shot
learning?

3-2. 小样本学习在现实中有哪些应用或用例？



------------------------------------------------------------------------

