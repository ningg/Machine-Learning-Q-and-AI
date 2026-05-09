








# Chapter 28: The k in k-Fold Cross-Validation

> 澄清 *k*-fold 中 $k$ 对训练相似度、方差—偏差与算力开销的影响，并给出按目标（近似最终模型 vs. 探查数据敏感度）选型与两步法的思路。

[](#chapter-28-the-k-in-k-fold-cross-validation)



**k-fold cross-validation is a common choice for evaluating machine
learning classifiers because it lets us use all training data to
simulate how well a machine learning algorithm might perform on new
data. What are the advantages and disadvantages of choosing a large k?**

Selecting a larger *k* means each fold leaves out fewer training points per round—so successive models resemble a full-data fit—but it also repeats training more often and yields smaller validation shards that can fluctuate sharply.

可把问题拆成三件事：相邻折之间训练集的相似程度、单次训练消耗的样本体量与迭代次数（算力账单）、以及很小的验证片段如何扰动指标的方差。

We can think of *k*-fold cross-validation as a workaround for model
evaluation when we have limited data. In machine learning model
evaluation, we care about the generalization performance of our model,
that is, how well it performs on new data. In *k*-fold cross-validation,
we use the training data for model selection and evaluation by
partitioning it into *k* validation rounds and folds. If we have *k*
folds, we have *k* iterations, leading to *k* different models, as
illustrated in Figure [28.1](#fig-ch28-fig01).

可把 *k*-折理解为：在同一份候选训练数据上轮转留出，从而在有限样本下粗略回答“换新数据会发生什么”。形式上，每轮都只留一块作验证并用其余折重训整套管线，迭代 $k$ 次即得 $k$ 条性能轨迹（见图 [28.1](#fig-ch28-fig01)）。

<a id="fig-ch28-fig01"></a>

<div align="center">
  <img src="./images/ch28-fig01.png" alt="An example of k-fold cross-validation for model evaluation where k = 5" width="78%" />
  <div><b>Figure 28.1</b></div>
</div>

Using *k*-fold cross-validation, we usually evaluate the performance of
a particular hyperparameter configuration by computing the average
performance over the *k* models. This performance reflects or
approximates the performance of a model trained on the complete training
dataset after evaluation.

对某一固定超参向量，我们常把 $k$ 折得分的算术平均视作其泛化性能的代理——经验上近似于“若在完整训练数据上重新训练后再测”会如何。

The following sections cover the trade-offs of selecting values for *k*
in *k*-fold cross-validation and address the challenges of large *k*
values and their computational demands, especially in deep learning
contexts. We then discuss the core purposes of *k* and how to choose an
appropriate value based on specific modeling needs.

下面分两节展开：其一是 $k$ 与相似度—方差—算力的张力；其二结合建模目的讨论如何挑 $k$ 以及是否与调参拆分。

## Trade-offs in Selecting Values for k

> 论述大 $k$ 使模型彼此更像全量训练版本、提高单次训练体量，却会压缩验证折并放大估计方差图景的复杂性。

[](#trade-offs-in-selecting-values-for-k)

If *k* is too large, the training sets are too similar between the
different rounds of cross-validation. The *k* models are thus very
similar to the model we obtain by training on the whole training set. In
this case, we can still leverage the advantage of *k*-fold
cross-validation: evaluating the performance for the entire training set
via the held-out validation fold in each round. (Here, we obtain the
training set by concatenating all *k* -- 1 training folds in a given
iteration.) However, a disadvantage of a large *k* is that it is more
challenging to analyze how the machine learning algorithm with the
particular choice of hyperparameter setting behaves on different
training datasets.

$k$ 很大时，每轮只差一点点样本，学到的决策边界高度相关：好处是均值分数更贴近“用几乎全数据训练”；坏处是你更难从分数分布里读出“换一批训练数据会发生什么”的诊断信息——因为训练折几乎总是同一分布的高重叠子集。

Besides the issue of too-similar datasets, running *k*-fold
cross-validation with a large value of *k* is also computationally more
demanding. A larger *k* is more expensive since it increases both the
number of iterations and the training set size at each iteration. This
is especially problematic if we work with relatively large models that
are expensive to train, such as contemporary deep neural networks.

算力维度上，大 $k$ 同时推高**迭代次数**和**每轮训练样本量**：对深度模型尤其刺耳，因为要重复训练一整套优化。

A common choice for *k* is typically 5 or 10, for practical and
historical reasons. A study by Ron Kohavi (see References at the end of this
chapter) found that *k* = 10 offers a good bias and variance trade-off
for classical machine learning algorithms, such as decision trees and
naive Bayes classifiers, on a handful of small datasets.

实务上常以 $k\in\{5,10\}$ 为默认脚注式选择；Ron Kohavi 的经典研究（参考文献）在多款小数据集与传统学习器上演示 $k{=}10$ 常落在可用偏差–方差折中点附近。

For example, in 10-fold cross-validation, we use 9/10 (90 percent) of
the data for training in each round, whereas in 5-fold cross-validation,
we use only 4/5 (80 percent) of the data, as shown in
Figure [28.2](#fig-ch28-fig02).

例如在 10 折场景里单轮训练的样本量是分母的 90%，而 5 折只占 80%；图 [28.2](#fig-ch28-fig02) 给出直观对比。

<a id="fig-ch28-fig02"></a>

<div align="center">
  <img src="./images/ch28-fig02.png" alt="A comparison of 5-fold and 10-fold cross-validation" width="78%" />
  <div><b>Figure 28.2</b></div>
</div>

However, this does not mean large training sets are bad, since they can
reduce the pessimistic bias of the performance estimate (mostly a good
thing) if we assume that the model training can benefit from more
training data. (See
Figure [5.1](./ch05/_books_ml-q-and-ai-ch05.md#fig-ch05-fig01) on page  for an example of a learning curve.)

更大的训练折意味着单轮估计更乐观地贴近“全数据”性能，只要模型仍能从额外样本里受益；学习曲线（第 5 章图 [5.1](./ch05/_books_ml-q-and-ai-ch05.md#fig-ch05-fig01)）展示了这种随训练量变化的偏差。

In practice, both a very small and a very large *k* may increase
variance. For instance, a larger *k* makes the training folds more
similar to each other since a smaller proportion is left for the
held-out validation sets. Since the training folds are more similar, the
models in each round will be more similar. In practice, we may observe
that the variance of the held-out validation fold scores is more similar
for larger values of *k*. On the other hand, when *k* is large, the
validation sets are small, so they may contain more random noise or be
more susceptible to quirks of the data, leading to more variation in the
validation scores across the different folds. Even though the models
themselves are more similar (since the training sets are more similar),
the validation scores may be more sensitive to the particulars of the
small validation sets, leading to higher variance in the overall
cross-validation score.

实务解读要小心：极小或极大的 $k$ 都可能推高“你看到的那条汇总分数”的方差图景——极大 $k$ 时模型彼此酷似，验证折却非常小，单次验证可能对偶然样本极度敏感（尽管模型差异不大）。这也是为何不能只看均值而忽略分数散布。

## Determining Appropriate Values for k

> 把“我们想回答的问题”排到前面：是想逼近最终全数据模型，还是想扫描训练抽样敏感度；顺带提示嵌套交叉验证与留出测试的重要性。

[](#determining-appropriate-values-for-k)

When deciding upon an appropriate value of *k*, we are often guided by
computational performance and conventions. However, it's worthwhile to
define the purpose and context of using *k*-fold cross-validation. For
example, if we care primarily about approximating the predictive
performance of the final model, using a large *k* makes sense. This way,
the training folds are very similar to the combined training dataset,
yet we still get to evaluate the model on all data points via the
validation folds.

若想回答“我几乎用满训练样本时模型有多好”，偏大 $k$ 更合拍：训练折高度接近全集，却仍能通过轮换验证让所有点各当一次留出观察。

On the other hand, if we care to evaluate how sensitive a given
hyperparameter configuration and training pipeline is to different
training datasets, then choosing a smaller number for *k* makes more
sense.

若更关心管线对数据扰动的敏感度，偏小 $k$ 让各轮训练样本差异更明显，更可读出鲁棒画像。

Since most practical scenarios consist of two steps -- tuning
hyperparameters and evaluating the performance of a model -- we can also
consider a two-step procedure. For instance, we can use a smaller *k*
during hyperparameter tuning. This will help speed up the hyperparameter
search and probe the hyperparameter configurations for robustness (in
addition to the average performance, we can also consider the variance
as a selection criterion). Then, after hyperparameter tuning and
selection, we can increase the value of *k* to evaluate the model.

工程上可把“搜寻超参”（需要快且要扫风险）与“定妆评估”（可以更慢更稳）拆分：前半段用小 $k$ 兼看均值与方差，后半段再以更大 $k$ 复核——但要牢记：若始终啃同一份数据，仍然会掺入选型偏差。

However, reusing the same dataset for model selection and evaluation
introduces biases, and it is usually better to use a separate test set
for model evaluation. Also, `nested cross-validation` may be preferred as
an alternative to *k*-fold cross-validation.

更稳妥的路线是：**留出独立测试集**专司最终打分，或对“选模型 + 报告泛化”使用嵌套交叉验证（nested cross-validation），以降低把同一噪声多次刷进指标里的风险。（中文读者可参考公开的 nested CV 讲义，例如 <https://ljalphabeta.gitbooks.io/python-/content/nested.html> 的示意流程。）

## Exercises

> LOOCV 与病态准确率的讨论；以及 *k*-折除调参评估外的其他用途枚举。

[](#exercises)

28-1. Suppose we want to provide a model with as much training data as
possible. We consider using *leave-one-out cross-validation (LOOCV)*, a
special case of *k*-fold cross-validation where *k* is equal to the
number of training examples, such that the validation folds contain only
a single data point. A colleague mentions that LOOCV is defective for
discontinuous loss functions and performance measures such as
classification accuracy. For instance, for a validation fold consisting
of only one example, the accuracy is always either 0 (0 percent) or 1
(100 percent). Is this really a problem?

28-2. This chapter discussed model selection and model evaluation as two
use cases of *k*-fold cross-validation. Can you think of other use
cases?

练习题 28-1 引导你辨析：离散指标在小留出上的极端取值如何进入整体方差公式；是否真的“坏”要看你要优化的风险与是否要光滑 surrogate。28-2 则鼓励联想到特征筛选、校准集构造、以及与 bootstrap 组合的汇报策略等延伸用途。

## References

> 模型评估综述与奠定 $k=5/10$ 常识的 Kohavi (1995) 原始论文入口。

[](#references)

- For a longer and more detailed explanation of why and how to use
  *k*-fold cross-validation, see my article: "Model Evaluation, Model
  Selection, and Algorithm Selection in Machine Learning" (2018),
  <https://arxiv.org/abs/1811.12808>.

- The paper that popularized the recommendation of choosing *k* = 5 and
  *k* = 10: Ron Kohavi, "A Study of Cross-Validation and Bootstrap for
  Accuracy Estimation and Model Selection" (1995),
  <https://dl.acm.org/doi/10.5555/1643031.1643047>.


------------------------------------------------------------------------
