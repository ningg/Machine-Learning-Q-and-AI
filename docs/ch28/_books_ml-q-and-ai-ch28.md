







# Chapter 28: The k in k-Fold Cross-Validation
[](#chapter-28-the-k-in-k-fold-cross-validation)



**k-fold cross-validation is a common choice for evaluating machine
learning classifiers because it lets us use all training data to
simulate how well a machine learning algorithm might perform on new
data. What are the advantages and disadvantages of choosing a large k?**

> 本章讨论了 k-fold 交叉验证，并讨论了它的优缺点。
> 
> - k-fold 交叉验证是一种常用的评估机器学习分类器的方法，它让我们使用所有训练数据来模拟机器学习算法在新数据上的表现。
> 
> - 选择较大的 k 值时，训练集之间的差异较小，因此模型之间的差异也较小。
> 
> - 选择较小的 k 值时，训练集之间的差异较大，因此模型之间的差异也较大。

We can think of *k*-fold cross-validation as a workaround for model
evaluation when we have limited data. In machine learning model
evaluation, we care about the generalization performance of our model,
that is, how well it performs on new data. In *k*-fold cross-validation,
we use the training data for model selection and evaluation by
partitioning it into *k* validation rounds and folds. If we have *k*
folds, we have *k* iterations, leading to *k* different models, as
illustrated in Figure [28.1](#fig-ch28-fig01).

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

> 使用 k-fold 交叉验证，我们通常通过计算 *k* 个模型的平均性能，来评估**特定超参配置**的性能。

The following sections cover the trade-offs of selecting values for *k*
in *k*-fold cross-validation and address the challenges of large *k*
values and their computational demands, especially in deep learning
contexts. We then discuss the core purposes of *k* and how to choose an
appropriate value based on specific modeling needs.

## Trade-offs in Selecting Values for k
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

Besides the issue of too-similar datasets, running *k*-fold
cross-validation with a large value of *k* is also computationally more
demanding. A larger *k* is more expensive since it increases both the
number of iterations and the training set size at each iteration. This
is especially problematic if we work with relatively large models that
are expensive to train, such as contemporary deep neural networks.

A common choice for *k* is typically 5 or 10, for practical and
historical reasons. A study by Ron Kohavi (see Refrence at the end of this
chapter) found that *k* = 10 offers a good bias and variance trade-off
for classical machine learning algorithms, such as decision trees and
naive Bayes classifiers, on a handful of small datasets.

> 5 或 10 是 k-fold 交叉验证的常见选择，这是出于实际和历史原因。Ron Kohavi 的研究（见本章末尾的参考文献）发现，*k* = 10 在小型数据集上对经典机器学习算法（如决策树和朴素贝叶斯分类器）提供了良好的偏差和方差权衡。

For example, in 10-fold cross-validation, we use 9/10 (90 percent) of
the data for training in each round, whereas in 5-fold cross-validation,
we use only 4/5 (80 percent) of the data, as shown in
Figure [28.2](#fig-ch28-fig02).

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
the validation scores may be more sensitive to the particularities of
the small validation sets, leading to higher variance in the overall
cross-validation score.

> k 过大或过小，都会增加方差。

## Determining Appropriate Values for k
[](#determining-appropriate-values-for-k)

When deciding upon an appropriate value of *k*, we are often guided by
computational performance and conventions. However, it's worthwhile to
define the purpose and context of using *k*-fold cross-validation. For
example, if we care primarily about approximating the predictive
performance of the final model, using a large *k* makes sense. This way,
the training folds are very similar to the combined training dataset,
yet we still get to evaluate the model on all data points via the
validation folds.

> 决定适当的 k 值时，先考虑使用 k-fold 交叉验证的目的和上下文。例如，如果我们主要关心近似最终模型的预测性能，使用较大的 k 是有意义的。这样，训练集非常相似于组合训练数据集，但我们仍然可以通过验证集评估模型。

On the other hand, if we care to evaluate how sensitive a given
hyperparameter configuration and training pipeline is to different
training datasets, then choosing a smaller number for *k* makes more
sense.

> 如果我们主要关心给定超参配置和训练管道对不同训练数据集的敏感性，那么选择较小的 k 是有意义的。

Since most practical scenarios consist of two steps -- tuning
hyperparameters and evaluating the performance of a model -- we can also
consider a two-step procedure. For instance, we can use a smaller *k*
during hyperparameter tuning. This will help speed up the hyperparameter
search and probe the hyperparameter configurations for robustness (in
addition to the average performance, we can also consider the variance
as a selection criterion). Then, after hyperparameter tuning and
selection, we can increase the value of *k* to evaluate the model.

> 大多数实际场景都包括两个步骤：调整超参和评估模型性能。因此，我们也可以考虑一个两步流程。
> 
> - 例如，在调整超参时，我们可以使用较小的 k。这将帮助加速超参搜索，并测试超参配置的稳健性（除了平均性能，我们还可以考虑方差作为选择标准）。
> - 然后，在调整超参和选择后，我们可以增加 k 值来评估模型。

However, reusing the same dataset for model selection and evaluation
introduces biases, and it is usually better to use a separate test set
for model evaluation. Also, `nested cross-validation` may be preferred as
an alternative to *k*-fold cross-validation.

> 然而，重复使用相同的数据集进行模型选择和评估，会引入偏差，通常最好使用单独的测试集进行模型评估。此外，嵌套交叉验证可能比 k-fold 交叉验证更可取。
>
> 更多细节： [嵌套交叉验证](https://ljalphabeta.gitbooks.io/python-/content/nested.html)

## Exercises
[](#exercises)

28-1. Suppose we want to provide a model with as much training data as
possible. We consider using *leave-one-out cross-validation (LOOCV)*, a
special case of *k*-fold cross-validation where *k* is equal to the
number of training examples, such that the validation folds contain only
a single data point. A colleague mentions that LOOCV is defective for
discontinuous loss functions and performance measures such as
classification accuracy. For instance, for a validation fold consisting
of only one example, the accuracy is always either 0 (0 percent) or 1
(99 percent). Is this really a problem?

28-2. This chapter discussed model selection and model evaluation as two
use cases of *k*-fold cross-validation. Can you think of other use
cases?

## References
[](#references)

- For a longer and more detailed explanation of why and how to use
  *k*-fold cross-validation, see my article: "Model Evaluation, Model
  Selection, and Algorithm Selection in Machine Learning"? (2018),
  <https://arxiv.org/abs/1811.12808>.

- The paper that popularized the recommendation of choosing *k* = 5 and
  *k* = 10: Ron Kohavi, "A Study of Cross-Validation and Bootstrap for
  Accuracy Estimation and Model Selection"? (1995),
  <https://dl.acm.org/doi/10.5555/1643031.1643047>.


------------------------------------------------------------------------

