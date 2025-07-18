







# Chapter 29: Training and Test Set Discordance
[](#chapter-29-training-and-test-set-discordance)



**Suppose we train a model that performs much better on the test dataset
than on the training dataset. Since a similar model configuration
previously worked well on a similar dataset, we suspect something might
be unusual with the data. What are some approaches for looking into
training and test set discrepancies, and what strategies can we use to
mitigate these issues?**

Before investigating the datasets in more detail, we should check for
technical issues in the **data loading** and **evaluation** code. For instance,
a simple sanity check is to temporarily replace the test set with the
training set and to reevaluate the model. In this case, we should see
identical training and test set performances (since these datasets are
now identical). If we notice a discrepancy, we likely have a bug in the
code; in my experience, such bugs are frequently related to incorrect
shuffling or inconsistent (often missing) data normalization.

> 在进一步检查数据集之前，我们应该检查数据加载和评估代码中的技术问题。
> - 例如，一个简单的健全性检查是暂时将测试集替换为训练集，并重新评估模型。
> - 在这种情况下，我们应该看到训练和测试集的性能相同（因为这些数据集现在相同）。
> - 如果我们注意到差异，我们代码可能有bug；根据经验，这种错误，通常是数据洗牌不均匀或数据归一化不一致（通常缺失）。


If the test set performance is much better than the training set
performance, we can rule out overfitting. More likely, there are
substantial differences in the training and test data distributions.
These distributional differences may affect both the features and the
targets. Here, plotting the target or label distributions of training
and test data is a good idea. For example, a common issue is that the
test set is missing certain class labels if the dataset was not shuffled
properly before splitting it into training and test data. For small
tabular datasets, it is also feasible to compare feature distributions
in the training and test sets using histograms.

Looking at feature distributions is a good approach for tabular data,
but this is trickier for image and text data. A relatively easy and more
general approach to check for discrepancies between training and test
sets is adversarial validation.

`Adversarial validation`, illustrated in

<a id="fig-ch29-fig01"></a>

<div align="center">
  <img src="./images/ch29-fig01.png" alt="adversarial validation illustration" width="78%" />
</div>

is a technique to identify the degree of
similarity between the training and test data. We first merge the
training and test sets into a single dataset, and then we create a
binary target variable that distinguishes between training and test
data. For instance, we can use a new *Is test?* label where we assign
the label 0 to training data and the label 1 to test data. We then use
*k*-fold cross-validation or repartition the dataset into a training set
and a test set and train a machine learning model as usual. Ideally, we
want the model to perform poorly, indicating that the training and test
data distributions are similar. On the other hand, if the model performs
well in predicting the *Is test?* label, it suggests a discrepancy
between the training and test data that we need to investigate further.

> **对抗验证**，用于识别训练和测试数据之间的**相似程度**。
> 
> - 我们首先将训练和测试集合并为一个数据集，然后创建一个二元目标变量，用于区分训练和测试数据。
> - 例如，我们可以使用一个新的 *Is test?* 标签，将标签 0 分配给训练数据，将标签 1 分配给测试数据。
> - 然后，我们使用 k-fold 交叉验证或重新划分数据集为训练集和测试集，并像往常一样训练机器学习模型。
> - 理想情况下，我们希望模型表现不佳，表明训练和测试数据分布相似。
> - 另一方面，如果模型在预测 *Is test?* 标签时表现良好，则表明训练和测试数据之间存在差异，我们需要进一步调查。


What mitigation techniques should we use if we detect a training-test
set discrepancy using adversarial validation? If we're working with a
tabular dataset, we can remove features one at a time to see if this
helps address the issue, as spurious features can sometimes be highly
correlated with the target variable. To implement this strategy, we can
use sequential feature selection algorithms with an updated objective.
For example, instead of maximizing classification accuracy, we can
minimize classification accuracy. For cases where removing features is
not so trivial (such as with image and text data), we can also
investigate whether removing individual training instances that are
different from the test set can address the discrepancy issue.

> 如果我们使用对抗验证检测到训练-测试集差异，我们应该使用什么缓解技术？
> 如果我们使用表格数据集，我们可以一次删除一个特征，看看是否有助于解决这个问题，因为虚假特征有时与目标变量高度相关。
> 为了实现这个策略，我们可以使用顺序特征选择算法，并更新目标函数。
> 例如，我们不再最大化分类准确率，而是最小化分类准确率。
> 对于图像和文本数据，我们也可以研究是否删除与测试集不同的训练实例是否有助于解决差异问题。



## Exercises
[](#exercises)

29-1. What is a good performance baseline for the adversarial prediction
task?

29-2. Since training datasets are often bigger than test datasets,
adversarial validation often results in an imbalanced prediction problem
(with a majority of examples labeled as *Is test?* being false instead
of true). Is this an issue, and if so, how can we mitigate that?


------------------------------------------------------------------------

