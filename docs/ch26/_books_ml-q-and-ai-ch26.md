








# Chapter 26: Confidence Intervals vs. Conformal Predictions

> 对照置信区间（总体参数）与预测区间 / conformal prediction（单点预测不确定性），说明覆盖保证、分布假设与用法差异。

[](#chapter-26-confidence-intervals-vs-conformal-predictions)



**What are the differences between confidence intervals and conformal
predictions, and when do we use one over the other?**

Confidence intervals and conformal predictions are both statistical
methods to estimate the range of plausible values for an unknown
population parameter. As discussed in
Chapter [\[ch25\]](./ch25/_books_ml-q-and-ai-ch25.md), a
confidence interval quantifies the level of confidence that a population
parameter lies within an interval. For instance, a 95 percent confidence
interval for the mean of a population means that if we were to take many
samples from the population and calculate the 95 percent confidence
interval for each sample, we would expect the true population mean
(average) to lie within these intervals 95 percent of the time.
Chapter [\[ch25\]](./ch25/_books_ml-q-and-ai-ch25.md)
covered several techniques for applying this method to estimate the
prediction performance of machine learning models. Conformal
predictions, on the other hand, are commonly used for creating
prediction intervals, which are designed to cover a true outcome with a
certain probability.

二者都用于刻画“未知量可能落在何处”，但对象不同：置信区间针对**总体参数**（例如真实泛化准确率），其频率学解释是在重复抽样下覆盖真值的比例；conformal prediction 更直接服务**单条预测**的输出集合/区间，并给定“真实结果落在集合内”的控制概率。第 25 章的方法主要把置信区间用于评估模型层面的性能；本章则把 conformal prediction 宽泛地看作构造**预测区间（prediction interval）**的一类现代工具。

This chapter briefly explains what a prediction interval is and how it
differs from confidence intervals, and then it explains how conformal
predictions are, loosely speaking, a method for constructing prediction
intervals.

本章先澄清预测区间相对置信区间回答的是哪一类不确定性，再说明 conformal prediction 如何在此基础上构造带有限样本覆盖保证的预测集合。

## Confidence Intervals and Prediction Intervals

> 用“人群平均身高 vs. 某个人的身高”类比，区分面向总体的参数区间与面向单次输出的预测区间。

[](#confidence-intervals-and-prediction-intervals)

Whereas a confidence interval focuses on parameters that characterize a
population as a whole, a *prediction interval* provides a range of
values for a single predicted target value. For example, consider the
problem of predicting people's heights. Given a sample of 10,000
people from the population, we might conclude that the mean (average)
height is 5 feet, 7 inches. We might also calculate a 95 percent
confidence interval for this mean, ranging from 5 feet, 6 inches to 5
feet, 8 inches.

对总体均值而言，我们只关心样本均值及其不确定性：它回答“平均身高是否在某一范围内”，而非某个具体个体的身高轨迹。

A *prediction interval*, however, is concerned with estimating not the
height of the population but the height of an individual person. For
example, given a weight of 185 pounds, a given person's prediction
interval may fall between 5 feet 8 inches and 6 feet.

预测区间则要覆盖**单次观测**：若已知体重等信息，我们更关心某位具体人士身高可能出现的范围，这已经与均值置信区间截然不同。

In a machine learning model context, we can use confidence intervals to
estimate a population parameter such as the accuracy of a model (which
refers to the performance on all possible prediction scenarios). In
contrast, a prediction interval estimates the range of output values for
a single given input example.

在机器学习里同理：准确率置信区间刻画的是“整条决策规则在整个总体上的表现”；预测区间／集合说的是“对这一输入向量，单次输出取值可能散落何处”。

## Prediction Intervals and Conformal Predictions

> 归纳：经典预测区间常与模型族/分布假定绑定；conformal 路线更一般，但可能要付更多算力。

[](#prediction-intervals-and-conformal-predictions)

Both conformal predictions and prediction intervals are statistical
techniques that estimate uncertainty for individual model predictions,
but they do so in different ways and under different assumptions.

二者都服务**单点预测的不确定性**，但 assumptions 与接口不同。

While prediction intervals often assume a particular data distribution
and are tied to a specific type of model, conformal prediction methods
are distribution free and can be applied to any machine learning
algorithm.

传统预测区间往往依附于指定模型与噪声/分布模型；conformal prediction 在较弱的可交换性等工作条件下可包装任意黑盒分类器或回归器。

In short, we can think of conformal predictions as a more flexible and
generalizable form of prediction intervals. However, conformal
predictions often require more computational resources than traditional
methods for constructing prediction intervals, which involve resampling
or permutation techniques.

可以把 conformal prediction 理解为“更通用的预测区间族”；代价常是额外的校准集、排序与集合构造步骤，算力与实现复杂度往往更高。

## Prediction Regions, Intervals, and Sets

> 统一术语：回归用区间、分类用集合，二者都可称 prediction region。

[](#prediction-regions-intervals-and-sets)

In the context of conformal prediction, the terms *prediction interval*,
*prediction set*, and *prediction region* are used to denote the
plausible outputs for a given instance. The type of term used depends on
the nature of the task.

conformal prediction 文献里会交替使用“区间 / 集合 / 区域”，本质是**模型认为与观测相容的输出范围**；叫法随任务类型而变。

In regression tasks where the output is a continuous variable, a
*prediction interval* provides a range within which the true value is
expected to fall with a certain level of confidence. For example, a
model might predict that the price of a house is between \$200,000 and
\$250,000.

回归场景下输出连续，预测结果常给出一个**实数区间**，并控制该区间覆盖真值的经验频率。

In classification tasks, where the output is a discrete variable (the
class labels), a *prediction set* includes all class labels that are
considered plausible predictions for a given instance. For example, a
model might predict that an image depicts either a cat, dog, or bird.

分类场景下标签离散，输出往往是一个**类别集合**（prediction set），必要时允许多标并列以换取覆盖保证。

*Prediction region* is a more general term that can refer to either a
prediction interval or a prediction set. It describes the set of outputs
considered plausible by the model.

prediction region 是上两者的统称：强调“输出空间中的一块允许区域”，而不限定连续或离散。

## Computing Conformal Predictions

> 走一遍训练—校准—测试流程：定义非一致性分数、在校准集上取分位阈值、再对新点生成预测集合。

[](#computing-conformal-predictions)

Now that we've introduced the difference between confidence intervals
and prediction regions and learned how conformal prediction methods are
related to prediction intervals, how exactly do conformal predictions
work?

In short, conformal prediction methods provide a framework for creating
prediction regions, sets of potential outcomes for a prediction task.
Given the assumptions and methods used to construct them, these regions
are designed to contain the true outcome with a certain probability.

若从实现角度翻译上一段设问：关键在于把分类器输出映射成可比的不一致性分数，再借校准集的经验分位数决定“保哪些标签”。

其核心是构造**非一致性分数（nonconformity score）**：分数越大表示该标签与模型输出越“不协调”；阈值则直接读自校准样本的排序。

For classifiers, a prediction region for a given input is a set of
labels such that the set contains the true label with a given confidence
(typically 95 percent), as illustrated in
Figure [26.1](#fig-ch26-fig01).

对分类任务，预测区域就是一组标签；在正确设定下它以高概率覆盖真实类（见图 [26.1](#fig-ch26-fig01)）。

<div align="center">
  <img src="./images/ch26-fig01.png" alt="Prediction regions for a classification task" width="78%" />
  <div><b>Figure 26.1</b></div>
</div>

As depicted in Figure [26.1](#fig-ch26-fig01), the ImageNet dataset consists of a subset of bird
species. Some bird species in ImageNet belong to one of the follow-
 ing classes: *hawk*, *duck*, *eagle*, or *goose*. ImageNet also
contains other animals, for example, cats. For a new image to classify
(here, an eagle), the conformal prediction set consists of classes such
that the true label, *eagle*, is contained within this set with 95
percent probability. Often, this includes closely related classes, such
as *hawk* and *goose* in this case. However, the prediction set can also
include less closely related class labels, such as *cat*.

图 [26.1](#fig-ch26-fig01) 的示例说明：即便真实类是 *eagle*，为换取 95% 的有限样本覆盖保证，预测集合也可能同时包含语义相近的 *hawk*/*goose*，甚至在某些分数配置下纳入远类 *cat*——这是“以集合换覆盖”的直接体现。

To sketch the concept of computing prediction regions step by step,
let's suppose we train a machine learning classifier for images.
Before the model is trained, the dataset is typically split into three
parts: a training set, a calibration set, and a test set. We use the
training set to train the model and the calibration set to obtain the
parameters for the conformal prediction regions. We can then use the
test set to assess the performance of the conformal predictor. A typical
split ratio might be 60 percent training data, 20 percent calibration
data, and 20 percent test data.

标准流程先把数据划成训练 / 校准 / 测试：训练集拟合模型；**校准集单独用来定阈值**（不参与训练）；测试集再汇报 conformal predictor 的经验覆盖与集合大小等指标。常见比例例如 60%/20%/20%。

The first step after training the model on the training set is to define
a *nonconformity measure*, a function that assigns a numeric score to
each instance in the calibration set based on how "unusual" it is.
This could be based on the distance to the classifier's decision
boundary or, more commonly, 1 minus the predicted probability of a class
label. The higher the score is, the more unusual the instance is.

第一步是定义非一致性度量：常见取法之一是对真类用 $1-\hat{p}(y\mid x)$，分数越大说明模型越“不信”该标注。

Before using conformal predictions for new data points, we use the
nonconformity scores from the calibration set to compute a quantile
threshold. This threshold is a probability level such that, for example,
95 percent of the instances in the calibration set (if we choose a 95
percent confidence level) have nonconformity scores below this
threshold. This threshold is then used to determine the prediction
regions for new instances, ensuring that the predictions are calibrated
to the desired confidence level.

阈值一般取校准分数的适当分位数（并常做有限样本修正），使得在可交换性假定下，对新点的预测集合在目标水平上有有限样本覆盖。

Once we have the threshold value, we can compute prediction regions for
new data. Here, for each possible class label (each possible output of
your classifier) for a given instance, we check whether its
nonconformity score is below the threshold. If it is, then we include it
in the prediction set for that instance.

阈值确定后，对每个候选标签计算其“假设为真类”时的非一致性分数；凡低于阈值的标签都收入该实例的预测集合。

## A Conformal Prediction Example

> 用一个三分类与 *score method* 演示：从概率到非一致性分数，再到预测集合。

[](#a-conformal-prediction-example)

Let's illustrate this process of making conformal predictions with an
example using a simple conformal prediction method known as the *score
method*. Suppose we train a classifier on a training set to distinguish
between three species of birds: sparrows, robins, and hawks. Suppose the
predicted probabilities for a calibration dataset are as follows:

* Sparrow \[0.95, 0.9, 0.85, 0.8, 0.75\]
* Robin \[0.7, 0.65, 0.6, 0.55, 0.5\]
* Hawk \[0.4, 0.35, 0.3, 0.25, 0.2\]

以下为 *score method* 第一步：列出校准集上各样本对**真类**的 softmax，再进入非一致性分数与阈值。

As depicted here, we have a calibration set consisting of 15 examples,
five for each of the three classes. Note that a classifier returns three
probability scores for each training example: one probability
corresponding to each of the three classes (*Sparrow*, *Robin*, and
*Hawk*). Here, however, we've selected only the probability for the
true class label. For example, we may obtain the values \[0.95, 0.02,
0.03\] for the first calibration example with the true label *Sparrow*.
In this case, we kept only 0.95.

这里每个校准样本只保留**真类对应的 softmax 概率**，共 15 个标量，来自三类各 5 个例子。

Next, after we obtain the previous probability scores, we can compute
the nonconformity score as 1 minus the probability, as follows:

* Sparrow \[0.05, 0.1, 0.15, 0.2, 0.25\]
* Robin \[0.3, 0.35, 0.4, 0.45, 0.5\]
* Hawk \[0.6, 0.65, 0.7, 0.75, 0.8\]

即对每个真类概率取 $1-p$ 作为非一致性分数。

Considering a confidence level of 0.95, we now select a threshold such
that 95 percent of these nonconformity scores fall below that threshold.
Based on the nonconformity scores in this example, this threshold is
0.8. We can then use this threshold to construct the prediction sets for
new instances we want to classify.

若要控制约 95% 的校准覆盖，需要相应地上调阈值；本例中演示选取阈值为 0.8（示意用），用于之后筛选标签。

Now suppose we have a new instance (a new image of a bird) that we want
to classify. We calculate the nonconformity score of this new bird
image, assuming it belongs to each bird species (class label) in the
training set:

* Sparrow 0.26
* Robin 0.45
* Hawk 0.9

In this case, the *Sparrow* and *Robin* nonconformity scores fall below
the threshold of 0.8. Thus, the prediction set for this input is
\[*Sparrow*, *Robin*\]. In other words, this tells us that, on average,
the true class label is included in the prediction set 95 percent of the
time.

当阈值为 0.8 时，*Sparrow* 与 *Robin* 的非一致性分数低于阈值，故进入预测集合；新图像上逐类“反事实”算分正对应上面列出的 0.26/0.45/0.9。严格实现需按校准分布计算分位数，这里仅示意流程。

A hands-on code example implementing the score method can be found in
the *supplementary/q26_conformal-prediction* subfolder at
<https://github.com/rasbt/MachineLearning-QandAI-book>.

可参考作者仓库目录 *supplementary/q26_conformal-prediction* 的可运行脚本，把本节步骤对齐到代码。

## The Benefits of Conformal Predictions

> 强调有限样本覆盖、与模型无关的通用性，以及与传统概率输出相比的松紧取舍。

[](#the-benefits-of-conformal-predictions)

In contrast to using class-membership probabilities returned from
classifiers, the major benefits of conformal prediction are its
theoretical guarantees and its generality. Conformal prediction methods
don't make any strong assumptions about the distribution of the data
or the model being used, and they can be applied in conjunction with any
existing machine learning algorithm to provide confidence measures for
predictions.

与直接解读 softmax 相比，conformal prediction 的卖点是：在标准可交换性条件下给出**有限样本**覆盖，而不是仅依赖渐近或启发式阈值；并且几乎不限制底层模型结构。

Confidence intervals have asymptotic coverage guarantees, which means
that the coverage guarantee holds in the limit as the sample (test set)
size goes to infinity. This doesn't necessarily mean that confidence
intervals work for only very large sample sizes, but rather that their
properties are more firmly guaranteed as the sample size increases.
Confidence intervals therefore rely on asymptotic properties, meaning
that their guarantees become more robust as the sample size grows.

许多经典置信区间构造依赖渐近正态等性质：样本越大，名义覆盖往往越接近理论值，但有限样本下仍可能有偏差。

In contrast, conformal predictions provide finite-sample guarantees,
ensuring that the coverage probability is achieved for any sample size.
For example, if we specify a 95 percent confidence level for a conformal
prediction method and generate 100 calibration sets with corresponding
prediction sets, the method will include the true class label for 95 out
of the 100 test points. This holds regardless of the size of the
calibration sets.

conformal prediction 在理想设定下对**有限校准规模**即可给出覆盖控制（实现细节需按具体算法校正分位数）；这与“靠增大测试集让渐近保证更紧”的置信区间叙事形成对照。

While conformal prediction has many advantages, it does not always
provide the tightest possible prediction intervals. Sometimes, if the
underlying assumptions of a specific classifier hold, that
classifier's own probability estimates might offer tighter and more
informative intervals.

代价是集合可能偏“松”：若模型概率经过良好校准并且类间结构简单，直接使用模型概率未必比 conformal 集合更宽，但需要额外验证。

## Recommendations

> 选型建议：要比较模型还是用模型——分别对应置信区间与 conformal / 预测区间。

[](#recommendations)

A confidence interval tells us about our level of uncertainty about the
model's properties, such as the prediction accuracy of a classifier. A
prediction interval or conformal prediction output tells us about the
level of uncertainty in a specific prediction from the model. Both are
very important in understanding the reliability and performance of our
model, but they provide different types of information.

简短建议：若想回答“这个模型整体上有多准”，用第 25 章的置信区间类工具；若要在部署后标记“哪些单点决策必须人工复核”，则 conformal prediction 更对口。

For example, a confidence interval for the prediction accuracy of a
model can be helpful for comparing and evaluating models and for
deciding which model to deploy. On the other hand, a prediction interval
can be helpful for using a model in practice and understanding its
predictions. For instance, it can help identify cases where the model is
unsure and may need additional data, human oversight, or a different
approach.

例如：离线比较多个候选模型可看准确率置信区间是否重叠；在线上则可用预测集合大小激增作为触发人工审核的信号。

## Exercises

> 练习题：集合大小的语义；回归场景下的置信区间与 conformal。

[](#exercises)

26-1. Prediction set sizes can vary between instances. For example, we
may encounter a prediction set size of 1 for a given instance and for
another, a set size of 3. What does the prediction set size tell us?

26-2. Chapters [\[ch25\]](./ch25/_books_ml-q-and-ai-ch25.md) and [\[ch26\]](./ch26/_books_ml-q-and-ai-ch26.md) focused on classification methods. Could we use
conformal prediction and confidence intervals for regression too?

26-1 让你把“集合元素个数”解释为模型在该点的把握程度（以及风险暴露）；26-2 提示回归同样可构造参数不确定性与 conformalized 区间两条并行路线。

## References

> 工具与文献入口：MAPIE、Molnar 小书、以及 conformal prediction 资源汇总。

[](#references)

- MAPIE is a popular library for conformal predictions in Python:
  <https://mapie.readthedocs.io/>.

- For more on the score method used in this chapter: Christoph Molnar,
  *Introduction to Conformal Prediction with Python* (2023),
  <https://christophmolnar.com/books/conformal-prediction/>.

- In addition to the score method, several other variants of confor-
   mal prediction methods exist. For a comprehensive collection of
  conformal prediction literature and resources, see the Awesome
  Conformal Prediction page:
  <https://github.com/valeman/awesome-conformal-prediction>.


------------------------------------------------------------------------
