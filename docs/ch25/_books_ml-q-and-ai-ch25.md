








# Chapter 25: Confidence Intervals

> 介绍评估分类器性能的置信区间构造思路：正态近似、训练集 bootstrap、测试集预测 bootstrap，以及更换随机种子重训；并说明各自假设与代价。

[](#chapter-25-confidence-intervals)



**What are the different ways to construct confidence intervals for
machine learning classifiers?**

There are several ways to construct `confidence intervals` for machine
learning models, depending on the model type and the nature of your
data. For instance, some methods are computationally expensive when
working with deep neural networks and are thus more suitable to less
resource-intensive machine learning models. Others require larger
datasets to be reliable.

为机器学习模型构造置信区间（confidence intervals）的途径有多种，具体取决于模型类型与数据特性。例如，某些方法在面对深度神经网络时计算代价很高，因此更适合较轻量的模型；另一些则需要较大样本才能可靠。

The following are the most common methods for constructing confidence
intervals:

- Constructing normal approximation intervals based on a test set

- Bootstrapping training sets

- Bootstrapping the test set predictions

- Confidence intervals from retraining models with different random
  seeds

上述四类依次为：单次划分测试集上的**正态近似**区间；**对训练集**做 bootstrap（多轮训练/评估）；**对测试集预测**做 bootstrap；以及**用不同随机种子多次重训**再汇总精度。

Before reviewing these in greater depth, let's briefly review the
definition and interpretation of confidence intervals.

在深入逐项讨论之前，先简要回顾置信区间的定义与解释方式。

## Defining Confidence Intervals

> 澄清置信区间的统计含义及其在机器学习里与“总体泛化准确率”的对应关系，并图示常见展示方式。

[](#defining-confidence-intervals)

A `confidence interval` is a type of method to estimate an unknown
population parameter. A **population parameter** is a specific measure of
a statistical population, for example, a mean (average) value or
proportion. By "specific" measure, I mean there is a single, exact
value for that parameter for the entire population. Even though this
value may not be known and often needs to be estimated from a sample, it
is a fixed and definite characteristic of the population. A *statistical
population*, in turn, is the complete set of items or individuals we
study.

置信区间用于估计未知的**总体参数**。**总体参数**是对整个统计总体某一数量的刻画，例如均值或比例；“具体”的意思是：对整个总体而言该参数只有一个确定的真值——即便未知且常需用样本估计，它仍是总体固定不变的特征；而**统计总体**就是我们研究的全体对象。

In a machine learning context, the population could be considered the
entire possible set of instances or data points that the model may
encounter, and the parameter we are often most interested in is the true
generalization accuracy of our model on this population.

在机器学习语境下，可把总体理解为模型未来可能遇到的全部实例；而我们最关心的参数之一，往往是模型在该总体上的**真实泛化准确率**。

The accuracy we measure on the test set estimates the true
generalization accuracy. However, it's subject to random error due to
the specific sample of test instances we happened to use. This is where
the concept of a confidence interval comes in. A 95 percent confidence
interval for the generalization accuracy gives us a range in which we
can be reasonably sure that the true generalization accuracy lies.

测试集上算出的准确率是对真实泛化准确率的估计，但会因我们恰好抽到的测试样本而带有随机波动；置信区间正是用来刻画这种不确定性：例如 95% 置信区间给出一个范围，使我们有理由认为真实泛化准确率落在其中。


For instance, if we take 100 different data samples and compute a 95 percent
confidence interval for each sample, approximately 95 of the 100
confidence intervals will contain the true population value (such as the
generalization accuracy), as illustrated in
Figure [25.1](#fig-ch25-fig01).

例如重复抽样 100 次并各算一条 95% 置信区间，则大约其中 95 条会包含真实总体值（例如泛化准确率），如图 [25.1](#fig-ch25-fig01) 所示。

<a id="fig-ch25-fig01"></a>

<div align="center">
  <img src="./images/ch25-fig01.png" alt="Confidence interval illustration" width="95%" />
  <div><b>Figure 25.1</b></div>
</div>

More concretely, if we were to draw 100 different representative test
sets from the population (for instance, the entire possible set of
instances that the model may encounter) and compute the 95 percent
confidence interval for the generalization accuracy from each test set,
we would expect about 95 of these intervals to contain the true
generalization accuracy.

更具体地说，若从总体中抽取 100 个有代表性的测试集并分别构造 95% 泛化准确率置信区间，我们预期约有 95 个区间包含真实泛化准确率。

We can display confidence intervals in several ways. It is common to use
a bar plot representation where the top of the bar represents the
parameter value (for example, model accuracy) and the whiskers denote
the upper and lower levels of the confidence interval (left chart of
Figure [25.2](#fig-ch25-fig02) ). Alternatively, the confidence intervals can be
shown without bars, as in the right chart of
Figure [25.2](#fig-ch25-fig02).

可视化上可以用多种方式呈现置信区间：常见做法是用柱状图顶端表示参数点估计（例如准确率），用误差棒（whiskers）表示区间上下界（见
Figure [25.2](#fig-ch25-fig02) 左图）；也可以不画柱体，只画区间（右图）。

This visualization is functionally useful in a number of ways. For
instance, when confidence intervals for two model performances do *not*
overlap, it's a strong visual indicator that the performances are
significantly different. Take the example of statistical
significance tests, such as t-tests: if two 95 percent confidence
intervals do not overlap, it strongly suggests that the difference
between the two measurements is statistically significant at the 0.05
level.

这种可视化在比较模型时很实用：若两条性能置信区间**完全不重叠**，通常强烈提示两者差异显著；这与显著性检验的直觉一致——两个 95% 区间若不重叠，往往意味着在 0.05 水平上差异显著。

On the other hand, if two 95 percent confidence intervals overlap, we
cannot automatically conclude that there's no significant difference
between the two measurements. Even when confidence intervals overlap,
there can still be a statistically significant difference.

反之，若两个 95% 区间**重叠**，也不能自动推出“无显著差异”；重叠时仍可能存在统计上显著的差别。

Alternatively, to provide more detailed information about the exact
quantities, we can use a table view to express the confidence intervals.
The two common notations are summarized in
Table [25.1](#confidence-intervals).

若需要更精确的数值表达，也可用表格给出区间；两种常见记法见
Table [25.1](#confidence-intervals)。

<a id="confidence-intervals">Table 25.1</a>

| 模型编号 | 置信区间（±表示法）      | 置信区间（下限, 上限）      |
|---------|-------------------------|-----------------------------|
| 1       | 89.1% ± 1.7%            | 89.1% (87.4%, 90.8%)        |
| 2       | 79.5% ± 2.2%            | 79.5% (77.3%, 81.7%)        |
| 3       | 95.2% ± 1.6%            | 95.2% (93.6%, 96.8%)        |



The $\pm$ notation is often preferred if the confidence interval is 
*symmetric*, meaning the upper and lower endpoints are equidistant from
the estimated parameter. Alternatively, the lower and upper confidence
intervals can be written explicitly.

当区间关于点估计**对称**时，常用 $\pm$ 记法；否则可显式写出上下界。

## The Methods

> 分述四种常用构造方式：适用场景、公式要点与局限。

[](#the-methods)

The following sections describe the four most common methods of
constructing confidence intervals.

下面各节概述构造置信区间的四种最常见做法。

### Method 1: Normal Approximation Intervals

> 单次训练–测试划分下用正态近似与标准误估计准确率区间；省事但依赖分布假定且看不到划分间波动。

[](#method-1-normal-approximation-intervals)

The normal approximation interval involves generating the confidence
interval from a single train-test split. It is often considered the
simplest and most traditional method for computing confidence intervals.
This approach is especially appealing in the realm of deep learning,
where training models is computationally costly. It's also desirable
when we are interested in evaluating a specific model, instead of models
trained on various data partitions like in *k*-fold cross-validation.

正态近似区间在单次 train–test 划分下即可得到置信区间，常被视为最简单、最“传统”的做法；特别适合深度学习中训练代价高的场景；当你想评估“这一版具体模型”、而不是像在 *k* 折交叉验证那样比较多种划分时也很有吸引力。

How does it work? In short, the formula for calculating the confidence
interval for a predicted parameter (for example, the sample mean,
denoted as $\bar{x}$), assuming a normal
distribution, is expressed as $\bar{x} \pm z \times \mathrm{SE}$.

思路简述如下：在给定近似正态的假定下，对某估计量（如样本均值 $\bar{x}$），置信区间为 $\bar{x} \pm z \times \mathrm{SE}$。

In this formula, *z* represents the *z*-score, which indicates a
particular value's number of standard deviations from the mean in a
standard normal distribution. *SE* represents the standard error of the
predicted parameter (in this case, the sample mean).

其中 $z$ 为标准正态下对应置信水平的临界值所用 *z-score*；$\mathrm{SE}$ 为该估计量（此处为样本均值）的**标准误**。

Most readers will be familiar with `z-score` tables that are
usually found in the back of introductory statistics textbooks. However,
a more convenient and preferred way to obtain `z-scores` is to
use functions like SciPy's `stats.zscore` function, which computes
the `z-scores` for given confidence levels.

教科书附录常附 *z* 表；更便利的是像 SciPy `stats.norm.ppf` 等方式按置信水平求得临界值。（注意：原书示例提到 `stats.zscore`——它通常用于对样本逐元素标准化；求置信区间的临界倍数时，一般用逆 CDF/`ppf` 类函数更贴切。）

For our scenario, the sample mean, denoted as
$\bar{x}$, corresponds to the test set accuracy,
$\mathrm{ACC}_{\mathrm{test}}$, a measure of successful predictions in the context of a
binomial proportion confidence interval.

对本章场景，样本均值可取为测试准确率 $\mathrm{ACC}_{\mathrm{test}}$，可视作二项比例的正态近似问题中的成功比例估计。

The standard error can be calculated under a normal approximation as
follows:

$$
\mathrm{SE} = \sqrt{ \frac{1}{n} \, \mathrm{ACC}_{\mathrm{test}} \left(1 - \mathrm{ACC}_{\mathrm{test}}\right) }
$$

In this equation, $n$ signifies the size of the test set. Substituting
the standard error back into the previous formula, we obtain the
following:

$$
\mathrm{ACC}_{\mathrm{test}} \pm z \sqrt{ \frac{1}{n} \, \mathrm{ACC}_{\mathrm{test}} \left(1 - \mathrm{ACC}_{\mathrm{test}}\right) }
$$

在正常近似下，标准误由前一式给出；$n$ 为测试集样本量；将 $\mathrm{SE}$ 代回 $\bar{x}\pm z\cdot\mathrm{SE}$ 即得测试准确率置信区间的常见写法。

Additional code examples to implement this method can also be found in
the *supplementary/q25_confidence-intervals* subfolder in the
supplementary code repository at
<https://github.com/rasbt/MachineLearning-QandAI-book>. While the normal
approximation interval method is very popular due to its simplicity, it
has some downsides. First, the normal approximation may not always be
accurate, especially for small sample sizes or for data that is not
normally distributed. In such cases, other methods of computing
confidence intervals may be more accurate. Second, using a single
train-test split does not provide information about the variability of
the model performance across different splits of the data. This can be
an issue if the performance is highly dependent on the specific split
used, which may be the case if the dataset is small or if there is a
high degree of variability in the data.

配套代码见 *supplementary/q25_confidence-intervals*（同上链接）。该方法简单好用，但要注意：近似失准常见于小样本与强偏态情形；单次划分也无法反映换划分导致的性能起伏——数据稀缺或划分敏感时需配合其他途径。

### Method 2: Bootstrapping Training Sets

> 通过对数据有放回抽样生成多轮训练/测试划分，用准确率分布的分位数构造区间；更“分布中立”但更贵且仍受限于所见数据。

[](#method-2-bootstrapping-training-sets)

Confidence intervals serve as a tool for approximating unknown
parameters. However, when we are restricted to just one estimate, such
as the accuracy derived from a single test set, we must make certain
assumptions to make this work. For example, when we used the normal
approximation interval described in the previous section, we assumed
normally distributed data, which may or may not hold.

置信区间用于逼近未知总体参数；若只有单次测试集上的准确率点估计，往往就要依赖额外假定——例如前文正态近似所要求的分布性质。

In a perfect scenario, we would have more insight into our test set
sample distribution. However, this would require access to many
independent test datasets, which is typically not feasible. A workaround
is the bootstrap method, which resamples existing data to estimate the
sampling distribution.

理想情况下我们希望直接了解测试性能的抽样分布；但独立测试集往往不可得，于是用 bootstrap 在有放回抽样中估计该分布。

In practice, when the test set is large enough, the normal distribution
approximation will hold, thanks to the central limit theorem. This
theorem states that the sum (or average) of a large number of
independent, identically distributed random variables will approach a
normal distribution, regardless of the underlying distribution of the
individual variables. It is difficult to specify what constitutes a
large-enough test set. However, under stronger assumptions than those of
the central limit theorem, we can at least estimate the rate of
convergence to the normal distribution using the Berry-Esseen theorem,
which gives a more quantitative estimate of how quickly the convergence
in the central limit theorem occurs.

测试集足够大时，许多统计量会因中心极限定理而近似正态，但“多大算够大”并无统一标尺；Berry–Esseen 定理在给定更强的矩条件下，可对收敛速率给出更可量化的界限。

In a machine learning context, we can take the original dataset and draw a random
sample *with replacement*. If the dataset has size $n$ and we draw a
random sample with replacement of size $n$, this implies that some data
points will likely be duplicated in this new sample, whereas other data
points are not sampled at all. We can then repeat this procedure for
multiple rounds to obtain multiple training and test sets. This process
is known as *out-of-bag bootstrapping*, illustrated in
Figure [25.4](#fig-ch25-fig04).

在机器学习场景中，可先对数据集做容量为 $n$ 的有放回抽样；新样本中会重复出现一些点也会漏掉一些点；重复多轮可得到多套训练–测试划分，即 **out-of-bag bootstrap**（图 [25.4](#fig-ch25-fig04)）。

<a id="fig-ch25-fig04"></a>

<div align="center">
  <img src="./images/ch25-fig04.png" alt="Out-of-bag bootstrapping illustration" width="95%" />
  <div><b>Figure 25.4</b></div>
</div>

Suppose we constructed *k* training and test sets. We can now take each
of these splits to train and evaluate the model to obtain *k* test set
accuracy estimates. Considering this distribution of test set accuracy
estimates, we can take the range between the 2.5th and 97.5th percentile
to obtain the 95 percent confidence interval, as illustrated in
Figure [25.5](#fig-ch25-fig05).

若有 $k$ 套划分，可训练 $k$ 个模型并得到 $k$ 个测试准确率；再在这组经验分布上取第 2.5 与第 97.5 百分位，得到近似的 95% 区间（图 [25.5](#fig-ch25-fig05)）。

<a id="fig-ch25-fig05"></a>

<div align="center">
  <img src="./images/ch25-fig05.png" alt="Distribution of test accuracies from 1,000 bootstrap samples, including a 95 percent confidence interval" width="78%" />
  <div><b>Figure 25.5</b></div>
</div>

Unlike the normal approximation interval method, we can consider this
out-of-bag bootstrap approach to be more agnostic to the specific
distribution. Ideally, if the assumptions for the normal approximation
are satisfied, both methodologies would yield identical outcomes.

与正态近似相比，该方法对总体分布形态的依赖更弱；若正态近似的条件成立，两种路径在理想情况下应给出相近结论。

Since bootstrapping relies on resampling the existing data used in those bootstrap rounds, its
downside is that it doesn't bring in any new information that could be
available in a broader population or unseen data. Therefore, it may not
always be able to generalize the performance of the model to new, unseen
data.

由于 bootstrap 只复用“已观测到的数据”，并不会补充来自更广总体的全新信息，因此对外推到完全未见场景仍需谨慎解读。

Note that we are using the bootstrap sampling approach in this chapter
instead of obtaining the train-test splits via *k*-fold
cross-validation, because of the bootstrap's theoretical grounding via
the central limit theorem discussed earlier. There are also more
advanced out-of-bag bootstrap methods, such as the .632 and .632+
estimates, which are reweighting the accuracy estimates.

本章采用自助划分而非 *k* 折，部分是为了与前面讨论的极限理论叙事衔接；工程上还存在 .632、.632+ 等对 out-of-bag 误差做加权修正的更精细版本。

### Method 3: Bootstrapping Test Set Predictions

> 固定一套已训模型，仅对测试集做 bootstrap；省重训但看不到训练数据微小变动带来的不稳定。

[](#method-3-bootstrapping-test-set-predictions)

An alternative approach to bootstrapping training sets is to bootstrap
test sets. The idea is to train the model on the existing training set as
usual and then to evaluate the model on bootstrapped test sets, as
illustrated in Figure [25.6](#fig-ch25-fig06). After obtaining the test set performance
estimates, we can then apply the percentile method described in the
previous section.

与“重抽样训练集”相对，这一做法固定一套训练数据先训好模型，再对**测试集**做有放回抽样；得到多条性能序列后同样可用分位数法构造区间（图 [25.6](#fig-ch25-fig06)）。

<a id="fig-ch25-fig06"></a>

<div align="center">
  <img src="./images/ch25-fig06.png" alt="Bootstrapping the test set" width="78%" />
  <div><b>Figure 25.6</b></div>
</div>


Contrary to the prior bootstrap technique, this method uses a trained
model and simply resamples the test set (instead of the training sets).
This approach is especially appealing for evaluating deep neural
networks, as it doesn't require retraining the model on the new data
splits. However, a disadvantage of this approach is that it doesn't
assess the model's variability toward small changes in the training
data.

其优点是评估深度模型时无需为每个 bootstrap 副本重训，计算上更省事；缺点是它**不包含**训练集扰动所带来的模型变异，只适合回答“对这一固定学到的模型，测试抽样不确定性如何”这一类问题。

### Method 4: Retraining Models with Different Random Seeds

> 汇总多次随机种子实验的均值与标准误，用 *t* 临界值近似区间；最贵但利于理解深度模型稳定性。

[](#method-4-retraining-models-with-different-random-seeds)


In deep learning, models are commonly retrained using various random
seeds since some random weight initializations may lead to much better
models than others. How can we build a confidence interval from these
experiments? If we assume that the sample means follow a normal
distribution, we can employ a previously discussed method where we
calculate the confidence interval around a sample mean, denoted as
$\bar{x}$, as follows:

$$
\bar{x} \pm z \cdot \mathrm{SE}
$$

深度学习中常会更换随机种子重训同一结构；若把这些重复试验得到的均值视作近似服从正态的样本均值，可先写出上述 $\bar{x} \pm z \cdot \mathrm{SE}$ 的范式。

Since in this context we often work with a relatively modest number of
samples (for instance, models from 5 to 10 random seeds), assuming a $t$
distribution is deemed more suitable than a normal distribution.
Therefore, we substitute the $z$ value with a $t$ value in the preceding
formula. (As the sample size increases, the $t$ distribution tends to
look more like the standard normal distribution, and the critical values
[$z$ and $t$] become increasingly similar.)

但实际重复次数往往很少（例如 5–10 个种子），用 **t 分布**临界值替代 $z$ 更谨慎；当样本量增大时，$t$ 与标准正态的临界值会逐渐接近。

Furthermore, if we are interested in the average accuracy, denoted as
$\overline{\mathrm{ACC}}_{\mathrm{test}}$, we consider $\mathrm{ACC}_{\mathrm{test},\,j}$
corresponding to a unique random seed $j$ as a sample. The number of
random seeds we evaluate would then constitute the sample size $n$. As
such, we would calculate:

$$
\overline{\mathrm{ACC}}_{\mathrm{test}} \pm t \cdot \mathrm{SE}
$$

若关注的是各随机种子下测试准确率 $\mathrm{ACC}_{\mathrm{test},\,j}$ 的平均水平，就把它当作 $r$ 次重复试验的观测，形式上仍写成 $\overline{\mathrm{ACC}}_{\mathrm{test}} \pm t \cdot \mathrm{SE}$。

Here, $\mathrm{SE}$ is the standard error, calculated as
$\mathrm{SE} = \mathrm{SD} / \sqrt{n}$, while

$$
\overline{\mathrm{ACC}}_{\mathrm{test}} = \frac{1}{r} \sum_{j=1}^{r} \mathrm{ACC}_{\mathrm{test},\,j}
$$

is the average accuracy, which we compute over the $r$ random seeds. The
standard deviation $\mathrm{SD}$ is calculated as follows:

$$
\mathrm{SD} = \sqrt{ \frac{ \sum_{j=1}^{r} \left( \mathrm{ACC}_{\mathrm{test},\,j} - \overline{\mathrm{ACC}}_{\mathrm{test}} \right)^2 }{ r-1 } }
$$

其中 $\mathrm{SE}=\mathrm{SD}/\sqrt{n}$；$\overline{\mathrm{ACC}}_{\mathrm{test}}$ 为 $r$ 个随机种子结果的样本均值；$\mathrm{SD}$ 则按上面最后一式对所有 $\mathrm{ACC}_{\mathrm{test},\,j}$ 做无偏样本标准差估计。

To summarize, calculating the confidence intervals using various random seeds
is another effective alternative. However, it is primarily beneficial
for deep learning models. It proves to be costlier than both the normal
approximation approach (method 1) and bootstrapping the test set
(method 3), as it necessitates retraining the model. On the bright
side, the outcomes derived from disparate random seeds provide us with a
robust understanding of the model's stability.

综上，多随机种子汇总是一条对深度模型特别自然的补充工具：它比方法 1 和方法 3 更“贵”（要反复训练），但能同时反映随机性带来的性能散布，从而更直接地刻画**训练稳定性**。

## Recommendations

> 用一张对照表收束四种路径：计算代价、分布假设与对“稳定性/划分敏感度”的诊断价值。

[](#recommendations)

Each possible method for constructing confidence intervals has its
unique advantages and disadvantages. The normal approximation interval
is cheap to compute but relies on the normality assumption about the
distribution. The out-of-bag bootstrap is agnostic to these assumptions
but is substantially more expensive to compute. A cheaper alternative is
bootstrapping the test only, but this involves bootstrapping a smaller
dataset and may be misleading for small or nonrepresentative test set
sizes. Lastly, constructing confidence intervals from different random
seeds is expensive but can give us additional insights into the
model's stability.

一条实用对照是：正态近似最省算力但强依赖分布假定；全集 out-of-bag bootstrap 对分布较“超脱”但更耗时；只对测试 bootstrap 在中间地带，却会低估训练扰动引发的差异；汇总多种子主要服务深度模型的**稳定性画像**，但需要反复训练买单。

## Exercises

> 练习题：置信水平与区间宽窄；更高效地评估 bootstrap 测试集准确率。

[](#exercises)

25-1. As mentioned earlier, the most common choice of confidence level
is 95 percent confidence intervals. However, 90 percent and 99 percent
are also common. Are 90 percent confidence intervals smaller or wider
than 95 percent confidence intervals, and why is this the case?

25-2. In the discussion of Method 3 bootstrapping the test set earlier in this chapter, we created test sets by bootstrapping and
then applied the already trained model to compute the test set accuracy
on each of these datasets. Can you think of a method or modification to
obtain these test accuracies more efficiently?

90% 与 95% 置信区间的宽窄关系需要从“覆盖率—区间长度”的取舍来理解；25-2 则引导你把“重复在 bootstrap 测试集上算准确率”与更省算的近似或加权估计联系起来。

## References

> 延伸阅读：置信区间可视化误区、二项比例区间、中心极限与 Berry–Esseen、以及 .632 bootstrap 系列文献。

[](#references)

- A detailed discussion of the pitfalls of concluding statistical
  significance from nonoverlapping confidence intervals: Martin
  Krzywinski and Naomi Altman, "Error Bars" (2013),
  <https://www.nature.com/articles/nmeth.2659>.

- A more detailed explanation of the binomial proportion confidence
  interval:
  <https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval>.

- For a detailed explanation of normal approximation intervals, see
  Section 1.7 of my article: "Model Evaluation, Model Selection, and
  Algorithm Selection in Machine Learning" (2018),
  <https://arxiv.org/abs/1811.12808>.

- Additional information on the central limit theorem for inde-
   pendent and identically distributed random variables:
  <https://en.wikipedia.org/wiki/Central_limit_theorem>.

- For more on the Berry-Esseen theorem:
  [https://en.wikipedia.org/wiki/Berry–Esseen_theorem](https://en.wikipedia.org/wiki/Berry%E2%80%93Esseen_theorem).

- The .632 bootstrap addresses a pessimistic bias of the regular
  out-of-bag bootstrapping approach: Bradley Efron, "Estimating the
  Error Rate of a Prediction Rule: Improvement on Cross-Validation"
  (1983), <https://www.jstor.org/stable/2288636>.

- The .632+ bootstrap corrects an optimistic bias introduced in the .632
  bootstrap: Bradley Efron and Robert Tibshirani, "Improvements on
  Cross-Validation: The .632+ Bootstrap Method" (1997),
  <https://www.jstor.org/stable/2965703>.

- A deep learning research paper that discusses bootstrapping the test
  set predictions: Benjamin Sanchez-Lengeling et al., "Machine
  Learning for Scent: Learning Generalizable Perceptual Representations
  of Small Molecules" (2019), <https://arxiv.org/abs/1910.10685>.


------------------------------------------------------------------------
