







# Chapter 23: Data Distribution Shifts
> 本章界定训练分布与部署后真实分布不一致的问题，分述协变量偏移、标签偏移、概念漂移与域偏移，并比较严重程度、监控要点与练习。
[](#chapter-23-data-distribution-shifts)



**What are the main types of data distribution shifts we may encounter
after model deployment?**

模型上线之后，我们可能遇到哪些主要的**数据分布偏移（data distribution shifts）**类型？

*Data distribution shifts* are one of the most common problems when
putting machine learning and AI models into production. In short, they
refer to the differences between the distribution of data on which a
model was trained and the distribution of data it encounters in the real
world. Often, these changes can lead to significant drops in model
performance because the model's predictions are no longer accurate.

*数据分布偏移*是把 ML/AI 模型投入生产时最常见的问题之一：简而言之，指**训练数据分布**与**线上/真实世界所见数据分布**之间的差异。此类变化常导致性能明显下滑，因为旧假设下的预测不再可靠。

> Tips:**数据分布偏移**是生产环境中使用模型时，最常见的问题。
> 
> - 指的是，模型在**训练时所使用的数据**分布，与在**实际应用中遇到的数据**分布之间的差异。
> - 通常，这些变化会导致`模型性能`显著下降，因为模型的预测不再准确。



There are several types of distribution shifts, some of which are more
problematic than others. The most common are covariate shift, concept
drift, label shift, and domain shift; all discussed in more detail in
the following sections.

分布偏移有多种，棘手程度不一；最常见包括**协变量偏移、概念漂移、标签偏移与域偏移**，下文分别展开。

> Tips:
> 
> - 数据分布偏移，有多种类型，其中最常见的是：协变量偏移、概念漂移、标签偏移和域偏移。
> - 这些偏移类型，将在后续章节中详细讨论。

## Covariate Shift
> 本节在 $p(x)$、$p(y)$、$p(y|x)$ 记号下定义协变量偏移，举例说明并点出对抗验证与重要性加权等应对思路。
[](#covariate-shift)

Suppose $p(x)$ describes the distribution of the input data (for
instance, the features), $p(y)$ refers to the distribution of the
target variable (or class label distribution), and $p(y|x)$ is the
distribution of the targets $y$ given the inputs $x$.

设 $p(x)$ 为输入（特征）分布，$p(y)$ 为目标（或类标签边缘）分布，$p(y|x)$ 为给定输入下目标的条件分布。

*Covariate shift* happens when the distribution of the input data,
$p(x)$, changes, but the conditional distribution of the output given
the input, $p(y|x)$, remains the same.

*协变量偏移（covariate shift）*指：输入分布 $p(x)$ 变化，而 $p(y|x)$ 不变。

> Tips:
> 
> - 协变量偏移，指的是，输入数据分布 $p(x)$ 发生变化，但输出条件分布 $p(y|x)$ 保持不变。
> - 协变量 covariate，一般是指 特征变量，通常会影响输出结果，但并不一定是主要因素.

<div align="center">
  <img src="./images/ch23-fig01.png" alt="Training data and new data distributions differ under covariate shift." width="65%" />
  <div><b>Figure 23.1</b></div>
</div>

For example, suppose we trained a model to predict whether an email is
spam based on specific features. Now, after we embed the email spam
filter in an email client, the email messages that customers receive
have drastically different features. For example, the email messages are
much longer and are sent from someone in a different time zone. However,
if the way those features relate to an email being spam or not doesn't
change, then we have a covariate shift.

例如训练邮件反垃圾模型后，嵌入客户端；用户收件特征整体变了（更长、来自不同时区等），但若“特征如何对应是否垃圾邮件”的规则未变，则属于协变量偏移。

Covariate shift is a very common challenge when deploying machine
learning models. It means that the data the model receives in a live or
production environment is different from the data on which it was
trained. However, because the relationship between inputs and outputs,
$p(y|x)$, remains the same under covariate shift, techniques are
available to adjust for it.

协变量偏移在部署中极常见：线上分布与训练集不同，但因 $p(y|x)$ 不变，仍有一类校正手段可用。

> Tips:
> 
> - 协变量偏移，模型在实际应用中遇到的**数据分布**，与在训练时所使用的数据分布**不同**。
> - 但是，由于输入和输出之间的关系 $p(y|x)$ 保持不变，因此有调整方法，例如：对抗验证、重要性加权等。

A common technique to detect covariate shift is *adversarial validation*, which is covered in more detail in
Chapter [\[ch29\]](./ch29/_books_ml-q-and-ai-ch29.md).
Once covariate shift is detected, a common method to deal with it is
**importance weighting**, which assigns different weights to the training
example to emphasize or de-emphasize certain instances during training.
Essentially, instances that are more likely to appear in the test
distribution are given more weight, while instances that are less likely
to occur are given less weight. This approach allows the model to focus
more on the instances representative of the test data during training,
making it more robust to covariate shift.

检测协变量偏移的常用做法是*对抗验证（adversarial validation）*（详见第 29 章）。发现偏移后，常见处理是**重要性加权（importance weighting）**：给训练样本不同权重，使更像在目标分布会出现的样本在训练中更受重视、更少出现的则被压低，从而让模型更贴近测试分布、提升对协变量偏移的鲁棒性。

## Label Shift
> 本节定义标签偏移（先验漂移），给出邮件垃圾比例的例子，并说明加权损失等对策。
[](#label-shift)

*Label shift*, sometimes referred to as *prior probability shift*,
occurs when the class label distribution $p(y)$ changes, but the
class-conditional distribution $p(y|x)$ remains unchanged. In
other words, there is a significant change in the label distribution or
target variable.

*标签偏移（label shift）*也称*先验概率偏移*：类别边缘分布 $p(y)$ 变，而 $p(y|x)$ 不变，即可理解为标签／目标变量的整体配比发生明显变化。

> Tips:
> 
> - 标签偏移，指的是，标签分布 $p(y)$ 发生变化，但条件分布 $p(y|x)$ 保持不变。
> - 标签偏移，通常与目标变量（或类标签分布）的变化有关。

As an example of such a scenario, suppose we trained an email spam
classifier on a balanced training dataset with 50 percent spam and 50
percent non-spam email. In contrast, in the real world, only 10 percent
of email messages are spam.

例如训练集里垃圾与非垃圾各占 50%，而真实世界里垃圾邮件仅占约 10%。

A common way to address label shifts is to update the model using the
`weighted loss function`, especially when we have an idea of the new
distribution of the labels. This is essentially a form of **importance
weighting**. By adjusting the weights in the loss function according to
the new label distribution, we are incentivizing the model to pay more
attention to certain classes that have become more common (or less
common) in the new data. This helps align the model's predictions more
closely with the current reality, improving its performance on the new
data.

对策之一是用**加权损失**更新模型，尤其在已知新标签分布时；本质上仍是**重要性加权**：按新分布调节各类在损失中的权重，促使模型更关注变多或变少的类别，从而更贴近当前数据。

> Tips: 损失函数加权，突出重要样本分类。

## Concept Drift
> 本节说明概念漂移即 $p(y|x)$ 的变化，并结合反垃圾例子讨论其难于处理之处。
[](#concept-drift)

*Concept drift* refers to the change in the mapping between the input
features and the target variable. In other words, concept drift is
typically associated with changes in the conditional distribution
$p(y|x)$, such as the relationship between the inputs $x$ and the output
$y$.

*概念漂移（concept drift）*指输入与目标之间映射关系发生变化，通常体现为条件分布 $p(y|x)$ 变了，即 $x$ 如何决定 $y$ 的规律改变。

> Tips:
> 
> - 概念漂移，指的是，输入特征与目标变量之间的映射关系发生变化。
> - 概念漂移，通常与**条件分布** $p(y|x)$ 的变化有关。

Using the example of the spam email classifier from the previous
section, the features of the email messages might remain the same, but
*how* those features relate to whether an email is spam might change.
This could be due to a new spamming strategy that wasn't present in
the training data. Concept drift can be much harder to dealt with than
the other distribution shifts discussed so far since it requires
continuous monitoring and potential model retraining.

延续邮件例子：邮件表层特征可能相仿，但“这些特征如何指向是否垃圾”的规则变了——例如出现训练期未见的新型诈骗话术。概念漂移往往比前述几类更难处理，需要持续监控并可能反复重训。

## Domain Shift
> 本节澄清文献中域偏移与概念漂移的混用，给出联合分布视角与公式，并回到邮件例子说明为何最难应对。
[](#domain-shift)

The terms *domain shift* and *concept drift* are used somewhat
inconsistently across the literature and are sometimes taken to be
interchangeable. In reality, the two are related but slightly different
phenomena. *Concept drift* refers to a change in the function that maps
from the inputs to the outputs, specifically to situations where the
relationship between features and target variables changes as more data
is collected over time.

文献中对*域偏移（domain shift）*与*概念漂移*的用法并不完全一致，有时还被混用；实际上二者相关却略有差别。*概念漂移*强调从输入到输出的映射在变，即特征与目标的**关系**随时间推移而改变。

> Tips: 
> 
> - 领域偏移，通常跟概念漂移有差异。
> - 领域偏移，指的是，输入数据分布 $p(x)$ 和输出条件分布 $p(y|x)$ 都发生变化。
> - 领域偏移，也被称为**联合分布偏移**，因为联合分布 $p(x, y)$ 是输入和输出分布的乘积。

In *domain shift*, the distribution of inputs, $p(x)$, and the
conditional distribution of outputs given inputs, $p(y|x)$, both change.
This is sometimes also called *joint distribution shift* due to the
joint distribution:

*域偏移*指 $p(x)$ 与 $p(y|x)$ **同时**变化；因联合分布可分解为二者乘积，故有时也称*联合分布偏移*：

$$
p(x, y) = p(y|x) \cdot p(x)
$$

We can thus think of domain shift as a combination of both covariate shift and concept drift. In addition, since we can obtain the marginal distribution $p(y)$ by integrating over the joint distribution $p(x, y)$ over the variable $x$ (mathematically expressed as
$p(y) = \int p(x, y) \, dx$), covariate drift and concept shift also imply label shift. (However, exceptions may exist where the change in $p(x)$ compensates for the change in $p(y|x)$ such that $p(y)$ may not change.) Conversely, label shift and concept drift usually also imply covariate shift.

因此域偏移可视为协变量偏移与概念漂移的“叠加”。又因 $p(y)$ 可通过对联合分布 $p(x,y)$ 关于 $x$ 积分得到（$p(y) = \int p(x, y) \, dx$），协变量侧与概念侧的变化通常也会牵动标签边缘分布（存在 $p(x)$ 变化与 $p(y|x)$ 变化相互抵消、使 $p(y)$ 看似不变的例外）。反过来，标签偏移与概念漂移往往也意味着输入分布层面需要随之调整。

To return once more to the example of email spam classification, domain
shift would mean that the features (content and structure of email)
*and* the relationship between the features and target both change over
time. For instance, spam email in 2023 might have different features
(new types of phishing schemes, new language, and so forth), and the
definition of what constitutes spam might have changed as well. This
type of shift would be the most challenging scenario for a spam filter
trained on 2020 data, as it would have to adjust to changes in both the
input data and the target concept.

再以邮件反垃圾为例：域偏移意味着邮件的内容／结构特征与“何为垃圾”的定义**一起**随时间而变；若模型仍停留在 2020 年的数据与标签语义上，却要面对 2023 年的新模式与新规则，这是最棘手的情形。

Domain shift is perhaps the most difficult type of shift to handle, but
monitoring model performance and data statistics over time can help
detect domain shifts early. Once they are detected, mitigation
strategies include collecting more labeled data from the target domain
and retraining or adapting the model.

域偏移或许最难处理，但通过持续监控模型表现与数据统计，有望尽早发现；一旦确认，可收集目标域标注并重训或做模型自适应等缓解。


## Types of Data Distribution Shifts
> 本节借助示意图比较各类偏移的相对“杀伤力”，强调现实语境中严重度取决于场景，并重申监控与预警的重要性。
[](#types-of-data-distribution-shifts)

<div align="center">
  <img src="./images/ch23-fig02.png" alt="Different types of data shifts in a binary classification context" width="52%" />
  <div><b>Figure 23.2</b></div>
</div>

As noted in the previous sections, some types of distribution shift are
more problematic than others. The least problematic among them is
typically `covariate shift`. Here, the distribution of the input features,
$p(x)$, changes between the training and testing data, but the
conditional distribution of the output given the inputs, $p(y|x)$,
remains constant. Since the underlying relationship between the inputs
and outputs remains the same, the model trained on the training data
can still apply, in principle, to the testing data and new data.

如前所述，不同偏移的麻烦程度不同。通常相对“温和”的是**协变量偏移**：训练与测试之间 $p(x)$ 变，而 $p(y|x)$ 不变；因输入—输出规律未改，原则上训练好的模型仍有望迁移到新数据。

The most problematic type of distribution shift is typically `joint distribution shift`, 
where both the input distribution $p(x)$ and the
conditional output distribution $p(y|x)$ change. This makes it
particularly difficult for a model to adjust, as the learned
relationship from the training data may no longer hold. The model has to
cope with both new input patterns and new rules for making predictions
based on those patterns.

最棘手的是**联合分布偏移**：$p(x)$ 与 $p(y|x)$ 同时变，训练期学到的关系可能整体失效，模型既要适应新输入模式，也要适应基于这些模式的新预测规则。

However, the "severity"? of a shift can vary widely depending on the
real-world context. For example, even a covariate shift can be extremely
problematic if the shift is severe or if the model cannot adapt to the
new input distribution. On the other hand, a joint distribution shift
might be manageable if the shift is relatively minor or if we have
access to a sufficient amount of labeled data from the new distribution
to retrain our model.

但现实里“严重度”取决于场景：协变量偏移若极其剧烈或模型无法适配新 $p(x)$，同样可能灾难性；反之，若联合偏移较温和，或能从新分布拿到足够标注重训，也未必不可控。

In general, it's crucial to monitor our models' performance and be
aware of potential shifts in the data distribution so that we can take
appropriate action if necessary.

总体而言，持续监控模型表现并对潜在分布变化保持警觉，才能在必要时及时采取行动。

> Tips: **监控**模型性能，及时发现潜在`数据分布偏移`，非常重要。

## Exercises
> 本节习题讨论重要性加权的局限，以及在无新标签时如何察觉偏移。
[](#exercises)

23-1. What is the big issue with importance weighting as a technique to
mitigate covariate shift?

23-1. 把重要性加权作为缓解协变量偏移的手段，主要问题是什么？

23-2. How can we detect these types of shifts in real-world scenarios,
especially when we do not have access to labels for the new data?

23-2. 在真实场景中，尤其当新数据没有标签时，如何检测这些偏移？

## References
> 本节给出域偏移缓解与域适应相关的综述文献链接。
[](#references)

- Recommendations and pointers to advanced mitigation techniques for
  avoiding domain shift: Abolfazl Farahani et al., "A Brief Review of
  Domain Adaptation"? (2020), <https://arxiv.org/abs/2010.03978>.

- 关于规避域偏移及进阶缓解技术的建议与索引：Farahani 等，《A Brief Review of Domain Adaptation?》(2020)，<https://arxiv.org/abs/2010.03978>。


------------------------------------------------------------------------

