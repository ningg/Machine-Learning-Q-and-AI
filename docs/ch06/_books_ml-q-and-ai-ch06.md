







# Chapter 6: Reducing Overfitting with Model Modifications
> 本章说明在已运用数据层面手段之后，如何通过修改模型结构与训练流程进一步抑制过拟合。
[](#chapter-6-reducing-overfitting-with-model-modifications)



**Suppose we train a neural network classifier in a supervised fashion
and already employ various dataset-related techniques to mitigate
overfitting. How can we change the model or make modifications to the
training loop to further reduce the effect of overfitting?**

假设我们已在监督训练中采用了多种与数据集相关的抗过拟合措施，还能如何通过改动模型或训练环节来进一步削弱过拟合的影响？

The most successful approaches against overfitting include
`regularization` techniques like `dropout` and `weight decay`. As a rule of
thumb, models with a larger number of parameters require more training
data to generalize well. Hence, decreasing the model size and capacity
can sometimes also help reduce overfitting. Lastly, building ensemble
models is among the most effective ways to combat overfitting, but it
comes with increased computational expense.

对抗过拟合较成功的路径包括 `dropout`、`weight decay` 等 `regularization`（正则化）技术；经验上参数量越大往往需要越多训练数据才能泛化良好，因此适度减小模型容量有时也有帮助；此外，集成多个模型效果很好，但计算开销更高。

> Tips: 减少`过拟合`，最有效的技术是`正则化`，包括`dropout`和`权重衰减`；此外，还可以`减小模型大小`和`构建集成模型`。

This chapter outlines the key ideas and techniques for several
categories of reducing overfitting with model modifications and then
compares them to one another. It concludes by discussing how to choose
between all types of overfitting reduction methods, including those
discussed in the previous chapter.

本章先按若干类别梳理通过模型修改减轻过拟合的核心思路与技术并相互对照，最后讨论如何在包含上一章数据手段在内的全部方法之间做出取舍。

## Common Methods
> 本节将模型与训练相关的抗过拟合技术归为正则化、选用更小模型与构建集成模型三大类。
[](#common-methods)

The various model- and training-related techniques to reduce overfitting
can be grouped into three broad categories: (1) adding `regularization`,
(2) choosing `smaller models`, and (3) building `ensemble models`.

与模型和训练相关的减拟合手段可粗略归为三类：（1）加入 `regularization`（正则化），（2）选择 `smaller models`（更小的模型），（3）构建 `ensemble models`（集成模型）。

### Regularization
> 本节解释正则化及其典型实现，并概述 dropout 与早停等作用机制。
[](#regularization)

We can interpret regularization as a penalty against complexity. Classic
regularization techniques for neural networks include $L_2$
regularization and the related weight decay method. We implement $L_2$
regularization by adding a penalty term to the loss function that is
minimized during training. This added term represents the size of the
weights, such as the squared sum of the weights. The following formula
shows an $L_2$ regularized loss

可将正则化理解为对模型复杂度的惩罚；神经网络的经典做法包括 $L_2$ 正则化及相关的权重衰减：在训练最小化的损失上加上刻画权重大小（如权重平方和）的惩罚项。下列公式给出带 $L_2$ 正则的损失：

$$RegularizedLoss=Loss+\frac{\lambda}{n} \sum_j w_{j}^{2}$$

where $\lambda$ is a `hyperparameter` that controls the
`regularization strength`.

其中 $\lambda$ 为控制 `regularization strength`（正则化强度）的 `hyperparameter`（超参数）。


During backpropagation, the optimizer minimizes the modified loss, now
including the additional penalty term, which leads to smaller model
weights and can improve generalization to unseen data.

反向传播中最小化加入惩罚项后的总损失，会使权重趋向更小，从而有望改善对未见数据的泛化。

> Tips: 正则化 `regularization`，通过`添加惩罚项`，来减少模型的`权重`。


`Weight decay` is similar to $L_2$ regularization but is applied to the
optimizer directly rather than modifying the loss function. Since weight
decay has the same effect as $L_2$ regularization, the two methods are
often used synonymously, but there may be subtle differences depending
on the implementation details and optimizer.

`Weight decay`（权重衰减）与 $L_2$ 正则类似，但直接作用在优化器而非改写损失；二者效果常被视作等价而混用，具体实现与优化器不同时仍可能有细微差别。

Many other techniques have regularizing effects. For brevity's sake,
we'll discuss just two more widely used methods: `dropout` and `early stopping`.

还有许多技术具有正则化效果；为简短起见，这里再介绍两种常用方法：`dropout` 与 `early stopping`（早停）。

`Dropout` reduces overfitting by randomly setting some of the activations
of the hidden units to zero during training. Consequently, the neural
network cannot rely on particular neurons to be activated. Instead, it
learns to use a larger number of neurons and multiple independent
representations of the same data, which helps to reduce overfitting.

`Dropout` 在训练时随机将部分隐藏单元激活置零，使网络无法依赖少数神经元；因而会动用更多神经元并形成多种独立表征，从而有助于减轻过拟合。

In `early stopping`, we monitor the model's performance on a validation
set during training and stop the training process when the performance
on the validation set begins to decline, as illustrated in
Figure [6.1](#fig-ch06-fig01).

在 `early stopping` 中，我们在训练过程中监控模型在验证集上的表现，并在验证性能开始下滑时终止训练，见图 [6.1](#fig-ch06-fig01)。

> Tips: 早停 `early stopping`，通过`监控模型在验证集上的性能`，来停止训练过程。

<a id="fig-ch06-fig01"></a>

<div align="center">
  <img src="./images/ch06-fig01.png" alt="Early stopping" width="78%" />
  <div><b>Figure 6.1</b></div>
</div>

In Figure [6.1](#fig-ch06-fig01), we can see that the validation accuracy increases
as the training and validation accuracy gap closes. The point where the
training and validation accuracy is closest is the point with the least
amount of overfitting, which is usually a good point for early stopping.

从图 [6.1](#fig-ch06-fig01) 可见，随着训练与验证准确率差距收窄，验证准确率上升；二者最接近的点通常过拟合最轻，也是早停的较佳截断点。

### Smaller Models
> 本节讨论剪枝、知识蒸馏等缩小模型规模的思路及其在训练流程中的角色。
[](#smaller-models)

Classic bias-variance theory suggests that reducing model size can
reduce overfitting. The intuition behind this theory is that, as a
general rule of thumb, the smaller the number of model parameters, the
smaller its capacity to memorize or overfit to noise in the data. The
following paragraphs discuss methods to reduce the model size, including
`pruning`, which removes parameters from a model, and `knowledge distillation`, which transfers knowledge to a smaller model.

经典偏差–方差理论认为缩小模型有助于抑制过拟合；直观上参数量越少，记住数据噪声的能力通常越弱。下文讨论减小模型规模的方法，包括移除参数的 `pruning`（剪枝）以及向更小模型迁移知识的 `knowledge distillation`（知识蒸馏）。

> Tips: 减小模型大小，包括`剪枝`和`知识蒸馏`。

Besides reducing the number of layers and shrinking the layers' widths
as a hyperparameter tuning procedure, another approach to obtaining
smaller models is `iterative pruning`, in which we train a large
model to achieve good performance on the original dataset. We then
iteratively remove parameters of the model, retraining it on the dataset
such that it maintains the same predictive performance as the original
model. (The lottery ticket hypothesis, discussed in
Chapter [\[ch04\]](./ch04/_books_ml-q-and-ai-ch04.md),
uses iterative pruning.)

除通过调参减少层数与宽度外，还可采用 `iterative pruning`（迭代剪枝）：先训练大模型在原数据上达到较好性能，再反复删参并在数据集上重训，使其保持与原模型相当的预测能力。（上文括号中所述彩票假设亦采用迭代剪枝。）

> Tips: 前置减少模型的参数，包括层数和宽度。后置的`迭代剪枝`，也是常用方法。

Another common approach to obtaining smaller models is **knowledge distillation**. 
The general idea behind this approach is to transfer
knowledge from a large, more complex model (the *teacher*) to a smaller
model (the *student*). Ideally, the student achieves the same predictive
accuracy as the teacher, but it does so more efficiently due to the
smaller size. As a nice side effect, the smaller student may overfit
less than the larger teacher model.

另一条常见路径是 **knowledge distillation**：把大模型（*teacher*）的知识迁移到小模型（*student*）；理想情况下学生以更小体量达到与教师相当的准确率，并往往比大教师更少过拟合。

Figure [6.2](#fig-ch06-fig02) diagrams the original knowledge distillation
process. Here, the `teacher` is first trained in a regular supervised
fashion to classify the examples in the dataset well, using a
conventional cross-entropy loss between the predicted scores and ground
truth class labels. While the smaller `student` network is trained on the
same dataset, the training objective is to minimize both 

图 [6.2](#fig-ch06-fig02) 示意经典蒸馏流程：先用常规监督与交叉熵把 `teacher` 训练好；在同一数据集上训练较小的 `student` 时，目标同时最小化

(a) the cross entropy between the outputs and the class labels and

（a）学生输出与类别标签之间的交叉熵，以及

(b) the difference between its outputs and the teacher outputs (measured 
using *Kullback-Leibler* divergence, which quantifies the difference between
two probability distributions by calculating how much one distribution
diverges from the other in terms of information content).

（b）学生与教师输出之差（用 *Kullback-Leibler* 散度度量两分布的信息差异）。

<a id="fig-ch06-fig02"></a>

<div align="center">
  <img src="./images/ch06-fig02.png" alt="image" width="78%" />
  <div><b>Figure 6.2</b></div>
</div>

By minimizing the Kullback-Leibler divergence--the difference between
the teacher and student score distributions--the student learns to
mimic the teacher while being smaller and more efficient.

通过最小化师生分数分布间的 KL 散度，学生以更紧凑的结构模仿教师。

> Tips: 知识蒸馏 `knowledge distillation`，通过`将知识从大模型`，`蒸馏`到`小模型`，来提高小模型的性能。

### Caveats with Smaller Models
> 本节澄清剪枝与蒸馏带来的泛化提升并不能简单等同于首选的抗过拟合手段。
[](#caveats-with-smaller-models)

While pruning and knowledge distillation can also enhance a model's
generalization performance, these techniques are not primary or
effective ways of reducing overfitting.

剪枝与知识蒸馏虽也可能改善泛化，但它们并非抑制过拟合的首要或最有效途径。

> Tips: `剪枝`和`知识蒸馏`，`可以`提高模型的泛化性能，但`不是`减少过拟合的`主要`方法。

Early research results indicate that pruning and knowledge distillation
can improve the generalization performance, presumably due to smaller
model sizes. However, counterintuitively, recent research studying
phenomena like double descent and grokking also showed that larger,
overparameterized models have improved generalization performance if
they are trained beyond the point of overfitting. `Double descent`
refers to the phenomenon in which models with either a small or an
extremely large number of parameters have good generalization
performance, while models with a number of parameters equal to the
number of training data points have poor generalization performance.
*Grokking* reveals that as the size of a dataset decreases, the need for
optimization increases, and generalization performance can improve well
past the point of overfitting.

早期研究表明剪枝与蒸馏或因模型变小而改善泛化；但近年关于 double descent、grokking 等现象的工作又指出，过参数化的大模型若在过拟合点之后继续训练，泛化也可能变好。`Double descent`（双降）指参数量很小或极大时泛化较好，而参数量约等于训练样本数时反而较差；*Grokking* 则揭示数据规模缩小时优化需求上升，泛化可在明显过拟合之后仍持续改善。

> Tips: `双降` `double descent`，是一种现象，模型参数的数量，在最佳量级之前和之后，都模型泛化效果，都会变差。但是，`涌现/顿悟` `grokking` 现象，展示出当模型参数数量超大时，泛化性能又会变好。 ??? FIXME

How can we reconcile the observation that pruned models can exhibit
better generalization performance with contradictory observations from
studies of double descent and grokking? Researchers recently showed that
the improved training process partly explains the reduction of
overfitting due to pruning. Pruning involves more extended training
periods and a replay of learning rate schedules that may be partly
responsible for the improved generalization performance.

如何把剪枝模型泛化变好与双降、grokking 的看似矛盾结论统一起来？新近研究认为训练过程的改进部分解释了剪枝带来的过拟合缓解：剪枝往往伴随更长训练与学习率日程的重演，这些都可能促成泛化提升。


Pruning and knowledge distillation remain excellent ways to improve the
computational efficiency of a model. However, while they can also
enhance a model's generalization performance, these techniques are not
primary or effective ways of reducing overfitting.

剪枝与蒸馏仍是提升计算效率的利器；但若论抑制过拟合，它们依然不是首选或最有效的单独手段。

> Tips: 剪枝和知识蒸馏，可以提高模型的泛化性能，但不是减少过拟合的主要方法。

### Ensemble Methods
> 本节说明集成学习的基本直觉、常见组合方式及其代价与适用场景。
[](#ensemble-methods)

Ensemble methods combine predictions from multiple models to improve the
overall prediction performance. However, the downside of using multiple
models is an increased computational cost.

集成方法合并多模型预测以提升整体表现，但使用多模型也意味着更高的计算成本。

We can think of ensemble methods as asking a committee of experts to
weigh in on a decision and then combining their judgments in some way to
make a final decision. Members in a committee often have different
backgrounds and experiences. While they tend to agree on basic
decisions, they can overrule bad decisions by majority rule. This
doesn't mean that the majority of experts is always right, but there
is a good chance that the majority of the committee is more often right,
on average, than every single member.

可把集成想象成专家委员会分别表态再汇总裁决：成员背景各异，在基础问题上往往一致，又能用多数票否决明显糟糕的判断；多数并不保证永远正确，但平均而言多数意见常常优于任一单个成员。

The most basic example of an ensemble method is majority voting. Here,
we train *k* different classifiers and collect the predicted class label
from each of these *k* models for a given input. We then return the most
frequent class label as the final prediction. (Ties are usually resolved
using a confidence score, randomly picking a label, or picking the class
label with the lowest index.)

最简单的集成是多数投票：训练 *k* 个分类器，对同一输入收集 *k* 个类别预测，再取出现次数最多的标签（平局时常靠置信度、随机或取最小索引类别等方式打破）。

Ensemble methods are more prevalent in classical machine learning than
deep learning because it is more computationally expensive to employ
multiple models than to rely on a single one. In other words, deep
neural networks require significant computational resources, making them
less suitable for ensemble methods.

集成在传统机器学习中更常见，因为深度学习下单模型已很昂贵，再复制多份更难承受。

Random forests and gradient boosting are popular examples of ensemble
methods. However, by using majority voting or stacking, for example, we
can combine any group of models: an ensemble may consist of a support
vector machine, a multilayer perceptron, and a nearest-neighbor
classifier. Here, stacking (also known as *stacked generalization*)is a
more advanced variant of majority voting that involves training a new
model to combine the predictions of several other models rather than
obtaining the label by majorit yvote.

随机森林与梯度提升是集成范例；也可用多数投票或 stacking 等任意组合模型——例如 SVM、多层感知机与 kNN 并存。stacking（亦称 *stacked generalization*）比简单多数票更进一步，用新模型去学习如何融合若干基模型输出，而不是仅靠 majorit yvote 表决得到标签。

A popular industry technique is to build models from *k-fold
cross-validation*, a model evaluationt echnique in which we train and
evaluate a model on *k* training folds.We then compute the average
performance metric across all *k* iterations to estimate the overall
performance measure of the model. After evaluation, we can either train
the model on the entire training dataset or combine the individual
models as an ensemble, as shown in
Figure [6.2](#fig-ch06-fig03).

工业界常基于 *k-fold cross-validation* 构造模型：在各训练折上轮流训练与评估，并对 *k* 轮指标取平均以估计整体性能；评估结束后既可仅用全部训练数据重训单一模型，也可将各折模型组合为集成，见图 [6.2](#fig-ch06-fig03)。

<a id="fig-ch06-fig03"></a>

<div align="center">
  <img src="./images/ch06-fig03.png" alt="[k]{.upright}-fold cross-validation for creating model ensembles" width="78%" />
  <div><b>Figure 6.3</b></div>
</div>

As shown in Figure [6.2](#fig-ch06-fig03), the *k*-fold ensemble approach trains each of the
*k* models on the respective *k* "" 1 training folds in each round.
After evaluating the models on the validation folds, we can combine them
into a majority vote classifier or build an ensemble using stacking, a
technique that combines multiple classification or regression models via
a meta-model.

如图所示，*k* 折集成在每轮对各模型使用相应的 *k* "" 1 份训练折；在验证折上评估结束后，可将它们合成多数投票分类器，或通过元模型采用 stacking 融合多个分类或回归模型。

While the ensemble approach can potentially reduce overfitting and
improve robustness, this approach is not always suitable. For instance,
potential downsides include managing and deploying an ensemble of
models, which can be more complex and computationally expensive than
using a single model.

集成有望降低过拟合并增强稳健性，但并非总适用：运维与部署多模型往往比单模型更复杂、更耗算力。

## Other Methods
> 本节补充若干并非专为减拟合设计但在实践中常具正则化效果的结构与优化技巧。
[](#other-methods)

So far, this book has covered some of the most prominent techniques to
reduce overfitting. Chapter [\[ch05\]](./ch05/_books_ml-q-and-ai-ch05.md) covered techniques that aim to reduce overfitting
from a data perspective. Additional techniques for reducing overfitting
with model modifications include skip-connections (found in residual
networks, for example), look-ahead optimizers, stochastic weight
averaging, multitask learning, and snapshot ensembles.

本书至此已介绍多种主流抗过拟合手段；第五章从数据视角讨论了相关技术。通过模型修改还能借助残差网络中的跳跃连接、lookahead 优化器、随机权重平均、多任务学习与快照集成等进一步抑制过拟合。

While they are not originally designed to reduce overfitting, layer
input normalization techniques such as batch normalization (BatchNorm)
and layer normalization (LayerNorm) can stabilize training and often
have a regularizing effect that reduces overfitting. Weight
normalization, which normalizes the model weights instead of layer
inputs, could also lead to better generalization performance. However,
this effect is less direct since weight normalization (WeightNorm)
doesn't explicitly act as a regularizer like weight decay does.

BatchNorm、LayerNorm 等虽非专为减拟合而设计，却能稳定训练并常带有正则化副作用；WeightNorm 归一化权重而非层输入，也可能改善泛化，但其作用不如权重衰减那样直接充当正则项。

## Choosing a Regularization Technique
> 本节给出在数据与模型两侧多种正则化手段并存时的实践取舍建议。
[](#choosing-a-regularization-technique)

Improving data quality is an essential first step in reducing
overfitting. However, for recent deep neural networks with large numbers
of parameters, we need to do more to achieve an acceptable level of
overfitting. Therefore, data augmentation and pretraining, along with
established techniques such as dropout and weight decay, remain crucial
overfitting reduction methods.

提升数据质量是抑制过拟合的首要步骤；但对当今参数众多的深度网络，仅靠数据往往不够，因此数据增强、预训练以及 dropout、权重衰减等成熟手段仍然关键。

In practice, we can and should use multiple methods at once to reduce
overfitting for an additive effect. To achieve the best results, treat
selecting these techniques as a hyperparameter optimization problem.

实践中通常应叠加多种方法以获得加成效果；具体组合最好视作超参数搜索问题来系统调优。

## Exercises
> 本节习题对比早停与调训练轮数，并反思集成方法的代价。
[](#exercises)

6-1. Supposewe'reusingearlystoppingasamechanismtoreduceover-
 fitting--inparticular,amodernearly-stoppingvariantthatcreates
checkpoints of the best model (for instance, the model with the high-
 est validation accuracy) during training so that we can load it after
the training has completed. This mechanism can be enabled in most modern
deep learning frameworks. However, a colleague recommends tuning the
number of training epochs instead. What are some of the advantages and
disadvantages of each approach?

习题 6-1：若用现代早停在训练中保存验证指标最佳检查点并在结束后加载，与同事主张改为调节训练轮数相比，各有哪些利弊？

6-2. Ensemble models have been established as a reliable and successful
method for decreasing overfitting and enhancing the reliability of
predictive modeling efforts. However, there's always a trade-off. What
are some of the drawbacks associated with ensemble techniques?

习题 6-2：集成模型在降低过拟合与提升预测可靠性方面成效卓著，但必有权衡——集成有哪些缺点？

## References
> 本节列出权重衰减、蒸馏、偏差–方差、彩票假设、双降、grokking、剪枝、Dropout 解释及正则“鸡尾酒”调参等相关参考文献。
[](#references)

- For more on the distinction between $L_2$ regularization and weight
  decay: Guodong Zhang et al., "Three Mechanisms of Weight Decay
  Regularization"? (2018), <https://arxiv.org/abs/1810.12281>.

Zhang 等关于 $L_2$ 正则化与 weight decay 区别的论文（2018）。

- Research results indicate that pruning and knowledge distillation can
  improve generalization performance, presumably due to smaller model
  sizes: Geoffrey Hinton, Oriol Vinyals, and Jeff Dean, "Distilling
  the Knowledge in a Neural Network"? (2015),
  <https://arxiv.org/abs/1503.02531>.

Hinton 等关于知识蒸馏的经典论文（2015）。

- Classic bias-variance theory suggests that reducing model size can
  reduce overfitting: Jerome H. Friedman, Robert Tibshirani, and Trevor
  Hastie, "Model Selection and Bias-Variance Tradeoff,"? Chapter 2.9,
  in *The Elements of Statistical Learning* (Springer, 2009).

《统计学习要素》中关于模型选择与偏差–方差权衡的章节（2009）。

- The lottery ticket hypothesis applies knowledge distillation to find
  smaller networks with the same predictive performance as the original
  one: Jonathan Frankle and Michael Carbin, "The Lottery Ticket
  Hypothesis: Finding Sparse, Trainable Neural Networks"? (2018),
  <https://arxiv.org/abs/1803.03635>.

Frankle 与 Carbin 提出彩票假设的论文（2018）。

- For more on double descent:
  <https://en.wikipedia.org/wiki/Double_descent>.

双降现象的维基条目。

- The phenomenon of grokking indicates that generalization perfor-
   mance can improve well past the point of overfitting: Alethea Power
  et al., "Grokking: Generalization Beyond Overfitting on Small
  Algorithmic Datasets"? (2022), <https://arxiv.org/abs/2201.02177>.

Power 等关于 grokking 的论文（2022）。

- Recent research shows that the improved training process partly
  explains the reduction of overfitting due to pruning: Tian Jin et al.,
  "Pruning's Effect on Generalization Through the Lens of Training
  and Regularization"? (2022), <https://arxiv.org/abs/2210.13738>.

Jin 等从训练与正则视角解释剪枝如何影响泛化的研究（2022）。

- Dropout was previously discussed as a regularization technique, but it
  can also be considered an ensemble method that approximates a weighted
  geometric mean of multiple networks: Pierre Baldi and Peter J.
  Sadowski, "Understanding Dropout"? (2013),
  [*https://proceedings.neurips.cc/paper/2013/hash/71f6278d140af599*](https://proceedings.neurips.cc/paper/2013/hash/71f6278d140af599e06ad9bf1ba03cb0-Abstract.html)
  [*e06ad9bf1ba03cb0-Abstract.html*](https://proceedings.neurips.cc/paper/2013/hash/71f6278d140af599e06ad9bf1ba03cb0-Abstract.html).

Baldi 与 Sadowski 将 Dropout 解释为近似多网络加权几何均值的工作（2013）。

- Regularization cocktails need to be tuned on a per-dataset basis:
  Arlind Kadra et al., "Well-Tuned Simple Nets Excel on Tabular
  Datasets"? (2021), <https://arxiv.org/abs/2106.11189>.

Kadra 等强调正则组合需按数据集调参的研究（2021）。


------------------------------------------------------------------------

