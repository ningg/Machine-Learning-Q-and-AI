







# Chapter 27: Proper Metrics
[](#chapter-27-proper-metrics)



**What are the three properties of a distance function that make it a
*proper* metric?**

Metrics are foundational to mathematics, computer science, and various
other scientific domains. Understanding the fundamental properties that
define a good distance function to measure distances or differences
between points or datasets is important. For instance, when dealing with
functions like loss functions in neural networks, understanding whether
they behave like proper metrics can be instrumental in knowing how
optimization algorithms will converge to a solution.

> Tips: 
> 
> - `Metrics` 度量，是`数学`、`计算机科学`和各种其他科学领域的基础。
> - 理解定义良好的`距离函数`的关键属性，对于测量点或数据集之间的距离或差异至关重要。
> - 例如，在处理神经网络中的损失函数时，了解它们是否表现出良好的距离函数属性，对于了解优化算法如何收敛到解决方案至关重要。
> - `距离函数`，具有三个关键属性：`非负性`、`对称性`和`三角不等式`。

This chapter analyzes two commonly utilized loss functions, 
the `mean squared error` and the `cross-entropy loss`, to demonstrate whether they
meet the criteria for proper metrics.

> Tips:  本章节，分析了两个常用的损失函数，`均方误差`和`交叉熵损失`，来演示它们是否符合良好的度量标准。

## The Criteria
[](#the-criteria)

To illustrate the criteria of a proper metric, consider two vectors or
points $\mathbf{v}$ and $\mathbf{w}$, and their distance $d(\mathbf{v}, \mathbf{w})$, as shown in
Figure [27.1](#fig-ch27-fig01).

<a id="fig-ch27-fig01"></a>

<div align="center">
  <img src="./images/ch27-fig01.png" alt="The Euclidean distance between two 2D vectors" width="78%" />
  <div><b>Figure 27.1</b></div>
</div>

The criteria of a proper metric are the following:

- The distance between two points is always **non-negative**, $d(\mathbf{v}, \mathbf{w}) \geq 0$, and can be 0 only if the two points are identical, that is, $\mathbf{v} = \mathbf{w}$.

- The distance is **symmetric**; for instance, $d(\mathbf{v}, \mathbf{w}) = d(\mathbf{w}, \mathbf{v})$.

- The distance function satisfies the **triangle inequality** for any three points: $\mathbf{v}$, $\mathbf{w}$, $\mathbf{x}$, meaning:

  $$
  d(\mathbf{v}, \mathbf{w}) \leq d(\mathbf{v}, \mathbf{x}) + d(\mathbf{x}, \mathbf{w})
  $$

> Tips: 距离函数，具有三个关键属性：`非负性`、`对称性`和`三角不等式`。

To better understand the triangle inequality, think of the points as
vertices of a triangle. If we consider any triangle, the sum of two of
the sides is always larger than the third side, as illustrated in
Figure [27.2](#fig-ch27-fig02).

<a id="fig-ch27-fig02"></a>

<div align="center">
  <img src="./images/ch27-fig02.png" alt="Triangle inequality" width="52%" />
  <div><b>Figure 27.2</b></div>
</div>

Consider what would happen if the triangle in equality depicted in
Figure [27.2](#fig-ch27-fig02) weren't true. If the sum of the lengths of sides
AB and BC was shorter than AC, then sides AB and BC would not meet to
form a triangle; instead, they would fall short of each other. Thus, the
fact that they meet and form a triangle demonstrates the triangle
inequality.



## The Mean Squared Error
[](#the-mean-squared-error)

The `mean squared error (MSE)` loss computes the squared Euclidean
distance between a target variable $y$ and a predicted target value
$\hat{y}$:

$$
\mathrm{MSE}=\frac{1}{n} \sum_{i=1}^n\left(y^{(i)} - \hat{y}^{(i)}\right)^2
$$

The index $i$ denotes the $i$th data point in the dataset or sample. Is
this loss function a proper metric?

For simplicity's sake, we will consider the `squared error (SE)` loss
between two data points (though the following insights also hold for the
MSE). As shown in the following equation, the SE loss quantifies the
squared difference between the predicted and actual values for a single
data point, while the MSE loss averages these squared differences over
all data points in a dataset:

$$
\mathrm{SE}(y, \hat{y})=\left(y - \hat{y}\right)^2
$$

In this case, the SE satisfies the first part of the first criterion:
the distance between two points is always `non-negative`. Since we are
raising the difference to the power of 2, it cannot be negative.

How about the second criterion, that the distance can be 0 only if the
two points are identical? Due to the subtraction in the SE, it is
intuitive to see that it can be 0 only if the prediction matches the
target variable, $y = \hat{y}$. As with the first
criterion, we can use the square to confirm that SE satisfies the second
criterion: we have $(y - \hat{y})^2 = (\hat{y} - y)^2$.

At first glance, it seems that the squared error loss also satisfies the
third criterion, the `triangle inequality`. Intuitively, you can check
this by choosing three arbitrary numbers, here 1, 2, 3:

- $(1 - 2)^2 \leq (1 - 3)^2 + (2 - 3)^2$

- $(1 - 3)^2 \leq (1 - 2)^2 + (2 - 3)^2$

- $(2 - 3)^2 \leq (1 - 2)^2 + (1 - 3)^2$

However, there are values for which this is not true. For example,
consider the values $a = 0$, $b = 2$, and $c = 1$. This gives us
$d(a, b) = 4$, $d(a, c) = 1$, and $d(b, c) = 1$, such that
we have the following scenario, which violates the triangle inequality:

- $(0 - 2)^2 \nleq (0 - 1)^2 + (2 - 1)^2$

- $(2 - 1)^2 \leq (0 - 1)^2 + (0 - 2)^2$

- $(0 - 1)^2 \leq (0 - 2)^2 + (1 - 2)^2$

Since it does not satisfy the triangle inequality via the example above,
we conclude that the (mean) squared error loss is not a proper metric.

However, if we change the squared error into the `root-squared error`

$$
\sqrt{(y - \hat{y})^2}
$$

the triangle inequality can be satisfied:

$$
\sqrt{(0 - 2)^2} \leq \sqrt{(0 - 1)^2} + \sqrt{(2 - 1)^2}
$$


You might be familiar with the `L2` distance or Euclidean
distance, which is known to satisfy the triangle inequality. These two
distance metrics are equivalent to the root-squared error when
considering two scalar values.

> Tips: 
> 
> - 如果将`平方误差`改为`平方根误差`，则三角不等式可以满足。
> - 平方根误差，是平方误差的平方根。


## The Cross-Entropy Loss
[](#the-cross-entropy-loss)

> Tips: 交叉熵损失，是衡量两个概率分布之间距离的损失函数。
>
> FIXME ??? 不理解

`Cross entropy` is used to measure the distance between two probability
distributions. In machine learning contexts, we use the discrete
cross-entropy loss (CE) between class label $y$ and the predicted
probability $\hat{p}$ when we train logistic regression or neural network
classifiers on a dataset consisting of $n$ training examples:

$$
\mathrm{CE}(\mathbf{y}, \mathbf{p}) = -\frac{1}{n} \sum_{i=1}^n y^{(i)} \times \log \left(p^{(i)}\right)
$$

Is this loss function a proper metric? Again, for simplicity's sake,
we will look at the cross-entropy function ($H$) between only two data
points:

$$
H(y, p) = - y \times \log(p)
$$

The cross-entropy loss satisfies one part of the first criterion: the
distance is always non-negative because the probability score is a
number in the range [0, 1]. Hence, $\log(p)$ ranges between
$-\infty$ and 0. The important part is that the $H$ function
includes a negative sign. Hence, the cross entropy ranges between
$0$ and $+\infty$ and thus satisfies one aspect of the first criterion
shown above.

However, the cross-entropy loss is not 0 for two identical points. For
example, $H(0.9, 0.9) = -0.9 \times \log(0.9) = 0.095$.

The second criterion shown above is also violated by the cross-entropy
loss because the loss is not symmetric: $-y \times \log(p) \neq -p \times \log(y)$.
Let's illustrate this with
a concrete, numeric example:

- If $y = 1$ and $p = 0.5$, then $-1 \times \log(0.5) = 0.693$.

- If $y = 0.5$ and $p = 1$, then $-0.5 \times \log(1) = 0$.

Finally, the cross-entropy loss does not satisfy the triangle
inequality, $H(r, p) \geq H(r, q) + H(q, p)$. Let's illustrate this with an example as well. Suppose we choose $r = 0.9$, $p = 0.5$, and $q = 0.4$. We have:

- $H(0.9, 0.5) = 0.624$
- $H(0.9, 0.4) = 0.825$
- $H(0.4, 0.5) = 0.277$

As you can see, $0.624 \geq 0.825 + 0.277$ does not hold here.

In conclusion, while the `cross-entropy loss` is a useful loss function
for training neural networks via (stochastic) gradient descent, it is
not a proper distance metric, as it does not satisfy any of the three
criteria. 

> Tips:
>
> - 交叉熵损失，是训练逻辑回归或神经网络分类器时，用于衡量两个概率分布之间距离的损失函数，这种损失函数在训练过程中，可以引导模型学习到更好的概率分布。
> - 但是，交叉熵损失，不是良好的度量标准，因为它不满足三角不等式。

## Exercises
[](#exercises)

27-1. Suppose we consider using the mean absolute error (MAE) as an
alternative to the root mean square error (RMSE) for measuring the
performance of a machine learning model, where

$$
\mathrm{MAE} = \frac{1}{n} \sum_{i=1}^n |y^{(i)} - \hat{y}^{(i)}|
$$

and

$$
\mathrm{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y^{(i)} - \hat{y}^{(i)})^2}
$$

However, a colleague argues that the MAE is not a proper distance metric
in metric space because it involves an absolute value, so we should use
the RMSE instead. Is this argument correct?

27-2. Based on your answer to the previous question, would you say that
the MAE is better or is worse than the RMSE?


------------------------------------------------------------------------

