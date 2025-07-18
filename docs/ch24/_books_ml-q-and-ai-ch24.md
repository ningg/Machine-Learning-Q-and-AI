








# Chapter 24: Poisson and Ordinal Regression
[](#chapter-24-poisson-and-ordinal-regression)



**When is it preferable to use `Poisson regression` over `Ordinal regression`, and vice versa?**

> 本章讨论了两种回归模型：泊松回归和序数回归，并讨论了它们的应用场景。
> 
> - 泊松回归用于**计数数据**，序数回归用于**有序数据**。
> 

We usually use `Poisson regression` when the target variable represents
count data (**positive integers**). As an example of count data, consider
the number of colds contracted on an airplane or the number of guests
visiting a restaurant on a given day. Besides the target variable
representing counts, the data should also be Poisson distributed, which
means that the mean and variance are roughly the same. (For large means,
we can use a normal distribution to approximate a Poisson distribution.)

> 泊松回归通常用于表示计数数据（正整数）的目标变量。例如，考虑飞机上感冒的人数或某天餐厅的客人数量。除了表示计数的目标变量外，数据还应服从泊松分布，这意味着均值和方差大致相同。（对于大均值，我们可以使用正态分布来近似泊松分布。） 
> 
> 更多细节： [泊松分布](https://baike.baidu.com/item/%E6%B3%8A%E6%9D%BE%E5%88%86%E5%B8%83/1442110)



*Ordinal data* is a subcategory of categorical data where the categories
have a natural order, such as 1 \< 2 \< 3, as illustrated in
Figure [1.1](#fig-ch24-fig01). Ordinal data is often represented as positive
integers and may look similar to count
data.Forexample,considerthestarratingonAmazon(1star,2stars,3stars, and
so on). However, ordinal regression does not make any assumptions about
the distance between the ordered categories. Consider the following
measure of disease severity: *severe \> moderate \> mild \> none*. While
we would typically map the disease severity variable to an integer representation
(4 \> 3 \> 2 \> 1), there is no assumption that the distance between 4
and 3 (severe and moderate) is the same as the distance between 2 and 1
(mild and none).

> 序数数据是分类数据的一个子类别，其中类别具有自然顺序，例如 1 < 2 < 3，如图 1.1 所示。序数数据通常表示为正整数，可能与计数数据相似。例如，考虑亚马逊上的星级评分（1 星、2 星、3 星等）。然而，序数回归对有序类别之间的距离没有任何假设。考虑以下疾病严重程度的衡量标准：*严重 > 中等 > 轻微 > 无*。虽然我们通常将疾病严重程度变量映射为整数表示（4 > 3 > 2 > 1），但没有任何假设认为 4 和 3（严重和中等）之间的距离与 2 和 1（轻微和无）之间的距离相同。
> 

<a id="fig-ch24-fig01"></a>

<div align="center">
  <img src="./images/ch24-fig01.png" alt="The distance between ordinal categories is arbitrary." width="78%" />
</div>

In short, we use Poisson regression for count data. We use Ordinal
regression when we know that certain outcomes are "higher" or
"lower" than others, but we are not sure how much or if it even
matters.

## Exercises
[](#exercises)

24-1. Suppose we want to predict the number of goals a soccer player
will score in a particular season. Should we solve this problem using
ordinal regression or Poisson regression?

24-2. Suppose we ask someone to sort the last three movies they have
watched based on their order of preference. Ignoring the fact that this
dataset is a tad too small for machine learning, which approach would be
best suited for this kind of data?


------------------------------------------------------------------------

