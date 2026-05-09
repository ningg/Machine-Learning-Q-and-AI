








# Chapter 24: Poisson and Ordinal Regression
> 本章对比泊松回归与序数回归：前者面向计数与近似泊松分布的响应，后者面向有序类别且不要求类间“间距”可度量，并配有练习。
[](#chapter-24-poisson-and-ordinal-regression)



**When is it preferable to use `Poisson regression` over `Ordinal regression`, and vice versa?**

在什么情况下更适合用**泊松回归（Poisson regression）**，又在什么情况下更适合用**序数回归（ordinal regression）**，反之亦然？

We usually use `Poisson regression` when the target variable represents
count data (**positive integers**). As an example of count data, consider
the number of colds contracted on an airplane or the number of guests
visiting a restaurant on a given day. Besides the target variable
representing counts, the data should also be Poisson distributed, which
means that the mean and variance are roughly the same. (For large means,
we can use a normal distribution to approximate a Poisson distribution.)

当目标变量是**计数**（正整数）时，通常用泊松回归；例如机舱内患感冒人数、某日餐厅来客数。除“取值为计数”外，数据还应大致服从**泊松分布**，即均值与方差接近（均值很大时，可用正态分布近似泊松；可参阅 [泊松分布（中文维基）](https://zh.wikipedia.org/wiki/%E6%B3%8A%E6%9D%BE%E5%88%86%E5%B8%83)）。

*Ordinal data* is a subcategory of categorical data where the categories
have a natural order, such as 1 \< 2 \< 3, as illustrated in
Figure [24.1](#fig-ch24-fig01). Ordinal data is often represented as positive
integers and may look similar to count
data. For example, consider the star rating on Amazon (1 star, 2 stars, 3 stars, and
so on). However, ordinal regression does not make any assumptions about
the distance between the ordered categories. Consider the following
measure of disease severity: *severe \> moderate \> mild \> none*. While
we would typically map the disease severity variable to an integer representation
(4 \> 3 \> 2 \> 1), there is no assumption that the distance between 4
and 3 (severe and moderate) is the same as the distance between 2 and 1
(mild and none).

*序数数据（ordinal data）*是有自然顺序的分类数据的子类，例如 $1 < 2 < 3$，见图 [24.1](#fig-ch24-fig01)。序数常编码为正整数，因而**表面上**像计数；例如电商星级（1 星、2 星……）。但序数回归**不要求**相邻等级之间的“距离”有意义或可比。再以疾病严重度 *重 > 中 > 轻 > 无* 为例：即便映射成 $4>3>2>1$，也**不假设**“4 与 3 的差距”等于“2 与 1 的差距”。

<a id="fig-ch24-fig01"></a>

<div align="center">
  <img src="./images/ch24-fig01.png" alt="The distance between ordinal categories is arbitrary." width="78%" />
  <div><b>Figure 24.1</b></div>
</div>

In short, we use Poisson regression for count data. We use Ordinal
regression when we know that certain outcomes are "higher" or
"lower" than others, but we are not sure how much or if it even
matters.

简言之：**计数且近似泊松**用泊松回归；当只知结果有**高下先后**、却不清楚间隔多大、甚至间隔是否重要时，用序数回归更合适。

## Exercises
> 本节通过进球预测与观影偏好排序两题，巩固泊松与序数两类响应的判别。
[](#exercises)

24-1. Suppose we want to predict the number of goals a soccer player
will score in a particular season. Should we solve this problem using
ordinal regression or Poisson regression?

24-1. 若要预测某球员一季进球**个数**，该用序数回归还是泊松回归？

24-2. Suppose we ask someone to sort the last three movies they have
watched based on their order of preference. Ignoring the fact that this
dataset is a tad too small for machine learning, which approach would be
best suited for this kind of data?

24-2. 若让某人按偏好对最近看过的三部电影**排序**（先忽略样本量太小这件事），哪种建模思路更贴切？


------------------------------------------------------------------------

