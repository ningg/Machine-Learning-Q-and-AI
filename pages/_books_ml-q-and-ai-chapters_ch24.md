# Machine Learning Q and AI {#machine-learning-q-and-ai .post-title style="text-align: left;"}

## 30 Essential Questions and Answers on Machine Learning and AI {#essential-questions-and-answers-on-machine-learning-and-ai .post-subtitle}

By Sebastian Raschka. [Free to read](#table-of-contents). Published by
[No Starch Press](https://nostarch.com/machine-learning-q-and-ai).\
Copyright Â© 2024-2025 by Sebastian Raschka.

![Machine Learning and Q and
AI](../images/2023-ml-ai-beyond.jpg){.right-image-shadow-30}

> Machine learning and AI are moving at a rapid pace. Researchers and
> practitioners are constantly struggling to keep up with the breadth of
> concepts and techniques. This book provides bite-sized bits of
> knowledge for your journey from machine learning beginner to expert,
> covering topics from various machine learning areas. Even experienced
> machine learning researchers and practitioners will encounter
> something new that they can add to their arsenal of techniques.

\

# Part 4: Predictive Performance and Model Evaluation [](#part-4-predictive-performance-and-model-evaluation)

## Chapter 24: Poisson and Ordinal Regression [](#chapter-24-poisson-and-ordinal-regression)

[]{#ch24 label="ch24"}

**When is it preferable to use Poisson regression over ordinal
regression, and vice versa?**

We usually use Poisson regression when the target variable represents
count data (positive integers). As an example of count data, consider
the number of colds contracted on an airplane or the number of guests
visiting a restaurant on a given day. Besides the target variable
representing counts, the data should also be Poisson distributed, which
means that the mean and variance are roughly the same. (For large means,
we can use a normal distribution to approximate a Poisson distribution.)

*Ordinal data* is a subcategory of categorical data where the categories
have a natural order, such as 1 \< 2 \< 3, as illustrated in
FigureÂ [1.1](#fig:ch24-fig01){reference="fig:ch24-fig01"
reference-type="ref"}. Ordinal data is often represented as positive
integers and may look similar to count
data.Forexample,considerthestarratingonAmazon(1star,2stars,3stars, and
so on). However, ordinal regression does not make any assumptions about
the distance between the ordered categories. Consider the following
measure of disease severity: *severe \> moderate \> mild \> none*. While
we would typicallymapthediseaseseverityvariabletoanintegerrepresentation
(4 \> 3 \> 2 \> 1), there is no assumption that the distance between 4
and 3 (severe and moderate) is the same as the distance between 2 and 1
(mild and none).

![The distance between ordinal categories is
arbitrary.](../images/ch24-fig01.png){#fig:ch24-fig01}

In short, we use Poisson regression for count data. We use ordinal
regression when we know that certain outcomes are â€œhigherâ€? or
â€œlowerâ€? than others, but we are not sure how much or if it even
matters.

### Exercises [](#exercises)

24-1. Suppose we want to predict the number of goals a soccer player
will score in a particular season. Should we solve this problem using
ordinal regression or Poisson regression?

24-2. Suppose we ask someone to sort the last three movies they have
watched based on their order of preference. Ignoring the fact that this
dataset is a tad too small for machine learning, which approach would be
best suited for this kind of data?

\

------------------------------------------------------------------------

