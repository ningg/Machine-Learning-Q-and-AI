







# Chapter 28: The k in k-Fold Cross-Validation [](#chapter-28-the-k-in-k-fold-cross-validation)



**k-fold cross-validation is a common choice for evaluating machine
learning classifiers because it lets us use all training data to
simulate how well a machine learning algorithm might perform on new
data. What are the advantages and disadvantages of choosing a large k?**

We can think of *k*-fold cross-validation as a workaround for model
evaluation when we have limited data. In machine learning model
evaluation, we care about the generalization performance of our model,
that is, how well it performs on new data. In *k*-fold cross-validation,
we use the training data for model selection and evaluation by
partitioning it into *k* validation rounds and folds. If we have *k*
folds, we have *k* iterations, leading to *k* different models, as
illustrated in Figure [1.1](#fig:ch28-fig01){reference="fig:ch28-fig01"
reference-type="ref"}.

![An example of [k]{.upright}-fold cross-validation for model evaluation
where [k]{.upright} = 5](../images/ch28-fig01.png){#fig:ch28-fig01}

Using *k*-fold cross-validation, we usually evaluate the performance of
a particular hyperparameter configuration by computing the average
performance over the *k* models. This performance reflects or
approximates the performance of a model trained on the complete training
dataset after evaluation.

The following sections cover the trade-offs of selecting values for *k*
in *k*-fold cross-validation and address the challenges of large *k*
values and their computational demands, especially in deep learning
contexts. We then discuss the core purposes of *k* and how to choose an
appropriate value based on specific modeling needs.

## Trade-offs in Selecting Values for k [](#trade-offs-in-selecting-values-for-k)

If *k* is too large, the training sets are too similar between the
different rounds of cross-validation. The *k* models are thus very
similar to the model we obtain by training on the whole training set. In
this case, we can still leverage the advantage of *k*-fold
cross-validation: evaluating the performance for the entire training set
via the held-out validation fold in each round. (Here, we obtain the
training set by concatenating all *k* â€" 1 training folds in a given
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
historical reasons. A study by Ron Kohavi (see â€œâ€? at the end of this
chapter) found that *k* = 10 offers a good bias and variance trade-off
for classical machine learning algorithms, such as decision trees and
naive Bayes classifiers, on a handful of small datasets.

For example, in 10-fold cross-validation, we use 9/10 (90 percent) of
the data for training in each round, whereas in 5-fold cross-validation,
we use only 4/5 (80 percent) of the data, as shown in
Figure [1.2](#fig:ch28-fig02){reference="fig:ch28-fig02"
reference-type="ref"}.

![A comparison of 5-fold and 10-fold
cross-validation](../images/ch28-fig02.png){#fig:ch28-fig02}

However, this does not mean large training sets are bad, since they can
reduce the pessimistic bias of the performance estimate (mostly a good
thing) if we assume that the model training can benefit from more
training data. (See
Figure [\[fig:ch05-fig01\]](#fig:ch05-fig01){reference="fig:ch05-fig01"
reference-type="ref"} on page  for an example of a learning curve.)

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

## Determining Appropriate Values for k [](#determining-appropriate-values-for-k)

When deciding upon an appropriate value of *k*, we are often guided by
computational performance and conventions. However, it's worthwhile to
define the purpose and context of using *k*-fold cross-validation. For
example, if we care primarily about approximating the predictive
performance of the final model, using a large *k* makes sense. This way,
the training folds are very similar to the combined training dataset,
yet we still get to evaluate the model on all data points via the
validation folds.

On the other hand, if we care to evaluate how sensitive a given
hyperparameter configuration and training pipeline is to different
training datasets, then choosing a smaller number for *k* makes more
sense.

Since most practical scenarios consist of two stepsâ€"tuning
hyperparameters and evaluating the performance of a modelâ€"we can also
consider a two-step procedure. For instance, we can use a smaller *k*
during hyperparameter tuning. This will help speed up the hyperparameter
search and probe the hyperparameter configurations for robustness (in
addition to the average performance, we can also consider the variance
as a selection criterion). Then, after hyperparameter tuning and
selection, we can increase the value of *k* to evaluate the model.

However, reusing the same dataset for model selection and evaluation
introduces biases, and it is usually better to use a separate test set
for model evaluation. Also, nested cross-validation may be preferred as
an alternative to *k*-fold cross-validation.

### Exercises [](#exercises)

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

## References [](#references)

- For a longer and more detailed explanation of why and how to use
  *k*-fold cross-validation, see my article: â€œModel Evaluation, Model
  Selection, and Algorithm Selection in Machine Learningâ€? (2018),
  <https://arxiv.org/abs/1811.12808>.

- The paper that popularized the recommendation of choosing *k* = 5 and
  *k* = 10: Ron Kohavi, â€œA Study of Cross-Validation and Bootstrap for
  Accuracy Estimation and Model Selectionâ€? (1995),
  <https://dl.acm.org/doi/10.5555/1643031.1643047>.

\

------------------------------------------------------------------------

