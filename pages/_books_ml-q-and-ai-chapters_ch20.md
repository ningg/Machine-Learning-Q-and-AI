







# Chapter 20: Production And Deployment [](#chapter-20-production-and-deployment)

## Stateless and Stateful Training [](#stateless-and-stateful-training)

[]{#ch20 label="ch20"}

**What is the difference between stateless and stateful training
workflows in the context of production and deployment systems?**

Stateless training and stateful training refer to different ways of
training a production model.

### Stateless (Re)training [](#stateless-retraining)

In stateless training, the more conventional approach, we first train an
initial model on the original training set and then retrain it as new
data arrives. Hence, stateless training is also commonly referred to as
stateless *retraining*.

As FigureÂ [1.1](#fig:ch20-fig01){reference="fig:ch20-fig01"
reference-type="ref"} shows, we can think of stateless retraining as a
sliding window approach in which we retrain the initial model on
different parts of the data from a given data stream.

![Stateless training replaces the model\
periodically.](../images/ch20-fig01.png){#fig:ch20-fig01}

For example, to update the initial model in
FigureÂ [1.1](#fig:ch20-fig01){reference="fig:ch20-fig01"
reference-type="ref"} (Model 1) to a newer model (Model 2), we train the
model on 30 percent of the initial data and 70 percent of the most
recent data at a given point in time.

Stateless retraining is a straightforward approach that allows us to
adapt the model to the most recent changes in the data and
feature-target relationships via retraining the model from scratch in
user-defined checkpoint intervals. This approach is prevalent with
conventional machine learning systems that cannot be fine-tuned as part
of a transfer or self-supervised learning workflow (see
ChapterÂ [\[ch02\]](../ch02){reference="ch02" reference-type="ref"}).
For example, standard implementations of tree-based models, such as
random forests and gradient boosting (XGBoost, CatBoost, and LightGBM),
fall into this category.

### Stateful Training [](#stateful-training)

In stateful training, we train the model on an initial batch of data and
then update it periodically (as opposed to retraining it) when new data
arrives.

AsillustratedinFigureÂ [1.2](#fig:ch20-fig02){reference="fig:ch20-fig02"
reference-type="ref"},we do not retrain the initial model (Model1.0)
from scratch; instead, we update or fine-tune it as new data arrives.
This approach is particularly attractive for models compatible with
transfer learning or self-supervised learning.

![Stateful training updates models
periodically.](../images/ch20-fig02.png){#fig:ch20-fig02}

The stateful approach mimics a transfer or self-supervised learning
workflow where we adopt a pretrained model for fine-tuning. However,
stateful training differs fundamentally from transfer and
self-supervised learning because it updates the model to accommodate
concept, feature, and label drifts. In contrast, transfer and
self-supervised learning aim to adopt the model for a different
classification task. For instance, in transfer learning, the target
labels often differ. In self-supervised learning, we obtain the target
labels from the dataset features.

One significant advantage of stateful training is that we do not need to
store data for retraining; instead, we can use it to update the model as
soon as it arrives. This is particularly attractive when data storage is
a concern due to privacy or resource limitations.

### Exercises [](#exercises)

20-1. Suppose we train a classifier for stock trading recommendations
using a random forest model, including the moving average of the stock
price as a feature. Since new stock market data arrives daily, we are
considering how to update the classifier daily to keep it up to date.
Should we take a stateless training or stateless retraining approach to
update the classifier?

20-2. Suppose we deploy a large language model (transformer) such as
ChatGPT that can answer user queries. The dialogue interface includes
thumbs-up and thumbs-down buttons so that users can give direct feedback
based on the generated queries. While collecting the user feedback, we
donâ€™t update the model immediately as new feedback arrives. However,
we are planning to release a new or updated model at least once per
month. Should we use stateless or stateful retraining for this model?

\

------------------------------------------------------------------------

