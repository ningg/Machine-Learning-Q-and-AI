








# Chapter 20: Stateless and Stateful Training
> 本章在生产训练/部署语境下对比无状态（周期重训）与有状态（增量更新）两类流程，并各辅图示与习题。
[](#stateless-and-stateful-training)



**What is the difference between stateless and stateful training
workflows in the context of production and deployment systems?**

无状态与有状态描述的是：新数据到来时，是周期性从零或近似从零重训，还是在既有权重上继续优化。

Stateless training and stateful training refer to different ways of
training a production model.

> Tips: **无状态训练** `stateless` 和 **有状态训练** `stateful`，是两种不同的训练方式；训练生产模型时，需要考虑用哪种方式。


## Stateless (Re)training
> 本节说明无状态训练先训初版再在数据流上按窗口重训的图景，并点出与传统批量学习模型的天然契合。
[](#stateless-retraining)

In stateless training, the more conventional approach, we first train an
`initial model` on the original training set and then retrain it as new
data arrives. Hence, stateless training is also commonly referred to as
stateless *retraining*.

常见做法是先在初始训练集上得到 `initial model`，新数据持续到达后按策略整批或按窗口再次训练，因此也称无状态 *retraining*。

> Tips: 无状态训练，是先训练一个`初始模型`，然后在新数据到达时，重新训练模型；可以简单认为是`树状结构`，初始模型是`父节点`、衍生出一堆`叶子节点`模型.

As Figure [20.1](#fig-ch20-fig01) shows, we can think of stateless retraining as a
sliding window approach in which we retrain the initial model on
different parts of the data from a given data stream.

可把它想象成对流式数据截取滑动窗口、周期性回到同一训练管线重训，使模型随时间对齐最新数据段。

> Tips: 图示中，`初始模型`是`父节点`，`新模型`是`叶子节点`，`新模型`是基于`初始模型`训练的；训练新模型时，会截取`滑动窗口`数据。

<a id="fig-ch20-fig01"></a>

<div align="center">
  <img src="./images/ch20-fig01.png" alt="Stateless training replaces the model periodically." width="78%" />
  <div><b>Figure 20.1</b></div>
</div>

For example, to update the initial model in
Figure [20.1](#fig-ch20-fig01) (Model 1) to a newer model (Model 2), we train the
model on 30 percent of the initial data and 70 percent of the most
recent data at a given point in time.

举例：从 Model 1 升到 Model 2 时，可在某一时刻混合 30% 早期样本与 70% 近期样本再训一版。

Stateless retraining is a straightforward approach that allows us to
adapt the model to the most recent changes in the data and
feature-target relationships via retraining the model from scratch in
user-defined checkpoint intervals. This approach is prevalent with
conventional machine learning systems that cannot be fine-tuned as part
of a transfer or self-supervised learning workflow (see
Chapter [\[ch02\]](./ch02/_books_ml-q-and-ai-ch02.md)).

做法直观：按固定节拍整体重训即可响应最新的特征–目标关系；它特别适合难以“增量更新权重”的传统监督管线（参见第 2 章关于迁移/自监督的讨论）。

> Tips: 传统的模型，中无状态训练，比较流行，比如`随机森林`、`梯度提升`等，这些都是无法`微调`的

For example, standard implementations of tree-based models, such as
random forests and gradient boosting (XGBoost, CatBoost, and LightGBM),
fall into this category.

例如随机森林与各类梯度提升库的主流实现通常属于此类。

## Stateful Training
> 本节描述有状态训练如何在初训后持续微调、并阐明其针对概念/特征/标签漂移与迁移学习“换任务”之间的本质不同。
[](#stateful-training)

In stateful training, we train the model on an initial batch of data and
then update it periodically (as opposed to retraining it) when new data
arrives.

有状态路线是首次批量训练后，新数据到来时在原参数附近继续更新，而非每次都完整重训。

> Tips: 有状态的训练，可以认为是 `链式结构`，初始模型 -> 新模型 -> 新模型 -> ... ， 每次都基于最新模型叠加而来.

As illustrated in Figure [20.2](#fig-ch20-fig02), we do not retrain the initial model (Model1.0)
from scratch; instead, we update or fine-tune it as new data arrives.
This approach is particularly attractive for models compatible with
transfer learning or self-supervised learning.

图 20.2 显示并不每次从 Model 1.0 彻底重算，而是在其权重上微调；这与支持迁移或自监督初始化的深度网络十分契合。

<a id="fig-ch20-fig02"></a>

<div align="center">
  <img src="./images/ch20-fig02.png" alt="Stateful training updates models periodically." width="78%" />
  <div><b>Figure 20.2</b></div>
</div>

The stateful approach mimics a transfer or self-supervised learning
workflow where we adopt a pretrained model for fine-tuning. However,
stateful training differs fundamentally from transfer and
self-supervised learning because it updates the model to accommodate
concept, feature, and label drifts. In contrast, transfer and
self-supervised learning aim to adopt the model for a different
classification task. For instance, in transfer learning, the target
labels often differ. In self-supervised learning, we obtain the target
labels from the dataset features.

表面上都像“拿一版预训练再微调”，但有状态强调同一部署任务下随时间的分布漂移（概念、特征、标注方式变化）；迁移与自监督则更常指换任务或从数据本身构造伪标签。

> Tips: 有状态的训练，跟`迁移学习`、`自监督学习`，有本质区别；有状态的训练，会更新模型，以适应概念、特征、标签的漂移；而迁移学习、自监督学习，是基于预训练模型，进行微调。

One significant advantage of stateful training is that we do not need to
store data for retraining; instead, we can use it to update the model as
soon as it arrives. This is particularly attractive when data storage is
a concern due to privacy or resource limitations.

若不能长期囤积原始样本（合规或资源），有状态增量可以在样本流过当下立即用于更新，而不必保留全史做周期全量重训。

> Tips: **有状态的训练**，不需要存储数据，可以及时更新模型；这在`隐私`或`资源有限`的情况下，特别有用。

## Exercises
> 本节通过金融日更随机森林与按月迭代大模型两题，巩固两类形态的算法与业务约束层面的选择。
[](#exercises)

20-1. Suppose we train a classifier for stock trading recommendations
using a random forest model, including the moving average of the stock
price as a feature. Since new stock market data arrives daily, we are
considering how to update the classifier daily to keep it up to date.
Should we take a stateful training or stateless retraining approach to
update the classifier?

<details><summary>Answer, Click to expand</summary>


When updating a random forest classifier daily with new stock market data, the choice between **stateless retraining** and **stateful retraining** depends on computational efficiency, data recency, and model integrity. Here's a structured analysis:

### 1. **Stateless Retraining (Full Retraining)**  
   - **Approach**: Train a **new model from scratch** daily using the **entire updated dataset** (historical data + new day's data).  
   - **Why it fits best for random forests**:  
     - Random forests are **not inherently incremental**. Trees are built independently via bootstrapping and feature randomization.  
     - Retraining from scratch ensures:  
       - **Consistent data representation** (e.g., recalculated moving averages reflect the full history).  
       - **Optimal tree structures** based on all available data, avoiding bias from sequential updates.  
     - Mitigates **concept drift** by re-optimizing splits with fresh data.  
   - **Trade-offs**:  
     - Computationally expensive as data grows (requires daily full training).  
     - Use a **sliding window** (e.g., 2 years of data) to cap training time and prioritize recent trends.  

### 2. **Stateful Retraining (Incremental Updates)**  
   - **Approach**: Update the **existing model** with new data only (e.g., add new trees or adjust leaf nodes).  
   - **Why it’s less suitable**:  
     - Random forests lack native support for incremental learning. Workarounds (e.g., warm starts in `sklearn`) merely **add new trees** without modifying existing ones. This:  
       - **Dilutes model cohesion** (old trees may become outdated).  
       - **Increases memory/compute** over time without guaranteeing improved accuracy.  
     - Moving averages and other temporal features **lose consistency** if not recalculated globally.  
   - **When it might work**:  
     - Only if new data is appended without affecting historical computations (rare for financial data).  
     - Prefer models **designed for streaming** (e.g., online gradient boosting, Hoeffding trees).  

### Recommendation: **Stateless Retraining**  
- **Key reasons**:  
  1. **Algorithmic compatibility**: Random forests thrive on complete, consistent datasets.  
  2. **Data integrity**: Ensures moving averages and technical indicators are recalculated correctly over the full series.  
  3. **Predictive stability**: Avoids "patchwork" models that mix outdated and new logic.  
- **Optimizations**:  
  - **Sliding window**: Retrain on a fixed-duration window (e.g., 500 trading days) to balance recency and speed.  
  - **Parallelization**: Leverage distributed computing (e.g., Spark ML) for faster daily jobs.  
  - **Model versioning**: Deploy the new model after validation to avoid downtime.  

### Practical Workflow  
1. **Daily update**:  
   - Fetch new data → recalculate features (e.g., moving averages) for the full dataset/window.  
   - Retrain the random forest from scratch.  
   - Validate against a holdout set (e.g., recent 30 days).  
2. **Fallback**: If retraining time is prohibitive:  
   - Reduce frequency (e.g., weekly) or window size.  
   - Switch to an **online-capable algorithm** (e.g., LGBM with `partial_fit`).  

### Conclusion  
For a random forest classifier with time-dependent features like moving averages, **stateless retraining** is **strongly recommended**. It preserves data consistency and model robustness, despite higher compute costs. If latency becomes critical, explore windowed retraining or alternative online-learning models instead of forcing stateful updates onto a batch-oriented algorithm.

</details>

20-2. Suppose we deploy a large language model (transformer) such as
ChatGPT that can answer user queries. The dialogue interface includes
thumbs-up and thumbs-down buttons so that users can give direct feedback
based on the generated queries. While collecting the user feedback, we
don't update the model immediately as new feedback arrives. However,
we are planning to release a new or updated model at least once per
month. Should we use stateless or stateful retraining for this model?


------------------------------------------------------------------------

