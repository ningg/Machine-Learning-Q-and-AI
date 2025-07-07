







# Introduction
[](#introduction)

Thanks to rapid advancements in deep learning, we have seen a
significant expansion of machine learning and AI in recent years.

This progress is exciting if we expect these advancements to create new
industries, transform existing ones, and improve the quality of life for
people around the world. On the other hand, the constant emergence of
new techniques can make it challenging and time-consuming to keep
abreast of the latest developments. Nonetheless, staying current is
essential for professionals and organizations that use these
technologies.

I wrote this book as a resource for readers and machine learning
practitioners who want to advance their expertise in the field and learn
about techniques that I consider useful and significant but that are
often overlooked in traditional and introductory textbooks and classes.
I hope you'll find this book a valuable resource for obtaining new
insights and discovering new techniques you can implement in your work.

> Tips: 本书会突出`核心概念`，并且，会给出`示例`，辅助理解。

## Who Is This Book For?
[](#who-is-this-book-for)

Navigating the world of AI and machine learning literature can often
feel like walking a tightrope, with most books positioned at either end:
broad beginner's introductions or deeply mathematical treatises. This
book illustrates and discusses important developments in these fields
while staying approachable and not requiring an advanced math or coding
background.

> Tips: 本书，并不要求读者有高等数学知识、也无需编码背景。简单来说，普通的高中毕业，也可以流畅阅读。

This book is for people with some experience with machine learning who
want to learn new concepts and techniques. It's ideal for those who
have taken a beginner course in machine learning or deep learning or
have read an equivalent introductory book on the topic. (Throughout this
book, I will use *machine learning* as an umbrella term for machine
learning, deep learning, and AI.)

> 本书中，会使用 *机器学习* 作为`统称`，包括机器学习、深度学习、AI。

## What Will You Get Out of This Book?
[](#what-will-you-get-out-of-this-book)

This book adopts a unique Q&A style, where each brief chapter is
structured around a central question related to fundamental concepts in
machine learning, deep learning, and AI. Every question is followed by
an explanation, with several illustrations and figures, as well as
exercises to test your understanding. Many chapters also include
references for further reading. These bite-sized nuggets of information
provide an enjoyable jumping-off point on your journey from machine
learning beginner to expert.

The book covers a wide range of topics. It includes new insights about
established architectures, such as convolutional networks, that allow
you to utilize these technologies more effectively. It also discusses
more advanced techniques, such as the inner workings of large language
models (LLMs) and vision transformers. Even experienced machine learning
researchers and practitioners will encounter something new to add to
their arsenal of techniques.


> Tips: 本书，会介绍`AI 领域`的**典型概念**、知识，但不是数学或编码书籍。阅读时，无需证明或编码、突出易读性。

While this book will expose you to new concepts and ideas, it's not a
math or coding book. You won't need to solve any proofs or run any
code while reading. In other words, this book is a perfect travel
companion or something you can read on your favorite reading chair with
your morning coffee or tea.

## How to Read This Book
[](#how-to-read-this-book)

Each chapter of this book is designed to be self-contained, offering you
the freedom to jump between topics as you wish. When a concept from one
chapter is explained in more detail in another, I've included chapter
references you can follow to fill in gaps in your understanding.

> 本书每个章节，都是独立的，你可以跳过一些章节，直接阅读你感兴趣的章节。

However, there's a strategic sequence to the chapters. For example,
the early chapter on embeddings sets the stage for later discussions on
self-supervised learning and few-shot learning. For the easiest reading
experience and the most comprehensive grasp of the content, my
recommendation is to approach the book from start to finish.

> 然而，本书的章节，是有`顺序`的，建议从前往后阅读；因为，把`最通用的概念`，放在了**最前章节**。

Each chapter is accompanied by optional exercises for readers who want
to test their understanding, with an answer key located at the end of
the book. In addition, for any papers referenced in a chapter or further
reading on that chapter's topic, you can find the complete citation
information in that chapter's "References" section.

The book is structured into five main parts centered on the most
important topics in machine learning and AI today.

> 整体内容，分为 5 个部分，都是`AI 领域`的`最重要的主题`。


> Tips: 下面**第一部分**，是`神经网络`和`深度学习`的`通用概念`，包含 嵌入、自监督学习、少样本学习、彩票假设、过拟合、多 GPU 训练范式等。

**Part I: Neural Networks and Deep Learning** covers questions about
deep neural networks and deep learning that are not specific to a
particular subdomain. For example, we discuss alternatives to supervised
learning and techniques for reducing overfitting, which is a common
problem when using machine learning models for real-world problems where
data is limited.

Chapter [\[ch01\]](../ch01/_books_ml-q-and-ai-ch01.md):
Embeddings, Latent Space, and Representations\
Delves into the distinctions and similarities between embedding vectors,
latent vectors, and representations. Elucidates how these concepts help
encode information in the context of machine learning.

Chapter [\[ch02\]](../ch02/_books_ml-q-and-ai-ch02.md):
Self-Supervised Learning\
Focuses on self-supervised learning, a method that allows neural
networks to utilize large, unlabeled datasets in a supervised manner.

Chapter [\[ch03\]](../ch03/_books_ml-q-and-ai-ch03.md):
Few-Shot Learning\
Introduces few-shot learning, a specialized supervised learning
technique tailored for small training datasets.

Chapter [\[ch04\]](../ch04/_books_ml-q-and-ai-ch04.md): The
Lottery Ticket Hypothesis\
Explores the idea that randomly initialized neural networks contain smaller, efficient subnetworks.

Chapter [\[ch05\]](../ch05/_books_ml-q-and-ai-ch05.md):
Reducing Overfitting with Data\
Addresses the challenge of overfitting in machine learning, discussing
strategies centered on data augmentation and the use of unlabeled data
to reduce overfitting.

Chapter [\[ch06\]](../ch06/_books_ml-q-and-ai-ch06.md):
Reducing Overfitting with Model Modifications\
Extends the conversation on overfitting, focusing on model-related
solutions like regularization, opting for simpler models, and ensemble
techniques.

Chapter [\[ch07\]](../ch07/_books_ml-q-and-ai-ch07.md):
Multi-GPU Training Paradigms\
Explains various training paradigms for multi-GPU setups to accelerate
model training, including data and model parallelism.

Chapter [\[ch08\]](../ch08/_books_ml-q-and-ai-ch08.md): The
Success of Transformers\
Explores the popular transformer architecture, highlighting features
like attention mechanisms, parallelization ease, and high parameter
counts.

Chapter [\[ch09\]](../ch09/_books_ml-q-and-ai-ch09.md):
Generative AI Models\
Provides a comprehensive overview of deep generative models, which are
used to produce various media forms, including images, text, and audio.
Discusses the strengths and weaknesses of each model type.

Chapter [\[ch10\]](../ch10/_books_ml-q-and-ai-ch10.md):
Sources of Randomness\
Addresses the various sources of randomness in the training of deep
neural networks that may lead to inconsistent and non-reproducible
results during both training and inference. While randomness can be
accidental, it can also be intentionally introduced by design.

> Tips: 下面**第二部分**，是`计算机视觉`的`典型概念`，包含 卷积神经网络、视觉变换器。

**Part II: Computer Vision** focuses on topics mainly related to deep
learning but specific to computer vision, many of which cover
convolutional neural networks and vision transformers.

Chapter [\[ch11\]](../ch11/_books_ml-q-and-ai-ch11.md):
Calculating the Number of Parameters\
Explains the\
procedure for determining the parameters in a convolutional neural
network, which is useful for gauging a model's storage and memory\
requirements.

Chapter [\[ch12\]](../ch12/_books_ml-q-and-ai-ch12.md):
Fully Connected and Convolutional Layers\
Illustrates the circumstances in which convolutional layers can
seamlessly replace fully connected layers, which can be useful for
hardware optimization or simplifying implementations.

Chapter [\[ch13\]](../ch13/_books_ml-q-and-ai-ch13.md):
Large Training Sets for Vision Transformers\
Probes the rationale behind vision transformers requiring more extensive
training sets compared to conventional convolutional neural networks.

> Tips: 下面**第三部分**，文本相关，是`自然语言处理`的`典型概念`，包含 分布式假设、数据增强、自注意力、编码器-解码器式变换器、使用和微调预训练变换器、评估生成式大语言模型等。

**Part III: Natural Language Processing** covers topics around working
with text, many of which are related to transformer architectures and
self-attention.

Chapter [\[ch14\]](../ch14/_books_ml-q-and-ai-ch14.md): The
Distributional Hypothesis\
Delves into the distributional hypothesis, a linguistic theory
suggesting that words appearing in the same contexts tend to possess
similar meanings, which has useful implications for training machine
learning models.

Chapter [\[ch15\]](../ch15/_books_ml-q-and-ai-ch15.md):
Data Augmentation for Text\
Highlights the significance of data augmentation for text, a technique
used to artificially increase dataset sizes, which can help with
improving model performance.

Chapter [\[ch16\]](../ch16/_books_ml-q-and-ai-ch16.md):
Self-Attention\
Introduces self-attention, a mechanism allowing each segment of a neural
network's input to refer to other parts. Self-attention is a key
mechanism in modern large language models.

Chapter [\[ch17\]](../ch17/_books_ml-q-and-ai-ch17.md):
Encoder- and Decoder-Style Transformers\
Describes the nuances of encoder and decoder transformer architectures and
explains which type of architecture is most useful for each language
processing task.

Chapter [\[ch18\]](../ch18/_books_ml-q-and-ai-ch18.md):
Using and Fine-Tuning Pretrained Transformers\
Explains different methods for fine-tuning pretrained large language
models and discusses their strengths and weaknesses.

Chapter [\[ch19\]](../ch19/_books_ml-q-and-ai-ch19.md):
Evaluating Generative Large Language Models\
Lists prominent evaluation metrics for language models like Perplexity, BLEU,
ROUGE, and BERTScore.

> Tips: 下面**第四部分**，是`生产`和`部署`的`典型概念`，包含 无状态和有状态训练、数据分布偏移等。

**Part IV: Production and Deployment** covers questions pertaining to
practical scenarios, such as increasing inference speeds and various
types of distribution shifts.

Chapter [\[ch20\]](../ch20/_books_ml-q-and-ai-ch20.md):
Stateless and Stateful Training\
Distinguishes between stateless and stateful training methodologies used
in deploying models.

Chapter [\[ch21\]](../ch21/_books_ml-q-and-ai-ch21.md):
Data-Centric AI\
Explores data-centric AI, which priori-\
 tizes refining datasets to enhance model performance. This approach
contrasts with the conventional model-centric approach, which emphasizes
improving model architectures or methods.

Chapter [\[ch22\]](../ch22/_books_ml-q-and-ai-ch22.md):
Speeding Up Inference\
Introduces techniques to enhance the speed of model inference without
tweaking the model's architecture or compromising accuracy.

Chapter [\[ch23\]](../ch23/_books_ml-q-and-ai-ch23.md):
Data Distribution Shifts\
Post-deployment, AI models\
may face discrepancies between training data and real-world data
distributions, known as data distribution shifts. These shifts can
deteriorate model performance. This chapter categorizes and elaborates
on common shifts like covariate shift, concept drift, label shift, and
domain shift.

> Tips: 下面**第五部分**，是`预测性能`和`模型评估`的`典型概念`，包含 泊松回归、置信区间、置信区间与一致性预测、交叉验证、训练和测试集不一致、有限标签数据等。

**Part V: Predictive Performance and Model Evaluation** dives deeper
into various aspects of squeezing out predictive performance, such as
changing the loss function, setting up *k*-fold cross-validation, and
dealing with limited labeled data.

Chapter [\[ch24\]](../ch24/_books_ml-q-and-ai-ch24.md):
Poisson and Ordinal Regression\
Highlights the differences between Poisson and ordinal regression.
Poisson regression is suitable for count data that follows a Poisson
distribution, like the number of colds contracted on an airplane. In
contrast, ordinal regression caters to ordered categorical data without
assuming equidistant categories, such as disease severity.

Chapter [\[ch25\]](../ch25/_books_ml-q-and-ai-ch25.md):
Confidence Intervals\
Delves into methods for constructing confidence intervals for machine
learning classifiers. Reviews the purpose of confidence intervals,
discusses how they estimate unknown population parameters, and
introduces techniques such as normal approximation intervals,
bootstrapping, and retraining with various random seeds.

Chapter [\[ch26\]](../ch26/_books_ml-q-and-ai-ch26.md):
Confidence Intervals vs. Conformal Predictions\
Discusses the distinction between confidence intervals and conformal
predictions and describes the latter as a tool for creating prediction
intervals that cover actual outcomes with specific probability.

Chapter [\[ch27\]](../ch27/_books_ml-q-and-ai-ch27.md):
Proper Metrics\
Focuses on the essential properties of a proper metric in mathematics
and computer science. Examines whether commonly used loss functions in
machine learning, such as mean squared error and cross-entropy loss,
satisfy these properties.

Chapter [\[ch28\]](../ch28/_books_ml-q-and-ai-ch28.md): The
*k* in *k*-Fold Cross-Validation\
Explores the role of the *k* in *k*-fold cross-validation and provides
insight into the advantages and disadvantages of selecting a large *k*.

Chapter [\[ch29\]](../ch29/_books_ml-q-and-ai-ch29.md):
Training and Test Set Discordance\
Addresses the scenario where a model performs better on a test dataset
than the training dataset. Offers strategies to discover and address
discrepancies\
 between training and test datasets, introducing the concept of
adversarial validation.

Chapter [\[ch30\]](../ch30/_books_ml-q-and-ai-ch30.md):
Limited Labeled Data\
Introduces various techniques to enhance model performance in situations
where data is limited. Covers data labeling, bootstrapping, and
paradigms such as transfer learning, active learning, and multimodal
learning.

## Online Resources
[](#online-resources)

I've provided optional supplementary materials on GitHub with code
examples for certain chapters to enhance your learning experience (see
<https://github.com/rasbt/MachineLearning-QandAI-book>). These materials
are designed as practical extensions and deep dives into topics covered
in the book. You can use them alongside each chapter or explore them
after reading to solidify and expand your knowledge.

Without further ado, let's dive in.


------------------------------------------------------------------------

