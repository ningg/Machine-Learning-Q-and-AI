





# Introduction
> 本页交代本书的写作背景、适合的读者、问答体例与阅读建议，并概述全书五大部分及各章主题，最后给出在线代码资源链接。
[](#introduction)

Thanks to rapid advancements in deep learning, we have seen a
significant expansion of machine learning and AI in recent years.

近年来深度学习快速发展，机器学习与 AI 的应用范围也随之大幅扩展。

This progress is exciting if we expect these advancements to create new
industries, transform existing ones, and improve the quality of life for
people around the world. On the other hand, the constant emergence of
new techniques can make it challenging and time-consuming to keep
abreast of the latest developments. Nonetheless, staying current is
essential for professionals and organizations that use these
technologies.

若这些进展能催生新产业、改造旧业态并改善全球生活质量，那无疑是令人振奋的。另一方面，新技术层出不穷，跟进最新进展既费时又费力；但对使用这些技术的从业者与机构而言，保持更新仍然十分必要。

I wrote this book as a resource for readers and machine learning
practitioners who want to advance their expertise in the field and learn
about techniques that I consider useful and significant but that are
often overlooked in traditional and introductory textbooks and classes.
I hope you'll find this book a valuable resource for obtaining new
insights and discovering new techniques you can implement in your work.

本书面向希望提升功力、并了解那些在传统入门教材中常被忽略、却十分重要的技术的读者与实践者。希望它能帮助你获得新洞见，并在工作中落地新技巧。

> Tips: 本书会突出`核心概念`，并且，会给出`示例`，辅助理解。

## Who Is This Book For?
> 本节对比市面上入门书与偏重数学专著的两极，说明本书的定位、预备知识要求，以及书中「机器学习」一词的用法。
[](#who-is-this-book-for)

Navigating the world of AI and machine learning literature can often
feel like walking a tightrope, with most books positioned at either end:
broad beginner's introductions or deeply mathematical treatises. This
book illustrates and discusses important developments in these fields
while staying approachable and not requiring an advanced math or coding
background.

AI 与机器学习读物往往两极：要么面向初学者的泛览，要么高度数学化的专著。本书在介绍重要进展的同时保持可读，不要求高等数学或编程背景。

> Tips: 本书，并不要求读者有高等数学知识、也无需编码背景。简单来说，普通的高中毕业，也可以流畅阅读。

This book is for people with some experience with machine learning who
want to learn new concepts and techniques. It's ideal for those who
have taken a beginner course in machine learning or deep learning or
have read an equivalent introductory book on the topic. (Throughout this
book, I will use *machine learning* as an umbrella term for machine
learning, deep learning, and AI.)

本书适合已具备一定机器学习基础、希望学习新概念与技术的读者；修过一门入门课或读过同等入门书尤为合适。（全书用 *machine learning* 统称机器学习、深度学习与 AI。）

> 本书中，会使用 *机器学习* 作为`统称`，包括机器学习、深度学习、AI。

## What Will You Get Out of This Book?
> 本节说明本书的问答体例、插图与练习安排、所涉主题广度，以及本书并非数学或编程教材的定位。
[](#what-will-you-get-out-of-this-book)

This book adopts a unique Q&A style, where each brief chapter is
structured around a central question related to fundamental concepts in
machine learning, deep learning, and AI. Every question is followed by
an explanation, with several illustrations and figures, as well as
exercises to test your understanding. Many chapters also include
references for further reading. These bite-sized nuggets of information
provide an enjoyable jumping-off point on your journey from machine
learning beginner to expert.

本书采用问答体例：每章围绕机器学习、深度学习与 AI 中的一个核心问题展开，配有解释、多幅插图与自测练习，许多章末附延伸阅读。这些信息块便于你从入门走向更熟练。

The book covers a wide range of topics. It includes new insights about
established architectures, such as convolutional networks, that allow
you to utilize these technologies more effectively. It also discusses
more advanced techniques, such as the inner workings of large language
models (LLMs) and vision transformers. Even experienced machine learning
researchers and practitioners will encounter something new to add to
their arsenal of techniques.

全书主题覆盖面广：既有对卷积网络等成熟架构的新视角，也讨论 LLM、视觉 Transformer 等更前沿内容；即便有经验的研究者与实践者也能从中拾取可纳入工具箱的新知。


> Tips: 本书，会介绍`AI 领域`的**典型概念**、知识，但不是数学或编码书籍。阅读时，无需证明或编码、突出易读性。

While this book will expose you to new concepts and ideas, it's not a
math or coding book. You won't need to solve any proofs or run any
code while reading. In other words, this book is a perfect travel
companion or something you can read on your favorite reading chair with
your morning coffee or tea.

本书重在概念与思想，不是数学或编程书：阅读时不必写证明或跑代码，适合旅途或晨间咖啡时随手翻阅。

## How to Read This Book
> 本节说明各章可独立阅读与建议顺序阅读的理由、练习与参考文献的位置，以及全书五大部分的划分逻辑。
[](#how-to-read-this-book)

Each chapter of this book is designed to be self-contained, offering you
the freedom to jump between topics as you wish. When a concept from one
chapter is explained in more detail in another, I've included chapter
references you can follow to fill in gaps in your understanding.

各章自成一体，可按兴趣跳转；若某概念在别章有更细讲解，文中会给出章节引用以便补齐背景。

> 本书每个章节，都是独立的，你可以跳过一些章节，直接阅读你感兴趣的章节。

However, there's a strategic sequence to the chapters. For example,
the early chapter on embeddings sets the stage for later discussions on
self-supervised learning and few-shot learning. For the easiest reading
experience and the most comprehensive grasp of the content, my
recommendation is to approach the book from start to finish.

但章节之间仍有策略性顺序：例如先读嵌入有助于理解后续自监督与少样本学习。若要最省力地建立整体图景，建议从头到尾通读。

> 然而，本书的章节，是有`顺序`的，建议从前往后阅读；因为，把`最通用的概念`，放在了**最前章节**。

Each chapter is accompanied by optional exercises for readers who want
to test their understanding, with an answer key located at the end of
the book. In addition, for any papers referenced in a chapter or further
reading on that chapter's topic, you can find the complete citation
information in that chapter's "References" section.

每章配有可选练习，书末附有答案要点；章内引用论文的完整书目信息见各章「References」。

The book is structured into five main parts centered on the most
important topics in machine learning and AI today.

全书按当今机器学习与 AI 最重要的主题分为五大部分。

> 整体内容，分为 5 个部分，都是`AI 领域`的`最重要的主题`。


> Tips: 下面**第一部分**，是`神经网络`和`深度学习`的`通用概念`，包含 嵌入、自监督学习、少样本学习、彩票假设、过拟合、多 GPU 训练范式等。

**Part I: Neural Networks and Deep Learning** covers questions about
deep neural networks and deep learning that are not specific to a
particular subdomain. For example, we discuss alternatives to supervised
learning and techniques for reducing overfitting, which is a common
problem when using machine learning models for real-world problems where
data is limited.

**第一部分：神经网络与深度学习**讨论不局限于某一子领域的深度网络与共性问题，例如监督学习之外的范式，以及数据有限时常见的过拟合与应对思路。

Chapter [\[ch01\]](./ch01/_books_ml-q-and-ai-ch01.md):
Embeddings, Latent Space, and Representations\
Delves into the distinctions and similarities between embedding vectors,
latent vectors, and representations. Elucidates how these concepts help
encode information in the context of machine learning.

第 1 章：嵌入、潜空间与表示——辨析嵌入向量、潜向量与表示的异同，以及它们在机器学习中表示信息的方式。

Chapter [\[ch02\]](./ch02/_books_ml-q-and-ai-ch02.md):
Self-Supervised Learning\
Focuses on self-supervised learning, a method that allows neural
networks to utilize large, unlabeled datasets in a supervised manner.

第 2 章：自监督学习——让神经网络以「类监督」方式利用大规模无标签数据。

Chapter [\[ch03\]](./ch03/_books_ml-q-and-ai-ch03.md):
Few-Shot Learning\
Introduces few-shot learning, a specialized supervised learning
technique tailored for small training datasets.

第 3 章：少样本学习——面向极小训练集的专门监督学习设定。

Chapter [\[ch04\]](./ch04/_books_ml-q-and-ai-ch04.md): The
Lottery Ticket Hypothesis\
Explores the idea that randomly initialized neural networks contain smaller, efficient subnetworks.

第 4 章：彩票假设——随机初始化的网络中可能存在更小却高效的子网络。

Chapter [\[ch05\]](./ch05/_books_ml-q-and-ai-ch05.md):
Reducing Overfitting with Data\
Addresses the challenge of overfitting in machine learning, discussing
strategies centered on data augmentation and the use of unlabeled data
to reduce overfitting.

第 5 章：用数据缓解过拟合——数据增强与利用无标签数据等策略。

Chapter [\[ch06\]](./ch06/_books_ml-q-and-ai-ch06.md):
Reducing Overfitting with Model Modifications\
Extends the conversation on overfitting, focusing on model-related
solutions like regularization, opting for simpler models, and ensemble
techniques.

第 6 章：用模型与训练改动缓解过拟合——正则、更小模型与集成等。

Chapter [\[ch07\]](./ch07/_books_ml-q-and-ai-ch07.md):
Multi-GPU Training Paradigms\
Explains various training paradigms for multi-GPU setups to accelerate
model training, including data and model parallelism.

第 7 章：多 GPU 训练范式——数据并行、模型并行等加速手段。

Chapter [\[ch08\]](./ch08/_books_ml-q-and-ai-ch08.md): The
Success of Transformers\
Explores the popular transformer architecture, highlighting features
like attention mechanisms, parallelization ease, and high parameter
counts.

第 8 章：Transformer 的成功——注意力、易并行与高参数量等特点。

Chapter [\[ch09\]](./ch09/_books_ml-q-and-ai-ch09.md):
Generative AI Models\
Provides a comprehensive overview of deep generative models, which are
used to produce various media forms, including images, text, and audio.
Discusses the strengths and weaknesses of each model type.

第 9 章：生成式 AI 模型——图像、文本、音频等媒介的深度生成模型概览与优劣。

Chapter [\[ch10\]](./ch10/_books_ml-q-and-ai-ch10.md):
Sources of Randomness\
Addresses the various sources of randomness in the training of deep
neural networks that may lead to inconsistent and non-reproducible
results during both training and inference. While randomness can be
accidental, it can also be intentionally introduced by design.

第 10 章：随机性的来源——训练与推理中导致结果不一致的多种随机因素，以及有意与无意引入的随机性。

> Tips: 下面**第二部分**，是`计算机视觉`的`典型概念`，包含 卷积神经网络、视觉变换器。

**Part II: Computer Vision** focuses on topics mainly related to deep
learning but specific to computer vision, many of which cover
convolutional neural networks and vision transformers.

**第二部分：计算机视觉**聚焦视觉任务中的深度学习，多涉及卷积网络与视觉 Transformer。

Chapter [\[ch11\]](./ch11/_books_ml-q-and-ai-ch11.md):
Calculating the Number of Parameters\
Explains the\
procedure for determining the parameters in a convolutional neural
network, which is useful for gauging a model's storage and memory\
requirements.

第 11 章：参数数量计算——估算卷积网络的存储与显存需求。

Chapter [\[ch12\]](./ch12/_books_ml-q-and-ai-ch12.md):
Fully Connected and Convolutional Layers\
Illustrates the circumstances in which convolutional layers can
seamlessly replace fully connected layers, which can be useful for
hardware optimization or simplifying implementations.

第 12 章：全连接与卷积层——何种情形下卷积可替代全连接以利于硬件或实现简化。

Chapter [\[ch13\]](./ch13/_books_ml-q-and-ai-ch13.md):
Large Training Sets for Vision Transformers\
Probes the rationale behind vision transformers requiring more extensive
training sets compared to conventional convolutional neural networks.

第 13 章：视觉 Transformer 为何需要更大训练集——相对传统 CNN 的数据规模要求。

> Tips: 下面**第三部分**，文本相关，是`自然语言处理`的`典型概念`，包含 分布式假设、数据增强、自注意力、编码器-解码器式变换器、使用和微调预训练变换器、评估生成式大语言模型等。

**Part III: Natural Language Processing** covers topics around working
with text, many of which are related to transformer architectures and
self-attention.

**第三部分：自然语言处理**围绕文本与 Transformer、自注意力等主题。

Chapter [\[ch14\]](./ch14/_books_ml-q-and-ai-ch14.md): The
Distributional Hypothesis\
Delves into the distributional hypothesis, a linguistic theory
suggesting that words appearing in the same contexts tend to possess
similar meanings, which has useful implications for training machine
learning models.

第 14 章：分布式假设——语境相近则语义相近的语言学思想及其对模型训练的启示。

Chapter [\[ch15\]](./ch15/_books_ml-q-and-ai-ch15.md):
Data Augmentation for Text\
Highlights the significance of data augmentation for text, a technique
used to artificially increase dataset sizes, which can help with
improving model performance.

第 15 章：文本数据增强——人工扩充语料以提升性能。

Chapter [\[ch16\]](./ch16/_books_ml-q-and-ai-ch16.md):
Self-Attention\
Introduces self-attention, a mechanism allowing each segment of a neural
network's input to refer to other parts. Self-attention is a key
mechanism in modern large language models.

第 16 章：自注意力——输入各部分互相关注的机制，是现代 LLM 的核心组件之一。

Chapter [\[ch17\]](./ch17/_books_ml-q-and-ai-ch17.md):
Encoder- and Decoder-Style Transformers\
Describes the nuances of encoder and decoder transformer architectures and
explains which type of architecture is most useful for each language
processing task.

第 17 章：编码器式与解码器式 Transformer——结构差异与适用任务。

Chapter [\[ch18\]](./ch18/_books_ml-q-and-ai-ch18.md):
Using and Fine-Tuning Pretrained Transformers\
Explains different methods for fine-tuning pretrained large language
models and discusses their strengths and weaknesses.

第 18 章：使用与微调预训练 Transformer——多种微调路径及其利弊。

Chapter [\[ch19\]](./ch19/_books_ml-q-and-ai-ch19.md):
Evaluating Generative Large Language Models\
Lists prominent evaluation metrics for language models like Perplexity, BLEU,
ROUGE, and BERTScore.

第 19 章：评估生成式大语言模型——Perplexity、BLEU、ROUGE、BERTScore 等常见指标。

> Tips: 下面**第四部分**，是`生产`和`部署`的`典型概念`，包含 无状态和有状态训练、数据分布偏移等。

**Part IV: Production and Deployment** covers questions pertaining to
practical scenarios, such as increasing inference speeds and various
types of distribution shifts.

**第四部分：生产与部署**涵盖推理加速、数据分布偏移等落地问题。

Chapter [\[ch20\]](./ch20/_books_ml-q-and-ai-ch20.md):
Stateless and Stateful Training\
Distinguishes between stateless and stateful training methodologies used
in deploying models.

第 20 章：无状态与有状态训练——部署场景下的两种训练/更新范式。

Chapter [\[ch21\]](./ch21/_books_ml-q-and-ai-ch21.md):
Data-Centric AI\
Explores data-centric AI, which priori-\
 tizes refining datasets to enhance model performance. This approach
contrasts with the conventional model-centric approach, which emphasizes
improving model architectures or methods.

第 21 章：以数据为中心的 AI——通过打磨数据集提升性能，与以模型为中心路线的对照。

Chapter [\[ch22\]](./ch22/_books_ml-q-and-ai-ch22.md):
Speeding Up Inference\
Introduces techniques to enhance the speed of model inference without
tweaking the model's architecture or compromising accuracy.

第 22 章：加速推理——在不改架构、不明显牺牲精度的前提下提速。

Chapter [\[ch23\]](./ch23/_books_ml-q-and-ai-ch23.md):
Data Distribution Shifts\
Post-deployment, AI models\
may face discrepancies between training data and real-world data
distributions, known as data distribution shifts. These shifts can
deteriorate model performance. This chapter categorizes and elaborates
on common shifts like covariate shift, concept drift, label shift, and
domain shift.

第 23 章：数据分布偏移——上线后训练分布与真实分布不一致及其对性能的影响，并梳理协变量偏移、概念漂移、标签偏移、域偏移等。

> Tips: 下面**第五部分**，是`预测性能`和`模型评估`的`典型概念`，包含 泊松回归、置信区间、置信区间与一致性预测、交叉验证、训练和测试集不一致、有限标签数据等。

**Part V: Predictive Performance and Model Evaluation** dives deeper
into various aspects of squeezing out predictive performance, such as
changing the loss function, setting up *k*-fold cross-validation, and
dealing with limited labeled data.

**第五部分：预测性能与模型评估**深入损失设计、*k* 折交叉验证、标签稀缺等榨取预测性能的主题。

Chapter [\[ch24\]](./ch24/_books_ml-q-and-ai-ch24.md):
Poisson and Ordinal Regression\
Highlights the differences between Poisson and ordinal regression.
Poisson regression is suitable for count data that follows a Poisson
distribution, like the number of colds contracted on an airplane. In
contrast, ordinal regression caters to ordered categorical data without
assuming equidistant categories, such as disease severity.

第 24 章：泊松回归与序数回归——计数数据与有序分类各自适用的设定与区别。

Chapter [\[ch25\]](./ch25/_books_ml-q-and-ai-ch25.md):
Confidence Intervals\
Delves into methods for constructing confidence intervals for machine
learning classifiers. Reviews the purpose of confidence intervals,
discusses how they estimate unknown population parameters, and
introduces techniques such as normal approximation intervals,
bootstrapping, and retraining with various random seeds.

第 25 章：置信区间——分类器上构造置信区间的方法、意义及正态近似、自助法、多随机种子重训等。

Chapter [\[ch26\]](./ch26/_books_ml-q-and-ai-ch26.md):
Confidence Intervals vs. Conformal Predictions\
Discusses the distinction between confidence intervals and conformal
predictions and describes the latter as a tool for creating prediction
intervals that cover actual outcomes with specific probability.

第 26 章：置信区间与 conformal prediction——二者差异及后者构造具有覆盖保证的预测区间。

Chapter [\[ch27\]](./ch27/_books_ml-q-and-ai-ch27.md):
Proper Metrics\
Focuses on the essential properties of a proper metric in mathematics
and computer science. Examines whether commonly used loss functions in
machine learning, such as mean squared error and cross-entropy loss,
satisfy these properties.

第 27 章：proper scoring rule（恰当评分）——数学与机器学习损失（如 MSE、交叉熵）是否满足恰当性。

Chapter [\[ch28\]](./ch28/_books_ml-q-and-ai-ch28.md): The
*k* in *k*-Fold Cross-Validation\
Explores the role of the *k* in *k*-fold cross-validation and provides
insight into the advantages and disadvantages of selecting a large *k*.

第 28 章：*k* 折交叉验证中的 *k*——取大或取小的利弊。

Chapter [\[ch29\]](./ch29/_books_ml-q-and-ai-ch29.md):
Training and Test Set Discordance\
Addresses the scenario where a model performs better on a test dataset
than the training dataset. Offers strategies to discover and address
discrepancies\
 between training and test datasets, introducing the concept of
adversarial validation.

第 29 章：训练集与测试集表现倒挂——如何发现与处理分布差异，并介绍对抗验证思路。

Chapter [\[ch30\]](./ch30/_books_ml-q-and-ai-ch30.md):
Limited Labeled Data\
Introduces various techniques to enhance model performance in situations
where data is limited. Covers data labeling, bootstrapping, and
paradigms such as transfer learning, active learning, and multimodal
learning.

第 30 章：标签数据有限——标注、自助法以及迁移学习、主动学习、多模态等范式。

## Online Resources
> 本节给出作者在 GitHub 上提供的补充代码与深度学习材料链接，便于与正文对照练习。
[](#online-resources)

I've provided optional supplementary materials on GitHub with code
examples for certain chapters to enhance your learning experience (see
<https://github.com/rasbt/MachineLearning-QandAI-book>). These materials
are designed as practical extensions and deep dives into topics covered
in the book. You can use them alongside each chapter or explore them
after reading to solidify and expand your knowledge.

作者在 GitHub 上提供了部分章节的代码示例与补充材料（<https://github.com/rasbt/MachineLearning-QandAI-book>），可作为正文的实践延伸与深挖；可边读边用，也可读后再做以巩固。

Without further ado, let's dive in.

闲话少叙，开始阅读吧。


------------------------------------------------------------------------

