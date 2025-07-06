







# Introduction [](#introduction)

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

## Who Is This Book For? [](#who-is-this-book-for)

Navigating the world of AI and machine learning literature can often
feel like walking a tightrope, with most books positioned at either end:
broad beginner's introductions or deeply mathematical treatises. This
book illustrates and discusses important developments in these fields
while staying approachable and not requiring an advanced math or coding
background.

This book is for people with some experience with machine learning who
want to learn new concepts and techniques. It's ideal for those who
have taken a beginner course in machine learning or deep learning or
have read an equivalent introductory book on the topic. (Throughout this
book, I will use *machine learning* as an umbrella term for machine
learning, deep learning, and AI.)

## What Will You Get Out of This Book? [](#what-will-you-get-out-of-this-book)

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

While this book will expose you to new concepts and ideas, it's not a
math or coding book. You won't need to solve any proofs or run any
code while reading. In other words, this book is a perfect travel
companion or something you can read on your favorite reading chair with
your morning coffee or tea.

## How to Read This Book [](#how-to-read-this-book)

Each chapter of this book is designed to be self-contained, offering you
the freedom to jump between topics as you wish. When a concept from one
chapter is explained in more detail in another, I've included chapter
references you can follow to fill in gaps in your understanding.

However, there's a strategic sequence to the chapters. For example,\
the early chapter on embeddings sets the stage for later discussions on
self-supervised learning and few-shot learning. For the easiest reading
experience and the most comprehensive grasp of the content, my
recommendation is to approach the book from start to finish.

Each chapter is accompanied by optional exercises for readers who want
to test their understanding, with an answer key located at the end of
the book. In addition, for any papers referenced in a chapter or further
reading on that chapter's topic, you can find the complete citation
information in that chapter's "References" section.

The book is structured into five main parts centered on the most
important topics in machine learning and AI today.

**Part I: Neural Networks and Deep Learning** covers questions about
deep neural networks and deep learning that are not specific to a
particular subdomain. For example, we discuss alternatives to supervised
learning and techniques for reducing overfitting, which is a common
problem when using machine learning models for real-world problems where
data is limited.

Chapter [\[ch01\]](./chapters_ch01/_books_ml-q-and-ai-chapters_ch01.md):
Embeddings, Latent Space, and Representations\
Delves into the distinctions and similarities between embedding vectors,
latent vectors, and representations. Elucidates how these concepts help
encode information in the context of machine learning.

Chapter [\[ch02\]](./chapters_ch02/_books_ml-q-and-ai-chapters_ch02.md):
Self-Supervised Learning\
Focuses on self-supervised learning, a method that allows neural
networks to utilize large, unlabeled datasets in a supervised manner.

Chapter [\[ch03\]](./chapters_ch03/_books_ml-q-and-ai-chapters_ch03.md):
Few-Shot Learning\
Introduces few-shot learning, a specialized supervised learning
technique tailored for small training datasets.

Chapter [\[ch04\]](./chapters_ch04/_books_ml-q-and-ai-chapters_ch04.md): The
Lottery Ticket Hypothesis\
Explores the idea that ran-\
 domlyinitializedneuralnetworkscontainsmaller,efficient subnetworks.

Chapter [\[ch05\]](./chapters_ch05/_books_ml-q-and-ai-chapters_ch05.md):
Reducing Overfitting with Data\
Addresses the challenge of overfitting in machine learning, discussing
strategies centered on data augmentation and the use of unlabeled data
to reduce overfitting.

Chapter [\[ch06\]](./chapters_ch06/_books_ml-q-and-ai-chapters_ch06.md):
Reducing Overfitting with Model Modifications\
Extends the conversation on overfitting, focusing on model-related
solutions like regularization, opting for simpler models, and ensemble
techniques.

Chapter [\[ch07\]](./chapters_ch07/_books_ml-q-and-ai-chapters_ch07.md):
Multi-GPU Training Paradigms\
Explains various training paradigms for multi-GPU setups to accelerate
model training, including data and model parallelism.

Chapter [\[ch08\]](./chapters_ch08/_books_ml-q-and-ai-chapters_ch08.md): The
Success of Transformers\
Explores the popular transformer architecture, highlighting features
like attention mechanisms, parallelization ease, and high parameter
counts.

Chapter [\[ch09\]](./chapters_ch09/_books_ml-q-and-ai-chapters_ch09.md):
Generative AI Models\
Provides a comprehensive overview of deep generative models, which are
used to produce various media forms, including images, text, and audio.
Discusses the strengths and weaknesses of each model type.

Chapter [\[ch10\]](./chapters_ch10/_books_ml-q-and-ai-chapters_ch10.md):
Sources of Randomness\
Addresses the various sources of randomness in the training of deep
neural networks that may lead to inconsistent and non-reproducible
results during both training and inference. While randomness can be
accidental, it can also be intentionally introduced by design.

**Part II: Computer Vision** focuses on topics mainly related to deep
learning but specific to computer vision, many of which cover
convolutional neural networks and vision transformers.

Chapter [\[ch11\]](./chapters_ch11/_books_ml-q-and-ai-chapters_ch11.md):
Calculating the Number of Parameters\
Explains the\
procedure for determining the parameters in a convolutional neural
network, which is useful for gauging a model's storage and memory\
requirements.

Chapter [\[ch12\]](./chapters_ch12/_books_ml-q-and-ai-chapters_ch12.md):
Fully Connected and Convolutional Layers\
Illustrates the circumstances in which convolutional layers can
seamlessly replace fully connected layers, which can be useful for
hardware optimization or simplifying implementations.

Chapter [\[ch13\]](./chapters_ch13/_books_ml-q-and-ai-chapters_ch13.md):
Large Training Sets for Vision Transformers\
Probes the rationale behind vision transformers requiring more extensive
training sets compared to conventional convolutional neural networks.

**Part III: Natural Language Processing** covers topics around working
with text, many of which are related to transformer architectures and
self-attention.

Chapter [\[ch14\]](./chapters_ch14/_books_ml-q-and-ai-chapters_ch14.md): The
Distributional Hypothesis\
Delves into the distributional hypothesis, a linguistic theory
suggesting that words appearing in the same contexts tend to possess
similar meanings, which has useful implications for training machine
learning models.

Chapter [\[ch15\]](./chapters_ch15/_books_ml-q-and-ai-chapters_ch15.md):
Data Augmentation for Text\
Highlights the significance of data augmentation for text, a technique
used to artificially increase dataset sizes, which can help with
improving model performance.

Chapter [\[ch16\]](./chapters_ch16/_books_ml-q-and-ai-chapters_ch16.md):
Self-Attention\
Introduces self-attention, a mechanism allowing each segment of a neural
network's input to refer to other parts. Self-attention is a key
mechanism in modern large language models.

Chapter [\[ch17\]](./chapters_ch17/_books_ml-q-and-ai-chapters_ch17.md):
Encoder- and Decoder-Style Transformers\
Describes\
the nuances of encoder and decoder transformer architectures and
explains which type of architecture is most useful for each language
processing task.

Chapter [\[ch18\]](./chapters_ch18/_books_ml-q-and-ai-chapters_ch18.md):
Using and Fine-Tuning Pretrained Transformers\
Explains different methods for fine-tuning pretrained large language
models and discusses their strengths and weaknesses.

Chapter [\[ch19\]](./chapters_ch19/_books_ml-q-and-ai-chapters_ch19.md):
Evaluating Generative Large Language Models\
Lists pro-\
 minent evaluation metrics for language models like Perplexity, BLEU,
ROUGE, and BERTScore.

**Part IV: Production and Deployment** covers questions pertaining to
practical scenarios, such as increasing inference speeds and various
types of distribution shifts.

Chapter [\[ch20\]](./chapters_ch20/_books_ml-q-and-ai-chapters_ch20.md):
Stateless and Stateful Training\
Distinguishes between stateless and stateful training methodologies used
in deploying models.

Chapter [\[ch21\]](./chapters_ch21/_books_ml-q-and-ai-chapters_ch21.md):
Data-Centric AI\
Explores data-centric AI, which priori-\
 tizes refining datasets to enhance model performance. This approach
contrasts with the conventional model-centric approach, which emphasizes
improving model architectures or methods.

Chapter [\[ch22\]](./chapters_ch22/_books_ml-q-and-ai-chapters_ch22.md):
Speeding Up Inference\
Introduces techniques to enhance the speed of model inference without
tweaking the model's architecture or compromising accuracy.

Chapter [\[ch23\]](./chapters_ch23/_books_ml-q-and-ai-chapters_ch23.md):
Data Distribution Shifts\
Post-deployment, AI models\
may face discrepancies between training data and real-world data
distributions, known as data distribution shifts. These shifts can
deteriorate model performance. This chapter categorizes and elaborates
on common shifts like covariate shift, concept drift, label shift, and
do-\
 main shift.

**Part V: Predictive Performance and Model Evaluation** dives deeper
into various aspects of squeezing out predictive performance, such as
changing the loss function, setting up *k*-fold cross-validation, and
dealing with limited labeled data.

Chapter [\[ch24\]](./chapters_ch24/_books_ml-q-and-ai-chapters_ch24.md):
Poisson and Ordinal Regression\
Highlights the differences between Poisson and ordinal regression.
Poisson regression is suitable for count data that follows a Poisson
distribution, like the number of colds contracted on an airplane. In
contrast, ordinal regression caters to ordered categorical data without
assuming equidistant categories, such as disease severity.

Chapter [\[ch25\]](./chapters_ch25/_books_ml-q-and-ai-chapters_ch25.md):
Confidence Intervals\
Delves into methods for constructing confidence intervals for machine
learning classifiers. Reviews the purpose of confidence intervals,
discusses how they estimate unknown population parameters, and
introduces techniques such as normal approximation intervals,
bootstrapping, and retraining with various random seeds.

Chapter [\[ch26\]](./chapters_ch26/_books_ml-q-and-ai-chapters_ch26.md):
Confidence Intervals vs. Conformal Predictions\
Discusses the distinction between confidence intervals and conformal
predictions and describes the latter as a tool for creating prediction
intervals that cover actual outcomes with specific probability.

Chapter [\[ch27\]](./chapters_ch27/_books_ml-q-and-ai-chapters_ch27.md):
Proper Metrics\
Focuses on the essential properties of a proper metric in mathematics
and computer science. Examines whether commonly used loss functions in
machine learning, such as mean squared error and cross-entropy loss,
satisfy these properties.

Chapter [\[ch28\]](./chapters_ch28/_books_ml-q-and-ai-chapters_ch28.md): The
*k* in *k*-Fold Cross-Validation\
Explores the role of the *k* in *k*-fold cross-validation and provides
insight into the advantages and disadvantages of selecting a large *k*.

Chapter [\[ch29\]](./chapters_ch29/_books_ml-q-and-ai-chapters_ch29.md):
Training and Test Set Discordance\
Addresses the scenario where a model performs better on a test dataset
than the training dataset. Offers strategies to discover and address
discrepancies\
 between training and test datasets, introducing the concept of
adversarial validation.

Chapter [\[ch30\]](./chapters_ch30/_books_ml-q-and-ai-chapters_ch30.md):
Limited Labeled Data\
Introduces various techniques to enhance model performance in situations
where data is limited. Covers data labeling, bootstrapping, and
paradigms such as transfer learning, active learning, and multimodal
learning.

## Online Resources [](#online-resources)

I've provided optional supplementary materials on GitHub with code
examples for certain chapters to enhance your learning experience (see
<https://github.com/rasbt/MachineLearning-QandAI-book>). These materials
are designed as practical extensions and deep dives into topics covered
in the book. You can use them alongside each chapter or explore them
after reading to solidify and expand your knowledge.

Without further ado, let's dive in.

\

------------------------------------------------------------------------

