







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
I hope youâ€™ll find this book a valuable resource for obtaining new
insights and discovering new techniques you can implement in your work.

## Who Is This Book For? [](#who-is-this-book-for)

Navigating the world of AI and machine learning literature can often
feel like walking a tightrope, with most books positioned at either end:
broad beginnerâ€™s introductions or deeply mathematical treatises. This
book illustrates and discusses important developments in these fields
while staying approachable and not requiring an advanced math or coding
background.

This book is for people with some experience with machine learning who
want to learn new concepts and techniques. Itâ€™s ideal for those who
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

While this book will expose you to new concepts and ideas, itâ€™s not a
math or coding book. You wonâ€™t need to solve any proofs or run any
code while reading. In other words, this book is a perfect travel
companion or something you can read on your favorite reading chair with
your morning coffee or tea.

## How to Read This Book [](#how-to-read-this-book)

Each chapter of this book is designed to be self-contained, offering you
the freedom to jump between topics as you wish. When a concept from one
chapter is explained in more detail in another, Iâ€™ve included chapter
references you can follow to fill in gaps in your understanding.

However, thereâ€™s a strategic sequence to the chapters. For example,\
the early chapter on embeddings sets the stage for later discussions on
self-supervised learning and few-shot learning. For the easiest reading
experience and the most comprehensive grasp of the content, my
recommendation is to approach the book from start to finish.

Each chapter is accompanied by optional exercises for readers who want
to test their understanding, with an answer key located at the end of
the book. In addition, for any papers referenced in a chapter or further
reading on that chapterâ€™s topic, you can find the complete citation
information in that chapterâ€™s â€œReferencesâ€? section.

The book is structured into five main parts centered on the most
important topics in machine learning and AI today.

**Part I: Neural Networks and Deep Learning** covers questions about
deep neural networks and deep learning that are not specific to a
particular subdomain. For example, we discuss alternatives to supervised
learning and techniques for reducing overfitting, which is a common
problem when using machine learning models for real-world problems where
data is limited.

ChapterÂ [\[ch01\]](../ch01){reference="ch01" reference-type="ref"}:
Embeddings, Latent Space, and Representations\
Delves into the distinctions and similarities between embedding vectors,
latent vectors, and representations. Elucidates how these concepts help
encode information in the context of machine learning.

ChapterÂ [\[ch02\]](../ch02){reference="ch02" reference-type="ref"}:
Self-Supervised Learning\
Focuses on self-supervised learning, a method that allows neural
networks to utilize large, unlabeled datasets in a supervised manner.

ChapterÂ [\[ch03\]](../ch03){reference="ch03" reference-type="ref"}:
Few-Shot Learning\
Introduces few-shot learning, a specialized supervised learning
technique tailored for small training datasets.

ChapterÂ [\[ch04\]](../ch04){reference="ch04" reference-type="ref"}: The
Lottery Ticket Hypothesis\
Explores the idea that ran-\
Â domlyinitializedneuralnetworkscontainsmaller,efficient subnetworks.

ChapterÂ [\[ch05\]](../ch05){reference="ch05" reference-type="ref"}:
Reducing Overfitting with Data\
Addresses the challenge of overfitting in machine learning, discussing
strategies centered on data augmentation and the use of unlabeled data
to reduce overfitting.

ChapterÂ [\[ch06\]](../ch06){reference="ch06" reference-type="ref"}:
Reducing Overfitting with Model Modifications\
Extends the conversation on overfitting, focusing on model-related
solutions like regularization, opting for simpler models, and ensemble
techniques.

ChapterÂ [\[ch07\]](../ch07){reference="ch07" reference-type="ref"}:
Multi-GPU Training Paradigms\
Explains various training paradigms for multi-GPU setups to accelerate
model training, including data and model parallelism.

ChapterÂ [\[ch08\]](../ch08){reference="ch08" reference-type="ref"}: The
Success of Transformers\
Explores the popular transformer architecture, highlighting features
like attention mechanisms, parallelization ease, and high parameter
counts.

ChapterÂ [\[ch09\]](../ch09){reference="ch09" reference-type="ref"}:
Generative AI Models\
Provides a comprehensive overview of deep generative models, which are
used to produce various media forms, including images, text, and audio.
Discusses the strengths and weaknesses of each model type.

ChapterÂ [\[ch10\]](../ch10){reference="ch10" reference-type="ref"}:
Sources of Randomness\
Addresses the various sources of randomness in the training of deep
neural networks that may lead to inconsistent and non-reproducible
results during both training and inference. While randomness can be
accidental, it can also be intentionally introduced by design.

**Part II: Computer Vision** focuses on topics mainly related to deep
learning but specific to computer vision, many of which cover
convolutional neural networks and vision transformers.

ChapterÂ [\[ch11\]](../ch11){reference="ch11" reference-type="ref"}:
Calculating the Number of Parameters\
Explains the\
procedure for determining the parameters in a convolutional neural
network, which is useful for gauging a modelâ€™s storage and memory\
requirements.

ChapterÂ [\[ch12\]](../ch12){reference="ch12" reference-type="ref"}:
Fully Connected and Convolutional Layers\
Illustrates the circumstances in which convolutional layers can
seamlessly replace fully connected layers, which can be useful for
hardware optimization or simplifying implementations.

ChapterÂ [\[ch13\]](../ch13){reference="ch13" reference-type="ref"}:
Large Training Sets for Vision Transformers\
Probes the rationale behind vision transformers requiring more extensive
training sets compared to conventional convolutional neural networks.

**Part III: Natural Language Processing** covers topics around working
with text, many of which are related to transformer architectures and
self-attention.

ChapterÂ [\[ch14\]](../ch14){reference="ch14" reference-type="ref"}: The
Distributional Hypothesis\
Delves into the distributional hypothesis, a linguistic theory
suggesting that words appearing in the same contexts tend to possess
similar meanings, which has useful implications for training machine
learning models.

ChapterÂ [\[ch15\]](../ch15){reference="ch15" reference-type="ref"}:
Data Augmentation for Text\
Highlights the significance of data augmentation for text, a technique
used to artificially increase dataset sizes, which can help with
improving model performance.

ChapterÂ [\[ch16\]](../ch16){reference="ch16" reference-type="ref"}:
Self-Attention\
Introduces self-attention, a mechanism allowing each segment of a neural
networkâ€™s input to refer to other parts. Self-attention is a key
mechanism in modern large language models.

ChapterÂ [\[ch17\]](../ch17){reference="ch17" reference-type="ref"}:
Encoder- and Decoder-Style Transformers\
Describes\
the nuances of encoder and decoder transformer architectures and
explains which type of architecture is most useful for each language
processing task.

ChapterÂ [\[ch18\]](../ch18){reference="ch18" reference-type="ref"}:
Using and Fine-Tuning Pretrained Transformers\
Explains different methods for fine-tuning pretrained large language
models and discusses their strengths and weaknesses.

ChapterÂ [\[ch19\]](../ch19){reference="ch19" reference-type="ref"}:
Evaluating Generative Large Language Models\
ListsÂ pro-\
Â minent evaluation metrics for language models like Perplexity, BLEU,
ROUGE, and BERTScore.

**Part IV: Production and Deployment** covers questions pertaining to
practical scenarios, such as increasing inference speeds and various
types of distribution shifts.

ChapterÂ [\[ch20\]](../ch20){reference="ch20" reference-type="ref"}:
Stateless and Stateful Training\
Distinguishes between stateless and stateful training methodologies used
in deploying models.

ChapterÂ [\[ch21\]](../ch21){reference="ch21" reference-type="ref"}:
Data-Centric AI\
Explores data-centric AI, which priori-\
Â tizes refining datasets to enhance model performance. This approach
contrasts with the conventional model-centric approach, which emphasizes
improving model architectures or methods.

ChapterÂ [\[ch22\]](../ch22){reference="ch22" reference-type="ref"}:
Speeding Up Inference\
Introduces techniques to enhance the speed of model inference without
tweaking the modelâ€™s architecture or compromising accuracy.

ChapterÂ [\[ch23\]](../ch23){reference="ch23" reference-type="ref"}:
Data Distribution Shifts\
Post-deployment, AI models\
may face discrepancies between training data and real-world data
distributions, known as data distribution shifts. These shifts can
deteriorate model performance. This chapter categorizes and elaborates
on common shifts like covariate shift, concept drift, label shift, and
do-\
Â main shift.

**Part V: Predictive Performance and Model Evaluation** dives deeper
into various aspects of squeezing out predictive performance, such as
changing the loss function, setting up *k*-fold cross-validation, and
dealing with limited labeled data.

ChapterÂ [\[ch24\]](../ch24){reference="ch24" reference-type="ref"}:
Poisson and Ordinal Regression\
Highlights the differences between Poisson and ordinal regression.
Poisson regression is suitable for count data that follows a Poisson
distribution, like the number of colds contracted on an airplane. In
contrast, ordinal regression caters to ordered categorical data without
assuming equidistant categories, such as disease severity.

ChapterÂ [\[ch25\]](../ch25){reference="ch25" reference-type="ref"}:
Confidence Intervals\
Delves into methods for constructing confidence intervals for machine
learning classifiers. Reviews the purpose of confidence intervals,
discusses how they estimate unknown population parameters, and
introduces techniques such as normal approximation intervals,
bootstrapping, and retraining with various random seeds.

ChapterÂ [\[ch26\]](../ch26){reference="ch26" reference-type="ref"}:
Confidence Intervals vs. Conformal Predictions\
Discusses the distinction between confidence intervals and conformal
predictions and describes the latter as a tool for creating prediction
intervals that cover actual outcomes with specific probability.

ChapterÂ [\[ch27\]](../ch27){reference="ch27" reference-type="ref"}:
Proper Metrics\
Focuses on the essential properties of a proper metric in mathematics
and computer science. Examines whether commonly used loss functions in
machine learning, such as mean squared error and cross-entropy loss,
satisfy these properties.

ChapterÂ [\[ch28\]](../ch28){reference="ch28" reference-type="ref"}: The
*k* in *k*-Fold Cross-Validation\
Explores the role of the *k* in *k*-fold cross-validation and provides
insight into the advantages and disadvantages of selecting a large *k*.

ChapterÂ [\[ch29\]](../ch29){reference="ch29" reference-type="ref"}:
Training and Test Set Discordance\
Addresses the scenario where a model performs better on a test dataset
than the training dataset. Offers strategies to discover and address
discrepancies\
Â between training and test datasets, introducing the concept of
adversarial validation.

ChapterÂ [\[ch30\]](../ch30){reference="ch30" reference-type="ref"}:
Limited Labeled Data\
Introduces various techniques to enhance model performance in situations
where data is limited. Covers data labeling, bootstrapping, and
paradigms such as transfer learning, active learning, and multimodal
learning.

## Online Resources [](#online-resources)

Iâ€™ve provided optional supplementary materials on GitHub with code
examples for certain chapters to enhance your learning experience (see
<https://github.com/rasbt/MachineLearning-QandAI-book>). These materials
are designed as practical extensions and deep dives into topics covered
in the book. You can use them alongside each chapter or explore them
after reading to solidify and expand your knowledge.

Without further ado, letâ€™s dive in.

\

------------------------------------------------------------------------

