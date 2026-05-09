



# Chapter 14: The Distributional Hypothesis
> 分布假设：共现上下文相似的词往往语义相近；本章说明其在 Word2vec、BERT、GPT 中的角色与局限。
[](#chapter-14-the-distributional-hypothesis)



**What is the `distributional hypothesis` in natural language processing
(NLP)? Where is it used, and how far does it hold true?**

自然语言处理（NLP）中的「分布假设」是什么？用在哪里？在多大程度上成立？

The `distributional hypothesis` is a linguistic theory suggesting that
words occurring in the same contexts tend to have similar meanings,
according to the original source, "Distributional Structure"? by
Zellig S. Harris. Succinctly, the more similar the meanings of two words
are, the more often they appear in similar contexts.

分布假设是语言学理论：同一上下文里出现的词往往含义相近，可追溯至 Zellig S. Harris 的 "Distributional Structure"。简言之，两词含义越接近，越常出现在相似上下文中。

> Tips: **分布假设**（distributional hypothesis），也称为分布语义学（distributional semantics），用于描述单词在上下文中的分布模式。它认为，在`相似的上下文`中出现的`单词`往往具有`相似的含义`。

Consider the sentence in
Figure [14.1](#fig-ch14-fig01), for example. The words *cats* and *dogs* often
occur in similar contexts, and we could replace *cats* with *dogs*
without making the sentence sound awkward. We could also replace *cats*
with *hamsters*, since both are mammals and pets, and the sentence would
still sound plausible. However, replacing *cats* with an unrelated word
such as *sandwiches* would render the sentence clearly wrong, and
replacing *cats* with the unrelated word *driving* would also make the
sentence grammatically incorrect.

例如图 [14.1](#fig-ch14-fig01) 中的句子：*cats* 与 *dogs* 常出现在相似上下文，互换后仍自然；换作 *hamsters* 也可行；换成 *sandwiches* 则语义荒谬，换成 *driving* 则语法不通。

> Tips: 图1.1中的句子。*cats*和*dogs*经常出现在相似的上下文中，我们可以将*cats*替换为*dogs*，而不会让句子听起来奇怪。
> - 我们也可以将*cats*替换为*hamsters*，因为它们都是哺乳动物和宠物，句子听起来仍然合理。
> - 但是，如果将*cats*替换为不相关的单词*sandwiches*，句子会变得明显错误，
> - 如果将*cats*替换为不相关的单词*driving*，句子也会变得语法错误。

<a id="fig-ch14-fig01"></a>

<div align="center">
  <img src="./images/ch14-fig01.png" alt="Common and uncommonwordsinagivencontext" width="78%" />
  <div><b>Figure 14.1</b></div>
</div>

It is easy to construct counterexamples using polysemous words, that is,
words that have multiple meanings that are related but not identical.
For example, consider the word *bank*. As a noun, it can refer to a
financial institution, the "rising ground bordering a river,"? the
"steep incline of a hill,"? or a "protective cushioning rim"?
(according to the Merriam-Webster dictionary). It can even be a verb: to
bank on something means to rely or depend on it. These different
meanings have different distributional properties and may not always
occur in similar contexts.

用多义词很容易构造反例：即多种相关但不完全相同的含义。例如 *bank* 可作名词指金融机构、河岸、陡坡或边缘护垫（据 Merriam-Webster），也可作动词表示指望、依赖；不同义项的分布性质不同，未必总出现在相似上下文中。

Nonetheless, the distributional hypothesis is quite useful. Word embeddings 
(introduced in Chapter [\[ch01\]](./ch01/_books_ml-q-and-ai-ch01.md)) such as Word2vec,
 as well as many large language transformer models, rely on this idea. 
 This includes the masked language model in BERT and the next-word pretraining task used in GPT.

尽管如此分布假设仍很有用：第 [\[ch01\]](./ch01/_books_ml-q-and-ai-ch01.md) 章介绍的 Word2vec 等词嵌入，以及许多大型语言 Transformer 都依赖这一思想，包括 BERT 的掩码语言建模与 GPT 的下一词预训练任务。

> Tips: 尽管存在反例，**分布假设**在实际应用中非常有用。
> - `Word2vec`等词嵌入（word embeddings）模型以及许多`大型语言模型`（large language models）都基于这个概念。
> - 这包括`BERT`中的`掩码`语言模型，和`GPT`中的`下一个词`预训练任务。

## Word2vec, BERT, and GPT
> Word2vec、BERT、GPT 三种建模范式如何用自监督目标把「共现」变成嵌入空间中的近邻关系。
[](#word2vec-bert-and-gpt)

The `Word2vec` approach uses a simple, two-layer neuralnetwork to encode
words into embedding vectors such that the embedding vectors of similar
words are both semantically and syntactically close. There are two ways
to train a Word2vec model: the `continuous bag-of-words` (CBOW) approach
and the `skip-gram` approach. When using CBOW, the Word2vec model learns to
predict the current words by using the surrounding context words.
Conversely, in the skip-gram model, Word2vec predicts the context words
from a selected word. While skip-gram is more effective for infrequent
words, CBOW is usually faster to train.

Word2vec 用浅层网络把词编码为嵌入，使相近词在向量空间中邻近；训练方式有 CBOW（用上下文预测中心词）与 skip-gram（由中心词预测上下文）；后者对罕见词往往更好，前者通常训练更快。

> Tips:  FIXME 跳字模型 没理解？？？
> - `Word2vec`是一种使用简单两层神经网络将单词编码为嵌入向量的方法。
> - 有两种训练Word2vec模型的方法：`连续词袋`（CBOW）方法和`跳字`（skip-gram）方法。
> - 当使用CBOW时，Word2vec模型学习通过使用周围上下文单词来预测当前单词。
> - 相反，在跳字模型中，Word2vec从选定的单词预测上下文单词。
> - 尽管跳字模型对于不常见的单词更有效，但CBOW通常训练速度更快。

After training, word embeddings are placed within the vector space so
that words with common contexts in the corpus--that is, words with
semantic and syntactic similarities--are positioned close to each
other, as illustrated in
Figure [14.2](#fig-ch14-fig02). Conversely, dissimilar words are located farther
apart in the embedding space.

训练完成后，语料中共现多、语义与句法相近的词在向量空间中彼此靠近，反之则更远，见图 [14.2](#fig-ch14-fig02)。

<a id="fig-ch14-fig02"></a>

<div align="center">
  <img src="./images/ch14-fig02.png" alt="Word2vec embeddings in a two-dimensional\vector space" width="78%" />
  <div><b>Figure 14.2</b></div>
</div>

BERT is an LLM based on the transformer architecture (see
Chapter [\[ch08\]](./ch08/_books_ml-q-and-ai-ch08.md))
that uses a masked language modeling approach that involves masking
(hiding) some of the words in a sentence. Its task is to predict these
masked words based on the other words in the sequence, as illustrated in
Figure [14.3](#fig-ch14-fig03). This is a form of the self-supervised learning
used to pretrain LLMs (see Chapter [\[ch02\]](./ch02/_books_ml-q-and-ai-ch02.md) for more on self-supervised learning). The
pretrained model produces embeddings in which similar words (or tokens)
are close in the embedding space.

BERT 是基于 Transformer（第 [\[ch08\]](./ch08/_books_ml-q-and-ai-ch08.md) 章）的大语言模型，用掩码语言建模：遮住句中部分词并据其余词预测被掩词，见图 [14.3](#fig-ch14-fig03)。这是预训练 LLM 常用的自监督形式（详见第 [\[ch02\]](./ch02/_books_ml-q-and-ai-ch02.md) 章）。预训练后相似词或 token 在嵌入空间中彼此接近。

<a id="fig-ch14-fig03"></a>

<div align="center">
  <img src="./images/ch14-fig03.png" alt="BERT's pretraining task involves predicting randomly masked words." width="78%" />
  <div><b>Figure 14.3</b></div>
</div>

GPT, which like BERT is also an LLM based on the transformer
architecture, functions as a decoder. Decoder-style models like GPT
learn to predict subsequent words in a sequence based on the preceding
ones, as illustrated in
Figure [14.4](#fig-ch14-fig04). GPT contrasts with BERT, an encoder model, as it
emphasizes predicting what follows rather than encoding the entire
sequence simultaneously.

GPT 与 BERT 同为基于 Transformer 的 LLM，但作解码器用：类 GPT 的解码器风格模型根据前文预测后续词，见图 [14.4](#fig-ch14-fig04)。与作为编码器的 BERT 不同，GPT 强调「往后预测」而非同时对整句双向编码。

<a id="fig-ch14-fig04"></a>

<div align="center">
  <img src="./images/ch14-fig04.png" alt="GPT is pretrained by predicting the next word." width="78%" />
  <div><b>Figure 14.4</b></div>
</div>

Where BERT is a bidirectional language model that considers the whole
input sequence, GPT only strictly parses previous sequence elements.
This means BERT is usually better suited for classification tasks,
whereas GPT is more suited for text generation tasks. Similar to BERT,
GPT produces high-quality contextualized word embeddings that capture
semantic similarity.

BERT 双向建模整段输入；GPT 严格依赖已生成的前缀，故前者更宜分类等任务，后者更宜生成；二者都能得到捕捉语义相似性的高质量上下文化嵌入。

> Tips: 
> - `BERT`是一种 **双向** 语言模型，考虑 **整个输入序列** 。
> - `GPT`只严格解析 **前一个序列元素** 。
> - 这意味着`BERT`通常更适合分类任务，而`GPT`更适合文本生成任务。
> - 与`BERT`类似，`GPT`产生高质量的 **上下文化单词嵌入** ，捕捉语义相似性。

## Does the Hypothesis Hold?
> 大数据上分布假设大体可用，支撑嵌入与语义技术及下游 NLP；反例存在但不妨碍其作为现代语言模型的基石。
[](#does-the-hypothesis-hold)

For large datasets, the distributional hypothesis more or less holds
true, making it quite useful for understanding and modeling language
patterns, word relationships, and semantic meanings. For example, this
concept enables techniques like word embedding and semantic analysis,
which, in turn, facilitate natural language processing tasks such as
text classification, sentiment analysis, and machine translation.

对大规模数据，分布假设大体成立，有助于建模语言模式、词关系与语义，并支撑词嵌入、语义分析及分类、情感分析、机器翻译等任务。

> Tips: 
> - 对于大型数据集，**分布假设**或多或少是正确的，对于理解语言模式、单词关系和语义含义非常有用。
> - 例如，这个概念启用了像`词嵌入`和`语义分析`这样的技术，这些技术反过来又促进了自然语言处理任务，如`文本分类`、`情感分析`和`机器翻译`。

In conclusion, while there are counterexamples in which the
distributional hypothesis does not hold, it is a very useful concept
that forms the cornerstone of modern language transformer models.

总之，尽管存在分布假设不成立的反例，它仍是极有用的概念，构成现代语言 Transformer 模型的基石之一。

## Exercises
> 习题：同音异义词与「类似分布假设」的其他模态。
[](#exercises)

14-1. Does the distributional hypothesis hold true in the case of
homophones, or words that sound the same but have different meanings,
such as *there* and *their*?

习题 14-1：对同音异义词（如 *there* 与 *their*），分布假设是否仍成立？

14-2. Can you think of another domain where a concept similar to the
distributional hypothesis applies? (Hint: think of other input
modalities for neural networks.)

习题 14-2：能否想到另一个领域也存在与分布假设类似的概念？（提示：神经网络的其他输入模态。）

## References
> 本节列分布假设原文、Word2vec、BERT、GPT、上下文表示可迁移性与语言模型即知识库等相关文献链接。
[](#references)

- The original source describing the distributional hypothesis:
  Zellig S. Harris, "Distributional Structure"? (1954),
  [*https://doi.org/10.1080/*](https://doi.org/10.1080/00437956.1954.11659520)
  [*00437956.1954.11659520*](https://doi.org/10.1080/00437956.1954.11659520).

Harris 阐述分布假设的经典文献（1954），DOI 链接见上。

- The paper introducing the Word2vec model: Tomas Mikolov et al.,
  "Efficient Estimation of Word Representations in Vector Space"?
  (2013), <https://arxiv.org/abs/1301.3781>.

Mikolov 等提出 Word2vec 的论文（2013）。

- The paper introducing the BERT model: Jacob Devlin et al., "BERT:
  Pre-training of Deep Bidirectional Transformers for Language
  Understanding"? (2018), <https://arxiv.org/abs/1810.04805>.

Devlin 等 BERT 论文（2018）。

- The paper introducing the GPT model: Alec Radford and Karthik
  Narasimhan, "Improving Language Understanding by Generative
  Pre-Training"? (2018),
  [*https://www.semanticscholar.org/paper/Improving*](https://www.semanticscholar.org/paper/Improving-Language-Understanding-by-Generative-Radford-Narasimhan/cd18800a0fe0b668a1cc19f2ec95b5003d0a5035)
  [*-Language-Understanding-by-Generative-Radford-Narasimhan/cd18800a0*](https://www.semanticscholar.org/paper/Improving-Language-Understanding-by-Generative-Radford-Narasimhan/cd18800a0fe0b668a1cc19f2ec95b5003d0a5035)
  [*fe0b668a1cc19f2ec95b5003d0a5035*](https://www.semanticscholar.org/paper/Improving-Language-Understanding-by-Generative-Radford-Narasimhan/cd18800a0fe0b668a1cc19f2ec95b5003d0a5035).

Radford、Narasimhan 关于生成式预训练改进语言理解的论文（2018）。

- BERT produces embeddings in which similar words (or tokens) are close
  in the embedding space: Nelson F. Liu et al., "Linguistic Knowledge
  and Transferability of Contextual Representations"? (2019),
  <https://arxiv.org/abs/1903.08855>.

Liu 等讨论语言学知识与上下文表示可迁移性（2019）。

- The paper showing that GPT produces high-quality contextualized word
  embeddings that capture semantic similarity: Fabio Petroni et al.,
  "Language Models as Knowledge Bases?"? (2019),
  <https://arxiv.org/abs/1909.01066>.

Petroni 等讨论语言模型作为知识库（2019）。


------------------------------------------------------------------------

