# Machine-Learning-Q-and-AI
大模型技术30讲（原版），30 Essential Questions and Answers on Machine Learning and AI

## 1.背景

买了一本《大模型技术30讲》，简单阅读了下，要点突出，对于入门、加深关键点理解，很有用。

但是，也存在问题：《大模型技术30讲》印刷质量真的偏差，而且大部分`术语`，都翻译为中文了，不利于中英对比，特别是 AI 领域基本都是英文的，需要我们熟悉`英文术语`。


因此，准备找到 [原始文档：Machine Learning Q and AI](https://sebastianraschka.com/books/ml-q-and-ai/)，并且，编程将其转存至 github.


## 2.目录

当前工程中，维护的 大模型技术 30 讲，目录如下：

- [Introduction](./pages/_books_ml-q-and-ai-chapters_introduction.md)

### Part I: Neural Networks and Deep Learning

- [Chapter 1: Embeddings, Latent Space, and Representations](./pages/_books_ml-q-and-ai-chapters_ch01.md)
- [Chapter 2: Self-Supervised Learning](./pages/_books_ml-q-and-ai-chapters_ch02.md)
- [Chapter 3: Few-Shot Learning](./pages/_books_ml-q-and-ai-chapters_ch03.md)
- [Chapter 4: The Lottery Ticket  Hypothesis](./pages/_books_ml-q-and-ai-chapters_ch04.md)
- [Chapter 5: Reducing Overfitting with Data](./pages/_books_ml-q-and-ai-chapters_ch05.md)
- [Chapter 6: Reducing Overfitting with Model Modifications](./pages/_books_ml-q-and-ai-chapters_ch06.md)
- [Chapter 7: Multi-GPU Training Paradigms](./pages/_books_ml-q-and-ai-chapters_ch07.md)
- [Chapter 8: The Success of Transformers](./pages/_books_ml-q-and-ai-chapters_ch08.md)
- [Chapter 9: Generative AI Models](./pages/_books_ml-q-and-ai-chapters_ch09.md)
- [Chapter 10: Sources of Randomness](./pages/_books_ml-q-and-ai-chapters_ch10.md)

### Part II: Computer Vision

- [Chapter 11: Calculating the Number of Parameters](./pages/_books_ml-q-and-ai-chapters_ch11.md)
- [Chapter 12: Fully Connected and Convolutional Layers](./pages/_books_ml-q-and-ai-chapters_ch12.md)
- [Chapter 13: Large Training Sets for Vision Transformers](./pages/_books_ml-q-and-ai-chapters_ch13.md)

### Part III: Natural Language Processing

- [Chapter 14: The Distributional Hypothesis](./pages/_books_ml-q-and-ai-chapters_ch14.md)
- [Chapter 15: Data Augmentation for Text](./pages/_books_ml-q-and-ai-chapters_ch15.md)
- [Chapter 16: Self-Attention](./pages/_books_ml-q-and-ai-chapters_ch16.md)
- [Chapter 17: Encoder- and Decoder-Style Transformers](./pages/_books_ml-q-and-ai-chapters_ch17.md)
- [Chapter 18: Using and Fine-Tuning Pretrained Transformers](./pages/_books_ml-q-and-ai-chapters_ch18.md)
- [Chapter 19: Evaluating Generative Large Language Models](./pages/_books_ml-q-and-ai-chapters_ch19.md)

### Part IV: Production and Deployment

- [Chapter 20: Stateless and Stateful Training](./pages/_books_ml-q-and-ai-chapters_ch20.md)
- [Chapter 21: Data-Centric AI](./pages/_books_ml-q-and-ai-chapters_ch21.md)
- [Chapter 22: Speeding Up Inference](./pages/_books_ml-q-and-ai-chapters_ch22.md)
- [Chapter 23: Data Distribution Shifts](./pages/_books_ml-q-and-ai-chapters_ch23.md)

### Part V: Predictive Performance and Model Evaluation

- [Chapter 24: Poisson and Ordinal Regression](./pages/_books_ml-q-and-ai-chapters_ch24.md)
- [Chapter 25: Confidence Intervals](./pages/_books_ml-q-and-ai-chapters_ch25.md)
- [Chapter 26: Confidence Intervals vs. Conformal Predictions](./pages/_books_ml-q-and-ai-chapters_ch26.md)
- [Chapter 27: Proper Metrics](./pages/_books_ml-q-and-ai-chapters_ch27.md)
- [Chapter 28: The k in k-Fold Cross-Validation](./pages/_books_ml-q-and-ai-chapters_ch28.md)
- [Chapter 29: Training and Test Set Discordance](./pages/_books_ml-q-and-ai-chapters_ch29.md)
- [Chapter 30: Limited Labeled Data](./pages/_books_ml-q-and-ai-chapters_ch30.md)








## 附录

### 附录A. 将网页内容，转录为 md，任务拆解

几个方面：

1. 范围：目录 和 目录对应的每一篇文章
2. 格式：网页内容，全部转换为 md 格式存储，并且 `md 文件命名` 跟网页保持一致
3. 图片：所有图片，都存储在 `images` 下
4. 删除部分无效内容
5. 调整格式：部分乱码、锚点

具体，对应 4 个脚本： 按照顺序执行.

* web_crawler.py 
* remove_header.py
* remove_print_book.py
* remove_after_separator.py   


### 附录B. GitHub 上锚点

在 GitHub 的 Markdown（.md）文件中，**可以使用锚点（Anchor）来实现文档内部的跳转**。锚点通常用于目录（Table of Contents）跳转到文档的某个标题位置。

#### 如何设置锚点？

GitHub 会自动为每个标题（#、##、### 等）生成一个锚点链接，格式如下：

1. **写标题**  
   例如：
   ```markdown
   ## 安装方法
   ```

2. **生成的锚点格式**  
   - 将标题全部转为小写
   - 移除标点
   - 用连字符（-）替换空格
   - 中文标题也适用，但空格会变成连字符

   例如，`## 安装方法` 的锚点就是 `#安装方法`。

3. **添加目录或内部跳转链接**  
   使用标准的 markdown 链接语法：

   ```markdown
   [安装方法](#安装方法)
   ```

   英文标题如 `## How To Install`，锚点是 `#how-to-install`：

   ```markdown
   [How To Install](#how-to-install)
   ```

#### 复杂标题的锚点规则

- 多个连续空格只变成一个连字符
- 中文/英文、数字、下划线、特殊字符、标点符号等都要去除或按规则处理
- 可以右键标题旁的小链条图标，复制锚点链接

### 例子

````markdown
## 目录
- [安装方法](#安装方法)
- [使用说明](#使用说明)

## 安装方法
内容...

## 使用说明
内容...
````

#### 手动自定义锚点？

**GitHub 不支持 HTML 的 `<a name="foo"></a>` 锚点写法**。只能用标题自动生成锚点。

---

**总结：**  
- 直接用 markdown 的标题，GitHub 自动生成锚点
- 用 `[描述](#锚点名)` 进行跳转
- 不支持 HTML `<a name="">` 自定义锚点


#### 图片锚点（需要手动适配）


在 GitHub 的 Markdown 文件中，**图片本身不能直接作为锚点**，也就是说，不能直接点击一个图片让页面跳转到该图片所在的位置。但可以通过以下方法实现“跳转到图片”或“让图片参与跳转”：

 方法三：利用 HTML 标签（不推荐）

Markdown 不支持为图片设置原生锚点，但你可以用 HTML 的 `<a id="pic1"></a>` 来“伪造”锚点：

````markdown
<a id="pic1"></a>
![图片描述](图片链接)
````

然后跳转链接：

````markdown
[跳到图片](#pic1)
````

但请注意，**GitHub 有时会忽略或过滤掉部分 HTML 标签**，这种方法在所有场景下不一定可靠，且不如标题锚点规范。

---

- **图片本身不能生成锚点，不能自动跳转到图片。**
- 不推荐用 HTML 锚点，兼容性不好。


