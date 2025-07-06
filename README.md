# Machine-Learning-Q-and-AI
大模型技术30讲（原版），30 Essential Questions and Answers on Machine Learning and AI

## 1.背景

买了一本《大模型技术30讲》，简单阅读了下，要点突出，对于入门、加深关键点理解，很有用。

但是，也存在问题：《大模型技术30讲》印刷质量真的偏差，而且大部分`术语`，都翻译为中文了，不利于中英对比，特别是 AI 领域基本都是英文的，需要我们熟悉`英文术语`。


因此，准备找到 [原始文档：Machine Learning Q and AI](https://sebastianraschka.com/books/ml-q-and-ai/)，并且，编程将其转存至 github.


## 2.目录

当前工程中，维护的 大模型技术 30 讲，目录如下：

- [Introduction](./docs/introduction/_books_ml-q-and-ai-chapters_introduction.md)

### Part I: Neural Networks and Deep Learning

- [Chapter 1: Embeddings, Latent Space, and Representations](./docs/chapters_ch01/_books_ml-q-and-ai-chapters_ch01.md)
- [Chapter 2: Self-Supervised Learning](./docs/chapters_ch02/_books_ml-q-and-ai-chapters_ch02.md)
- [Chapter 3: Few-Shot Learning](./docs/chapters_ch03/_books_ml-q-and-ai-chapters_ch03.md)
- [Chapter 4: The Lottery Ticket  Hypothesis](./docs/chapters_ch04/_books_ml-q-and-ai-chapters_ch04.md)
- [Chapter 5: Reducing Overfitting with Data](./docs/chapters_ch05/_books_ml-q-and-ai-chapters_ch05.md)
- [Chapter 6: Reducing Overfitting with Model Modifications](./docs/chapters_ch06/_books_ml-q-and-ai-chapters_ch06.md)
- [Chapter 7: Multi-GPU Training Paradigms](./docs/chapters_ch07/_books_ml-q-and-ai-chapters_ch07.md)
- [Chapter 8: The Success of Transformers](./docs/chapters_ch08/_books_ml-q-and-ai-chapters_ch08.md)
- [Chapter 9: Generative AI Models](./docs/chapters_ch09/_books_ml-q-and-ai-chapters_ch09.md)
- [Chapter 10: Sources of Randomness](./docs/chapters_ch10/_books_ml-q-and-ai-chapters_ch10.md)

### Part II: Computer Vision

- [Chapter 11: Calculating the Number of Parameters](./docs/chapters_ch11/_books_ml-q-and-ai-chapters_ch11.md)
- [Chapter 12: Fully Connected and Convolutional Layers](./docs/chapters_ch12/_books_ml-q-and-ai-chapters_ch12.md)
- [Chapter 13: Large Training Sets for Vision Transformers](./docs/chapters_ch13/_books_ml-q-and-ai-chapters_ch13.md)

### Part III: Natural Language Processing

- [Chapter 14: The Distributional Hypothesis](./docs/chapters_ch14/_books_ml-q-and-ai-chapters_ch14.md)
- [Chapter 15: Data Augmentation for Text](./docs/chapters_ch15/_books_ml-q-and-ai-chapters_ch15.md)
- [Chapter 16: Self-Attention](./docs/chapters_ch16/_books_ml-q-and-ai-chapters_ch16.md)
- [Chapter 17: Encoder- and Decoder-Style Transformers](./docs/chapters_ch17/_books_ml-q-and-ai-chapters_ch17.md)
- [Chapter 18: Using and Fine-Tuning Pretrained Transformers](./docs/chapters_ch18/_books_ml-q-and-ai-chapters_ch18.md)
- [Chapter 19: Evaluating Generative Large Language Models](./docs/chapters_ch19/_books_ml-q-and-ai-chapters_ch19.md)

### Part IV: Production and Deployment

- [Chapter 20: Stateless and Stateful Training](./docs/chapters_ch20/_books_ml-q-and-ai-chapters_ch20.md)
- [Chapter 21: Data-Centric AI](./docs/chapters_ch21/_books_ml-q-and-ai-chapters_ch21.md)
- [Chapter 22: Speeding Up Inference](./docs/chapters_ch22/_books_ml-q-and-ai-chapters_ch22.md)
- [Chapter 23: Data Distribution Shifts](./docs/chapters_ch23/_books_ml-q-and-ai-chapters_ch23.md)

### Part V: Predictive Performance and Model Evaluation

- [Chapter 24: Poisson and Ordinal Regression](./docs/chapters_ch24/_books_ml-q-and-ai-chapters_ch24.md)
- [Chapter 25: Confidence Intervals](./docs/chapters_ch25/_books_ml-q-and-ai-chapters_ch25.md)
- [Chapter 26: Confidence Intervals vs. Conformal Predictions](./docs/chapters_ch26/_books_ml-q-and-ai-chapters_ch26.md)
- [Chapter 27: Proper Metrics](./docs/chapters_ch27/_books_ml-q-and-ai-chapters_ch27.md)
- [Chapter 28: The k in k-Fold Cross-Validation](./docs/chapters_ch28/_books_ml-q-and-ai-chapters_ch28.md)
- [Chapter 29: Training and Test Set Discordance](./docs/chapters_ch29/_books_ml-q-and-ai-chapters_ch29.md)
- [Chapter 30: Limited Labeled Data](./docs/chapters_ch30/_books_ml-q-and-ai-chapters_ch30.md)








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

<details><summary>点击展开</summary>

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

#### 例子

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

Markdown 不支持为图片设置原生锚点，但你可以用 HTML 的 `<a name="pic1"></a>` 来“伪造”锚点：

````markdown
<a name="pic1"></a>
![图片描述](图片链接)
````

然后跳转链接：

````markdown
[跳到图片](#pic1)
````

> Tips: pic1 锚点的命名中，不能包含 `:` 符号。

但请注意，**GitHub 有时会忽略或过滤掉部分 HTML 标签**，这种方法在所有场景下不一定可靠，且不如标题锚点规范。

---

- **图片本身不能生成锚点，不能自动跳转到图片。**
- 不推荐用 HTML 锚点，兼容性不好。

</details>

### 附录C. 数学公式

<details><summary>点击展开</summary>

#### 公式标识

是的，GitHub 的 Markdown 文件现在支持数学公式的展示（自 2022 年 3 月起）。你可以在 `.md` 文件中使用 LaTeX 语法，通过以下两种方式书写数学公式：

1. **行内公式**  
   使用单个美元符号包裹，例如：  
   ```
   $E = mc^2$
   ```
   显示为：$E = mc^2$

2. **块级公式**  
   使用两个美元符号包裹，并单独成行，例如：  
   ```
   $$
   E = mc^2
   $$
   ```
   显示为：
   $$
   E = mc^2
   $$

**注意事项**：
- 公式渲染仅在 GitHub 的仓库、issues、pull requests、discussions、README 等页面支持，Gist、Wiki 目前不支持。
- 如果你在本地编辑器预览，可能无法看到公式渲染效果，建议推送到 GitHub 后查看。


#### 角标

在 LaTeX 语法中，**角标**分为两种：

##### 1. 上标（Superscript）

使用 `^` 符号。例如：
```
x^2
```
在行内公式中写作：`$x^2$`  
效果：$x^2$

如果上标内容超过一个字符，需要用大括号包裹：
```
x^{10}
```
效果：$x^{10}$

---

##### 2. 下标（Subscript）

使用 `_` 符号。例如：
```
x_1
```
在行内公式中写作：`$x_1$`  
效果：$x_1$

如果下标内容超过一个字符，也需要用大括号包裹：
```
x_{ij}
```
效果：$x_{ij}$

---

##### 组合例子

```
x_i^2
```
效果：$x_i^2$

```
x_{ij}^{10}
```
效果：$x_{ij}^{10}$

---

**总结：**
- 上标用 `^`，下标用 `_`
- 多字符时用 `{}` 包裹

</details>



### 附录D. 在线文件模板 （Docsify）

<details><summary>点击展开</summary>

参考： [happy-llm](https://github.com/datawhalechina/happy-llm/tree/main)

> 如何使用 Docsify 构造 github pages？

要在 GitHub Pages 上使用 Docsify 搭建文档网站，步骤如下：

#### 1. 新建 GitHub 仓库
- 创建一个新的 GitHub 仓库（比如 yourname/your-repo）。

#### 2. 初始化项目文件
- 在本地或在 GitHub 网页端新建如下文件结构：

```
/
├── docs/
│   ├── index.html
│   └── README.md
└── .gitignore
```

##### index.html 内容如下（这是启动 Docsify 的核心页面）：
```html
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>Docsify 文档</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <!-- 可选：自定义主题 -->
  <link rel="stylesheet" href="//cdn.jsdelivr.net/npm/docsify@4/themes/vue.css" />
</head>
<body>
  <div id="app"></div>
  <script src="//cdn.jsdelivr.net/npm/docsify@4"></script>
</body>
</html>
```

##### README.md（文档主页）
```
# 欢迎使用 Docsify 文档

这是首页内容。
```

#### 3. 推送到 GitHub
- 将这些文件推送到你的仓库，通常建议放在 docs 目录下。

#### 4. 配置 GitHub Pages
- 打开你的仓库，点击 **Settings > Pages**
- Source 选择 `docs` 目录（或根目录，取决于你的 index.html 放在哪里）
- 保存后，GitHub 会分配一个访问地址（如 `https://yourname.github.io/your-repo/`）

### 5. 访问你的 Docsify 网站
- 访问上一步生成的 URL 即可看到 Docsify 驱动的文档网站。

---

### 可选：本地预览
- 安装 docsify-cli：
  ```bash
  npm i docsify-cli -g
  ```
- 在 docs 所在目录运行：
  ```bash
  docsify serve docs
  ```
- 浏览器访问 `http://localhost:3000` 进行本地预览。

---

### 常见问题

- **文档无法显示？**  
  检查 index.html 是否直接引用了 jsdelivr CDN 上的 docsify 脚本，且 index.html 路径正确。
- **自定义导航/侧边栏？**  
  新建 `_sidebar.md` 或 `_navbar.md`，参见 [Docsify 官方文档](https://docsify.js.org/#/zh-cn/)。

---

需要更详细的步骤或遇到具体问题，可以继续追问！

</details>



