# 章节英中双语与标题范围行（约定）

本仓库章节 Markdown（`docs/ch*/`、`docs/introduction/`）采用以下版式，与试点章（ch01、introduction）一致。

## 标题范围行

- 在 `#` / `##` / `###` 标题行与下一行 `[](#…)` 锚点之间插入**一行**以 `>` 开头的 blockquote。
- 内容为一句话概括本节讨论范围（非机械翻译标题）；可保留必要英文术语。

## 正文双语

- 每个英文叙述单元（通常为一个空行分隔的段落；列表可在整组后以一段中文概括）之后，紧跟一段**普通段落**中文译文（**不用** `>`）。
- 不修改任何既有以 `>` 开头的批注（含 `Tips:`、练习答案中的嵌套 `>`）。

## 术语（跨章尽量统一）

| English | 中文（文中可括号保留英文） |
| --- | --- |
| embedding | 嵌入 / 嵌入向量 |
| latent space | 潜空间 |
| representation | 表示 |
| self-supervised learning | 自监督学习 |
| fine-tuning | 微调 |
| overfitting | 过拟合 |
| conformal prediction | 一致性预测 / conformal prediction |

## 公式与资源

- 保持现有 `$$` 与行内 `$…$`，译文不拆公式。
- 图片路径、内部链接、`<a id="…">` 与 HTML 图注块勿改。
