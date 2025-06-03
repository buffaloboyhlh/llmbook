# 第一章： 特征工程

特征工程（Feature Engineering）是指将原始数据转化为更适合机器学习模型使用的特征集的过程。它在机器学习中非常重要，因为好的特征往往比复杂的模型更能提升效果。

换句话说：特征工程就是“**从数据中提炼出最能描述问题的特征**”。

**特征工程**，顾名思义，是对原始数据进行一系列工程处理，将其提炼为特征，作为输入供算法和模型使用。从本质上来讲，特征工程是一个表示和展现数据的过程。在实际工作中，特征工程旨在去除原始数据中的杂志和冗余，
设计更高效的特征以刻画求解的问题与预测模型之间的关系。

----

## 1.1 特征归一化

### 1️⃣ 什么是特征归一化？

特征归一化（Feature Normalization），也叫特征缩放（Feature Scaling），是把不同量纲（取值范围、分布）的特征，统一到相似的数值范围，以便模型更好地收敛和训练。

### 2️⃣ 为什么要归一化？

因为不同特征可能有不同的量纲、取值范围，直接放到模型里可能会导致：

+ 🚀 梯度更新时数值不平衡，影响模型收敛速度
+ 🚀 权重对不同特征的影响程度不同（偏向值大的特征）
+ 🚀 距离度量模型（如KNN、聚类）直接受数值大小影响

简单例子：

+ 特征 A：工资（单位：元），范围（2000, 10000）
+ 特征 B：年龄（单位：岁），范围（18, 60）
  如果不归一化，模型会偏向工资，因为数值大。

### 3️⃣ 常用的归一化方法

#### 📌 (1) Min-Max 归一化（最大最小缩放）

$$
x’ = \frac{x - \min(x)}{\max(x) - \min(x)}
$$

将特征压缩到 [0, 1] 之间。

+ **适合**：数据分布接近均匀，且没有明显异常值。
+ **优点**：保留原始数据分布。
+ **缺点**：对异常值敏感。

#### 📌 (2) Z-Score 标准化（零均值单位方差）

$$
x’ = \frac{x - \mu}{\sigma}
$$

其中：

+ $\mu$ 是均值
+ $\sigma$ 是标准差

转换后的数据均值 0，标准差 1。

优点：适合大多数机器学习模型，对异常值有一定鲁棒性。

常用于：SVM、线性回归、神经网络等。

#### 📌 (3) L2 归一化（向量模归一化）

公式（对样本向量 $\mathbf{x}=[x_1, x_2, …, x_n]$）：

$$
\mathbf{x}’ = \frac{\mathbf{x}}{\|\mathbf{x}\|_2}
$$

将样本向量的欧氏模长缩放为 1，适合距离度量（如文本余弦相似度）。

#### 📌 (4) Robust Scaler（中位数缩放）

对中位数和 IQR（四分位数间距）归一化：

$$
x’ = \frac{x - \text{median}(x)}{\text{IQR}}
$$

优点：对异常值鲁棒性强。
适合数据有异常值的场景。

## 1.2 类别型特征 

###  什么是类别型特征？

类别型特征是指 取值是离散的、通常代表类别而不是数值大小关系的特征。

🔹 例如：

+ 颜色（红、黄、蓝）
+  性别（男、女）
+ 地区（北京、上海、广州）

类别型特征通常是：

+ 无序类别（Nominal）：如颜色、性别，类别之间没有大小关系。
+ 有序类别（Ordinal）：如教育程度（小学、初中、高中、本科、硕士），类别有顺序，但仍是离散值。

### 1️⃣ 序号编码（Label Encoding）

将每个类别 映射为唯一的整数。

```text
颜色: red, green, blue
Label Encoding: 0, 1, 2
```
在 Python 中可以用：
```python
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
labels = encoder.fit_transform(['red', 'green', 'blue'])
print(labels)  # 输出: [2, 1, 0]
```

###  2️⃣ 独热编码（One-Hot Encoding）

对每个类别创建一个二进制特征，表示是否属于该类别。

```text
颜色: red, green, blue
One-Hot Encoding:
red   -> [1, 0, 0]
green -> [0, 1, 0]
blue  -> [0, 0, 1]
```

```python
import pandas as pd

df = pd.DataFrame({'color': ['red', 'green', 'blue']})
df_onehot = pd.get_dummies(df, columns=['color'])
print(df_onehot)
```

### 3️⃣ 二进制编码（Binary Encoding）

先将类别编号（Label Encoding），然后把这些数字用 二进制表示，每一位二进制位作为一个新特征。

```text
类别: A, B, C, D
Label 编码: 0, 1, 2, 3
二进制表示:
0 -> 00
1 -> 01
2 -> 10
3 -> 11
```
转换结果：

```text
A -> [0, 0]
B -> [0, 1]
C -> [1, 0]
D -> [1, 1]
```

在 Python 中可以用：

```python
!pip install category_encoders

import pandas as pd
import category_encoders as ce

df = pd.DataFrame({'color': ['red', 'green', 'blue']})
encoder = ce.BinaryEncoder(cols=['color'])
df_bin = encoder.fit_transform(df)
print(df_bin)
```

## 1.3 文本表示模型

###  1️⃣ 什么是文本表示模型？

**文本表示模型的目标：**
把「文本」转换成「计算机能处理的向量形式（特征向量）」，为后续的任务（分类、检索、生成等）提供输入。

###  2️⃣ 文本表示模型发展脉络

文本表示技术经历了以下发展阶段：

+ 1️⃣ 词袋模型（Bag of Words, BOW）
+ 2️⃣ TF-IDF
+ 3️⃣ 分布式词向量（Word2Vec、GloVe、FastText）
+ 4️⃣ 上下文相关词向量（ELMo、ULMFiT）
+ 5️⃣ 上下文相关句向量（BERT、RoBERTa、GPT 等 Transformer 系列）

###  3️⃣ 各阶段模型详解

####  📌 (1) 词袋模型（BOW）

词袋模型是一种最基本、最经典的文本表示方法。它的核心思想就是：
> 把文本看作是一个词汇集合（忽略词序），通过统计每个词出现的频率，形成一个向量表示。

通俗来说：

+ 文本 = 一袋词
+ 不在乎顺序、只在乎词出现没出现、出现了多少次

假设有一个 词汇表（Vocabulary），包含所有出现过的词。
每个文本，就用一个向量来表示，向量的长度 = 词汇表的大小。

**语料：**

```text
文本1：我 爱 吃 苹果
文本2：我 不 爱 吃 香蕉
```

**构建词汇表（出现过的词）：**

```text
['我', '爱', '吃', '苹果', '不', '香蕉']
```

**每个文本的向量表示：**

```text
文本1: [1, 1, 1, 1, 0, 0]
文本2: [1, 1, 1, 0, 1, 1]
```
每个数字表示对应词在文本中出现的次数。

#### 📌 (2) TF-IDF

TF-IDF（Term Frequency - Inverse Document Frequency）是一种衡量词语在文档中重要性的加权方法。
它是在 BOW 模型的基础上，引入“逆文档频率”，让常见词（例如“的”、“是”等）权重更低，突显出真正有区分力的词。

#####  TF-IDF 公式

TF-IDF 由两个部分组成：

###### 📌 (1) 词频 TF（Term Frequency）

表示词在文档中的出现频率：

$$
\text{TF}(t, d) = \frac{\text{词 } t \text{ 在文档 } d \text{ 中出现的次数}}{\text{文档 } d \text{ 中的总词数}}
$$

###### 📌 (2) 逆文档频率 IDF（Inverse Document Frequency）

表示词在整个语料库中的重要性（全局信息）：

$$
\text{IDF}(t) = \log \frac{N}{\text{DF}(t)}
$$

其中：

+ N：文档总数
+ $\text{DF}(t)$：包含词 t 的文档数

###### 📌 (3) TF-IDF

最终的 TF-IDF 权重：

$$
\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)
$$

```python
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    '我 爱 吃 苹果',
    '我 不 爱 吃 香蕉',
    '你 爱 吃 西瓜'
]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

print("词汇表:", vectorizer.get_feature_names_out())
print("TF-IDF 矩阵:\n", X.toarray())
```
!!! success "输出结果"
    
    词汇表: ['不' '你' '吃' '我' '爱' '苹果' '西瓜' '香蕉']

    TF-IDF 矩阵:

    [[0.     0.     0.446 0.446 0.     0.631 0.     0.    ]
    [0.446 0.     0.349 0.349 0.     0.     0.     0.631]
    [0.     0.631 0.349 0.     0.349 0.     0.631 0.    ]]
    

#### 📌 (3) Word2Vec / GloVe / FastText（分布式词向量）

#### 📌 (4) ELMo / ULMFiT（上下文相关词向量）

#### 📌 (5) BERT / RoBERTa / GPT（上下文相关句向量 / 句子向量）

### 4️⃣ 文本表示的输出形式

+ **词向量（Word Embedding）**：每个词一个向量
+ **句向量（Sentence Embedding）**：整句一个向量
+ **文档向量（Document Embedding）**：整篇文章一个向量




    
