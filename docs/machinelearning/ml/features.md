# 第一章：特征工程

特征工程（Feature Engineering）是指将原始数据转化为更适合机器学习模型使用的特征集的过程。它在机器学习中非常重要，因为好的特征往往比复杂的模型更能提升效果。

换句话说：特征工程就是“**从数据中提炼出最能描述问题的特征**”。

**特征工程**，顾名思义，是对原始数据进行一系列工程处理，将其提炼为特征，作为输入供算法和模型使用。从本质上来讲，特征工程是一个表示和展现数据的过程。在实际工作中，特征工程旨在去除原始数据中的杂志和冗余，
设计更高效的特征以刻画求解的问题与预测模型之间的关系。

----

## 一、特征归一化

### 1.1 什么是特征归一化？

**特征归一化（Normalization）** 是将数值型特征压缩到相同的尺度，通常是 [0, 1] 区间，以避免不同量纲和数值范围对模型造成影响。

它是特征工程中非常重要的预处理步骤，尤其适用于基于“距离计算”或“梯度优化”的模型。


### 1.2 为什么需要归一化？

#### 场景 1：不同特征的数值范围差异大

比如：

+ 身高（150~200cm）
+ 体重（50~100kg）
+ 收入（1w~100w）

这些不同量纲的数值直接输入模型，会让“量纲大的特征”主导模型学习。

#### 场景 2：使用以下模型时必须归一化

+ KNN / K-means（基于距离）
+ SVM（核函数计算）
+ Logistic / 线性回归（收敛速度受梯度影响）
+ 神经网络（梯度稳定性）

### 1.3 归一化的常见方法

#### 1. Min-Max 归一化（缩放到 0~1）

$$
x’ = \frac{x - x_{\min}}{x_{\max} - x_{\min}}
$$

**特点：**

+ 适合数值分布稳定，边界已知的情况；
+ 对异常值敏感（容易压缩其他值）。

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
```

#### 2. MaxAbs 归一化（缩放到 [-1, 1]）

适合稀疏数据，如文本向量、TF-IDF 等。

$$
x’ = \frac{x}{|x_{\max}|}
$$

```python
from sklearn.preprocessing import MaxAbsScaler

scaler = MaxAbsScaler()
X_scaled = scaler.fit_transform(X)
```


####  3. L2 归一化（单位向量归一化）

让每一行（样本）模长为 1，常用于文本/向量表示。

**公式（L2 范数）：**

$$
    x’ = \frac{x}{\|x\|_2}
$$

```python
from sklearn.preprocessing import Normalizer

scaler = Normalizer(norm='l2')
X_scaled = scaler.fit_transform(X)
```

### 1.4 归一化 vs 标准化（Standardization）


| 比较项       | 标准化（Standardization） | 归一化（Normalization） |
|--------------|----------------------------|--------------------------|
| 适用场景     | 大多数模型（如 SVM、逻辑回归、KNN、PCA） | 特征有明确边界的情况（如图像像素） |
| 输出范围     | 均值 0，标准差 1           | 一般在 [0, 1] 或 [-1, 1] |
| 是否受异常值影响 | 是                        | 是（更敏感）             |
| 是否依赖分布 | 适合正态分布               | 无要求                   |
| 代码实现（sklearn） | `StandardScaler`      | `MinMaxScaler`           |

### 1.5 实战

```python
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 示例数据
df = pd.DataFrame({
    "height": [160, 170, 180],
    "weight": [50, 70, 90]
})

scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

print(df_scaled)
```
!!! Example "输出结果"
  
    ```text
          height  weight
      0     0.0     0.0
      1     0.5     0.5
      2     1.0     1.0
    ```

----

## 二、类别型特征

**类别型特征（Categorical Features）指的是取值为离散类别**而非数值大小有意义的特征。比如颜色（红、绿、蓝）、城市（北京、上海）、性别（男、女）等。这些特征无法直接输入大多数机器学习算法，需先进行编码和处理。

### 2.1 类别型特征的类型

类别型特征主要分为两类：

**1. 名义型（Nominal）**

+ 没有顺序或大小的概念。
+ 示例：颜色（红、绿、蓝）、职业、城市。

**2. 序数型（Ordinal）**

+ 具有明确的顺序关系，但没有明确的间距或量化含义。
+ 示例：教育程度（小学 < 初中 < 高中 < 大学 < 硕士 < 博士）。


### 2.2 常见的编码方式

#### 1. 标签编码（Label Encoding）

使用整数代替类别。

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
data['color_encoded'] = le.fit_transform(data['color'])
```

!!! warning

    ⚠️ 适合有序类别（Ordinal），不适合无序类别（Nominal），否则可能引入虚假顺序信息。


#### 2. 独热编码（One-Hot Encoding）

将每个类别转化为一个独立的二进制列。

```python
import pandas as pd

pd.get_dummies(data['color'], prefix='color')
```

!!! warning

    ⚠️ 维度可能会爆炸（高基数特征），适合树模型（如 XGBoost、LightGBM）不敏感于稀疏特征。


#### 3. 二进制编码（Binary Encoding）

将类别先转换为整数，再转换为二进制编码。

```text
类别A → 1 → 001
类别B → 2 → 010
类别C → 3 → 011
```

!!! info

    优点：比 One-Hot 更节省空间，适合中高基数类别。

#### 4. 频率编码（Frequency Encoding）

用某个类别出现的频率进行编码。

```python
freq = data['city'].value_counts(normalize=True)
data['city_freq'] = data['city'].map(freq)
```

#### 5. 目标编码（Target Encoding / Mean Encoding）

用某个类别对应的目标变量的均值进行编码。

**示例（分类问题）：**
```python
means = data.groupby('city')['target'].mean()
data['city_encoded'] = data['city'].map(means)
```
!!! warning

    ⚠️ 有泄露风险，需使用交叉验证或训练集均值。


### 2.3 实战

```python
import pandas as pd
df = pd.DataFrame({
    'color': ['red', 'green', 'blue', 'green', 'red'],
    'target': [1, 0, 1, 0, 1]
})

# Label Encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['label_enc'] = le.fit_transform(df['color'])

# One-Hot
df_onehot = pd.get_dummies(df['color'], prefix='color')

# Target Encoding
mean_map = df.groupby('color')['target'].mean()
df['target_enc'] = df['color'].map(mean_map)

print(df)
print(df_onehot)
```

----

## 三、文本表示模型

文本表示模型（Text Representation Models）是自然语言处理（NLP）中的核心技术，目标是将文本（词、句子、文档）转换为模型可处理的数值向量，以便进行分类、聚类、搜索、问答、生成等任务。

### 3.1 概述：为什么需要文本表示

机器无法直接理解人类语言，需要将其转化为向量。好的表示方法应该满足：

+ 语义保留性：相似文本向量应接近
+ 上下文敏感性：一个词在不同语境中应有不同表示
+ 压缩性：高效表示，避免维度灾难
+ 可泛化性：适用于下游任务，如分类、匹配、检索等

### 3.2 传统表示方法

#### 1. One-Hot Encoding

- 每个词对应一个高维向量（如词表中第 123 个词为 `[0,...,1,...,0]`）  
- 缺点：
    - 稀疏  
    - 不含语义  
    - 向量之间无距离含义（如“猫”和“狗”距离和“猫”和“自行车”一样）  

#### 2. Bag of Words (BoW)

- 统计词频：`[the: 4, cat: 1, sat: 1, on: 1, mat: 1]`  
- 缺点：忽略词序、无法捕捉上下文信息

#### 3. TF-IDF

- TF：Term Frequency，词频  
- IDF：Inverse Document Frequency，文档反频率  
- 优点：考虑了词语的重要性  
- 缺点：向量仍然稀疏、维度高、不含语义关系  

```python
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = ["我 爱 自然语言处理", "自然语言 很 有趣"]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
print(X.toarray())
```

### 3.3 分布式词向量

####  1. Word2Vec

- 提出者：Google（2013）
- 两种模型结构：
    - **CBOW（Continuous Bag-of-Words）**：根据上下文词预测中心词。
    - **Skip-Gram**：根据中心词预测上下文词。
- 优点：
    - 能捕捉词与词之间的语义/句法关系。
    - 支持线性推理关系：如 `vector("王") - vector("男人") + vector("女人") ≈ vector("女王")`。


```python
from gensim.models import Word2Vec

# 训练语料（分词后的句子）
sentences = [["我", "爱", "自然语言处理"], ["自然语言", "处理", "很", "有趣"]]

# 训练模型
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=1)  # sg=1 表示使用 Skip-Gram

# 获取词向量
vector = model.wv['自然语言']
print(vector.shape)  # 输出: (100,)
```

#### 2. GloVe（Global Vectors）

- **提出者**：斯坦福大学 NLP 团队（2014 年）
- **论文标题**：[GloVe: Global Vectors for Word Representation](https://aclanthology.org/D14-1162/)
- **核心思想**：基于**词与词之间的共现频率矩阵**，通过矩阵因式分解获得词向量表示。

##### 基本原理

GloVe 不是预测模型（如 Word2Vec），而是**基于计数的模型**。它将一个大型语料库中所有词对的共现频率统计为矩阵 \( X_{ij} \)，其中：

- \( X_{ij} \)：词 \( j \) 在词 \( i \) 的上下文中出现的次数

然后通过构造损失函数，拟合以下关系：

\[
w_i^T \cdot \tilde{w}_j + b_i + \tilde{b}_j = \log(X_{ij})
\]

其中：

- \( w_i \)：词 \( i \) 的向量  
- \( \tilde{w}_j \)：上下文词 \( j \) 的向量  
- \( b_i, \tilde{b}_j \)：偏置项  
- \( X_{ij} \)：共现次数  

通过优化这个目标函数，学习每个词和上下文词的低维稠密向量。

##### 相比 Word2Vec 的优势

| 特征         | GloVe                         | Word2Vec                        |
|--------------|-------------------------------|----------------------------------|
| 建模方式     | 基于全局共现矩阵              | 基于局部上下文窗口              |
| 训练机制     | 矩阵因式分解（线性回归）      | 神经网络上下文预测              |
| 上下文理解   | 全局语料统计，语义更稳定      | 局部窗口，语义动态，适配更好    |
| 能否在线训练 | 否（需预计算共现矩阵）        | 是                              |


##### 使用 GloVe 的词向量

GloVe 官方提供了多种预训练模型：

| 维度  | 语料        | 下载地址                                              |
|-------|-------------|--------------------------------------------------------|
| 50d   | Wikipedia 6B| [glove.6B.zip](https://nlp.stanford.edu/data/glove.6B.zip) |
| 100d  | Wikipedia 6B| 同上                                                  |
| 300d  | Common Crawl 840B | [glove.840B.300d.zip](https://nlp.stanford.edu/data/glove.840B.300d.zip) |


##### 加载预训练 GloVe 向量（Python 示例）

```python
import numpy as np

# 加载 GloVe 文件
def load_glove_embeddings(file_path):
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

# 使用示例
glove_path = "glove.6B.100d.txt"
glove_vectors = load_glove_embeddings(glove_path)
print(glove_vectors["king"])  # 打印 "king" 的词向量
```

##### GloVe 的意义

+ 更好地捕捉语义相似性：如 “ice” 和 “snow” 会靠近，而 “ice” 和 “steam” 会被区分。
+ 适用于需要静态词表示的 NLP 任务，如聚类、KMeans 可视化、信息检索等。

####  3. FastText

+ 提出者：Facebook AI（2016）
+ 创新点：将词拆解为子词（n-gram）进行训练
+ 解决 OOV 问题：未见过的词也可以通过其子词组合推导向量。
+ 更适合中文和形态复杂语言。

**特点:**

+ "apple" 会被分成：< ap, app, ppl, ple, le > 等字符 n-gram。
+ 每个词的向量 = 其子词向量平均或加权和。


##### 训练示例（使用 fasttext）

```bash
# 安装 fasttext（如果使用 Python，建议用 pip 安装 fasttext-wheel）
pip install fasttext-wheel
```

```python
import fasttext

# 训练模型（文本文件每行为一句话，已进行分词）
model = fasttext.train_unsupervised('corpus.txt', model='skipgram')

# 获取词向量
vec = model.get_word_vector("自然语言处理")
```

### 3.4 上下文相关的表示方法


传统的词向量（如 Word2Vec、GloVe、FastText）是**静态的**：一个词在所有上下文中始终对应同一个向量。

然而，在实际语言中，一个词在不同上下文中含义可能不同。例如：

- “银行”（bank）在“金融银行”和“河岸”中语义完全不同。

**上下文相关词向量的核心思想**

> 为每个词在不同上下文中生成**动态的表示向量**。

#### 1. ELMo（Embeddings from Language Models）

- **提出者**：AllenNLP（2018）
- **本质**：利用双向 LSTM（BiLSTM）语言模型，结合上下文生成词的动态表示。
- **关键特性**：
    - 每个词的向量由其上下文共同决定。
    - 对词义歧义问题有较强建模能力。

##### ELMo 表示方式

一个词的表示由多个 LSTM 层的输出加权求和得到：

\[
\text{ELMo}_t = \gamma \sum_{k=0}^{L} s_k h_{t,k}
\]

- \( h_{t,k} \)：第 \( k \) 层 BiLSTM 对第 \( t \) 个词的输出
- \( s_k \)：权重参数（通过训练学习）
- \( \gamma \)：缩放因子

##### 优点

- 跨句建模，语言理解更深入。
- 可用于下游任务微调。


#### 2. GPT 系列（Generative Pre-trained Transformer）

- **GPT**：使用 Transformer 的 decoder 架构，左到右地建模文本（单向）。
- **预训练任务**：语言建模（预测下一个词）
- **适合任务**：生成类（如续写、对话生成）

##### GPT 特点：

- 上下文建模能力强，适合开放式生成任务。
- 靠右侧上下文预测下一个词。

#### 3. BERT（Bidirectional Encoder Representations from Transformers）

- **提出者**：Google AI（2018）
- **模型结构**：Transformer Encoder（双向）
- **预训练任务**：
    - **Masked Language Modeling (MLM)**：随机遮盖一些词并预测它们。
    - **Next Sentence Prediction (NSP)**：判断两个句子是否为相邻句。

#####  BERT 特点

- **双向建模**：能同时看到左右上下文。
- **上下文感知表示**：词向量依赖所在句子语义。
- **可微调性强**：几乎适用于所有下游 NLP 任务（分类、问答、NER 等）。

##### BERT 应用示例（使用 Transformers 库）

```python
from transformers import BertTokenizer, BertModel
import torch

# 加载预训练模型和分词器
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# 编码输入文本
inputs = tokenizer("Natural Language Processing is amazing!", return_tensors="pt")
outputs = model(**inputs)

# 获取最后一层的词向量
last_hidden_states = outputs.last_hidden_state
print(last_hidden_states.shape)  # (batch_size, seq_len, hidden_size)
```

## 四、Word2Vec

Word2Vec 是最经典的词嵌入（word embedding）方法之一，由 Google 于 2013 年提出，用神经网络学习词的稠密向量表示。

### 4.1 为什么需要 Word2Vec？

传统的词表示方法（如 one-hot 编码）有以下缺点：

+ 维度高：词表大时，向量维度也非常大。
+ 稀疏：绝大多数维度都是 0。
+ 不含语义关系：如 “king” 与 “queen” 之间没有任何向量关系。

!!! info

    Word2Vec 通过训练学习出具有语义信息的低维稠密向量（Dense Vector），能让“语义相近”的词在向量空间中“靠得更近”。


### 4.2 Word2Vec 的两种模型

#### 1. CBOW（Continuous Bag of Words）

- **输入**：上下文词（context）
- **输出**：目标词（target word）
- **目标**：从上下文词预测中心词。

##### 示例：

- 给定句子：“I love natural language processing”
- 中心词为 `"natural"`，上下文词为 `[I, love, language, processing]`
- CBOW 的任务是：根据上下文 `[I, love, language, processing]` 来预测 `"natural"`

##### 特点：

- 更适合小语料
- 训练速度快，效果稳定

#### 2. Skip-Gram

- **输入**：中心词（target）
- **输出**：上下文词（context）
- **目标**：从中心词预测其上下文词。

#####  示例：

- 给定中心词 `"natural"`
- 预测其上下文词 `[I, love, language, processing]`

##### 特点：

- 更适合大语料
- 对低频词效果更好
- 训练时间相对较长，但语义表达能力强

##### CBOW vs Skip-Gram 对比

| 模型      | 输入           | 输出            | 优势                | 劣势                 |
|-----------|----------------|------------------|---------------------|----------------------|
| CBOW      | 上下文词       | 中心词           | 快速，稳定          | 低频词效果一般       |
| Skip-Gram | 中心词         | 上下文词         | 低频词学习能力强    | 训练较慢             |


### 4.3 原理公式

以 Skip-Gram 模型为例，其核心目标是：

> 给定一个中心词，预测其上下文中出现的词语。

#### Skip-Gram 的目标函数：


最大化语料中每个词的上下文词的条件概率乘积：

\[
\mathcal{L} = \prod_{t=1}^{T} \prod_{\substack{-c \leq j \leq c \\ j \ne 0}} P(w_{t+j} \mid w_t)
\]

其中：

- \( T \)：语料中的词总数  
- \( c \)：上下文窗口大小  
- \( w_t \)：当前位置的词（中心词）  
- \( w_{t+j} \)：中心词周围的上下文词

#### 条件概率的建模（Softmax）：

\[
P(w_O \mid w_I) = \frac{\exp\left(v_{w_O}^\top \cdot v_{w_I}\right)}{\sum_{w=1}^{V} \exp\left(v_{w}^\top \cdot v_{w_I}\right)}
\]

- \( v_{w_I} \)：输入词 \( w_I \) 的词向量（中心词）
- \( v_{w_O} \)：输出词 \( w_O \) 的词向量（上下文词）
- \( V \)：词表大小（vocabulary size）

#### 问题：Softmax 计算代价高

- 每次更新都需要计算整个词表中所有词的得分和归一化
- 词表通常很大（上万甚至百万级），导致计算效率低

#####  解决方案（加速技巧）：

###### 1. Hierarchical Softmax（层次化 Softmax）


- 构建一棵霍夫曼树（Huffman Tree）
- 每个词是树上的一个叶子节点
- 只需沿路径更新节点，复杂度降为 \( O(\log V) \)

###### 2. Negative Sampling（负采样）

- 每次只优化一个正样本 + 少量负样本（如 5-20 个）
- 训练速度显著提升，且效果稳定
- 本质上是用 logistic 回归训练一个二分类器：

\[
\log \sigma(v_{w_O}^\top v_{w_I}) + \sum_{i=1}^{k} \mathbb{E}_{w_i \sim P_n(w)} \left[ \log \sigma(-v_{w_i}^\top v_{w_I}) \right]
\]

- \( \sigma(x) \)：sigmoid 函数
- \( P_n(w) \)：负样本分布
- \( k \)：负样本数量


### 4.4 使用 Gensim 训练 Word2Vec（实战）

#### 训练 Word2Vec 模型

```python
from gensim.models import Word2Vec

# 训练语料：每句话是一个分词后的列表
sentences = [["我", "爱", "自然", "语言", "处理"], ["语言", "模型", "很", "有趣"]]

# 训练模型
model = Word2Vec(
    sentences, 
    vector_size=100,  # 词向量维度
    window=5,         # 上下文窗口大小
    min_count=1,      # 最小词频
    sg=1              # sg=1 表示 Skip-Gram，sg=0 表示 CBOW
)

# 查看某个词的词向量
print(model.wv["语言"])

# 找出与“语言”最相似的词
print(model.wv.most_similar("语言"))
```

####  保存和加载模型

```python
# 保存模型
model.save("word2vec.model")

# 加载模型
from gensim.models import Word2Vec
model = Word2Vec.load("word2vec.model")
```

### 4.5 Word2Vec 的应用场景

+ 文本分类（词向量作为输入）
+ 命名实体识别（NER）
+ 情感分析
+ 相似词推荐 / 搜索引擎
+ 可视化（如 t-SNE 降维）

### 4.6 向量推理的神奇之处

Word2Vec 向量空间可以执行语义类比：

$$
    \text{vector}(“国王”) - \text{vector}(“男人”) + \text{vector}(“女人”) \approx \text{vector}(“女王”)
$$

```python
result = model.wv.most_similar(positive=["女", "国王"], negative=["男"])
print(result)  # 可能得到 "女王"
```

### 4.7 常见问题


| 问题                                  | 解答说明 |
|---------------------------------------|----------|
| Word2Vec 能处理未登录词（OOV）吗？   | ❌ 不能，必须出现在训练语料中，未见过的词没有向量。 |
| Word2Vec 可以用于中文吗？             | ✅ 可以，但必须先对中文进行分词处理（如使用jieba）。 |
| 如何选择 vector_size 参数？           | 常用为 100、200、300，推荐根据任务复杂度和语料量做调优。 |
| CBOW 和 Skip-Gram 哪个更好？          | CBOW 快速且适用于高频词，Skip-Gram 更适合表示低频词语义。 |
| 训练语料越多越好吗？                  | ✅ 是的，语料越大，词向量越能捕捉复杂语义关系。 |
| Word2Vec 能表示多义词吗？             | ❌ 不能，每个词只有一个静态向量，无法区分语境。 |
| 是否需要标准化词向量？                | 训练后通常会进行归一化，以便进行向量相似度计算（如余弦相似度）。 |
| Word2Vec 和 TF-IDF 有什么区别？       | TF-IDF 是稀疏表示，Word2Vec 是低维稠密表示，且包含语义信息。 |


**提示：**

- 若任务对词语的上下文语义区分度要求很高（如情感分析、问答系统），建议使用上下文感知模型（如 ELMo 或 BERT）。
- Word2Vec 是轻量级、计算成本低的首选向量表示方式，适合很多嵌入需求。

### 4.8 总结

| 项目         | 内容                                                                 |
|--------------|----------------------------------------------------------------------|
| 模型名称     | Word2Vec                                                             |
| 核心结构     | CBOW（上下文预测中心词）<br>Skip-Gram（中心词预测上下文）            |
| 输入         | 单词 ID 或上下文词                                                   |
| 输出         | 对应词向量                                                           |
| 优点         | 语义表达能力强、训练速度快、结构简单、适用于多种下游任务             |
| 缺点         | 静态表示（无上下文变化）<br>无法处理 OOV 和多义词                     |
| 训练方法     | 使用浅层神经网络 + Softmax 或负采样（Negative Sampling）             |
| 加速技巧     | Negative Sampling、Hierarchical Softmax                              |
| 应用场景     | 文本分类、聚类、相似度计算、情感分析、搜索排序、推荐系统等            |
| 替代方案     | GloVe、FastText、ELMo、BERT 等上下文相关模型                         |















