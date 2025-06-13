# 第四章：降维

### 1️⃣ 什么是降维？

降维是指将高维数据映射到低维空间的过程，在保留尽可能多的重要信息的同时减少特征数量。

#### ✨ 降维的作用：

- 📊 降低计算复杂度
- 🔍 缓解“维度灾难”
- 🧠 减少过拟合
- 🧩 提高模型泛化能力
- 🖼️ 便于数据可视化（2D/3D）

### 2️⃣ 降维方法分类

| 方法类型     | 降维方法       | 特点与说明                     |
|--------------|----------------|--------------------------------|
| 线性降维     | PCA            | 保留最大方差方向，快速稳定     |
|              | LDA            | 监督式，增强类间区分           |
| 非线性降维   | t-SNE          | 保持局部结构，适合可视化       |
|              | UMAP           | 更快、保拓扑，适合大规模数据   |
| 神经网络方法 | AutoEncoder    | 通过神经网络压缩再重构         |
| 特征选择     | 基于模型/统计  | 删除无关或冗余特征             |


### 3️⃣ 主成分分析（PCA）

PCA 是最常用的线性降维方法。

#### 原理：
- 找出数据中方差最大的方向作为新坐标轴（主成分）
- 主成分相互正交
- 用前 \(k\) 个主成分近似原始数据

#### 数学推导：

1. 标准化数据矩阵 \( X \in \mathbb{R}^{n \times d} \)
2. 计算协方差矩阵：
   $$
   C = \frac{1}{n} X^T X
   $$
3. 对 \(C\) 做特征值分解，得：
   $$
   C = V \Lambda V^T
   $$
4. 取前 \(k\) 个特征向量 \(v_1, ..., v_k\) 构成投影矩阵 \(W_k\)
5. 得到降维后的数据：
   $$
   X_{\text{new}} = X W_k
   $$

### 4️⃣ 线性判别分析（LDA）

- 有监督方法
- 最大化类间距离，最小化类内距离
- 适用于分类任务中的降维

###  5️⃣ t-SNE（t-Distributed Stochastic Neighbor Embedding）

- 保持高维空间中数据点的**邻近结构**
- 适合高维数据的**2D 可视化**
- 非常适合于探索性数据分析

### 6️⃣ UMAP（Uniform Manifold Approximation and Projection）


- 保持数据在高维空间中的**拓扑结构**
- 速度比 t-SNE 更快，支持增量式更新
- 适合用于大规模数据集降维与聚类前处理

### 7️⃣ 自动编码器（AutoEncoder）

AutoEncoder 是一种无监督神经网络降维方法：

- 编码器将数据压缩成低维表示 \(z\)
- 解码器将 \(z\) 恢复为原始数据
- 通过重构误差进行训练

结构：

输入层 → 编码器（降维） → 解码器（重建） → 输出层

### 8️⃣ 特征选择 vs 特征提取

| 类型       | 特征选择                 | 特征提取                     |
|------------|--------------------------|------------------------------|
| 方法       | 删除冗余/无关特征       | 构造新特征组合               |
| 示例       | 过滤法、Wrapper、L1正则 | PCA, AutoEncoder             |
| 是否构造新特征 | 否                       | 是                           |


### 9️⃣ Python 示例：PCA + 可视化

```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

# 加载数据
data = load_iris()
X = data.data
y = data.target

# 降维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 可视化
plt.figure(figsize=(8, 6))
for label in set(y):
    plt.scatter(X_pca[y == label, 0], X_pca[y == label, 1], label=f"Class {label}")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA on Iris Dataset")
plt.legend()
plt.grid(True)
plt.show()
```

### 🔟 总结

| 方法         | 类型       | 是否监督 | 特点                         |
|--------------|------------|----------|------------------------------|
| PCA          | 线性       | 否       | 快速稳定，保留最大方差方向   |
| LDA          | 线性       | 是       | 提高类间区分性               |
| t-SNE        | 非线性     | 否       | 局部保持，适合高维可视化     |
| UMAP         | 非线性     | 否       | 保拓扑结构，适合大规模数据   |
| AutoEncoder  | 神经网络   | 否       | 非线性压缩，可自定义编码结构 |


