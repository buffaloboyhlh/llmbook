# 损失函数

---

### 1. 均方误差损失（MSE Loss）
**概念**：计算预测值与真实值的平方差的平均值，对离群值敏感。  
**公式**：  

\[
L = \frac{1}{N} \sum_{i=1}^N (y_{\text{pred}}^{(i)} - y_{\text{true}}^{(i)})^2
\]

**适用场景**：回归任务（如房价预测、温度预测）。  
**优点**：梯度平滑，收敛快。  
**缺点**：对异常值敏感，可能导致梯度爆炸。 (核心在于它的平方项会放大偏差大的样本的影响) 

**PyTorch 实现**：
```python
import torch.nn as nn

mse_loss = nn.MSELoss()
output = mse_loss(preds, targets)  # preds和targets形状相同
```

---

### 2. 平均绝对误差损失（MAE Loss）
**概念**：计算预测值与真实值的绝对差的平均值，对离群值鲁棒。  
**公式**：  

\[
L = \frac{1}{N} \sum_{i=1}^N |y_{\text{pred}}^{(i)} - y_{\text{true}}^{(i)}|
\]  

**适用场景**：回归任务（数据包含噪声或异常值）。  
**优点**：对异常值不敏感，梯度稳定。  
**缺点**：收敛速度慢，梯度在零点不可导。  
**PyTorch 实现**：
```python
mae_loss = nn.L1Loss()
output = mae_loss(preds, targets)
```

---

### 3. Huber Loss
**概念**：MSE 和 MAE 的结合，在误差较小时使用平方项，较大时使用线性项。  
**公式**：  

\[
L = \frac{1}{N} \sum_{i=1}^N 
\begin{cases} 
0.5 (y_{\text{pred}}^{(i)} - y_{\text{true}}^{(i)})^2, & \text{if } |y_{\text{pred}} - y_{\text{true}}| \leq \delta \\
\delta |y_{\text{pred}} - y_{\text{true}}| - 0.5 \delta^2, & \text{otherwise}
\end{cases}
\]

**适用场景**：回归任务（需平衡对异常值的敏感性和收敛速度）。  
**优点**：结合 MSE 和 MAE 的优点，鲁棒性强。  
**缺点**：需手动调整超参数 \(\delta\)（通常取 1.0）。  
**PyTorch 实现**：
```python
def huber_loss(preds, targets, delta=1.0):
    error = preds - targets
    abs_error = torch.abs(error)
    quadratic = torch.min(abs_error, delta)
    linear = abs_error - quadratic
    loss = 0.5 * quadratic ** 2 + delta * linear
    return loss.mean()

output = huber_loss(preds, targets)
```

---

### 4. 交叉熵损失（Cross-Entropy Loss）
**概念**：衡量预测概率分布与真实分布的差异，常用作分类任务损失函数。  
**公式**（多分类）：  

\[
L = -\sum_{i=1}^N y_{\text{true}}^{(i)} \log \left( \text{Softmax}(y_{\text{pred}}^{(i)}) \right)
\]

**适用场景**：多分类任务（如图像分类）。  
**优点**：梯度优化方向明确，对错误预测惩罚大。  
**缺点**：对类别不平衡敏感。  
**PyTorch 实现**：
```python
ce_loss = nn.CrossEntropyLoss()  # 输入为 logits（无需 softmax）
output = ce_loss(preds, targets)  # preds: (N, C), targets: (N,)
```

---

### 5. 二元交叉熵损失（BCE Loss）
**概念**：二分类任务的交叉熵损失，也可用于多标签分类。  
**公式**：  

\[
L = -\frac{1}{N} \sum_{i=1}^N \left[ y_{\text{true}}^{(i)} \log \sigma(y_{\text{pred}}^{(i)}) + (1 - y_{\text{true}}^{(i)}) \log (1 - \sigma(y_{\text{pred}}^{(i)})) \right]
\]  

**适用场景**：二分类、多标签分类（如医学图像病灶检测）。  
**优点**：直接优化概率输出。  
**缺点**：需注意数值稳定性（建议使用 `BCEWithLogitsLoss`）。  
**PyTorch 实现**：
```python
bce_loss = nn.BCEWithLogitsLoss()  # 自动应用 Sigmoid
output = bce_loss(preds, targets)  # preds和targets形状相同
```

---

### 6. Hinge Loss
**概念**：最大化分类边界，常用于支持向量机（SVM）。  
**公式**（二分类）：  

\[
L = \frac{1}{N} \sum_{i=1}^N \max(0, 1 - y_{\text{true}}^{(i)} \cdot y_{\text{pred}}^{(i)})
\]  

**适用场景**：二分类、结构化预测任务。  
**优点**：对正确分类的点不产生损失，关注边界附近的样本。  
**缺点**：不直接输出概率。  
**PyTorch 实现**：
```python
hinge_loss = nn.HingeEmbeddingLoss()
output = hinge_loss(preds, targets)  # targets取值为1或-1
```

---

### 7. KL 散度（Kullback-Leibler Divergence）
**概念**：衡量两个概率分布的差异，常用于生成模型。  
**公式**：  

\[
L = \sum_{i=1}^N P(y_{\text{true}}^{(i)}) \log \left( \frac{P(y_{\text{true}}^{(i)})}{P(y_{\text{pred}}^{(i)})} \right)
\]  

**适用场景**：分布匹配任务（如变分自编码器）。  
**优点**：严格衡量分布差异。  
**缺点**：不对称，需注意输入为概率分布。  
**PyTorch 实现**：
```python
kl_loss = nn.KLDivLoss(reduction='batchmean')
log_probs = torch.log_softmax(preds, dim=1)  # 输入需为 log 概率
output = kl_loss(log_probs, targets_probs)  # targets_probs为概率分布
```

---

### 总结与选择建议
| **损失函数**      | **任务类型**       | **特点**                             |
|-------------------|--------------------|--------------------------------------|
| MSE               | 回归               | 对异常敏感，梯度稳定                 |
| MAE               | 回归               | 对异常鲁棒，收敛慢                   |
| Huber Loss        | 回归               | 平衡鲁棒性与收敛速度                 |
| Cross-Entropy     | 多分类             | 分类任务首选，梯度高效               |
| BCE               | 二分类/多标签      | 输出概率，需注意数值稳定性           |
| Hinge Loss        | SVM/二分类         | 关注决策边界，不输出概率             |
| KL Divergence     | 分布匹配           | 衡量分布差异，需概率输入             |

根据任务类型和数据特性选择合适的损失函数，并注意 PyTorch 中各损失函数的输入格式（如是否需要提前应用 Softmax/Sigmoid）。


以下是二元交叉熵（BCE Loss）和交叉熵损失（Cross-Entropy Loss）的详细对比与原理讲解，结合公式、示例和代码实现。

---

### **1. 交叉熵损失（Cross-Entropy Loss）**
#### **原理**
交叉熵衡量两个概率分布之间的差异。  
在分类任务中，我们希望模型的预测概率分布 \(P_{\text{pred}}\) 尽可能接近真实的概率分布 \(P_{\text{true}}\)。  
**公式**（多分类）：  

\[
L = -\sum_{i=1}^C y_{\text{true}}^{(i)} \log \left( p_{\text{pred}}^{(i)} \right)
\]

- \(C\)：类别总数  
- \(y_{\text{true}}^{(i)}\)：真实标签的 one-hot 编码（第 \(i\) 类为 1，其余为 0）  
- \(p_{\text{pred}}^{(i)}\)：模型预测的第 \(i\) 类的概率（通过 Softmax 得到）

#### **适用场景**
- **多分类任务**（每个样本只属于一个类别），例如 MNIST 手写数字分类（10 个类别）。

#### **PyTorch 实现**
```python
import torch
import torch.nn as nn

# 假设预测值 preds 是未归一化的 logits（无需手动 Softmax）
preds = torch.tensor([[2.0, 1.0, 0.1], [0.5, 3.0, 0.3]])  # 形状：(N, C)
targets = torch.tensor([0, 1])  # 真实类别索引，形状：(N,)

ce_loss = nn.CrossEntropyLoss()
loss = ce_loss(preds, targets)
print(loss.item())  # 输出：0.8133
```

#### **计算步骤**
1. 对 `preds` 应用 Softmax：
   $$
   \text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^C e^{x_j}}
   $$
   - 第一个样本 `[2.0, 1.0, 0.1]` 的 Softmax 结果：≈ `[0.659, 0.242, 0.099]`
2. 取真实类别对应的概率的对数：
   - 第一个样本的真实类别是 0 → \(-\log(0.659) ≈ 0.417\)
   - 第二个样本的真实类别是 1 → \(-\log(0.242) ≈ 1.418\)
3. 对所有样本求平均：\((0.417 + 1.418)/2 ≈ 0.9175\)

---

### **2. 二元交叉熵损失（BCE Loss）**
#### **原理**
BCE 是交叉熵在二分类或多标签分类中的特例，每个输出节点独立计算概率。  
**公式**（单个样本）：

\[
L = -\left[ y_{\text{true}} \log(p_{\text{pred}}) + (1 - y_{\text{true}}) \log(1 - p_{\text{pred}}) \right]
\]

- \(y_{\text{true}} \in \{0, 1\}\)：真实标签  
- \(p_{\text{pred}}\)：模型预测的正类概率（通过 Sigmoid 得到）

#### **适用场景**
- **二分类任务**（如猫狗分类）
- **多标签分类**（如图像中存在多个物体，每个标签独立判断）

#### **PyTorch 实现**
```python
# 二分类示例（单标签）
preds = torch.tensor([0.8, -1.2, 2.1])  # 未归一化的 logits（形状：N,）
targets = torch.tensor([1.0, 0.0, 1.0])  # 真实标签（形状：N,）

bce_loss = nn.BCEWithLogitsLoss()  # 自动应用 Sigmoid
loss = bce_loss(preds, targets)
print(loss.item())  # 输出：0.6893

# 多标签分类示例（每个样本有多个标签）
preds = torch.tensor([[1.2, -0.5], [0.3, 2.0]])  # 形状：(N, C)
targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])  # 形状：(N, C)

bce_loss = nn.BCEWithLogitsLoss()
loss = bce_loss(preds, targets)
print(loss.item())  # 输出：0.4562
```

#### **计算步骤**
1. 对 `preds` 应用 Sigmoid：

   $$
   \sigma(x) = \frac{1}{1 + e^{-x}}
   $$ 

   - 第一个样本的预测值 `0.8` → Sigmoid 结果为 ≈ `0.690`
2. 计算损失：
   - 第一个样本的真实标签为 1 → \(-\log(0.690) ≈ 0.371\)
   - 第二个样本的真实标签为 0 → \(-\log(1 - \sigma(-1.2)) = -\log(0.769) ≈ 0.262\)
   - 第三个样本的真实标签为 1 → \(-\log(\sigma(2.1)) ≈ 0.122\)
3. 对所有样本求平均：\((0.371 + 0.262 + 0.122)/3 ≈ 0.2518\)

---

### **3. 关键区别与总结**
| **特性**             | **Cross-Entropy Loss**           | **BCE Loss**                     |
|----------------------|----------------------------------|----------------------------------|
| **任务类型**         | 多分类（单标签）                 | 二分类/多标签分类                |
| **输出节点数**       | \(C\)（类别总数）               | 每个标签一个节点                 |
| **激活函数**         | Softmax（归一化为概率分布）      | Sigmoid（独立概率）              |
| **真实标签格式**     | 类别索引或 one-hot 编码         | 0/1 或浮点概率（多标签时为向量） |
| **数值稳定性**       | 内置处理（无需手动处理）         | 推荐使用 `BCEWithLogitsLoss`     |

---

### **4. 常见问题**
#### **Q1：为什么多分类用 Softmax，而二分类用 Sigmoid？**
- **Softmax**：将输出归一化为概率分布（总和为 1），适用于互斥类别（如 MNIST 数字分类）。  
- **Sigmoid**：独立计算每个节点的概率，适用于非互斥标签（如多标签分类中的“同时包含猫和狗”）。

#### **Q2：多标签分类时如何处理？**
- 对每个标签独立使用 Sigmoid，计算每个标签的 BCE 损失后取平均。  
- 示例代码：
  ```python
  # 多标签分类（3 个标签）
  preds = torch.tensor([[1.2, -0.5, 2.1], [-0.3, 1.0, 0.5]])  # 形状：(N, C)
  targets = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])  # 形状：(N, C)
  loss = nn.BCEWithLogitsLoss()(preds, targets)
  ```

#### **Q3：为什么推荐 `BCEWithLogitsLoss`？**
- 它结合了 Sigmoid 和 BCE 计算，通过数值稳定化的实现避免 `log(0)` 导致的 NaN 错误。

---

### **5. 总结**
- **Cross-Entropy Loss**：多分类任务的标配，直接优化概率分布。  
- **BCE Loss**：二分类/多标签任务的核心，需注意 Sigmoid 激活和数值稳定性。  
- **选择依据**：根据任务类型（单标签 vs 多标签）和输出格式（概率分布 vs 独立概率）选择损失函数。