# 第九章：前向神经网络

### 1️⃣ 简介

前向神经网络是最基础的人工神经网络结构，数据在网络中单向流动，不存在循环或反馈连接。它是现代深度学习的基础。

### 2️⃣ 基本结构

```text
输入层 → 隐藏层 → 输出层
```

- **输入层**：接收输入特征  
- **隐藏层**：一个或多个，每层由多个神经元构成  
- **输出层**：输出最终预测结果

每一层的神经元与下一层全连接。

###  3️⃣ 数学原理

对于任意一层：

$$
z^{(l)} = W^{(l)} \cdot a^{(l-1)} + b^{(l)} \\
a^{(l)} = f(z^{(l)})
$$

- \( W^{(l)} \)：权重矩阵（第 \(l\) 层）  
- \( b^{(l)} \)：偏置向量  
- \( f \)：激活函数，如 ReLU、Sigmoid、Tanh  
- \( a^{(l)} \)：第 \(l\) 层的输出

###  4️⃣ 常见激活函数

| 函数     | 表达式                                  | 作用                             |
|----------|-------------------------------------------|----------------------------------|
| Sigmoid  | \( \sigma(x) = \frac{1}{1+e^{-x}} \)       | 输出范围 (0,1)，用于概率         |
| Tanh     | \( \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} \) | 输出范围 (-1,1)                  |
| ReLU     | \( \text{ReLU}(x) = \max(0, x) \)          | 非线性，收敛速度快，广泛使用     |


###  5️⃣ Python 实现（使用 PyTorch）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义前向神经网络模型
class FeedForwardNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # 输入层到隐藏层
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # 隐藏层到输出层

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 激活函数 ReLU
        x = self.fc2(x)
        return x

# 示例
model = FeedForwardNN(input_dim=4, hidden_dim=8, output_dim=3)
print(model)
```

### 6️⃣ 训练流程

```python
# 模拟数据
X = torch.randn(100, 4)
y = torch.randint(0, 3, (100,))

# 定义损失和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 训练循环
for epoch in range(100):
    outputs = model(X)
    loss = criterion(outputs, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

### 7️⃣ 前向传播 vs 反向传播

| 项目         | 描述                                           |
|--------------|------------------------------------------------|
| 前向传播     | 从输入层到输出层计算预测值                     |
| 反向传播     | 根据损失函数，反向传播误差并更新权重参数       |
| 损失函数     | 衡量预测值与真实值的误差（如 CrossEntropy）     |


### 8️⃣ 应用场景

+ 图像分类（如 MNIST 手写数字）
+ 二分类/多分类任务
+ 回归预测
+ 特征提取
+ 时间序列建模（结合 RNN）

### 9️⃣ 常见问题

| 问题             | 原因或解决办法                          |
|------------------|-----------------------------------------|
| 梯度消失/爆炸    | 使用 ReLU、BatchNorm、权重初始化        |
| 过拟合           | 使用 Dropout、正则化、增加数据量        |
| 学习率太大或太小 | 使用优化器如 Adam，并调节学习率         |

###  🔟 总结

| 项目       | 描述                                     |
|------------|------------------------------------------|
| 架构       | 输入层 → 隐藏层 → 输出层                 |
| 数据流动   | 单向前向，无反馈                         |
| 激活函数   | 引入非线性，提升表达能力                 |
| 训练方法   | 前向传播 + 反向传播 + 梯度下降           |
| 实现方式   | 可使用 PyTorch / TensorFlow              |



