# 优化器

---

### **1. 优化器的作用**

优化器（Optimizer）是深度学习模型训练的核心组件，**通过反向传播的梯度信息调整模型参数**，最小化损失函数。  
- **核心目标**：找到损失函数的全局最小值（或足够好的局部最小值）  
- **关键问题**：如何高效更新参数？如何平衡收敛速度和稳定性？

---

### **2. 常见优化器详解**

---

#### **2.1 随机梯度下降（SGD）**
**原理**：沿当前梯度反方向更新参数，学习率（\(\eta\)）控制步长。  
**公式**：  

$$
\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta J(\theta_t)
$$

**优点**：  
- 简单直观，易于实现  
- 对凸函数保证收敛  

**缺点**：  
- 学习率固定，容易陷入局部极小值或鞍点  
- 在高维非凸优化中震荡严重  

**PyTorch 实现**：
```python
import torch.optim as optim

optimizer = optim.SGD(model.parameters(), lr=0.01)  # 基础SGD
```

---

#### **2.2 SGD with Momentum（动量法）**
**原理**：引入动量（动量系数 \(\beta\)），累积历史梯度方向，抑制震荡。  
**公式**：  

\[
v_{t} = \beta \cdot v_{t-1} + (1 - \beta) \cdot \nabla_\theta J(\theta_t)  
\]  

\[
\theta_{t+1} = \theta_t - \eta \cdot v_{t}
\]

**优点**：  
- 加速收敛，减少参数更新的震荡  
- 帮助跳出局部极小值  

**缺点**：  
- 需手动调整 \(\beta\)（通常取 0.9）  

**PyTorch 实现**：
```python
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

---

#### **2.3 AdaGrad**
**原理**：自适应调整学习率，对频繁更新的参数使用更小的学习率。  
**核心思想**：更新频繁的参数，步长减小；更新少的参数，步长保留大一点。
**公式**：  

\[
G_{t} = G_{t-1} + (\nabla_\theta J(\theta_t))^2  
\]  

\[
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \cdot \nabla_\theta J(\theta_t)
\]

**优点**：  
- 自动调整学习率，适合稀疏数据（如 NLP 任务）  

**缺点**：  
- 累积梯度平方导致学习率过早衰减至零  

**PyTorch 实现**：
```python
optimizer = optim.Adagrad(model.parameters(), lr=0.01, eps=1e-8)
```

---

#### **2.4 RMSProp**
**原理**：改进 AdaGrad，引入衰减系数（\(\beta\)），避免学习率过快衰减。  
**公式**：  

\[
G_{t} = \beta \cdot G_{t-1} + (1 - \beta) \cdot (\nabla_\theta J(\theta_t))^2  
\]  

\[
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \cdot \nabla_\theta J(\theta_t)
\]  

**优点**：  
- 缓解 AdaGrad 的学习率衰减问题  
- 适合非平稳目标函数（如 RNN）  

**缺点**：  
- 对初始学习率敏感  

**PyTorch 实现**：
```python
optimizer = optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99, eps=1e-8)
```

---

#### **2.5 Adam（Adaptive Moment Estimation）**
**原理**：结合动量法和 RMSProp，计算梯度的一阶矩（均值）和二阶矩（方差）。  
**公式**：  

\[
m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot \nabla_\theta J(\theta_t) \quad (\text{一阶矩})  
\]  

\[
v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot (\nabla_\theta J(\theta_t))^2 \quad (\text{二阶矩})  
\]

\[
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t} \quad (\text{偏差修正})  
\]  

\[
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \cdot \hat{m}_t
\]  

**优点**：  
- 自适应学习率，收敛速度快  
- 默认参数通常表现良好（\(\beta_1=0.9, \beta_2=0.999\)）  

**缺点**：  
- 可能在某些任务上不如 SGD + Momentum 泛化性好  

**PyTorch 实现**：
```python
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
```

---

### **3. 优化器对比与选择建议**
| **优化器**       | **适用场景**               | **优点**                          | **缺点**                          |
|------------------|--------------------------|----------------------------------|----------------------------------|
| SGD              | 简单任务、凸优化           | 稳定，调参简单                    | 收敛慢，易陷入局部最优            |
| SGD + Momentum   | 图像分类、非凸优化         | 加速收敛，减少震荡                | 需调整动量系数                    |
| AdaGrad          | 稀疏数据（如 NLP）         | 自动调整学习率                    | 学习率过早衰减                    |
| RMSProp          | RNN、非平稳目标           | 缓解 AdaGrad 衰减问题             | 对初始学习率敏感                  |
| Adam             | 通用任务（默认首选）       | 自适应性强，收敛快                | 可能过拟合，内存占用稍高          |

---

### **4. PyTorch 优化器通用配置**
#### **基础使用模板**
```python
model = MyModel()  # 定义模型
criterion = nn.CrossEntropyLoss()  # 定义损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 选择优化器

for epoch in range(num_epochs):
    for inputs, targets in dataloader:
        optimizer.zero_grad()  # 清空历史梯度
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新参数
```

#### **学习率调整**
```python
# 动态调整学习率（如每 30 个 epoch 衰减 0.1 倍）
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
for epoch in range(100):
    train(...)
    scheduler.step()  # 更新学习率
```

---

### **5. 优化器调参技巧**
1. **学习率（lr）**：  
   - 初始值参考：SGD (0.01~0.1)，Adam (0.001~0.0001)  
   - 使用学习率预热（Warmup）或周期性调度（如 Cosine 退火）  

2. **动量系数（\(\beta\)）**：  
   - Momentum 中通常取 0.9  
   - Adam 的 \(\beta_1=0.9, \beta_2=0.999\) 适用于大多数任务  

3. **权重衰减（Weight Decay）**：  
   - 防止过拟合，相当于 L2 正则化  
   - 示例：`optim.Adam(..., weight_decay=1e-4)`

---

### **6. 总结**
- **默认选择**：优先尝试 **Adam**（快速收敛，调参简单），复杂任务可换用 **SGD + Momentum**（需精细调参）。  
- **学习率调整**：配合学习率调度器（如 `StepLR` 或 `CosineAnnealingLR`）可进一步提升性能。  
- **实践原则**：不同任务需实验对比，没有绝对最优的优化器。