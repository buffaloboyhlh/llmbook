# 学习率调度器（Learning Rate Scheduler）

---

#### **一、概念**

学习率调度器是深度学习训练过程中动态调整学习率的工具。学习率（Learning Rate, LR）是控制模型参数更新步长的超参数，直接影响模型收敛速度和性能。

- **固定学习率**：训练全程使用相同的学习率，可能导致收敛慢（学习率过小）或震荡（学习率过大）。
- **动态学习率**：根据训练进度（如 epoch、batch 或指标变化）调整学习率，平衡收敛速度与稳定性。

---

#### **二、常见调度器类型与公式**

以下是几种经典的学习率调度器及其数学公式：

##### 1. **StepLR（阶梯下降）**

- **原理**：每隔固定步长（如每 `step_size` 个 epoch）将学习率乘以衰减因子 `gamma`。
- **公式**：

  $$
  \text{LR} = \text{初始学习率} \times \gamma^{\lfloor \frac{\text{epoch}}{\text{step\_size}} \rfloor}
  $$

- **适用场景**：简单任务或初步实验。

##### 2. **MultiStepLR（多阶段下降）**

- **原理**：在预设的多个里程碑（如 `milestones=[30, 80]`）处衰减学习率。
- **公式**：

  $$
  \text{LR} = \text{初始学习率} \times \gamma^{\text{当前里程碑数}}
  $$

- **适用场景**：复杂任务分阶段优化。

##### 3. **ExponentialLR（指数衰减）**

- **原理**：每个 epoch 按指数函数持续衰减学习率。
- **公式**：

  $$
  \text{LR} = \text{初始学习率} \times \gamma^{\text{epoch}}
  $$

- **适用场景**：需要平滑衰减的场景。

##### 4. **CosineAnnealingLR（余弦退火）**

- **原理**：按余弦函数周期性地调整学习率。
- **公式**：

$$
\text{LR} = \text{LR}_{\text{min}} + \frac{1}{2}(\text{LR}_{\text{max}} - \text{LR}_{\text{min}})(1 + \cos(\frac{\text{epoch}}{T_{\text{max}}} \pi))
$$

- **适用场景**：图像分类等复杂任务，逃离局部最优。

##### 5. **ReduceLROnPlateau（动态调整）**

- **原理**：当验证损失停滞时自动降低学习率。
- **公式**：无固定公式，依赖监控指标的变化。
- **适用场景**：验证集表现停滞时的自适应调整。

---

#### **三、优缺点对比**

| 调度器               | 优点          | 缺点             |
|-------------------|-------------|----------------|
| StepLR            | 简单直观，计算高效   | 需手动设置步长和衰减因子   |
| MultiStepLR       | 灵活支持多阶段调整   | 需预定义里程碑        |
| ExponentialLR     | 平滑衰减，避免突变   | 可能过早降至过低值      |
| CosineAnnealing   | 逃离局部最优，收敛稳定 | 计算成本略高         |
| ReduceLROnPlateau | 自适应，无需预设周期  | 依赖验证集指标，可能延迟响应 |

---

#### **四、PyTorch 实现代码**

以下是 PyTorch 中常用调度器的代码示例：

##### 1. **StepLR**

```python
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

for epoch in range(100):
    train(...)
    scheduler.step()  # 每个 epoch 后更新
```

##### 2. **CosineAnnealingLR**

```python
from torch.optim.lr_scheduler import CosineAnnealingLR

scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)
```

##### 3. **ReduceLROnPlateau**

```python
from torch.optim.lr_scheduler import ReduceLROnPlateau

scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

val_loss = validate(...)
scheduler.step(val_loss)  # 需传入监控指标
```

##### 4. **OneCycleLR（高级）**

```python
from torch.optim.lr_scheduler import OneCycleLR

scheduler = OneCycleLR(optimizer, max_lr=0.1, total_steps=100)

for batch in dataloader:
    train_batch(...)
    scheduler.step()  # 每个 batch 后更新
```

---

#### **五、选择建议**

- **简单任务**：优先尝试 `StepLR` 或 `MultiStepLR`。
- **复杂任务**：使用 `CosineAnnealingLR` 或 `OneCycleLR`。
- **验证集依赖**：选择 `ReduceLROnPlateau`。
- **小数据集**：谨慎使用动态调度，避免过拟合。

---

#### **六、总结**

学习率调度器通过动态调整训练步长，显著提升模型收敛速度和性能。实际应用中需结合任务特点选择策略，并通过实验验证效果。