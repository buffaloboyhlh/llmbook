# 可视化

## 一、torchinfo

### 1.1 安装

```bash
pip install torchinfo 
```

### 1.2 使用torchinfo

```python
from torchinfo import summary
import torch.nn as nn

model = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(16*16*16, 10)
)

summary(model, input_size=(1, 3, 32, 32))  # batch_size=1, channel=3, height=32, width=32
```

输出结果：
```text
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
Sequential                               [1, 10]                   --
├─Conv2d: 1-1                            [1, 16, 32, 32]           448
├─ReLU: 1-2                              [1, 16, 32, 32]           --
├─MaxPool2d: 1-3                         [1, 16, 16, 16]           --
├─Flatten: 1-4                           [1, 4096]                 --
├─Linear: 1-5                            [1, 10]                   40,970
==========================================================================================
Total params: 41,418
Trainable params: 41,418
Non-trainable params: 0
Total mult-adds (M): 0.50
==========================================================================================
Input size (MB): 0.01
Forward/backward pass size (MB): 0.13
Params size (MB): 0.17
Estimated Total Size (MB): 0.31
==========================================================================================
```

!!! info
    
    ## 📋 参数详解（`torchinfo.summary`）
    
    | 参数名           | 类型               | 默认值                                  | 作用 |
    |------------------|--------------------|------------------------------------------|------|
    | `model`          | `nn.Module`        | 必须                                     | 要分析的 PyTorch 模型 |
    | `input_size`     | tuple / list       | `None`                                   | 输入张量的尺寸，如 `(1, 3, 224, 224)` |
    | `input_data`     | tensor / list      | `None`                                   | 用真实张量代替 `input_size`，适合复杂输入 |
    | `batch_dim`      | int                | `0`                                      | 指定 batch 的维度索引，通常是第 0 维 |
    | `col_names`      | tuple              | `("output_size", "num_params")`          | 控制输出列字段，常见值包括 `"input_size"`, `"output_size"`, `"num_params"`, `"mult_adds"` |
    | `col_width`      | int                | `25`                                     | 控制输出列的宽度，影响打印时对齐效果 |
    | `depth`          | int / None         | `3`                                      | 控制显示的模块嵌套层数，设为 `None` 显示全部 |
    | `device`         | `"cpu"` / `"cuda"` | `"cpu"`                                  | 指定运行模型的设备，用于计算内存和形状 |
    | `dtypes`         | list of `torch.dtype` | `None`                               | 指定每个输入张量的数据类型，如 `[torch.float32]` |
    | `verbose`        | int                | `1`                                      | 控制是否打印输出，设为 `0` 可静默 |
    | `row_settings`   | tuple              | `("var_names", "depth")`                 | 控制行显示格式，例如变量名、缩进风格 |
    
    ---
    
    ### ✅ 示例用法
    
    ```python
    from torchinfo import summary
    summary(
        model,
        input_size=(1, 3, 224, 224),
        col_names=("input_size", "output_size", "num_params"),
        depth=2,
        device="cuda",
        verbose=1
    )
    ```


## 二、TensorBoard

---

### **1. 为什么在 PyTorch 中使用 TensorBoard？**
- **功能需求**：TensorBoard 提供了丰富的可视化功能（如标量曲线、图像、模型结构、直方图、嵌入投影等），帮助开发者理解模型行为和调试训练过程。
- **PyTorch 支持**：PyTorch 通过 `torch.utils.tensorboard` 模块原生支持 TensorBoard，无需依赖 TensorFlow。
- **核心优势**：实时监控训练指标、对比不同实验、分析模型性能。

---

### **2. 安装与准备**
#### **安装依赖**
确保已安装 PyTorch 和 TensorBoard：
```bash
pip install torch torchvision tensorboard
```

#### **导入关键模块**
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter  # 核心写入器
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
```

---

### **3. 基础使用流程**
#### **3.1 创建 SummaryWriter**
`SummaryWriter` 是 PyTorch 中向 TensorBoard 写入数据的主要接口。
```python
# 创建写入器，指定日志保存目录
writer = SummaryWriter(log_dir="logs/experiment_1")
```

#### **3.2 记录标量（Scalars）**
记录训练损失、准确率等标量数据：
```python
for epoch in range(100):
    # 模拟训练过程
    train_loss = 0.1 * (0.9 ** epoch)
    val_accuracy = 0.8 + 0.1 * (1 - 0.9 ** epoch)
    
    # 记录标量到 TensorBoard
    writer.add_scalar('Loss/Train', train_loss, epoch)
    writer.add_scalar('Accuracy/Val', val_accuracy, epoch)
```

#### **3.3 启动 TensorBoard**
在终端运行以下命令：
```bash
tensorboard --logdir=logs/experiment_1 --port=6006
```
访问 `http://localhost:6006` 查看可视化界面。

---

### **4. 核心功能详解**
#### **4.1 图像可视化**
记录输入数据、特征图或生成图像：
```python
# 示例：记录一批 MNIST 图像
transform = transforms.Compose([transforms.ToTensor()])
train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=8, shuffle=True)

# 获取一批图像
images, labels = next(iter(train_loader))
grid = torchvision.utils.make_grid(images)  # 将图像拼接为网格

# 记录图像
writer.add_image('MNIST Samples', grid, 0)
```

#### **4.2 模型结构可视化**
可视化神经网络的计算图：
```python
# 定义简单模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = Net()

# 记录模型结构
dummy_input = torch.rand(1, 1, 28, 28)  # 虚拟输入（与 MNIST 图像尺寸匹配）
writer.add_graph(model, dummy_input)
```

#### **4.3 直方图与分布**
监控权重或激活值的分布变化：
```python
# 在训练过程中记录权重
for name, param in model.named_parameters():
    writer.add_histogram(f'Weights/{name}', param, epoch)
    writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
```

#### **4.4 PR 曲线与 ROC 曲线**
记录分类模型的性能曲线：
```python
from sklearn.metrics import precision_recall_curve, roc_curve
import numpy as np

# 模拟预测概率和真实标签
y_true = np.random.randint(0, 2, size=100)
y_scores = np.random.rand(100)

# 计算 PR 曲线
precision, recall, _ = precision_recall_curve(y_true, y_scores)
writer.add_pr_curve('PR Curve', y_true, y_scores, 0)

# 计算 ROC 曲线
fpr, tpr, _ = roc_curve(y_true, y_scores)
writer.add_roc_curve('ROC Curve', y_true, y_scores, 0)
```

#### **4.5 嵌入投影（Embedding Projector）**
可视化高维数据（如特征向量）：
```python
# 生成特征和标签
features = torch.randn(100, 256)  # 100个样本，256维特征
labels = torch.randint(0, 10, (100,))

# 记录嵌入向量
writer.add_embedding(
    features,
    metadata=labels,
    tag='MNIST Embeddings',
    global_step=epoch
)
```

---

### **5. 高级功能**
#### **5.1 对比多个实验**
通过不同日志目录区分实验：
```python
# 实验1
writer_exp1 = SummaryWriter(log_dir="logs/exp1_lr0.01")
# 实验2
writer_exp2 = SummaryWriter(log_dir="logs/exp2_lr0.001")
```
启动 TensorBoard 时指定父目录：
```bash
tensorboard --logdir=logs --port=6006
```

#### **5.2 超参数调优（HParams）**
使用 TensorBoard 的 HParams 插件对比超参数：
```python
# 记录超参数和指标
with SummaryWriter(log_dir="logs/hparams") as w:
    # 定义超参数
    hparams = {'lr': 0.01, 'batch_size': 64}
    metrics = {'accuracy': 0.85}
    
    # 写入 HParams
    w.add_hparams(hparams, metrics)
```

#### **5.3 性能分析（Profiler）**
使用 PyTorch Profiler 分析模型性能：
```python
from torch.profiler import profile, record_function, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    with record_function("model_inference"):
        model(dummy_input)

# 记录分析结果到 TensorBoard
writer.add_profiler_trace(prof)
```

---

### **6. 完整训练示例**
```python
# 完整训练流程示例
model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    for i, (images, labels) in enumerate(train_loader):
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 记录损失
        writer.add_scalar('Loss/Train', loss.item(), epoch * len(train_loader) + i)
    
    # 记录模型权重
    for name, param in model.named_parameters():
        writer.add_histogram(f'Weights/{name}', param, epoch)
        writer.add_histogram(f'Gradients/{name}', param.grad, epoch)

writer.close()  # 关闭写入器
```

---

### **7. 常见问题**
1. **TensorBoard 不显示数据**：
      - 检查日志路径是否与 `SummaryWriter` 指定的路径一致。
      - 确保 `writer.close()` 被调用或使用 `with` 语句自动关闭。
2. **多 GPU 训练**：
      - 在分布式训练中，确保只在主进程记录数据。
3. **远程访问**：
      - 使用 `tensorboard --logdir=logs --host 0.0.0.0` 允许远程访问。

---

### **8. 总结**

通过 `SummaryWriter`，PyTorch 可以无缝集成 TensorBoard，实现以下功能：

- **训练监控**：实时跟踪损失、准确率等指标。
- **模型调试**：可视化权重分布、梯度流动。
- **数据检查**：查看输入数据、特征图。
- **性能优化**：分析计算瓶颈和超参数影响。

掌握这些操作后，您可以更高效地开发和优化 PyTorch 模型！


## 三、PR曲线和ROC曲线

---

### **1. PR 曲线（Precision-Recall Curve）**
#### **1.1 核心概念**
- **精确率（Precision）**：  
  **预测为正类的样本中，真实为正类的比例**。
  $$
  \text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}
  $$
- **召回率（Recall）**：  
  **真实为正类的样本中，被正确预测为正类的比例**。  
  $$
  \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}
  $$

#### **1.2 PR 曲线的绘制**
1. **阈值调整**：  
   分类模型输出概率（如逻辑回归的预测概率），通过调整分类阈值（如从 1 到 0），计算不同阈值下的 Precision 和 Recall。
2. **连接点**：  
   将所有（Recall, Precision）点按阈值从高到低连接，形成 PR 曲线。

#### **1.3 应用场景**
- **类别不平衡问题**：当负样本远多于正样本时，PR 曲线比 ROC 曲线更敏感。
- **关注正类精度**：例如医疗诊断（减少误诊）、欺诈检测（减少漏检）。

#### **1.4 代码示例**
```python
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

# 模拟数据（真实标签和预测概率）
y_true = [1, 0, 1, 1, 0, 1, 0, 1]    # 真实标签
y_scores = [0.8, 0.3, 0.6, 0.7, 0.4, 0.9, 0.2, 0.5]  # 预测概率

# 计算 PR 曲线
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

# 绘制 PR 曲线
plt.figure()
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR Curve')
plt.show()
```

---

### **2. ROC 曲线（Receiver Operating Characteristic Curve）**
#### **2.1 核心概念**
- **真正例率（TPR, True Positive Rate）**：  
  等同召回率（Recall）。  
  $$
  \text{TPR} = \frac{\text{TP}}{\text{TP} + \text{FN}}
  $$
- **假正例率（FPR, False Positive Rate）**：  
  **真实为负类的样本中，被错误预测为正类的比例**。  
  $$
  \text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}
  $$

#### **2.2 ROC 曲线的绘制**
1. **阈值调整**：  
   调整分类阈值（从 1 到 0），计算不同阈值下的 TPR 和 FPR。
2. **连接点**：  
   将所有（FPR, TPR）点按阈值从高到低连接，形成 ROC 曲线。

#### **2.3 应用场景**
- **平衡分类任务**：当正负样本数量接近时，ROC 曲线更直观。
- **关注整体性能**：例如广告点击率预测、信用评分。

#### **2.4 代码示例**
```python
from sklearn.metrics import roc_curve, roc_auc_score

# 计算 ROC 曲线
fpr, tpr, thresholds = roc_curve(y_true, y_scores)

# 计算 AUC 值（曲线下面积）
auc = roc_auc_score(y_true, y_scores)

# 绘制 ROC 曲线
plt.figure()
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # 绘制对角线
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
```

---

### **3. PR 曲线 vs. ROC 曲线**
| **特性**               | **PR 曲线**                     | **ROC 曲线**                    |
|------------------------|--------------------------------|---------------------------------|
| **关注点**             | 正类的精度和召回率              | 整体分类性能（TPR 和 FPR）       |
| **类别不平衡敏感度**   | 高（负样本多时更敏感）          | 低（受负样本数量影响较小）        |
| **AUC 意义**           | 高 AUC 表示高 Precision 和 Recall | 高 AUC 表示高 TPR 和低 FPR       |
| **适用场景**           | 类别高度不平衡（如欺诈检测）     | 类别平衡或关注整体性能（如广告推荐） |

---

### **4. 关键指标：AUC**
- **AUC-PR**：PR 曲线下的面积，值越接近 1 越好。  
- **AUC-ROC**：ROC 曲线下的面积，值越接近 1 越好。  
  **AUC-ROC > 0.5** 表示模型优于随机猜测。

---

### **5. 实际应用中的选择**
#### **5.1 如何选择 PR 或 ROC？**
- **类别不平衡**：优先使用 PR 曲线（如正样本占比 < 20%）。
- **类别平衡**：优先使用 ROC 曲线。

#### **5.2 阈值选择**
- **PR 曲线**：选择 Precision 和 Recall 的平衡点（如 F1 分数最大）。
- **ROC 曲线**：选择 TPR 高且 FPR 低的阈值（靠近左上角）。

---

### **6. 在 PyTorch 中记录 PR/ROC 到 TensorBoard**
```python
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_recall_curve, roc_curve
import numpy as np

writer = SummaryWriter()

# 模拟模型输出和真实标签
y_true = np.random.randint(0, 2, size=100)       # 真实标签
y_scores = np.random.rand(100)                   # 预测概率

# 记录 PR 曲线
precision, recall, _ = precision_recall_curve(y_true, y_scores)
writer.add_pr_curve('PR Curve', y_true, y_scores, 0)

# 记录 ROC 曲线
fpr, tpr, _ = roc_curve(y_true, y_scores)
writer.add_roc_curve('ROC Curve', y_true, y_scores, 0)

writer.close()
```

---

### **7. 总结**
- **PR 曲线**：聚焦正类的精度和召回率，适合类别不平衡任务。  
- **ROC 曲线**：反映模型整体分类能力，适合平衡任务。  
- **AUC 值**：量化曲线下面积，用于模型性能对比。  
- **实际应用**：根据数据分布和业务需求选择合适指标。