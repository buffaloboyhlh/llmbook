# Trainer API

以下是 **Hugging Face Transformers 库中 `Trainer API` 的详细用法指南**，帮助你快速掌握模型训练与微调的核心流程：

---

### **1. Trainer API 的优势**

- **自动化训练流程**：封装训练循环、梯度下降、评估和保存模型
- **分布式训练支持**：开箱即用支持多GPU/TPU训练
- **灵活的参数配置**：通过 `TrainingArguments` 控制超参数
- **内置评估与日志**：支持 TensorBoard、W&B 等日志工具

---

### **2. 基础使用流程**

#### **步骤 1：安装依赖**
```bash
pip install transformers datasets torch accelerate
```

#### **步骤 2：准备数据集**
使用 `datasets` 库加载数据：
```python
from datasets import load_dataset

dataset = load_dataset("imdb")  # 示例：IMDB 电影评论数据集
print(dataset["train"][0])  # 查看样本结构
```

#### **步骤 3：数据预处理**
使用 Tokenizer 转换文本：
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def preprocess_function(examples):
    return tokenizer(
        examples["text"], 
        padding="max_length", 
        truncation=True, 
        max_length=512
    )

tokenized_dataset = dataset.map(preprocess_function, batched=True)
```

#### **步骤 4：定义训练参数**
```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",          # 输出目录
    evaluation_strategy="epoch",     # 每 epoch 评估一次
    learning_rate=2e-5,
    per_device_train_batch_size=8,   # 每个设备的批次大小
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",            # 日志目录
    report_to="tensorboard",         # 使用 TensorBoard
    save_strategy="epoch",           # 每 epoch 保存模型
    load_best_model_at_end=True,     # 训练结束时加载最佳模型
)
```

#### **步骤 5：定义模型**
```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", 
    num_labels=2  # 分类标签数
)
```

#### **步骤 6：定义评估指标**
```python
import numpy as np
from datasets import load_metric

metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
```

#### **步骤 7：创建 Trainer**
```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    compute_metrics=compute_metrics,
)
```

#### **步骤 8：开始训练**
```python
trainer.train()  # 启动训练
trainer.evaluate()  # 最终评估
```

---

### **3. 核心功能详解**

#### **(1) 自定义训练行为**
通过覆盖 `Trainer` 方法实现：
```python
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = torch.nn.CrossEntropyLoss()(logits, labels)
        return (loss, outputs) if return_outputs else loss
```

#### **(2) 使用回调函数**
内置回调示例：
```python
from transformers import EarlyStoppingCallback

# 添加早停回调
trainer.add_callback(EarlyStoppingCallback(
    early_stopping_patience=3  # 连续3次评估无提升则停止
))
```

#### **(3) 分布式训练**
自动支持多GPU/TPU：
```bash
# 启动命令（需安装 accelerate）
accelerate launch --num_processes 4 train.py
```

#### **(4) 混合精度训练**
在 `TrainingArguments` 中启用：
```python
training_args = TrainingArguments(
    fp16=True,  # 使用 FP16 混合精度（NVIDIA GPU）
    bf16=True   # 使用 BF16 混合精度（AMD/TPU）
)
```

---

### **4. 模型保存与加载**

#### **(1) 保存最佳模型**
```python
# 自动根据评估指标保存最佳模型（需设置 load_best_model_at_end=True）
trainer.save_model("./best_model")
```

#### **(2) 从检查点恢复训练**
```python
trainer.train(resume_from_checkpoint=True)  # 自动加载最新检查点
```

---

### **5. 可视化训练过程**
使用 TensorBoard 查看日志：
```bash
tensorboard --logdir ./logs
```

---

### **6. 完整代码示例（文本分类）**
```python
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset
import numpy as np
from datasets import load_metric

# 加载数据集
dataset = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 数据预处理
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 定义模型
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# 定义评估指标
metric = load_metric("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# 配置训练参数
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# 创建 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    compute_metrics=compute_metrics,
)

# 开始训练
trainer.train()
```

---

### **7. 常见问题解决**

| 问题现象 | 解决方案 |
|---------|----------|
| **CUDA内存不足** | 减小 `per_device_batch_size` 或使用梯度累积 (`gradient_accumulation_steps`) |
| **训练速度慢** | 启用混合精度 (`fp16=True`)、使用更小模型或升级硬件 |
| **评估指标异常** | 检查 `compute_metrics` 函数和数据标签对齐 |
| **无法恢复检查点** | 确保检查点路径包含 `checkpoint-xxx` 文件夹 |

---

### **8. 进阶技巧**
1. **参数高效微调 (PEFT)**  
   使用 `peft` 库实现 LoRA 等高效微调方法：
   ```python
   from peft import LoraConfig, get_peft_model

   peft_config = LoraConfig(
       r=8,
       lora_alpha=16,
       target_modules=["query", "value"],
       lora_dropout=0.1,
       bias="none"
   )
   model = get_peft_model(model, peft_config)
   ```

2. **自定义数据整理器**  
   继承 `DataCollator` 实现特殊数据处理：
   ```python
   from transformers import DataCollatorWithPadding

   data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
   ```

3. **与 Hugging Face Hub 集成**  
   训练后直接上传模型：
   ```python
   trainer.push_to_hub("my-awesome-model")
   ```

---

### **9. 关键参数说明**

| 参数 | 作用 | 示例值 |
|------|------|--------|
| `learning_rate` | 初始学习率 | `2e-5`, `5e-4` |
| `per_device_train_batch_size` | 单设备批次大小 | `8`, `16` |
| `gradient_accumulation_steps` | 梯度累积步数 | `2`, `4` |
| `warmup_steps` | 学习率预热步数 | `500`, `1000` |
| `logging_steps` | 日志记录间隔 | `50`, `100` |
| `eval_steps` | 评估间隔（当 `strategy="steps"`） | `500` |

---

通过 `Trainer API`，你可以快速实现从数据准备到模型部署的完整训练流程。对于需要更细粒度控制的场景，可以结合自定义训练循环（使用 `torch.utils.data.DataLoader`）与 `Trainer` 的模块化组件。


### **10. TrainingArguments参数说明**

---

##### 一、核心参数分类说明

###### 1. **输出与保存路径**
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `output_dir` | str | **必填** | 模型、日志和检查点的保存目录 |
| `logging_dir` | str | `None` | TensorBoard 日志目录（默认与 `output_dir` 一致） |
| `report_to` | str/list | `"all"` | 日志报告方式（`"tensorboard"`, `"wandb"`, `"none"` 等） |
| `save_total_limit` | int | `None` | 最多保留的检查点数量（超出则删除旧检查点） |

---

###### 2. **训练超参数**
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `num_train_epochs` | float | `3.0` | 训练的总轮数（可设为小数进行部分 epoch 训练） |
| `per_device_train_batch_size` | int | `8` | **每个设备（GPU/CPU）的训练批次大小** |
| `per_device_eval_batch_size` | int | `8` | 每个设备的评估批次大小 |
| `learning_rate` | float | `5e-5` | 初始学习率（AdamW 优化器的基准值） |
| `weight_decay` | float | `0.0` | 权重衰减（L2 正则化强度） |
| `gradient_accumulation_steps` | int | `1` | **梯度累积步数**（模拟更大批次，解决显存不足问题） |

---

###### 3. **优化器与调度器**
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `optim` | str | `"adamw_torch"` | 优化器类型（`"adamw_torch"`, `"adamw_apex_fused"`, `"sgd"` 等） |
| `lr_scheduler_type` | str | `"linear"` | 学习率调度器类型（`"linear"`, `"cosine"`, `"constant"` 等） |
| `warmup_ratio` | float | `0.0` | 学习率 warmup 占总训练步数的比例 |
| `warmup_steps` | int | `0` | 直接指定 warmup 步数（优先级高于 `warmup_ratio`） |

---

###### 4. **评估与保存策略**
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `evaluation_strategy` | str | `"no"` | 评估策略（`"no"`, `"steps"`, `"epoch"`） |
| `eval_steps` | int | `None` | 当 `evaluation_strategy="steps"` 时，每多少步评估一次 |
| `save_strategy` | str | `"steps"` | 模型保存策略（`"steps"`, `"epoch"`） |
| `save_steps` | int | `500` | 当 `save_strategy="steps"` 时，每多少步保存一次 |
| `load_best_model_at_end` | bool | `False` | 训练结束后加载最佳模型（需配合 `metric_for_best_model`） |
| `metric_for_best_model` | str | `None` | 用于选择最佳模型的指标（如 `"eval_loss"` 或自定义指标） |

---

###### 5. **硬件与性能优化**
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `fp16` / `bf16` | bool | `False` | 启用混合精度训练（`fp16` 适用于 NVIDIA GPU，`bf16` 适用于 A100+） |
| `no_cuda` | bool | `False` | 禁用 CUDA（强制使用 CPU 训练） |
| `ddp_find_unused_parameters` | bool | `True` | 分布式训练时检测未使用的参数（解决 DDP 报错问题） |
| `dataloader_num_workers` | int | `0` | 数据加载的进程数（建议设为 CPU 核心数，加速数据读取） |
| `gradient_checkpointing` | bool | `False` | **梯度检查点技术**（用时间换显存，适合大模型训练） |

---

###### 6. **日志与调试**
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `logging_strategy` | str | `"steps"` | 日志记录策略（`"steps"`, `"epoch"`, `"no"`） |
| `logging_steps` | int | `500` | 每多少步记录一次日志 |
| `logging_first_step` | bool | `False` | 是否记录第一步的日志 |
| `debug` | str | `""` | 调试模式（`"underflow_overflow"` 检测梯度异常） |

---

##### 二、关键参数详解与实战技巧

###### 1. **批次大小与梯度累积**
- **公式**：实际总批次大小 = `per_device_train_batch_size` × `gradient_accumulation_steps` × GPU 数量
- **显存不足**：若遇到 OOM（内存不足），可减小 `per_device_train_batch_size` 并增大 `gradient_accumulation_steps`。

```python
# 示例：在单卡 24GB 显存下训练大型模型
per_device_train_batch_size = 4
gradient_accumulation_steps = 8  # 等效于批次大小 4×8=32
```

---

###### 2. **学习率与调度器**
- **Warmup**：避免初始学习率过大导致震荡。推荐在前 5%~10% 的训练步数中线性增加学习率。
- **调度器选择**：
  - `"linear"`：线性衰减（BERT 的默认选择）
  - `"cosine"`：余弦退火（适合微调任务）
  - `"constant"`：恒定学习率（调试时使用）

```python
learning_rate = 2e-5
warmup_ratio = 0.1  # 10% 的训练步数用于 warmup
lr_scheduler_type = "cosine"
```

---

###### 3. **混合精度训练**
- **`fp16`**：在 NVIDIA GPU 上使用，需安装 `apex` 或 `amp` 库。
- **`bf16`**：在 Ampere 架构（如 A100）上效果更好，支持更广的数值范围。

```python
fp16 = True  # 启用混合精度
```

---

###### 4. **模型保存与最佳模型加载**
- **自动保存最佳模型**：设置 `load_best_model_at_end=True` 和 `metric_for_best_model`。
- **多指标选择**：若需根据验证集准确率保存模型：

```python
evaluation_strategy = "steps"
eval_steps = 500
metric_for_best_model = "eval_accuracy"  # 假设 compute_metrics 返回该字段
load_best_model_at_end = True
```

---

###### 5. **分布式训练配置**
- **多 GPU 训练**：使用 `torchrun` 或 `accelerate` 库启动，无需修改代码。
- **参数优化**：
  ```python
  ddp_find_unused_parameters = False  # 若模型存在未使用的参数（如某些条件分支），需设为 True
  ```

---

##### 三、完整示例代码

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5.0,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=2,  # 总批次大小 = 16×2=32（单卡）
    learning_rate=3e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1",
    fp16=True,
    dataloader_num_workers=4,
    logging_steps=100,
    report_to="wandb",  # 集成 Weights & Biases
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
```

---

##### 四、常见问题与调试

###### 1. **显存不足（OOM）**
- 降低 `per_device_train_batch_size`
- 启用 `gradient_checkpointing=True`
- 启用 `fp16`/`bf16`
- 增加 `gradient_accumulation_steps`

###### 2. **训练速度慢**
- 增加 `dataloader_num_workers`
- 确保 `pin_memory=True`（默认启用）
- 检查是否启用混合精度训练

###### 3. **评估指标不更新**
- 确认 `evaluation_strategy` 不是 `"no"`
- 检查 `compute_metrics` 函数是否正确返回指标

---

通过合理配置 `TrainingArguments`，您可以精确控制训练流程的每个细节，从基础训练到分布式、混合精度等高级场景均可覆盖。