以下是 **Hugging Face `datasets` 库** 的详细使用教程，涵盖数据加载、处理、流式加载及实战示例。

---

## 1. 安装与基础用法

### 安装库
```bash
pip install datasets
```

### 加载内置数据集
```python
from datasets import load_dataset

# 加载常见数据集（如 GLUE 中的 MNLI）
dataset = load_dataset("glue", "mnli")  # 返回 DatasetDict 对象

# 查看数据集结构
print(dataset)  # 输出数据集划分（train/validation/test）
print(dataset["train"][0])  # 查看第一条训练数据
```

---

## 2. 数据集操作

### 访问数据
```python
# 获取训练集、验证集、测试集
train_data = dataset["train"]
val_data = dataset["validation_matched"]  # MNLI 的特殊验证集

# 访问单条数据
print(train_data[0])

# 批量访问
for batch in train_data.select(range(10)):  # 前10条
    print(batch["premise"], batch["hypothesis"], batch["label"])
```

### 数据集划分
```python
# 分割训练集为 train 和 test（自定义比例）
split_dataset = dataset["train"].train_test_split(test_size=0.1)
print(split_dataset)  # 包含 train 和 test 子集
```

---

## 3. 处理自定义数据

### 加载本地文件
支持 CSV、JSON、文本等格式：
```python
# 加载 CSV 文件
custom_dataset = load_dataset("csv", data_files={"train": "data/train.csv", "test": "data/test.csv"})

# 加载 JSON
custom_dataset = load_dataset("json", data_files="data/*.json")

# 加载文本（每行一个样本）
text_dataset = load_dataset("text", data_files="data/texts.txt")
```

### 自定义数据生成器
```python
from datasets import Dataset

# 从 Python 字典或列表创建
data = {"text": ["Hello!", "How are you?"], "label": [0, 1]}
custom_ds = Dataset.from_dict(data)

# 从生成器创建（适合大型数据）
def data_generator():
    for i in range(1000):
        yield {"text": f"Sample {i}", "label": i % 2}

streaming_ds = Dataset.from_generator(data_generator)
```

---

## 4. 数据预处理

### 使用 `map` 函数
```python
# 示例：分词处理
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(
        examples["premise"],
        examples["hypothesis"],
        truncation=True,
        max_length=128,
        padding="max_length"
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)
```

### 过滤数据
```python
# 过滤长度过短的文本
filtered_dataset = dataset.filter(
    lambda example: len(example["premise"].split()) > 5
)
```

---

## 5. 流式模式处理大数据
对于超大数据集（如 >100GB），使用流式加载避免内存溢出：
```python
# 流式加载（逐条读取）
stream_dataset = load_dataset("c4", "en", streaming=True)
for example in iter(stream_dataset["train"]):  # 逐个样本加载
    print(example["text"])
    break  # 示例仅读取第一条

# 批量处理（流式 + map）
tokenized_stream = stream_dataset.map(tokenize_function, batched=True)
```

---

## 6. 转换为 PyTorch/TensorFlow 格式

### PyTorch DataLoader
```python
from torch.utils.data import DataLoader

# 设置数据格式为 PyTorch Tensor
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# 创建 DataLoader
dataloader = DataLoader(tokenized_dataset["train"], batch_size=32, shuffle=True)
```

### TensorFlow
```python
tf_dataset = tokenized_dataset["train"].to_tf_dataset(
    columns=["input_ids", "attention_mask"],
    label_cols=["label"],
    batch_size=32,
    shuffle=True
)
```

---

## 7. 实战示例：文本分类

### 完整流程
```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer

# 加载数据集
dataset = load_dataset("imdb")

# 分词处理
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
def tokenize(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length")

dataset = dataset.map(tokenize, batched=True)

# 转换为 PyTorch 格式
dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# 定义模型
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# 使用 Trainer 训练
trainer = Trainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
)
trainer.train()
```

---

## 8. 高级功能

### 数据集缓存
- 默认缓存路径：`~/.cache/huggingface/datasets`
- 自定义缓存路径：
  ```python
  dataset = load_dataset("glue", "mnli", cache_dir="./my_cache")
  ```

### 性能优化
- **批量处理**：`map(..., batched=True)` 比逐条处理快 10 倍以上。
- **多进程**：
  ```python
  dataset.map(..., num_proc=4)  # 使用 4 个进程
  ```

### 数据集可视化
```python
# 查看特征结构
print(dataset["train"].features)

# 生成统计摘要
import pandas as pd
df = pd.DataFrame(dataset["train"])
print(df.describe())
```

---

通过 `datasets` 库，你可以高效管理 NLP、CV、语音等任务的数据，结合 `transformers` 库实现端到端模型训练。