# AutoModel和AutoTokenizer

---

### **1. 核心概念**
- **`AutoModel`**: 根据模型名称自动加载预训练模型的基类（如BERT、GPT-2等）。
- **`Tokenizer`**: 负责将文本转换为模型可接受的数值输入（如分词、添加特殊标记等）。

---

### **2. 基本用法**

#### **步骤 1：安装库**
```bash
pip install transformers torch
```

#### **步骤 2：加载模型和分词器**
```python
from transformers import AutoModel, AutoTokenizer

# 指定模型名称（Hugging Face Hub 中的名称）
model_name = "bert-base-uncased"

# 加载预训练的分词器和模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
```

---

### **3. 文本预处理（Tokenizer 的用法）**

#### **示例 1：基础分词**
```python
text = "Hugging Face Transformers is amazing!"

# 分词并转换为模型输入格式
inputs = tokenizer(text, return_tensors="pt")  # 返回PyTorch张量

print(inputs)
# 输出:
# {'input_ids': tensor([[ 101, 17662,  1437, 11303,  ... ]]),
#  'attention_mask': tensor([[1, 1, 1, ..., 1]])}
```

#### **重要参数**
- `padding=True`: 自动填充到批次内最长长度
- `truncation=True`: 截断超过模型最大长度的文本
- `max_length=512`: 指定最大长度
- `return_tensors="pt"`: 返回PyTorch张量（可选 `"tf"` 为TensorFlow）

#### **示例 2：批量处理**
```python
texts = ["Hello, world!", "How are you?"]
inputs = tokenizer(
    texts, 
    padding=True, 
    truncation=True, 
    return_tensors="pt"
)
```

---

### **4. 模型推理（AutoModel 的用法）**

#### **示例 1：获取隐藏状态**
```python
# 将输入传递给模型
outputs = model(**inputs)

# 提取最后一层隐藏状态
last_hidden_states = outputs.last_hidden_state
print(last_hidden_states.shape)  # (batch_size, sequence_length, hidden_dim)
```

#### **示例 2：特定任务模型**
针对不同任务需使用对应的 `AutoModelForXXX` 类：
```python
from transformers import AutoModelForSequenceClassification

# 加载文本分类模型
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

# 推理
inputs = tokenizer("I love this movie!", return_tensors="pt")
outputs = model(**inputs)

# 输出logits
logits = outputs.logits
print(logits)  # tensor([[ 4.1234, -3.4567]])
```

---

### **5. 进阶用法**

#### **自定义输入格式**
```python
# 手动添加特殊标记（如BERT的[CLS]和[SEP]）
text = "[CLS] " + text + " [SEP]"
tokens = tokenizer.tokenize(text)
```

#### **保存与加载本地模型**
```python
# 保存到本地
model.save_pretrained("./my_model")
tokenizer.save_pretrained("./my_model")

# 从本地加载
model = AutoModel.from_pretrained("./my_model")
tokenizer = AutoTokenizer.from_pretrained("./my_model")
```

#### **GPU加速**
```python
model = model.to("cuda")  # 将模型移动到GPU
inputs = {k: v.to("cuda") for k, v in inputs.items()}  # 输入数据移动到GPU
```

---

### **6. 完整流程示例（情感分析）**
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# 加载模型和分词器
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# 输入文本
text = "Hugging Face makes NLP easy and fun!"

# 预处理
inputs = tokenizer(text, return_tensors="pt")

# 推理
with torch.no_grad():
    outputs = model(**inputs)

# 后处理
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
labels = ["NEGATIVE", "POSITIVE"]
print(f"Prediction: {labels[predictions.argmax()]} ({predictions.max().item():.2f})")
# 输出: Prediction: POSITIVE (0.99)
```

---

### **7. 关键注意事项**

1. **模型与任务匹配**：确保使用正确的 `AutoModelForXXX` 类（如 `AutoModelForQuestionAnswering` 用于问答任务）。
2. **输入格式**：不同模型可能需要不同的特殊标记（如BERT需要 `[CLS]` 和 `[SEP]`，但Tokenizer会自动处理）。
3. **内存管理**：大模型可能需要 `from_pretrained(..., device_map="auto")` 进行分布式加载。
4. **微调训练**：在模型上添加自定义分类头并调用 `model.train()`。

---

### **8. 常见问题解决**

- **`CUDA out of memory`**: 减小批次大小或使用梯度累积。
- **`Input length exceeds max_length`**: 设置 `truncation=True` 或增大 `max_length`。
- **`No module named 'transformers'`**: 检查是否安装了正确的库版本。

---

通过掌握 `AutoModel` 和 `Tokenizer` 的用法，你可以灵活实现从文本预处理到模型推理的完整流程，并为后续的模型微调奠定基础。