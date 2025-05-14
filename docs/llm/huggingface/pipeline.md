# pipeline

---

### **1. 安装 Transformers 库**
确保安装以下依赖：
```bash
pip install transformers
pip install torch  # 推荐安装 PyTorch 作为后端
```

---

### **2. Pipeline 的基本用法**

`pipeline` 是 Hugging Face 提供的高级 API，封装了以下步骤：

1. **预处理**（文本分词、转换为张量）
2. **模型推理**（调用预训练模型）
3. **后处理**（将输出转换为可读结果）

#### **示例 1：快速使用预置任务**
```python
from transformers import pipeline

# 选择一个任务，例如文本分类
classifier = pipeline("text-classification")

# 输入文本
result = classifier("I love using Hugging Face Transformers!")
print(result)
# 输出示例: [{'label': 'POSITIVE', 'score': 0.9998}]
```

---

### **3. 支持的任务类型**

Hugging Face 提供多种预置任务，例如：

- **文本分类** (`text-classification`)
- **文本生成** (`text-generation`)
- **命名实体识别** (`ner`)
- **问答系统** (`question-answering`)
- **摘要生成** (`summarization`)
- **翻译** (`translation`)
- **情感分析** (`sentiment-analysis`)

---

### **4. 自定义模型与分词器**
可以指定模型名称或本地路径来自定义 Pipeline：

#### **示例 2：使用指定模型**
```python
from transformers import pipeline

# 指定模型名称（Hugging Face Hub 上的模型）
translator = pipeline(
    "translation_en_to_fr",
    model="Helsinki-NLP/opus-mt-en-fr"
)

text = "Hello, how are you?"
result = translator(text)
print(result)  # 输出: [{'translation_text': 'Bonjour, comment allez-vous?'}]
```

---

### **5. 多输入与批量处理**
传递列表进行批量推理以提升效率：
```python
ner = pipeline("ner", grouped_entities=True)
texts = [
    "My name is John and I work at Google in New York.",
    "Apple Inc. was founded by Steve Jobs in California."
]
results = ner(texts)
print(results)
```

---

### **6. 参数调整**
通过参数控制生成结果（适用于生成类任务）：
```python
generator = pipeline("text-generation", model="gpt2")
result = generator(
    "In the future, AI will",
    max_length=50,
    num_return_sequences=2,
    temperature=0.7
)
print(result)
```

---

### **7. 使用 GPU 加速**
若已安装 PyTorch 的 GPU 版本，可通过 `device` 参数指定设备：
```python
classifier = pipeline("text-classification", device=0)  # 使用第一个 GPU
```

---

### **8. 本地模型与分词器**
若需离线使用，可提前下载模型到本地：
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
```

---

### **9. 常见问题**
- **内存不足**：尝试使用较小的模型（如 `distilbert` 系列）或减少 `batch_size`。
- **任务支持**：查看[官方文档](https://huggingface.co/docs/transformers/main_classes/pipelines)获取完整任务列表。
- **自定义任务**：继承 `pipeline` 类实现自定义逻辑。

---

通过 `pipeline`，你可以快速实现 NLP 任务的原型验证。如需更精细控制（如自定义训练），可深入学习 `AutoModel` 和 `Tokenizer` 的用法。