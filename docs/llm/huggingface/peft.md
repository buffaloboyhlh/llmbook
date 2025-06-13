# PEFT 教程 

以下是 **Hugging Face `peft` 库（Parameter-Efficient Fine-Tuning）的详细用法指南**，帮助你高效微调大模型（如LLaMA、GPT等）而无需更新全部参数：

---

### **1. PEFT 的核心思想**
- **参数高效微调**：仅更新模型的一小部分参数（如适配器层、LoRA权重），显著减少显存占用。
- **支持多种技术**：
  - **LoRA**（Low-Rank Adaptation）: 通过低秩矩阵分解注入可训练参数
  - **Prefix Tuning**：在输入前添加可学习的提示向量
  - **Adapter**：在Transformer层中插入小型适配模块
  - **IA3**：通过缩放激活调整模型

---

### **2. 安装与依赖**
```bash
pip install peft transformers accelerate torch
```

---

### **3. 核心模块**
- `LoraConfig`: 配置 LoRA 微调参数
- `get_peft_model()`: 将基础模型转换为 PEFT 模型
- `PeftModel`: 封装后的 PEFT 模型类
- `PeftModelForCausalLM`: 支持生成任务的 PEFT 模型

---

### **4. LoRA 微调完整流程（以文本生成为例）**

#### **步骤 1：加载基础模型和分词器**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "facebook/opt-125m"  # 示例模型
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

#### **步骤 2：定义 LoRA 配置**
```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,                  # 低秩矩阵的维度
    lora_alpha=32,        # 缩放因子（类似学习率）
    target_modules=["q_proj", "v_proj"],  # 要注入LoRA的模块名
    lora_dropout=0.05,    # Dropout概率
    bias="none",          # 是否训练偏置参数
    task_type="CAUSAL_LM" # 任务类型（因果语言模型）
)
```

#### **步骤 3：创建 PEFT 模型**
```python
peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()  # 查看可训练参数占比
# 输出示例: trainable params: 0.8M || all params: 125M || trainable%: 0.64%
```

#### **步骤 4：准备数据集**
```python
from datasets import load_dataset

dataset = load_dataset("imdb")  # 示例数据集
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=256)

tokenized_dataset = dataset.map(tokenize_function, batched=True)
```

#### **步骤 5：配置训练参数**
```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./lora_outputs",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    num_train_epochs=3,
    fp16=True,  # 混合精度训练
    logging_steps=10,
    save_strategy="epoch"
)
```

#### **步骤 6：创建 Trainer 并训练**
```python
from transformers import Trainer

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=lambda data: {"input_ids": torch.stack([f["input_ids"] for f in data])}
)

trainer.train()
```

#### **步骤 7：保存与加载适配器**
```python
# 保存适配器权重
peft_model.save_pretrained("./lora_adapter")

# 加载适配器到基础模型
from peft import PeftModel

loaded_model = PeftModel.from_pretrained(
    base_model,  # 原始基础模型
    "./lora_adapter"
)

# 合并适配器到基础模型（可选）
merged_model = loaded_model.merge_and_unload()
```

---

### **5. 关键参数详解（以 `LoraConfig` 为例）**
| 参数 | 作用 | 推荐值 |
|------|------|--------|
| `r` | 低秩矩阵的秩 | 4-64（越大能力越强，但显存占用增加） |
| `target_modules` | 要注入LoRA的模块名称 | 根据模型结构选择（见下表） |
| `lora_alpha` | 缩放因子（控制LoRA权重的影响力） | 通常设为 `2*r` |
| `lora_dropout` | 防止过拟合的Dropout率 | 0.05-0.2 |

#### **常见模型的 `target_modules` 选择**
| 模型类型 | 可选的模块名称 |
|---------|----------------|
| LLaMA   | `q_proj`, `v_proj` |
| GPT-2    | `c_attn` |
| BERT     | `query`, `value` |
| RoBERTa  | `query`, `value` |

---

### **6. 其他 PEFT 方法示例**

#### **(1) Prefix Tuning**
```python
from peft import PrefixTuningConfig

config = PrefixTuningConfig(
    task_type="CAUSAL_LM",
    num_virtual_tokens=10,  # 前缀token数量
    encoder_hidden_size=512  # 前缀编码维度
)
model = get_peft_model(model, config)
```

#### **(2) Adapter**
```python
from peft import AdaptionPromptConfig

config = AdaptionPromptConfig(
    adapter_layers=2,  # 适配器层数
    adapter_len=4,     # 适配器长度
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, config)
```

---

### **7. 进阶技巧**

#### **(1) 多任务适配器组合**
```python
# 加载多个适配器
model = PeftModel.from_pretrained(base_model, "adapter1")
model.load_adapter("adapter2", adapter_name="task2")

# 切换适配器
model.set_adapter("task2")
```

#### **(2) 8-bit 量化训练**
```python
from transformers import BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # 4-bit量化
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config
)
```

#### **(3) 与 Trainer API 深度集成**
```python
# 直接使用 Hugging Face Trainer
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=dataset,
    compute_metrics=compute_metrics
)
```

---

### **8. 常见问题解决**

| 问题现象 | 解决方案 |
|---------|----------|
| **显存不足** | 减小 `batch_size`，启用4-bit量化 (`BitsAndBytesConfig`) |
| **`KeyError: 'lora'`** | 检查 `target_modules` 名称是否与模型层匹配 |
| **训练损失不下降** | 增大 `r` 或 `lora_alpha`，检查数据质量 |
| **加载适配器报错** | 确保基础模型结构与适配器训练时一致 |

---

### **9. 最佳实践**
1. **从小开始**：初始使用 `r=8`, `alpha=16`，逐步调大
2. **模块选择**：优先在 `query` 和 `value` 投影层添加LoRA
3. **混合精度**：始终启用 `fp16=True` 或 `bf16=True`
4. **模型保存**：同时保存适配器权重和基础模型配置

---

通过 `peft` 库，你可以用消费级GPU微调数十亿参数的大模型。结合量化技术（如QLoRA），甚至能在单卡上微调Llama 2-70B等超大规模模型！