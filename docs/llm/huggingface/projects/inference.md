# 中文句子关系推断

本章将使用神经网络判断两个句子是否是连续的关系，以人类的角度来讲，阅读两个句子，很容易就能判断出这两个句子是相连的，还是无关的
**0 是相连的 ； 1 是不相连的。**
---

### Step1 数据加载&预处理

```python
from datasets import load_dataset
from transformers import BertTokenizer, BertModel
import torch
import random
import warnings
warnings.filterwarnings("ignore")

model_name = 'bert-base-chinese'
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

dataset = load_dataset(path="src/seamew/ChnSentiCor")
dataset = dataset.filter(function=lambda x: len(x['text']) > 40)

tokenizer = BertTokenizer.from_pretrained(model_name, return_tensors='pt')
pretrained_model = BertModel.from_pretrained(model_name).to(device)


def process_fn(batch_data):
    texts = []
    labels = []

    for text in batch_data['text']:
        label = random.randint(0, 1)
        labels.append(label)

        sentence1 = text[:20]
        sentence2 = text[20:40]

        if label == 1:
            j = random.randint(0, len(dataset['train']) - 1)
            sentence2 = dataset['train'][j]['text'][20:40]

        texts.append((sentence1, sentence2))

    tokenized = tokenizer.batch_encode_plus(batch_text_or_text_pairs=texts, padding="max_length", return_tensors="pt",
                                            max_length=45,add_special_tokens=True)

    tokenized['label'] = labels
    return tokenized


tokenized_dataset = dataset.map(process_fn, batched=True, remove_columns=['text'])
tokenized_dataset
```
**输出**

```text
DatasetDict({
    train: Dataset({
        features: ['label', 'input_ids', 'token_type_ids', 'attention_mask'],
        num_rows: 8001
    })
    validation: Dataset({
        features: ['label', 'input_ids', 'token_type_ids', 'attention_mask'],
        num_rows: 1012
    })
    test: Dataset({
        features: ['label', 'input_ids', 'token_type_ids', 'attention_mask'],
        num_rows: 991
    })
})
```

### Step2 定义下游模型

```python
# 定义下游任务模型

class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc = torch.nn.Linear(in_features=768, out_features=2, bias=True)
        self.fc.bias.data.zero_()

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        with torch.no_grad():
            output = pretrained_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        logits = self.fc(output.last_hidden_state[:, 0])
        if labels is not None:
            loss = torch.nn.CrossEntropyLoss()(logits, labels)
            return {"loss": loss, "logits": logits}

        return {"logits": logits}


model = MyModel()
print(model)
```

### Step3 定义训练参数

```python
# 定义训练参数
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir='outputs',
    save_strategy='epoch',
    save_total_limit=1,
    num_train_epochs=5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    weight_decay=0.01,
    learning_rate=5e-5,
    lr_scheduler_type="linear",
    logging_strategy='epoch',
    logging_dir="logs",
    report_to="tensorboard",
)

print(training_args)
```

### Step4 定义训练器

```python
import evaluate

accuray_metic = evaluate.load("accuracy")


accuracy_metric = evaluate.load("accuracy")

def compute_metrics(pred):
    predictions = pred.predictions.argmax(axis=1)
    labels = pred.label_ids
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)  # ✅ 参数也加上关键词更清晰
    return {"accuracy": accuracy["accuracy"]}  # ✅ 这里返回的是字典中的 "accuracy" 值


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['validation'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
```

**输出**

```text
<IPython.core.display.HTML object>
TrainOutput(global_step=1250, training_loss=0.3362057861328125, metrics={'train_runtime': 210.7087, 'train_samples_per_second': 189.859, 'train_steps_per_second': 5.932, 'total_flos': 0.0, 'train_loss': 0.3362057861328125, 'epoch': 4.983508245877061})
```

### Step5 评估模型

```python
trainer.evaluate()
```

**输出**

```text
<IPython.core.display.HTML object>
{'eval_loss': 0.3010992109775543,
 'eval_accuracy': 0.8656126482213439,
 'eval_runtime': 6.1257,
 'eval_samples_per_second': 165.205,
 'eval_steps_per_second': 41.301,
 'epoch': 4.983508245877061}
```

### Step6 验证模型的输出

```python
batch_data = tokenized_dataset['test'].select(range(10))

tlabel = batch_data['label']
print("真实标签：", tlabel)

model = model.to(device)
model.eval()

input_ids = torch.LongTensor(batch_data['input_ids']).to(device)
attention_mask = torch.LongTensor(batch_data['attention_mask']).to(device)
token_type_ids = torch.LongTensor(batch_data['token_type_ids']).to(device)

outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
logits = torch.softmax(outputs['logits'], dim=1)
result = torch.argmax(logits, dim=1)
print("预测标签：", result.tolist())
```

**输出**

```text
真实标签： [0, 1, 0, 0, 1, 0, 1, 1, 0, 0]
预测标签： [0, 0, 0, 0, 0, 1, 1, 1, 0, 0]
```