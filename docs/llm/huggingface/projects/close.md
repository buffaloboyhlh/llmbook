# 中文填空

### Step1 数据预处理

```python
from transformers import BertTokenizer, BertModel, TrainingArguments, Trainer
from datasets import load_dataset
import torch.nn as nn
import torch

model_name = 'bert-base-chinese'
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
# 加载数据集
dataset = load_dataset(path="src/seamew/ChnSentiCor")
# 编码器
tokenizer = BertTokenizer.from_pretrained(model_name, return_tensors='pt')


def precess_fn(examples):
    return tokenizer(examples['text'], truncation=True, max_length=30, padding='max_length', return_tensors='pt',
                     return_length=True)


tokenized_dataset = dataset.map(precess_fn, batched=True)
tokenized_dataset = tokenized_dataset.filter(function=lambda x: x['length'] >= 30)
tokenized_dataset = tokenized_dataset.remove_columns(['label', 'text'])
tokenized_dataset
```

### Step2 定义模型

```python
# 定义模型

pretrained_model = BertModel.from_pretrained(model_name).to(device)
for param in pretrained_model.parameters():
    param.requires_grad = False


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.decoder = nn.Linear(768, tokenizer.vocab_size)
        self.decoder.bias.data.zero_()  #将bias重置为0
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        with torch.no_grad():
            out = pretrained_model(input_ids, attention_mask, token_type_ids)

        logits = self.dropout(out.last_hidden_state[:,15])
        logits = self.decoder(logits)
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return {'loss': loss, 'logits': logits}  # 关键字loss和logits是huggingface transformer要求的

        return {'logits': logits}


model = MyModel()
print(model)
```

### Step3 训练参数

```python
train_args = TrainingArguments(
    output_dir="outputs",
    save_strategy="epoch",
    save_total_limit=1,
    learning_rate=5e-4,
    weight_decay=0.01,
    lr_scheduler_type="linear",
    warmup_steps=5,
    num_train_epochs=10,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    logging_dir="logs",
    logging_strategy="epoch",
    report_to="tensorboard",
    disable_tqdm=False,
)
```

### Step4 训练模型

```python
# 数据整理函数
def data_collator(batch_data):
    input_ids = [data['input_ids'] for data in batch_data]
    attention_masks = [data['attention_mask'] for data in batch_data]
    token_type_ids = [data['token_type_ids'] for data in batch_data]

    # 转换数据类型
    input_ids = torch.LongTensor(input_ids)
    attention_masks = torch.LongTensor(attention_masks)
    token_type_ids = torch.LongTensor(token_type_ids)

    # 把第15个词作为目标值
    labels = input_ids[:, 15].reshape(-1).clone()
    input_ids[:, 15] = tokenizer.get_vocab()[tokenizer.mask_token]  # 遮挡第15个词


    return {
        'input_ids': input_ids,
        'attention_mask': attention_masks,
        'token_type_ids': token_type_ids,
        'labels': labels,
    }


tainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['validation'],
    data_collator=data_collator,
    tokenizer=tokenizer,
)
tainer.train()
```

### Step5 查看训练效果

```python
model.eval()

datas = tokenized_dataset['test'].select(range(5))
datas = data_collator(datas)
TLabels = datas['labels']
print("真实数据：",tokenizer.decode(TLabels))

model.to(device=device)
outputs = model(input_ids=datas['input_ids'].to(device), attention_mask=datas['attention_mask'].to(device), token_type_ids=datas['token_type_ids'].to(device),labels=datas['labels'].to(device))

preds =  outputs['logits'].argmax(1).tolist()
print("预测数据：",tokenizer.decode(preds))
```

