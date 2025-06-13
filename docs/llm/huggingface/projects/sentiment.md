# 中文情感分类

### Step1 数据准备

本章所使用的数据集依然是ChnSentiCorp数据集，这是一个情感分类数据集，每条数据中包括一句购物评价，以及一个标识，表明这条评价是一条好评还是一条差评。在ChnSentiCorp数据集中，被评价的商品包括书籍、酒店、计算机配件等。对于人类来讲，即使不给予标识，也能通过评价内容大致判断出这是一条好评还是一条差评；对于神经网络，也将通过这个任务来验证它的有效性。

```python
from datasets import load_dataset

dataset = load_dataset('src/seamew/ChnSentiCor') # 使用huggingface-cli 下载的离线数据
dataset
```
**输出**

```text
DatasetDict({
    train: Dataset({
        features: ['label', 'text'],
        num_rows: 9600
    })
    validation: Dataset({
        features: ['label', 'text'],
        num_rows: 1200
    })
    test: Dataset({
        features: ['label', 'text'],
        num_rows: 1200
    })
})
```

### Step2 加载编码器

```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
tokenizer
```
**输出**

```text
BertTokenizer(name_or_path='bert-base-chinese', vocab_size=21128, model_max_length=512, is_fast=False, padding_side='right', truncation_side='right', special_tokens={'unk_token': '[UNK]', 'sep_token': '[SEP]', 'pad_token': '[PAD]', 'cls_token': '[CLS]', 'mask_token': '[MASK]'}, clean_up_tokenization_spaces=True, added_tokens_decoder={
	0: AddedToken("[PAD]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	100: AddedToken("[UNK]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	101: AddedToken("[CLS]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	102: AddedToken("[SEP]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	103: AddedToken("[MASK]", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
}
)
```
在输出中可以看到vocab_size=21128，词库的大小

#### 编码句子

```python
out = tokenizer.batch_encode_plus(
    batch_text_or_text_pairs=['从明天起, 做一个幸福的人。','喂马,劈柴,周游世界'],
    truncation=True,
    max_length=17,
    padding='max_length',
    return_tensors='pt',
    return_length=True,
)

out
```
**输出**

```text
{'input_ids': tensor([[ 101,  794, 3209, 1921, 6629,  117,  976,  671,  702, 2401, 4886, 4638,
          782,  511,  102,    0,    0],
        [ 101, 1585, 7716,  117, 1207, 3395,  117, 1453, 3952,  686, 4518,  102,
            0,    0,    0,    0,    0]]), 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), 'length': tensor([15, 12]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]])}
```

#### 解码句子

```python
tokenizer.decode(out['input_ids'][0])
```
**输出**

```text
'[CLS] 从 明 天 起, 做 一 个 幸 福 的 人 。 [SEP] [PAD] [PAD]'
```

### Step3 定义数据集

本次任务为情感分类任务，所以需要一个情感分类数据集进行模型的训练和测试，此处加载ChnSentiCorp数据集

```python
import torch
from torch.utils.data import DataLoader, Dataset


class MyDataset(Dataset):
    def __init__(self, split='train'):
        self.dataset = load_dataset('src/seamew/ChnSentiCor', split=split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx]['text']
        label = self.dataset[idx]['label']
        return text, label

dataset = MyDataset('train')
len(dataset),dataset[20]
```
**输出**

```text
(9600, ('非常不错，服务很好，位于市中心区，交通方便，不过价格也高！', 1))
```

### Step4 定义数据整理函数

```python
def collate_fn(batch_data):
    texts, labels = zip(*batch_data)
    # 编码
    tokenized_Data = tokenizer.batch_encode_plus(
        batch_text_or_text_pairs=texts,
        truncation=True,
        max_length=500,
        padding='max_length',
        return_length=True,
        return_tensors='pt',
    )

    input_ids = tokenized_Data['input_ids']
    attention_mask = tokenized_Data['attention_mask']
    token_type_ids = tokenized_Data['token_type_ids']
    labels = torch.tensor(labels).long()

    # 把数据转移到设备上
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    token_type_ids = token_type_ids.to(device)
    labels = labels.to(device)
    return input_ids, attention_mask, token_type_ids, labels
```

### Step5 定义数据加载器

```python
loader = DataLoader(dataset=dataset, batch_size=16, collate_fn=collate_fn,drop_last=True,shuffle=True)

len(loader)
```

#### 查看数据样本

```python
for i, (input_ids,attention_mask,token_type_ids,labels) in enumerate(loader):
    break

input_ids.shape,attention_mask.shape,token_type_ids.shape,labels.shape
```

**输出**

```text
(torch.Size([16, 500]),
 torch.Size([16, 500]),
 torch.Size([16, 500]),
 torch.Size([16]))
```

### Step6 定义模型

```python
from transformers import BertModel

model = BertModel.from_pretrained("bert-base-chinese")

model.to(device) # 转移到设备上

# 冻结模型参数
for param in model.parameters():
    param.requires_grad = False
```

#### 查看模型的输出形状

```python
out = model(input_ids,attention_mask,token_type_ids)
out.last_hidden_state.shape
```
**输出**

```text
torch.Size([16, 500, 768]) 
```

从预训练模型的计算结果可以看出，这也是16句话的结果，每句话包括500个词，每个词被抽成一个768维的向量。


### Step7 定义下游任务模型

完成以上工作，现在可以定义下游任务模型了。下游任务模型的任务是对backbone抽取的特征进行进一步计算，得到符合业务需求的计算结果。对于本章的任务来讲，需要计算一个二分类的结果，和数据集中真实的label保持一致，代码如下

```python
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # 使用预训练的模型提取数据特征
        with torch.no_grad():
            out = model(input_ids, attention_mask, token_type_ids)
        # 对抽取的特征只取第1个字的结果做分类即可;“注意：之所以只取了第0个词的特征做后续的判断计算，这和预训练模型BERT的训练方法有关系”
        out = self.fc(out.last_hidden_state[:, 0])
        out = out.softmax(dim=1)
        return out


classifier = MyModel()
classifier.to(device)
out = classifier(input_ids, attention_mask, token_type_ids)
print(out)
```
**输出**

```text
tensor([[0.5787, 0.4213],
        [0.6246, 0.3754],
        [0.3652, 0.6348],
        [0.3738, 0.6262],
        [0.5788, 0.4212],
        [0.4522, 0.5478],
        [0.4285, 0.5715],
        [0.4864, 0.5136],
        [0.5243, 0.4757],
        [0.4276, 0.5724],
        [0.3951, 0.6049],
        [0.4388, 0.5612],
        [0.5446, 0.4554],
        [0.5176, 0.4824],
        [0.3299, 0.6701],
        [0.4360, 0.5640]], device='mps:0')
```

### Step8 训练模型

```python
from torch.optim import AdamW
import torch.nn as nn
from transformers import get_scheduler


def train_model():
    # 定义优化器
    optimizer = AdamW(classifier.parameters(), lr=5e-4)
    # 定义 损失函数
    loss_fn = nn.CrossEntropyLoss()
    # 定义学习率调机器
    scheduler = get_scheduler(
        name="linear",  # 学习率调度策略的名字
        optimizer=optimizer,  # 训练中使用的优化器
        num_warmup_steps=0,  # 预热步数
        num_training_steps=len(loader)  # 总的训练步数
    )
    # 将模型切换到训练模式
    classifier.train()
    # 按批次遍历训练集中的数据
    for idx, (input_ids,attention_mask,token_type_ids,labels) in enumerate(loader):

        optimizer.zero_grad()
        # 模型输出
        out = classifier(input_ids, attention_mask, token_type_ids)

        loss = loss_fn(out, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # 输出各项数据的情况，便于观察训练效果
        if idx % 30 == 0:
            pred_labels = torch.argmax(out, dim=1)
            accuracy = (pred_labels == labels).sum().item() / len(labels)  # 精确度
            lr = scheduler.get_last_lr()[0]
            print(f"【{idx}/{len(loader)}】 损失值：{loss.item()} 精确度:{accuracy} lr:{lr}")

train_model()
```

### Step9 评估模型

```python
def test_model():
    classifier.eval()
    # 定义数据集
    dataset = MyDataset('test')
    test_loader = DataLoader(dataset, batch_size=16, collate_fn=collate_fn, drop_last=True)
    correct = 0
    total = 0
    with torch.no_grad():
        for input_ids, attention_mask, token_type_ids, labels in test_loader:
            out = classifier(input_ids, attention_mask, token_type_ids)
            pred_labels = torch.argmax(out, dim=1)
            correct += (pred_labels == labels).sum().item()
            total += labels.size(0)

    print(f"精确度：{correct/total*100:.2f}%")

test_model()
```

**输出**

```text
精确度：87.58%
```

