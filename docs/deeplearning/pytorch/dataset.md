# 数据预处理

以下是 PyTorch 中 **`Dataset`**、**`DataLoader`** 和 **`transforms`** 三个核心 API 的使用教程，涵盖基础用法和实际示例。

---

## 1. Dataset: 数据容器
`Dataset` 是 PyTorch 中用于封装数据的基类，需继承并实现 `__len__` 和 `__getitem__` 方法。

### 自定义 Dataset 示例
```python
from torch.utils.data import Dataset
import os
from PIL import Image

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, labels, transform=None):
        self.img_dir = img_dir
        self.labels = labels  # 假设 labels 是列表或数组
        self.transform = transform
        self.img_names = os.listdir(img_dir)  # 假设所有文件是图像

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
```

---

## 2. Transforms: 数据预处理
`transforms` 提供图像预处理工具，常用组合如 `Compose`。

### 内置 Transforms 示例
```python
from torchvision import transforms

# 定义组合变换
transform = transforms.Compose([
    transforms.Resize(256),          # 调整大小
    transforms.RandomCrop(224),      # 随机裁剪
    transforms.RandomHorizontalFlip(),  # 随机水平翻转（数据增强）
    transforms.ToTensor(),           # 转为 Tensor [0,1]
    transforms.Normalize(            # 标准化（ImageNet 均值方差）
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 自定义 Transform（如随机擦除）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
])
```

---

## 3. DataLoader: 数据批量加载
`DataLoader` 从 `Dataset` 中加载数据，支持批量加载、多进程加速等。

### 基础用法
```python
from torch.utils.data import DataLoader

# 创建 Dataset 实例
dataset = CustomImageDataset(
    img_dir="data/images",
    labels=[0, 1, 0, 1],  # 示例标签
    transform=transform
)

# 创建 DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,   # 训练时建议打乱
    num_workers=4,  # 多进程加载（根据 CPU 核心数调整）
    drop_last=True  # 丢弃最后不足一个 batch 的数据
)

# 在训练循环中使用
for epoch in range(num_epochs):
    for batch_images, batch_labels in dataloader:
        # 将数据送入 GPU
        batch_images = batch_images.to(device)
        batch_labels = batch_labels.to(device)

        # 训练代码...
```

---

## 4. 综合示例：图像分类全流程

### 步骤 1: 定义 Dataset + Transforms
```python
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = CustomImageDataset(
    img_dir="train_data",
    labels=train_labels,
    transform=train_transform
)
```

### 步骤 2: 创建 DataLoader
```python
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4
)
```

### 步骤 3: 在模型训练中使用
```python
model = models.resnet18(pretrained=True)
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(10):
    model.train()
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = torch.nn.CrossEntropyLoss()(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## 5. 高级技巧 & 注意事项

### 自定义 Transforms
```python
class RandomRotation:
    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, img):
        angle = torch.empty(1).uniform_(-self.degrees, self.degrees).item()
        return transforms.functional.rotate(img, angle)

transform = transforms.Compose([
    RandomRotation(30),
    transforms.ToTensor()
])
```

### 多进程问题
- **`num_workers`**: 建议设为 CPU 核心数，但需避免内存不足。
- **`pin_memory=True`**: 当使用 GPU 时加速数据传输。

### 其他数据类型
- **文本数据**: 需自定义 `Dataset` 加载文本文件或 tokenize。
- **非图像数据**: 移除图像相关 transforms，使用 `torch.FloatTensor` 转换。

---

通过合理组合 `Dataset`、`DataLoader` 和 `transforms`，可以高效管理数据流程，提升模型训练效率。