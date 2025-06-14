# 第十三章：生成式对抗网络

---

## 🧠 一、什么是 GAN？

**生成式对抗网络（GAN）** 是一种强大的生成模型，由 Ian Goodfellow 等人在 2014 年提出，核心思想是通过两个神经网络之间的博弈训练，从而生成以假乱真的数据样本。

### ✨ GAN 包含两个主要部分：

- **生成器（Generator，$G$）**：接受随机噪声 $z$，生成伪造数据 $G(z)$。
- **判别器（Discriminator，$D$）**：判断输入数据是真实的（来自训练集）还是伪造的（来自生成器）。

二者通过对抗训练，不断优化，使得生成器能生成更逼真的样本，而判别器能更准确地区分真假。

---

## 📐 二、基本原理与数学公式

### 1. 目标函数（Minimax 博弈）

GAN 的目标是一个极小极大问题：

$$
\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]
$$

其中：

- $x$：真实样本
- $z$：生成器的输入噪声
- $G(z)$：生成样本
- $D(x)$：判别器认为 $x$ 为真实样本的概率
- $p_{\text{data}}$：真实数据的分布
- $p_z$：噪声分布（通常为标准正态分布）

---

## 🧪 三、PyTorch 实现（MNIST 示例）

### 1. 模型结构

```python
import torch
import torch.nn as nn

# 生成器
class Generator(nn.Module):
    def __init__(self, noise_dim=100, output_dim=784):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, output_dim), nn.Tanh()
        )
    def forward(self, z):
        return self.model(z)

# 判别器
class Discriminator(nn.Module):
    def __init__(self, input_dim=784):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)
```

---

### 2. 训练过程

```python
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 初始化模型与优化器
G = Generator()
D = Discriminator()
criterion = nn.BCELoss()
optimizer_G = optim.Adam(G.parameters(), lr=0.0002)
optimizer_D = optim.Adam(D.parameters(), lr=0.0002)

# 加载 MNIST 数据
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1)),
    transforms.Normalize(0.5, 0.5)
])
dataloader = DataLoader(
    datasets.MNIST(root="./data", train=True, transform=transform, download=True),
    batch_size=64, shuffle=True
)

# 训练循环
for epoch in range(50):
    for real_data, _ in dataloader:
        batch_size = real_data.size(0)
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        # 训练判别器
        z = torch.randn(batch_size, 100)
        fake_data = G(z)
        real_output = D(real_data)
        fake_output = D(fake_data.detach())
        D_loss = criterion(real_output, real_labels) + criterion(fake_output, fake_labels)

        optimizer_D.zero_grad()
        D_loss.backward()
        optimizer_D.step()

        # 训练生成器
        output = D(fake_data)
        G_loss = criterion(output, real_labels)

        optimizer_G.zero_grad()
        G_loss.backward()
        optimizer_G.step()

    print(f"Epoch {epoch}: D_loss={D_loss.item():.4f}, G_loss={G_loss.item():.4f}")
```

---

## 📊 四、常见问题与改进

| 问题                 | 原因与解决方法                                         |
|----------------------|--------------------------------------------------------|
| 训练不稳定           | 使用 WGAN、调整网络结构、使用归一化等                 |
| 模式崩溃（Mode Collapse） | 加入噪声、改进网络结构（如 Unrolled GAN）            |
| 判别器过强或过弱     | 保持 G 和 D 的训练平衡；使用标签平滑、梯度惩罚等       |

---

## 🚀 五、常见 GAN 变体

| 变体       | 特点描述                                             |
|------------|------------------------------------------------------|
| DCGAN      | 使用 CNN 架构，提升图像生成效果                      |
| CGAN       | 条件 GAN，生成特定类别样本（如输入标签）            |
| WGAN       | 使用 Wasserstein 距离，提升训练稳定性                |
| WGAN-GP    | WGAN 加入梯度惩罚，进一步改善收敛性                  |
| LSGAN      | 使用最小二乘损失，减少梯度消失                       |
| StyleGAN   | 高质量人脸生成，控制风格与内容                       |
| CycleGAN   | 无需成对图像进行风格迁移（如将马变成斑马）           |

---

## 🧱 六、应用场景

- 图像生成、上色与修复
- 图像超分辨率（SRGAN）
- 图像风格转换（CycleGAN）
- 数据增强（Data Augmentation）
- 音乐与语音生成
- 文本生成（seqGAN 等）
- 医疗图像合成与数据补全

---

## 📚 七、推荐资源

- 论文原文：[Generative Adversarial Nets (2014)](https://arxiv.org/abs/1406.2661)
- 实践教程：TensorFlow / PyTorch 官方文档
- GitHub 示例项目：`pytorch/examples`, `eriklindernoren/PyTorch-GAN`

