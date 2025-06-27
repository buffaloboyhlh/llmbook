# KNN 算法

## 一、原理篇

## 二、代码篇

### 1、数据准备和可视化

```python
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = ['Heiti TC']

# 加载数据集 MNIST

X = np.loadtxt("mnist_x", delimiter=" ")  # 数据
Y = np.loadtxt("mnist_y")  # 对应的标签

# 可视化第一条数据
num_data = np.reshape(np.array(X[0], dtype=int), (28, 28))
plt.figure()
plt.imshow(num_data, cmap='gray')
```

### 2、划分数据集

```python
# 划分数据集

split = int(len(Y) * 0.8)

# 打乱数据
np.random.seed(42)
idx = np.random.permutation(np.arange(len(X)))  # np.random.permutation 用于生成一个随机排列
X = X[idx]
Y = Y[idx]

x_tain, x_test = X[:split], X[split:]
y_tain, y_test = Y[:split], Y[split:]

print(x_tain.shape)
print(x_test.shape)
```

### 3、实现代码（手撕版）

#### 欧氏距离

```python
def distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))
```

#### KNN

```python
class KNN:
    def __init__(self, k, label_num):
        self.k = k
        self.label_num = label_num  # 类别的数量

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def get_close_k_neighbors(self, x):
        # 计算已知样本和目标样本 x 之间的距离
        distances = list(map(lambda a: distance(a, x), self.x_train))

        # 按照距离大小排序，并得到对应的下标
        knn_index = np.argsort(distances)
        # 获取最近的 k个下标
        knn_index = knn_index[:self.k]
        return knn_index

    def get_label(self, x):
        knn_index = self.get_close_k_neighbors(x)
        # 统计类别
        label_statistic = np.zeros(shape=[self.label_num], dtype=int)
        for index in knn_index:
            label = int(self.y_train[index])
            label_statistic[label] += 1

        # 返回数量最多的类别
        return np.argmax(label_statistic)

    def predict(self, x_test):
        # 预测样本类别
        labels = np.zeros(shape=[len(x_test)], dtype=int)
        for i, x in enumerate(x_test):
            labels[i] = self.get_label(x)

        return labels
```

#### 验证效果

```python
for k in range(1, 11):
    knn = KNN(k, label_num=10)
    knn.fit(x_tain, y_tain)
    pred_label = knn.predict(x_test)

    acc = np.mean(pred_label == y_test)
    print(f"k={k}, 预测的准确率为：{acc * 100:.2f}%")
```

!!! Example "输出结果"

    ```text
        k=1, 预测的准确率为：89.00%
        k=2, 预测的准确率为：89.00%
        k=3, 预测的准确率为：91.00%
        k=4, 预测的准确率为：90.00%
        k=5, 预测的准确率为：88.00%
        k=6, 预测的准确率为：87.00%
        k=7, 预测的准确率为：85.00%
        k=8, 预测的准确率为：86.50%
        k=9, 预测的准确率为：86.00%
        k=10, 预测的准确率为：87.00%
    ```

### 4、实现代码（sklearn版）

## 三、项目实战

### 色彩风格迁移

#### 数据可视化

```python
import numpy as np
from matplotlib import pyplot as plt
from skimage import io
from skimage.color import rgb2lab, lab2rgb
from sklearn.neighbors import KNeighborsRegressor # KNN 回归器
import os

path = "style_transfer"

data_dir = os.path.join(path, "vangogh")
fig = plt.figure(figsize=(16,8))

for i,file in enumerate(os.listdir(data_dir)[:3]):
    img = io.imread(os.path.join(data_dir, file))
    ax = fig.add_subplot(1,3,i+1)
    ax.imshow(img)
    ax.set_title(file)

plt.show()
```

#### 提取训练数据

```python
block_size = 1 # block_size 表示向外扩展的层数，扩展 1 层即 3*3

def read_style_image(filename,size = block_size):
    # 读入风格图像，得到映射 x->y
    # 其中 X 存储 3*3像素的灰度值，Y 存储中心格子的色彩值
    # 读取图像文件，设图像宽为 W,高为 H.得到 W*H*3 的 RGB 矩阵

    img = io.imread(filename)
    fig = plt.figure()
    plt.imshow(img)
    plt.show()

    # 将 RGB 矩阵转换为 LAB 表示法的矩阵，大小仍然是 w*h*3，三维分别是 L、A、B
    img = rgb2lab(img)

    # 获取图像的宽度和高度
    width, height = img.shape[:2]

    X = []
    Y = []

    # 枚举全部可能的中心格子
    for x in range(size,width-size):
        for y in range(size,height-size):
            # 保存所有窗口
            X.append(img[x-size:x+size+1,y-size:y+size+1,0].flatten()) # 3*3像素窗口的灰度值
            Y.append(img[x,y,1:])

    return X,Y
```

#### 构建模型

```python
X, Y = read_style_image(os.path.join(path, "style.jpg"))

knn = KNeighborsRegressor(n_neighbors=4,weights="distance")
knn.fit(X,Y)
```

#### 风格迁移函数

```python
def rebuild(img,size=block_size):
    fig = plt.figure()
    plt.imshow(img)
    plt.show()

    # 将内容图像转为 LAB 表示
    img = rgb2lab(img)
    width, height = img.shape[:2]

    # 初始化输出图像对应的矩阵
    photo = np.zeros((width,height,3))

    # 枚举内容图像的中心格子，保存所有窗口
    X = []
    for x in range(size,width-size):
        for y in range(size,height-size):
            # 得到中心格子对应的窗口
            window = img[x-size:x+size+1,y-size:y+size+1,0].flatten()
            X.append(window)

    X = np.array(X)

    # 用 KNN 回归器预测颜色
    pred_ab = knn.predict(X).reshape(width-2*size,height-2*size,-1)

    # 设置输出图像
    photo[:,:,0] = img[:,:,0]
    photo[size:width-size,size:height-size,1:] = pred_ab

    # 由于最外面 size 层 无法构造窗口，简单起见，我们直接把这些像素剪掉
    photo = photo[size:width-size,size:height-size,:]
    return photo
```

#### 效果展示

```python
content = io.imread(os.path.join(path, "input.jpg"))
new_photo = rebuild(content)

new_photo = lab2rgb(new_photo)

fig = plt.figure()
plt.imshow(new_photo)
plt.show()
```







