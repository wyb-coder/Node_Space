##### 1. Pytorch基础

1.基础功能函数：

```python
import torch
dir(torch)						# 显示其下所有方法、子包
help(torch.cuda.is_available)	# 方法简介
```

2.数据加载：

- Dataset：提供方式获取数据及其lable。
- Dataloader：为网络提供不同的数据形式，以Dataset为基础。

```python
# 基础图像处理
from PIL import Image
img = Image.open(img_path)
print(img.size)
img.show()

# 基础文件处理
train_dir_path = "./data/hymenoptera_data/train"
label = "ants"
ants_dir_path = os.path.join(train_dir_path, label)		# 自动拼接文件路径，无需显示指出“/”
data_path_list = os.listdir(ants_dir_path)				# 获取文件名列表
print(type(data_path_list[0]))
```

3.TensorBoard：训练过程可视化

```python
img = Image.open(img_dir)
img_arr = np.array(img)

writer = SummaryWriter("logs")
# “类名”、“像素信息”、“步数”、“图象格式(高、宽、通道)”
writer.add_image("test", img_arr, 1, dataformats="HWC")

for i in range(100):
    # 标量，即数字
    writer.add_scalar("y = x", i, i)
writer.close()

tensorboard --logdir=Pytorch/logs --port=6007
```

4.Transforms：数据处理工具箱

- Tensor结构：三维数组，例如例如(3, 224, 224)的张量，打印时会显示成三块，每块是一个(224, 224)的矩阵。

```python
image = Image.open(image_path)
image_tensor = tranforms.ToTensor()(img)
image_norm_tensor = tranforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(image_tensor)

image_resize = transforms.Resize((512, 512))(image)
transforms.Resize(512)				# 短边缩放到512，长边按比例调整
# transform.Compose 写入一系列操作，按序执行，注意上一个操作的输入与下一个操作的输出一一对应
transform.Compose([transform.Resize(512), tranforms.ToTensor()])(image)

transforms.RandomCrop(512)(image)	# 随机裁剪，通常用于数据增强       
```



##### 1.2 Pytorch for NN(Neural Networks)

1.基本框架：`touch.nn`

```python
from torch import nn
nn.Module
```

2.卷积(Convolution)：使用卷积核提取数据特征，同时缩小数据规模。

- stride：每次移动的步长(格子数)
- padding：是否做填充
- out_channel：输出通道数，可通过不同卷积核生成并行结果

3.池化层(Pooling)：池化核，类似卷积，不过取最大/小值

- Max Pooling：取窗口内最大值，保留最强激活特征
- Average Pooling：取平均值，平滑特征分布

4.线性层(Linear)：全连接层，将输入特征进行线性变换。常用于模型的输出层、`特征维度转换`

5.丢弃层(Dropout)：正则化技术、预防过拟合。随机将部分神经元的输出设为0，打破神经元之间的依赖

6.稀疏层(Sparse)：稀疏输入或稀疏权重计算



































