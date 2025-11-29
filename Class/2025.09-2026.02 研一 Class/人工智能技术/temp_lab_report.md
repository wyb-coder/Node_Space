2025～2026年秋季学期
====================

**《人工智能技术》课程实验报告**

![](media/image1.emf){width="2.203472222222222in"
height="2.1756944444444444in"}

> **实验四 计算智能**

**学 生 姓 名： *王耀彬 ***

**学 号： *2025354100103 ***

**专 业 班 级： *智能专研25 ***

> **\
> 实验四 计算智能**

**1．实验目的**

理解人工神经网络（或者深度学习）的结构和原理，掌握人工神经网络（或者深度学习）的训练过程，利用现有的人工神经网络（或者深度学习）进行分类操作，并通过调节参数以提高分类性能。

**2．实验内容**

编写一个人工神经网络（或者深度学习）程序，实现简单的分类。

**3．实验报告要求**

（1）简述实验原理，并请给出实现算法及程序流程图。

本次实验计划基于“Food-11”数据集设计神经网络以进行图像分类。旨在通过深度学习技术解决多类别的食物图像识别问题。

详细实现：构建基于卷积神经网络（CNN）的深度学习模型，并针对图像分类任务中的常见问题（如过拟合、样本不平衡等）实施了特定的改进策略。包括数据预处理与增强、引入残差神经网络（ResNet）、设计Focal
Loss损失函数、分层K折交叉验证、模型集成推理与测试时增强（TTA）。下面是实现技术概述。

1.  残差神经网络（Residual
    Network）：为了解决随着网络深度增加导致的梯度消失和网络退化问题，本实验自定义了基于Residual
    Block（残差块）的卷积神经网络。模型主体由四个阶段组成，每个阶段包含不同数量的残差块（分别为2,
    3, 3, 1层）。残差块通过引入跳跃连接（Skip
    Connection），将输入特征直接叠加到卷积输出上，实现了恒等映射的学习。这种结构保证了信息在前向传播和梯度在反向传播时的顺畅流动，使得构建更深层的网络成为可能，从而能够提取更丰富的图像语义特征。

2.  数据增强策略（Data
    Augmentation）：为增强模型的鲁棒性并防止过拟合，在训练阶段引入了多样化的数据增强操作。代码中使用了RandomResizedCrop（随机裁剪）、RandomHorizontalFlip（随机翻转）、RandomRotation（随机旋转）、RandomAffine（随机仿射变换）以及
    RandomErasing（随机擦除）这些变换模拟了真实场景下的图像扰动，强制模型学习图像的不变性特征，而非记忆特定的像素排列。

3.  损失函数优化（Focal
    Loss）：针对可能存在的样本难易不均问题，本实验采用了Focal
    Loss替代标准的交叉熵损失函数。Focal
    Loss通过引入聚焦参数γ和平衡参数α，降低了易分类样本的权重，使模型在训练过程中更加关注那些难以分类的“困难样本”。这一改进对于提升模型在决策边界上的判别能力具有重要意义。

4.  训练与验证策略（Stratified K-Fold &
    Optimizer）：为了充分利用有限的数据集并评估模型的真实性能，采用了分层K折交叉验证（Stratified
    K-Fold），设置K=4
    。该方法确保了每个折（Fold）中各类别的比例与原始数据集一致，避免了因数据划分偏差导致的评估误差。优化器选用Adam，并配合
    CosineAnnealingWarmRestarts（余弦退火热重启）学习率调度策略，帮助模型跳出局部极小值，寻找全局更优解。

5.  模型集成（Ensemble
    Learning）：在推理阶段，实验并未依赖单一模型，而是加载了K折交叉验证中保存的K个最佳模型权重。对测试集数据进行预测时，采用软投票（Soft
    Voting）机制，即将K个模型的输出概率向量相加后取最大值对应的类别作为最终预测结果。集成学习通过组合多个基学习器，有效地降低了预测方差，提升了系统的整体泛化性能。

下面是本次实验设计的程序流程图：

（2）源程序清单。要求符合一般的程序书写风格，并包括必要的注释。

"""

Baseline：基准参数结构模型

"""

 

\#
==================================================================================

\#                                   Import Model

\#
==================================================================================

**import** numpy as np

**import** pandas as pd

**import** torch

**import** os

**import** torch.nn as nn

**import** torchvision.transforms as transforms

**from** PIL **import** Image

\# "ConcatDataset" and "Subset" are possibly useful when doing
semi-supervised learning.

**from** torch.utils.data **import** ConcatDataset, DataLoader, Subset,
Dataset

**from** torchgen.api.types **import** layoutT

**from** torchvision.datasets **import** DatasetFolder, VisionDataset

**from** tqdm.auto **import** tqdm

**import** random

**import** torchvision.models as models

**import** torch.nn.functional as F

**from** torch.autograd **import** Variable

**from** sklearn.model\_selection **import** StratifiedKFold

 

 

 

\#
==================================================================================

\#                               Image Transforms

\#
==================================================================================

test\_tfm **=** transforms.Compose(\[

    transforms.Resize((128, 128)),

    transforms.ToTensor(),

    \# transforms.Normalize(mean=\[0.485,0.456,0.406\],
std=\[0.229,0.224,0.225\]),

\])

 

train\_tfm **=** transforms.Compose(\[

    transforms.RandomResizedCrop(128, scale**=**(0.7, 1.0)),

    transforms.RandomHorizontalFlip(0.5),

    transforms.RandomVerticalFlip(0.5),

    transforms.RandomRotation(180),

    transforms.RandomAffine(30),

    transforms.RandomGrayscale(0.2),

    \# transforms.ColorJitter(brightness=0.2, contrast=0.2,
saturation=0.2, hue=0.1),

    transforms.ToTensor(),

    \# transforms.Normalize(mean=\[0.485,0.456,0.406\],
std=\[0.229,0.224,0.225\]),

    transforms.RandomErasing(0.2)  \# 随机擦除

\])

 

 

 

\#
==================================================================================

\#                                   Dataset

\#
==================================================================================

**class** FoodDataset(Dataset):

 

    **def** \_\_init\_\_(self, path**=**None, tfm**=**test\_tfm,
files**=**None):

        super(FoodDataset).\_\_init\_\_()

        self.path **=** path

        **if** path:

            self.files **=** sorted(\[os.path.join(path, x) **for** x
**in** os.listdir(path) **if** x.endswith(".jpg")\])

        **else**:

            self.files **=** files

        self.transform **=** tfm

        print('Num of element: ', len(self.files))

 

    **def** \_\_len\_\_(self):

        **return** len(self.files)

 

    **def** \_\_getitem\_\_(self, idx):

        fname **=** self.files\[idx\]

        im **=** Image.open(fname)

        im **=** self.transform(im)

        \# im = self.data\[idx\]

        **try**:

            label **=** int(fname.split("/")\[**-**1\].split("\_")\[0\])

        **except**:

            label **=** **-**1  \# test has no label

        **return** im, label

 

 

 

\#
==================================================================================

\#                               Model Structure

\#
==================================================================================

**class** Residual\_Block(nn.Module):

    **def** \_\_init\_\_(self, in\_channels, out\_channels,
stride**=**1):

        super().\_\_init\_\_()

        self.conv1 **=** nn.Sequential(

            nn.Conv2d(in\_channels, out\_channels, kernel\_size**=**3,
stride**=**stride, padding**=**1),

            nn.BatchNorm2d(out\_channels),

            nn.ReLU(inplace**=**True)

        )

        self.conv2 **=** nn.Sequential(

            nn.Conv2d(out\_channels, out\_channels, kernel\_size**=**3,
stride**=**1, padding**=**1),

            nn.BatchNorm2d(out\_channels),

            \# 不激活，先残差连接，再激活。

        )

        self.relu **=** nn.ReLU(inplace**=**True)

        self.downsample **=** None

        **if** stride !**=** 1 **or** in\_channels !**=** out\_channels:

            self.downsample **=** nn.Sequential(

                nn.Conv2d(in\_channels, out\_channels,
kernel\_size**=**1, stride**=**stride),

                nn.BatchNorm2d(out\_channels),

            )

 

    **def** forward(self, x):

        residual **=** x

        out **=** self.conv1(x)

        out **=** self.conv2(out)

 

        **if** self.downsample:

            \# 对其特征，确保加法对齐

            residual **=** self.downsample(x)

        out **+=** residual

        **return** self.relu(out)

 

**class** Classifier(nn.Module):

    **def** \_\_init\_\_(self, block, num\_layers, num\_classes**=**11):

        super(Classifier, self).\_\_init\_\_()

        self.preConv **=** nn.Sequential(

            nn.Conv2d(3, 32, kernel\_size**=**7, stride**=**2,
padding**=**3, bias**=**False),

            nn.BatchNorm2d(32),

            nn.ReLU(inplace**=**True)

        )

        self.layer0 **=** self.makeResidualBlocks(block, 32, 64,
num\_layers\[0\], stride**=**2)

        self.layer1 **=** self.makeResidualBlocks(block, 64, 128,
num\_layers\[1\], stride**=**2)

        self.layer2 **=** self.makeResidualBlocks(block, 128, 256,
num\_layers\[2\], stride**=**2)

        self.layer3 **=** self.makeResidualBlocks(block, 256, 512,
num\_layers\[3\], stride**=**2)

 

        self.fc **=** nn.Sequential(

            nn.Dropout(0.4),

            nn.Linear(512 **\*** 4 **\*** 4, 512),

            nn.BatchNorm1d(512),

            nn.ReLU(inplace**=**True),

            nn.Dropout(0.2),

            nn.Linear(512, num\_classes)

        )

 

 

    **def** forward(self, x):

        out **=** self.preConv(x)

        out **=** self.layer0(out)

        out **=** self.layer1(out)

        out **=** self.layer2(out)

        out **=** self.layer3(out)

        out **=** self.fc(out.view(out.size(0), **-**1))

        **return** out

 

    **def** makeResidualBlocks(self, block, in\_channels, out\_channels,
num\_layer, stride**=**1):

        layers **=** \[block(in\_channels, out\_channels, stride)\]

        **for** i **in** range(1, num\_layer):

            layers.append(block(out\_channels, out\_channels))

        **return** nn.Sequential(**\***layers)

 

 

**class** FocalLoss(nn.Module):

    **def** \_\_init\_\_(self, class\_num, alpha**=**None, gamma**=**2,
size\_average**=**True):

        super().\_\_init\_\_()

        **if** alpha **is** None:

            self.alpha **=** Variable(torch.ones(class\_num, 1))

        **else**:

            **if** isinstance(alpha, Variable):

                self.alpha **=** alpha

            **else**:

                self.alpha **=** Variable(alpha)

        self.gamma **=** gamma

        self.class\_num **=** class\_num

        self.size\_average **=** size\_average

 

    **def** forward(self, inputs, targets):

        N **=** inputs.size(0)

        C **=** inputs.size(1)

        P **=** F.softmax(inputs, dim**=**1)

 

        class\_mask **=** inputs.data.new(N, C).fill\_(0)

        class\_mask **=** Variable(class\_mask)

        ids **=** targets.view(**-**1, 1)

        class\_mask.scatter\_(1, ids.data, 1.)

 

        **if** inputs.is\_cuda **and** **not** self.alpha.is\_cuda:

            self.alpha **=** self.alpha.cuda()

        alpha **=** self.alpha\[ids.data.view(**-**1)\]

        probs **=** (P **\*** class\_mask).sum(1).view(**-**1, 1)

        log\_p **=** probs.log()

        batch\_loss **=** **-**alpha **\*** (torch.pow((1 **-** probs),
self.gamma)) **\*** log\_p

        **if** self.size\_average:

            loss **=** batch\_loss.mean()

        **else**:

            loss **=** batch\_loss.sum()

 

        **return** loss

**class** MyCrossEntropy(nn.Module):

    **def** \_\_init\_\_(self, class\_num):

        **pass**

 

 

 

\#
==================================================================================

\#                                   Config

\#
==================================================================================

os.environ\["CUDA\_VISIBLE\_DEVICES"\] **=** "7"

\_exp\_name **=** "Real\_K-Fold"

myseed **=** 5201314  \# set a random seed for reproducibility

torch.backends.cudnn.deterministic **=** True

torch.backends.cudnn.benchmark **=** False

np.random.seed(myseed)

torch.manual\_seed(myseed)

**if** torch.cuda.is\_available():

    torch.cuda.manual\_seed\_all(myseed)

 

batch\_size **=** 256

\_dataset\_dir **=** "./Data"

num\_layers **=** \[2, 3, 3, 1\] \# residual number layers

alpha **=** torch.Tensor(\[1, 2.3, 0.66, 1, 1.1, 0.75, 2.3, 3.5, 1.1,
0.66, 1.4\]).view(**-**1,1)

 

k\_fold **=** 4

 

 

 

\#
==================================================================================

\#                                       K-Fold

\#
==================================================================================

train\_dir **=** "./Data/training"

val\_dir **=** "./Data/validation"

train\_files **=** \[os.path.join(train\_dir, x) **for** x **in**
os.listdir(train\_dir) **if** x.endswith('.jpg')\]

val\_files **=** \[os.path.join(val\_dir, x) **for** x **in**
os.listdir(val\_dir) **if** x.endswith('.jpg')\]

total\_files **=** train\_files **+** val\_files

 

\# random.shuffle(total\_files)

 

total\_labels **=** \[int(f.split("/")\[**-**1\].split("\_")\[0\])
**for** f **in** total\_files\]

 

\# 2. 将文件和标签转为 numpy 数组，便于索引

total\_files\_np **=** np.array(total\_files)

total\_labels\_np **=** np.array(total\_labels)

 

\# 3. 初始化 StratifiedKFold

\#    我们使用 myseed 来确保分折结果可以复现

skf **=** StratifiedKFold(n\_splits**=**k\_fold, shuffle**=**True,
random\_state**=**myseed)

 

num **=** len(total\_files) **//** k\_fold

 

 

\#
==================================================================================

\#                                   Training

\#
==================================================================================

\# "cuda" only when GPUs are available.

device **=** "cuda" **if** torch.cuda.is\_available() **else** "cpu"

 

\# The number of training epochs and patience.

n\_epochs **=** 300

 

patience **=** 20 \# If no improvement in 'patience' epochs, early stop

 

**for** fold, (train\_indices, val\_indices) **in**
enumerate(skf.split(total\_files\_np, total\_labels\_np)):

    print(f'======================== Starting Fold:{fold}
======================== ')

    model **=** Classifier(Residual\_Block, num\_layers,
num\_classes**=**11).to(device)

    criterion **=** FocalLoss(11, alpha**=**alpha)

    optimizer **=** torch.optim.Adam(model.parameters(), lr**=**0.0005,
weight\_decay**=**1e**-**5)

    scheduler **=**
torch.optim.lr\_scheduler.CosineAnnealingWarmRestarts(optimizer,
T\_0**=**16, T\_mult**=**1)

 

    stale **=** 0

    best\_acc **=** 0

 

    train\_data **=** total\_files\_np\[train\_indices\].tolist()

    val\_data **=** total\_files\_np\[val\_indices\].tolist()

 

    train\_set **=** FoodDataset(tfm**=**train\_tfm,
files**=**train\_data)

    train\_loader **=** DataLoader(train\_set,
batch\_size**=**batch\_size, shuffle**=**True, num\_workers**=**0,
pin\_memory**=**True)

 

    valid\_set **=** FoodDataset(tfm**=**test\_tfm, files**=**val\_data)

    valid\_loader **=** DataLoader(valid\_set,
batch\_size**=**batch\_size, shuffle**=**False, num\_workers**=**0,
pin\_memory**=**True)

 

    **for** epoch **in** range(n\_epochs):

        model.train()

 

        train\_loss **=** \[\]

        train\_accs **=** \[\]

        lr **=** optimizer.param\_groups\[0\]\["lr"\]

        pbar **=** tqdm(train\_loader)

        pbar.set\_description(f'T: {epoch + 1:03d}/{n\_epochs:03d}')

 

        **for** batch **in** pbar:

            \# A batch consists of image data and corresponding labels.

            imgs, labels **=** batch

            \# imgs = imgs.half()

            \# print(imgs.shape,labels.shape)

 

            \# Forward the data. (Make sure data and model are on the
same device.)

            logits **=** model(imgs.to(device))

 

            \# Calculate the cross-entropy loss.

            \# We don't need to apply softmax before computing
cross-entropy as it is done automatically.

            loss **=** criterion(logits, labels.to(device))

 

            \# Gradients stored in the parameters in the previous step
should be cleared out first.

            optimizer.zero\_grad()

 

            \# Compute the gradients for parameters.

            loss.backward()

 

            \# Clip the gradient norms for stable training.

            grad\_norm **=**
nn.utils.clip\_grad\_norm\_(model.parameters(), max\_norm**=**10)

 

            \# Update the parameters with computed gradients.

            optimizer.step()

 

            \# Compute the accuracy for current batch.

            acc **=** (logits.argmax(dim**=-**1) **==**
labels.to(device)).float().mean()

 

            \# Record the loss and accuracy.

            train\_loss.append(loss.item())

            train\_accs.append(acc)

            pbar.set\_postfix({'lr': lr, 'b\_loss': loss.item(),
'b\_acc': acc.item(),

                              'loss': sum(train\_loss) **/**
len(train\_loss),

                              'acc': (sum(\[a.item() **for** a **in**
train\_accs\]) **/** len(train\_accs))

})

 

        scheduler.step()

 

        model.eval()

 

        \# These are used to record information in validation.

        valid\_loss **=** \[\]

        valid\_accs **=** \[\]

 

        \# Iterate the validation set by batches.

        pbar **=** tqdm(valid\_loader)

        pbar.set\_description(f'V: {epoch + 1:03d}/{n\_epochs:03d}')

        **for** batch **in** pbar:

            \# A batch consists of image data and corresponding labels.

            imgs, labels **=** batch

            \# imgs = imgs.half()

 

            \# We don't need gradient in validation.

            \# Using torch.no\_grad() accelerates the forward process.

            with torch.no\_grad():

                logits **=** model(imgs.to(device))

 

            \# We can still compute the loss (but not the gradient).

            loss **=** criterion(logits, labels.to(device))

 

            \# Compute the accuracy for current batch.

            acc **=** (logits.argmax(dim**=-**1) **==**
labels.to(device)).float().mean()

 

            \# Record the loss and accuracy.

            valid\_loss.append(loss.item())

            valid\_accs.append(acc)

            pbar.set\_postfix({'v\_loss': sum(valid\_loss) **/**
len(valid\_loss),

                              'v\_acc': sum(valid\_accs).item() **/**
len(valid\_accs)})

 

            \# break

 

        \# The average loss and accuracy for entire validation set is
the average of the recorded values.

        valid\_loss **=** sum(valid\_loss) **/** len(valid\_loss)

        valid\_acc **=** sum(valid\_accs) **/** len(valid\_accs)

 

        **if** valid\_acc &gt; best\_acc:

            print(f"Best model found at fold {fold} epoch {epoch + 1},
acc={valid\_acc:.5f}, saving model")

            torch.save(model.state\_dict(),
f"Fold\_{fold}\_{\_exp\_name}\_best.ckpt")

            \# only save best to prevent output memory exceed error

            best\_acc **=** valid\_acc

            stale **=** 0

        **else**:

            stale **+=** 1

            **if** stale &gt;**=** patience:

                print(f"No improvment {patience} consecutive epochs,
early stopping")

                **break**

 

 

\#
==================================================================================

\#                                   Testing

\#
==================================================================================

test\_set **=** FoodDataset(os.path.join(\_dataset\_dir,"test"),
tfm**=**test\_tfm)

test\_loader **=** DataLoader(test\_set, batch\_size**=**batch\_size,
shuffle**=**False, num\_workers**=**0, pin\_memory**=**True)

 

models **=** \[\]

**for** fold **in** range(k\_fold):

    model\_best **=** Classifier(Residual\_Block,
num\_layers).to(device)

    model\_best.load\_state\_dict(torch.load(f"Fold\_{fold}\_{\_exp\_name}\_best.ckpt"))

    model\_best.eval()

    models.append(model\_best)

 

prediction **=** \[\]

with torch.no\_grad():

    **for** data,\_ **in** test\_loader:

        test\_preds **=** \[\]

        **for** model\_best **in** models:

            test\_preds.append(model\_best(data.to(device)).cpu().data.numpy())

        test\_preds **=** np.sum(test\_preds, axis**=**0)

        test\_label **=** np.argmax(test\_preds, axis**=**1)

        prediction **+=** test\_label.squeeze().tolist()

 

\#create test csv

**def** pad4(i):

    **return** "0"**\***(4**-**len(str(i)))**+**str(i)

df **=** pd.DataFrame()

df\["Id"\] **=** \[pad4(i) **for** i **in**
range(1,len(test\_set)**+**1)\]

df\["Category"\] **=** prediction

output\_name **=** "submission\_" **+** \_exp\_name **+** ".csv"

df.to\_csv(output\_name,index **=** False)

（3）实验结果及分析。

训练阶段分析：
在引入数据增强的初期，训练Loss下降相对平缓，但随着Epoch的增加，模型逐渐适应了变换后的数据分布。配合余弦退火策略，学习率在每个周期内动态调整，使得Loss曲线呈现出周期性的震荡下行趋势，表明模型正在参数空间中进行精细化搜索。

验证阶段分析：验证集准确率（Validation
Accuracy）是评估模型泛化能力的核心指标。实验中，通过K-Fold得到的四个模型在各自的验证集上均达到了较高的精度。Focal
Loss的使用有效改善了少数类或相似类别的识别效果。

集成效果分析：最终的测试结果表明，相较于任意单折模型的预测，采用模型集成后的分类结果更加稳健。集成策略成功平滑了单体模型的预测噪声，修正了部分模型的片面错误。

本次实验取自Kaggle:[*ML2022Spring-HW3 |
Kaggle*](https://www.kaggle.com/competitions/ml2022spring-hw3b/submissions)，提交结果，分数如下

![](media/image2.png){width="5.76875in" height="1.9444444444444444in"}

**4．本次实验总结体会**

本次实验通过完整的深度学习工程实践，加深了我对计算智能核心概念的理解。深入理解并实践了卷积神经网络的基本范式，包括残差结构的设计与标准训练流程的搭建，确保了模型具备处理复杂图像分类任务的基础能力。

在掌握基础网络架构之后，通过引入 Focal Loss
解决了样本不均衡带来的优化难题，并利用 Stratified K-Fold
和模型集成技术打破了单一模型和单一数据集划分的局限性，认识到算法性能的提升往往源于对数据分布特性的深刻洞察。

**5．对本实验方法的改进建议**

计划可使用迁移学习技术，加载在ImageNet上预训练的ResNet或EfficientNet权重，显著加快收敛速度并提升特征提取的层级。
