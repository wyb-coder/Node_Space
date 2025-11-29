#### 1 *Machine Learning*概论 + *Regression*

##### 1.1 课程概述*What is ML*：==*Looking For Function*==

1.*What is ML*：==*Looking For Function*==

  - ```
    Input：vector、Matrix、Sequence
    ```

  - *Supervised Learning*：有监督学习/有标签(*lables*)

    - 分类、预测、==回归==

  - *Self-supervised Learning*：无监督学习

    - 聚类、**降维**、异常检测
    - *Downstream Tasks*：下游任务
    - *Pre-trained Model(BERT) = Foundation Model* 预训练
    - *Generative Adversarial Network*：生成式对抗学习(有、无混合)，输入大量 `X/Y`，自动学习映射关系
    - *Reinforcement Learning*：强化学习，无从了解如何标签

  - *Anomaly Detection*：异常检测，错误类型输出

  - *Explainable AI*：可解释性，回答为什么

  - *Model Attack*：模型攻击，对图片加入隐藏信息，欺骗模型

  - *Domain Adaptation*：迁移学习的重要分支

    - *Domain Adaptation*

  - *Network Compression*：网络压缩

  - *Life-long Learning*

  - *Meta Learning = Learning to Learning*：机器自主学习发现算法



##### 1.2 机器学习基本概念

1.*What is ML*：==Looking For Function==

  - *Function with Unknown Parameters*
  - $y=b+wx_{1}$：`bias`、`weight`、`feature`
  - *Optimization*：Find $w^{*}，b^{*}$

2.激活函数：*Sigmoid Function*：$c~sigmoid(b+xx_{1})$

- $$
  y=c\frac{1}{1+e^{-(b+wx_{1})}}
  $$

  - $y=b+\sum_{i}c_{i}sigmoid(b_{i}+w_{i}x_{1})$：对单一的 $x_{1}$，做“激活”，则足以逼近任何形式的复杂函数。

  - $$
    y=b+\sum_{i}c_{i}sigmoid(b_{i}+\sum_{j}w_{ij}x_{j})
    $$

  - `递进的过程`，对每一个特征 $(x_{i})$ 激活后，再将所有特征相加，则最终表现为“向量”的形式。

  - $$
    \vec{y}=b+\vec{c}^{T}\sigma(\vec{b}+\vec{w}\vec{x})
    $$

    *ReLU*：$c~max(0，b+wx_{1})$

  - ==非线性 = 不是直线平方也能无限逼近==

3.损失函数：$L(\theta)$，以后，以 $\theta$ 代表未知参数

4.梯度下降：$g=\nabla L(\theta)$

  - 代表函数在参数 $\theta$ 上升`最快的方向`，因此，`参数更新沿着梯度的反方向。`
  - 前向传播和反向传播是逐层的，即计算速度是一层一层传递相同的。但是，每一层有自己 的函数、自己的梯度下降，因此，==不同层、不同节点的梯度大小和收敛速度不一样==
  - ==实际工程中：将全部*N*划分为*batch，*分别计算损失函数 $L_{1}，L_{2}...L_{N}$，每走 完一轮 称为一个*epoch*==
  - *Update*是指一次更新参数，则运行完一个*batch*为更新
  - $(batch=update)\times S=epoch=len(DataSet)$
  - 反向传播：原本的，如果正向计算梯度，每一层的结果依赖于下一层的偏导数，则`直接更换方向`，其实所做的计算量是一样的，只不过让其更符合直觉。并且，反向传播过程中，$\sigma^{’}(z_x)$是一个常数，了解为，`建立了一个反向的神经网络，而激活函数为常数`。
  - *SGD*：随机梯度下降：*Mini-batch + GD*
  - *Batch GD*：全量梯度下降

5.*HyperParameter*：人为设定的参数 == 超参数

  - 例如：*Learning Rate、batch size、Sigmoid*

6.*Neural Network (NN)*：神经网络

  - 有很多隐藏层(*Hidden Layer*)越叠越深*Deep Learning*
  - *Question*：足够“长”的神经网络也能够拟合任何函数，为什么增加深度而不是长度(特征有限?(×)特征决定是每一个神经元中`weight`的数目，而不是有多少个神经元)
  - *MLP(多层感知机)*：*NN*的最基础形式，多层全连接层组成的前馈网络

7.*Regression*：

  - *正则化*：在损失函数中加入 $\lambda$ 项

  - $$
    L=\sum_{n}(\vec{y}-(b+\sum w_{i}x_{i}))^{2}+\lambda\sum(w_{i})^{2}
    $$

  - ==鼓励某个weight更小==

  - 更小的权重意味着对参数的变化更不敏感 ==> 更平滑

##### Homework 1

1.任务描述：COVID-19 Cases Prediction

2.作业实现：[Homework1_COVID_Predict](D:\Study\CS Study\Code\Python\DeepLearning\DeepLearning_LHY\Homework\Homework1_COVID_Predict)

3.具体细节：

  - 手动选择特征列 ==> `sklearn` + `f_regression`自动选取前*k*重要的特征

  - ==LeakyReLU==：避免了*ReLU*的“神经元死亡”问题，因为负区间仍然有梯度

    ```python
    self.layers = nn.Sequential(
        nn.Linear(input_dim, 32),
        nn.LeakyReLU(0.05),
        nn.Linear(32, 16),
        nn.LeakyReLU(0.05),
        nn.Linear (16, 8),
        nn.LeakyReLU(0.05),
        nn.Linear (8, 4),
        nn.LeakyReLU(0.05),
        nn.Linear(4, 1)
    	)
    	#输入维度(每样本数行，特征数列)
    	tensor(batch_size, input_dim)
       	# 输出 神经网络压缩变换的始终是特征数
        tensor(batch_size, 1)
    	x = x.squeeze(1) # (B， 1) → (B) 	仅仅是因为损失函数习惯于该格式
    ```

  - ==L2正则化==：在优化目标里额外加一项`“惩罚大权重”`的约束，鼓励权重变小，避免过拟合。在优化器(*optimizer*)中实现。在更新参数时，`除了梯度下降，还额外加上一个与权 重成正比的衰减项`。优化器在更新参数时自动把*L2*惩罚加进去了。

4.任务指标与完成情况：

  - ![image-20251115120125381](D:\Study\CS Study\Note 2024-\All Photo\image-20251115120125381.png)
  - ![image-20251115120136177](D:\Study\CS Study\Note 2024-\All Photo\image-20251115120136177.png)







---







#### 2 Classification

1.初探分类任务：

  - 为什么`不可以直接当成 Regression`：
    - 某些分类点可能分布较大，导致分割曲线偏向
    - 直接定义分割函数 $(g(x)<0~Output=1...)$，与损失函数(直接统计分类错误的个数)，无法微分，则无法进行梯度下降。
    - 解决办法：`SVM`、`概率`

2.概率解决方法==朴素贝叶斯分类==：

  - 当损失函数不可微时，无法用梯度下降和反向传播来优化，于是我们转向一种概`率建模 + 直接统计`的方法：==朴素贝叶斯分类器==

  - 在*B*已经发生的基础上，*A*发生的概率 $P(A|B)=\frac{P(A\cap B)}{P(B)}$

  - 若*C*由互斥的*A、B*导致，则：
    - $P(C)=P(C|B)P(B)+P(C|A)P(A)$
    - $P(C)=P(C\cap A)+P(C\cap B)$
    
  - ==贝叶斯定理==，已经观测到*C*，反推是由*A*导致的概率：

  - $$
      P(A|C)=\frac{P(C|A)\times P(A)}{P(C)}
      $$

  - 朴素贝叶斯并不涉及“反向传播”，只统计参数。并不是观测到标签以优化特征权重，仅仅 做前向传播。

  - `朴素的统计每个样本的比例(先验概率) => 在某一类中某个特征出现的频率(条件概率) => 看到特征后，反推最可能的类别(贝叶斯 + 测试集输入)`

  - 朴素：假设特征之间相互独立

  - 显然，统计只应用于离散值(是否出现、出现数量)，针对连续特征，引入高斯分布，`假设在某个类别下，特征的分布服从某种概率分布`

      - `高斯分布：建模“特征在某个类别下的分布”。用概率密度来扮演“频率”的角色。每个类别有自己的高斯分布 ==> 自己的频率`
           - 计算 $P(C|A)$ 需要 $P(A|C)$

  - ==*Maximum Likelihood*(最大似然估计)==：选择概率分布，选择参数，使得在这个参数下，观测到这些样本数据的概率最大。

    - 最优均值：
      $$
      \mu^{*}=\frac{1}{N}\sum_{n=1}^{N}x^{n}
      $$
      最优协方差(矩阵)：
      $$
      \Sigma^{*}=\frac{1}{N}\sum_{n=1}^{N}(x_{n}-\mu^{*})(x_{n}-\mu^{*})^{T}
      $$
      
    - 输入*X*维(*X*个特征)，则 $\mu$ 为 $1\times X$
    
    - ==极大似然估计==：假设数据是由某个分布 $P_{\theta}(x)$ 生成的，其中 $\theta$ 是待估计参数。
    
      - 则给定训练数据 $x_{1}...x_{N}$ 的联合概率为：$L(\theta)=\prod_{i=1^{N}}P_{\theta}(x_{i})$ 即为似然函数
      - 极大似然估计：选择参数，使得观测到的数据在模型下的概率最大。$argmaxL(\theta)$
    
  - ==朴素贝叶斯，分类任务是以结果的类别来分开的，因此，是每个类别有一个自己的高斯分布每一个类别的所有特征都是该高斯分布的输入 ==> 最优均值、协方差可能是矩阵的形式。==

  - 每个类别都有自己的分布 ==> 模型参数过多 ==> 过拟合

  - ==高斯判别分析(GDA)==：

    - 所有的特征`使用统一的协方差矩阵，以减少模型参数，以对抗过拟合`
    - 不再假设特征独立，而是直接建模整个特征向量 $X$ 的联合分布，整个特征向量共享一个多维高斯分布。
    - 对原本的协方差矩阵做`加权平均`
    - 能捕捉特征相关性、可解释性强
    - 在高斯判别分析的基础上，实际上，通过化简，可以化为 $z=w^{T}+b$也就是`linear`的。

  - ![image-20251115121445737](D:\Study\CS Study\Note 2024-\All Photo\image-20251115121445737.png)

  - ![image-20251115121456940](D:\Study\CS Study\Note 2024-\All Photo\image-20251115121456940.png)

  - 显然，通过朴素贝叶斯分布，通过概率来拟合 $z=w^{T}+b$，而实际上，我们可以**通过其他方式找到*weight*与*bias*，即逻辑回归。**

  - ==高斯判别(自行计算参数$\mu$、$\Sigma$) + 梯度下降 = 逻辑回归==

3.==逻辑回归(*Logistic Regression*)==：

  - 回归 != 预测。线性回归主要用于预测。回归输出连续值，而`不同种类的回归可以利用该值做分类。`

  - 根据上述高斯判别分析，想到通过线性回归预测出概率值，而原本的输出概率通过转换，也就得到了*Sigmoid*激活函数。

  - `并不是凭空提出的*Sigmoid*函数，而是通过对高斯判别分析的公式转换，发现除了线性回归之外，存在一个转化，才得到了*sigmoid`。*

  - 因此，将线性回归引入，避免了复杂的求概率过程，而是通过前向传播后向传播训练模型(最简神经网络)，优化前半部分 + *Sigmoid*，最终 ==> 逻辑回归。

  - ==逻辑回归 == 线性回归 + Sigmoid==

  - 逻辑回归的损失函数：==交叉熵(Cross entropy)==
    - 交叉熵比较的是，输出向量$y^{’}$与`独热编码`之间的距离
    - 一个分布*q*去近似真实分布*p*时，所需的**平均编码长度**。交叉熵的推导：
    - ![image-20251115121910603](D:\Study\CS Study\Note 2024-\All Photo\image-20251115121910603.png)
    
    - $L(w, b)$：逻辑回归的似然函数，即整批数据的联合概率.
    
    - 显然，似然函数 $\uparrow$ 对分类置信度更高 ==> 模型越好
    
    - 该似然函数通过等价变换 <=> *NLL*(负对数似然) <=>最小化损失。
    
    - ![image-20251115122035741](D:\Study\CS Study\Note 2024-\All Photo\image-20251115122035741.png)
    
    - ![image-20251115122047562](D:\Study\CS Study\Note 2024-\All Photo\image-20251115122047562.png)
    
      - 补充：`伯努利分布 == 二元事件分类`
    
      $$
      P(y|x;w，b)=p^{y}(x)\times(1-p(x))^{1-y}
      $$
      
      - 这也就是为什么似然函数损失函数 $L(w,b)$ 会长上面的样子。
      
    - 为什么`不使用平方误差：波动过小，不利于梯度下降`，收敛过慢。
    
  - `逻辑回归不使用任何假设(高斯、伯努利)，因此区别于高斯判别`

  - *Discriminative*：判别式，直接学习后验概率或决策边界，而不同于生成模型(*Generative*)，来直接计算生成过程。

      - 判别式模型(*Discriminative*)：不关心“类是怎么生成的”，只关心“如何区分它们”。逻辑回归、SVM、神经网络
      - 生成式模型(*Generative*)：先学会“每个类长什么样”，再用贝叶斯公式算。朴素贝叶斯、GDA


4.==多分类(Multi-class Classification)==：

  - *Softmax*激活函数：$y\in[0，1]$
    - `输出与输入维度等大小的向量。将数值解释为概率`

  - 针对逻辑回归某些无法处理的情况，可以做特征变换，让机器自己做，则为 *cascade logistic regression*.
  - 所谓*cascade logistic regression*(级联逻辑回归)，本质上就是把多个逻辑 回归模型按层次串联，每一层对输入做特征变换或非线性映射，再交给下一层。这样 逐层堆叠的结构，就**自然过渡成了神经网络。**

5.*Model*训练：

  - **极小值*(local minima)*、鞍点(*saddle point*)**：都属于*critical point*，但鞍点仍旧存在下降的空间。
    
    - 区分方法：对于Loss不再变化的某一点，利用泰勒近似该点的函数。
    - $L(\theta)\approx L(\theta^{\prime})+\frac{1}{2}(\theta-\theta^{\prime})^{T}H(\theta-\theta^{\prime})$ 通过正负情况。
    - $H$为正定矩阵 == *Local minima*，其余同理。
    - *==H \== Hessian==*：海森矩阵：由多元函数的二阶偏导数组成的方阵，用来描述函数在某点的局部曲率特性。
    - 在三维图像中是极小值，但在四维空间中，可能存在路径可走。
    
  - *Batch*：

    - ==*Shuffle*==：随机分*batch*，每个*epoch*的*batch*内部不同。
    - ==更小的batch，梯度更新幅度通常更大、更“抖动”==

  - *Momentum*(动量)：

    - 不止向梯度的反方向移动参数。上一步的方向 + 梯度反方向的矢量和
    - *NAG、Adam*

  - *Adaptive Learning Rate*(自适应学习率)：每一个参数不同的学习率并自动适应性调整。

    - 调整依据：参数的历史梯度信息，==调整的是每个参数自己的学习率==。无需人工设计，自动微调

    - *Loss*变化幅度很小，但*gradient*变化幅度会很大：震荡

    - 在某方向上陡峭(变化幅度大)：期望学习率较低，反之
    - 通过数学实现这一点(*AdaGrad*)：$\theta_{i}^{t+1}\leftarrow\theta_{i}^{t}-\frac{n}{\sigma_{i}^{t}}g_{i}^{t}$
    - 其中
      $$
      \sigma_{i}^{t}=\sqrt{\frac{1}{t+1}\sum_{i=0}^{t}(g_{i}^{t})^{2}}
      $$
      
    - ==*RMSprop*(二阶动量)==：改进*AdaGrad*，用指数加权平均代替累积和，避免学习率过快衰减。更适合深度网络。

      - 为什么说是动量，本质都是更新梯度的方法。

    - *Adam == RMSProp + Momentum*

      - 变体：*AdamW、Nadam、AMSGrad*

    - 单纯的*AdaGrad*：引入问题，动量积累，导致一段时间内震荡。

      - 解决方法：*Learning Rate Scheduling*

  - ==*Learning Rate Scheduling*(学习率调度)==：

    - 按照预先设定的规则随训练进度改变学习率。调整的是`全局学习率`。`人为设计`的调度策略。
    - *Learning Rate Decay*：随时间逐步减小(快到终点)
    - ==*Warm Up*==：*Increase and then Decrease*(先变大后边小)，*Magic Function*，先在局部搜索与统计。
    - 与*Adaptive Learning Rate*相结合，一者负责全局，另一者做局部修正。

6.==New Optimizers==：

  - *SGD*：随机梯度下降
  - *SGD with Momentum*：记录历史梯度信息，保留“动量”
  - *Adagrad*：≠ *Momentum*：维护历史梯度平方和
  - *RMSProp*：指数加权平均代替累积和
  - *Adam*：*RMSProp + Momentum*：加入偏差校正，兼顾收敛速度与稳健性



##### Homework 2

1.任务描述：*Sound Classification*

2.作业实现：[Sound Classification](D:\Study\CS Study\Code\Python\DeepLearning\DeepLearning_LHY\Homework\Homework2_Sound_Classification)

3.*具体细节*：

  - *nn.BatchNorm1d(output_dim)*：对全连接层输出的每个特征维度做批量归一化，让训练更快、更稳、更容易收敛。
  - ==*concat_nframes*(帧拼接) *vs LSTM*(长短期记忆网络)：==
    - ==*concat_nframes*：==强制修改输入数据，将上下文(局部上下文)也作为输入。
    - ==*RNN*：循环神经网络==，也能使用上下文信息。每一步将当前输入和上一步的隐藏状态结合。
    - 在长序列中，梯度在反向传播时会逐渐衰减或爆炸，导致模型难以学习长期依 赖。
    - *==Bi-LSTM==*：动态上下文，利用整个序列的前后信息。*RNN* + 门控机制
    - 门控机制：三个门(遗忘门、输入门、输出门)和一个细胞状态。
    - 在长序列中“有选择地记忆和遗忘”，从而解决长期依赖问题。

4.任务指标与完成情况：

- ![image-20251115123753028](D:\Study\CS Study\Note 2024-\All Photo\image-20251115123753028.png)
- ![image-20251115123804186](D:\Study\CS Study\Note 2024-\All Photo\image-20251115123804186.png)








---





#### 3 卷积神经网络(*Convolutional Neural Network*)

1.*CNN*概述：专精于`图像处理`，同样可能适用于与图像处理存在共通的领域。

  - ==设定区域(*Receptive field*)==每一个神经元仅仅关注其对应区域。
    - 朴素的类人思考，学习图片中的某些特点特征(*Pattern*)

  - *field*的重叠、不同大小、考虑不同*RGB*通道等，都是被允许的。其完全由设计者自定义 ==> `一个区域对应一个/多个神经元`
  - *kernel size*：*Receptive field*的尺寸
  - *stride*：区域移动步长
  - *padding*：值补充，区域在步长的移动下，超出区域，则补充
  - 问题：某个特征出现在图片的不同位置，则某些神经元职能重复
    - 共享参数：某两个神经元使用完全一致的*weight、bios*，在神经网络内部神经元中修改。
    - 使用==卷积核(Filter)==：*Receptive Field + Parameter Sharing*每个卷积 核对应一个特征，扫面全部图像。实际为一个可学习的小矩阵，它通过卷积操作在输 入上滑动，提取不同的局部特征。多个*filter*组合起来，就能逐层构建出从低级纹理到高级语义的特征表示。

  - *==Convolutional Layer==*：使用*Filter*的层。
  - 卷积层的叠加，会**逐步学习更大范围的*Pattern***，每一次卷积都是在之前卷积的基础上**更进一步**的。
  - *Pooling*：基于*subsamping*思想，特定的采样不会对图片(*pattern*)有很大的影响。
    - *No Learning*，固定的操作
    - *Max Pooling、Min Pooling*，不同池化方法。
    - 基本思路：将图片变小，减少运算量。子采样(*subsamping*)
    - 与*Convolutional*交替使用。

  - ***Flatten***：拉伸为一维向量。

2.*CNN*应用：围棋：考虑为*CLassification*，预测下一个棋子的“类别”

  - ==*AlphaGo*：*CNN*为基础==，监督学习 + 强化学习 + 价值网络 + 蒙特卡洛树搜索 (*MCTS*)
  - ==在懂原理的基础上，认识到某一部分带来的抽象/具体的影响，进一步自主的设计网络结构。例如*AlphaGo*不使用*Pooling*==
  - *CNN*对图片增强技术(放大、缩小)等，应对的不好

3.*Why DeepLearning work Gooooooood?*

  - *DeepLearning*：较少的参数，得到相对较低的*Loss*

  - 单一*Hidden Layer*`即可拟合任意精度函数`，隐藏层 == 神经元 + 激活函数。也需注意，显然，每个神经元的*weight*的个数取决于输入权重的数量，但每个神经元的*weight*是其完全独立的。

    - 理解为，*Full Connection*，`每一个特征都完整的输入了每个神经元，每个神经元独立的负责某一段的曲线，将所有曲线拼合，则得到全部的函数。`

    $$
    y(x)=\sum_{i=1}^{N}w_{i}\times\sigma(a_{i}x+b_{i})+c
    $$

  - 回到之前的问题，为什么是*DeepLearning* 而不是*FatLearning*?

    - *Deep Structure*：效率更高，较少参数，更好效果，更快速度。
    - 深层网络能够逐层提取特征，不同幅值不同、特征之间本身差距巨大的交由不同的层 来拟合。==逐步的拟合、拟合程度逐步精细，已经效果很好的(*Loss*)则无需再深入==。
      - 类比：折纸后剪窗花 *vs* 直接剪

    - *FatLearning*：`暴力使用巨量参数直接拟合目标函数`，泛化性差、同任务目标下所需 参数多。更多的参数，则显然更易*overfitting*


4.*==Spatial Transforms Layer (Neworks) == (STN)==*：空间变换器，可嵌入神经网络中的可微分模块。令网络自动学习并执行几何变换。

  - *CNN*不很好的支持平移，几乎完全不支持旋转、缩放、裁剪
    - ***CNN*本身具有平移等变性，但池化、边界效应与下采样等会致使该特性失效。**况且， 即使具有对平移很好的适应，对其他几何变换不适应而适应其中一种，也是完全不重 要的，不如找到方法`提升对所有几何变换的泛化。`

  - 也是一个*Layer*，由*weight、bios*组成，可训练。通过对*weight*做针对性微调，以决定几何变换的映射关系。
    - 对*weight*做“微调”，实际上，就是将*Spatial Transforms Layer*的输出，`送入一个参数已冻结的/根据其变化的 NN 中`
    - 归根结底，`几何变换 === 对系数矩阵做乘法`
    - 则目标为，找出该系数矩阵，令神经网络拟合该矩阵

  - ==令网络自动学习并执行几何变换。的意思是，帮助*CNN*将不是正确角度等几何变换的图片，转换为正确角度==，因此，经过*Spatial Transforms Layer*，*CNN*得到了几乎固定的输入。



##### Homework 3

1.任务描述：*Food Image Classification*

2.作业实现：[Homework3_Image_Classification](D:\Study\CS Study\Code\Python\DeepLearning\DeepLearning_LHY\Homework\Homework3_Image_Classification)

3.具体细节：

  - *Training Augmentation*(数据增强)：
  - *Cross Validation*(交叉验证)：把数据集划分成多个子集，轮流作为训练集和验证集，最终综合各次验证结果来评估模型性能。
    - *K-Fold、LOOCV...*：*K*折交叉验证、留一法、留P法

  - *Ensemble*：集成学习，将多个模型的预测结果组合起来，形成一个更强的整体模型。
    - *Bagging*：有放回的随机采样(*Random Forest*)
    - *Boosting*：AdaBoost、Gradient、Boosting、XGBOOST、LightGBM


4.任务指标与完成情况：

  - ![image-20251115125140377](D:\Study\CS Study\Note 2024-\All Photo\image-20251115125140377.png)
  - 



















