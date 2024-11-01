## 日志名称说明
2-8cifar100_IB0.1_Centralized_tasks7_Alpha0.5_components1
```
2-8
cifar100 数据集名称（共有三个cifar10、imageNetLT）
IB0.1 数据不平衡程度（共有三种IB0.5，IB0.01）
Centralized （分为两种，Decentralized）
tasks7（组成联邦的计算机数量）
Alpha0.5    （狄利克雷分布函数的浓度参数）
components1 （组件，此处默认为不分组）
```

# 实验笔记
### 改动：
utils/ResNet32Feature.py 为imageNet-LT单独定义了骨干网络
utils/ResNet50Feature.py 增加了BBN_ResNet50_model（）骨干网络
utils/loss.py 增加聚焦损失函数
utils/args.py 添加新的超参数
取消了所有骨干网络使用预训练权重
## 1-去除预训练模型带来的不公
修改文件：/home/yang/Desktop/code/FL_longtail/Casual_fedavg_INET/models.py
修改位置：97行左右，共有8处，model = models.vgg11(pretrained=False)


## 2-screen查看记录
step1 screen 模式下 Ctrl + a
step2 松开后按 [ 或者 Esc
step3 即可进入光标模式 按 ↑ 或者 ↓ 移动光标浏览历史信息 也可以 PgUp 或者 PgDn 快速翻页
step4 退出光标模式 Ctrl + C

## 维度计算版本问题
learners/learner.py
做了如下替换：
# loss = (loss_vec.T @ weights[indices]) / loss_vec.size(0)
loss = (loss_vec * weights[indices]).sum() / len(loss_vec)

## 2024-1-10 跑imageNet：
FedEM卡死在10个epoch处

## 2024-1-27
将FedEM移动到最下方，调小btchsize=16.

## 2024-1-30 
FedEM报错：
#loss = (loss_vec.T @ weights[indices]) / loss_vec.size(0)
#这两种方法在数学上是等价的，但在实现上略有不同。使用元素乘法方法更直观，因为它明确地表示了每个损失值与其相应权重的乘积，然后对这些乘积求和。而矩阵乘法方法则使用了线性代数的点积概念。
#如果调整为*后不启用CUDA断言无法训练FedEM
loss = (loss_vec * weights[indices]).sum() / len(loss_vec)

RuntimeError: CUDA error: device-side assert triggered
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

### 尝试用cup跑，找出错误原因
最大维度问题，在learner.py做了如下修改:
```py
'''
    nn.CrossEntropyLoss返回一个标量值，而Focal Loss可能返回一个向量，
    其中每个元素对应于输入数据的一个样本。需要确保Focal Loss返回的是一个向量.
    '''
    # if len(weights[indices].shape) == 0:
    #      weights[indices] = weights[indices].expand_as(loss_vec)
    '''
    # 但是这样修改报了一个维度错误：
    # index 210 is out of bounds for dimension 0 with size 42     
    # 首先需要检查indices中的值是否超出了weights张量的大小范围。你可以使用以下方法来检查并修正
    '''
    max_index = weights.size(0) - 1
    indices = torch.clamp(indices, 0, max_index)
    if len(weights[indices].shape) == 0:
        weights[indices] = weights[indices].unsqueeze(0).expand_as(loss_vec)
```
## 之前的实验没有使用ResNet50而是34，如果效果不佳，需要重新跑：
utils_old/utils.py 中已经切换为使用resnet50

## 2月1日
可行后需要在cifar10、100上测试curi的敏感度。
执行：
shells/m2d1_CIFAR10_alph04_component1_centrolized.sh
测试curi在CIFAR10上的

### 目前取得的curi范围比较宽，0-10，步长0.2，共有50个实验。需要观察完curi合适区间再做定夺
目前单独拿出来，BBN32网络效果是最好的去长尾设置，修改学习率，查看是否是没有调参的原因；
学习率设置为0.01确实比0.005强

之前的IB10精度可达0.88，目前尝试换网络为resnet18进行重新实验。
具体原因不明

## 2月2日，ResNet34使用Pertrain熟悉来训练网络。
shells/m2d2_CIFAR10_alph04_component1_centrolized_ResNet18.sh
warm_up+curi的curitemp参数是没有规律的

## 2月3日，ResNet34使用Pertrain熟悉来训练网络，在此基础上，单独测试curiavg的真正合适的curi
bash shells/m2d3_CIFAR10_alph04_component1_centrolized_ResNet34_IB10-50-100.sh



## 2月4日，IB10时合适的curi范围在0.01-1之间，修改了IB50IB100的实验范围
IB10最佳效果为curi=0.28附近

继续训练IB10-IB100
bash shells/m2d3_CIFAR10_alph04_component1_centrolized_ResNet34_IB10-50-100.sh



## 2-5 测试cifar100——centrolized训练效果
export n_components=1
export n_tasks=10
export alpha=0.6
当设置参数为以上内容时，IB10的CIFAR100最高在focalloss方法上，得到精度为0.34左右，需要提高到45%以上才行。


## 2-8重新处理cifar100的实验，测试一个合适的task数量
发现当task设置为7，alpha=0.5，会得到最优精度

## 2-9,开始重新训练CIFAR100的IB10-100
设置如下：
export n_tasks=7
export alpha=0.5
。。。
export curi=0.01
while [ $(echo "$curi < 0.45" | bc) -eq 1 ]
    。。。
    curi ADD
done

另外增加了RSloss的实现，预计55*3*20min =完成实验室间预计2.5天

## 2-11 2月9日的实验在2-8开头的日志文件中，结果可用于论文
### 补充：fadavg，IB50时的精度似乎没有IB100高，需要重新泡一下IB50的avg，--use_RSloss还没有使用，可以加上
运行：shells/m2d3_CIFAR10_alph04_component1_centrolized_ResNet34_IB10-50-100_add.sh
IB10的avg：0.6874  RSloss：0.7374
IB50的avg：0.6598  RSloss：0.5094
IB100的avg：0.453  RSloss：0.5347
重新跑Rsloss，IB50的RSloss：0.5978 IB100的RSloss：0.4875
### 2-13CIFAR10 decentralized 正常跑完实验




## 2-13 CIFAR100 decentralized call an error:

### utils_old/decentralized.py line 37: mixing_matrix *= adjacency_matrix
```sh
mixing_matrix *= adjacency_matrix
TypeError: unsupported operand type(s) for *: 'NoneType' and 'int'
```
问题在于，使用decentralized方法，多个task客户端通过p=0.5 的厄多斯-雷尼图相连，task数量并不能随意，不然会造成adjacency_matrix的维度不对无法聚合
修改n_components=1为n_components=3，可正常训练。

### 但是出现无法收敛的情况，鉴于2-5日日志：
export n_components=1
export n_tasks=10
export alpha=0.6
当设置参数为以上内容时，IB10的CIFAR100最高在focalloss方法上，得到精度为0.34左右，需要提高到45%以上才行
### 做一个测试：
export n_components=1
export n_tasks=10
export alpha=0.6
测试facalloss与avg在10.50.100IB上的精度:
IB50结果为avg：2.77  avg_localstep3:2.66  focalloss：2.96 精度还是太低

### 另一个测试，
task设置为8则是用7个客户端进行雷尼图相连
export n_components=1
export n_tasks=8
export alpha=0.5
可以正常运行，猜测结论：
```py
def compute_mixing_matrix(adjacency_matrix):
    network_mask = 1 - adjacency_matrix
    N = adjacency_matrix.shape[0]

    s = cp.Variable()
    W = cp.Variable((N, N))
    objective = cp.Minimize(s)

    constraints = [
        W == W.T,   #确保混合矩阵 W 是对称的。
        W @ np.ones((N, 1)) == np.ones((N, 1)), #确保混合矩阵的每一行之和为1。
        cp.multiply(W, network_mask) == np.zeros((N, N)),   #强制网络中没有边的地方混合矩阵的对应元素为0
        -s * np.eye(N) << W - (np.ones((N, 1)) @ np.ones((N, 1)).T) / N,
        W - (np.ones((N, 1)) @ np.ones((N, 1)).T) / N << s * np.eye(N), #通过这两个约束，将 W 限制在一定范围内，以确保最小化目标函数 s
        np.zeros((N, N)) <= W   #确保混合矩阵中的元素都是非负的
    ]
    prob = cp.Problem(objective, constraints)   #创建一个 CVXPY 问题实例，包括目标函数和约束条件。
    # prob.solve()    #求解问题，找到使得目标函数最小化的混合矩阵
    prob.solve(solver=cp.CVXOPT)  # 尝试不同的求解器，例如 CVXOPT
    #调试mixing_matrix即W=None的问题：
    print("优化状态:", prob.status)
    mixing_matrix = W.value #获取求解得到的混合矩阵的值
```
调试结论为print("优化状态:", prob.status)打印出没有优化成功，下一步切换凸优化器进行改进


### 更换策略，修改产生连边的系数，
/media/yang/Data_Set/FL/V26-FL/utils_old/utils.py line 693：
mixing_matrix = get_mixing_matrix(n=n_clients, p=0.5, seed=seed)
修改概论为0.5进行测试：
```sh
adjacency_matrix=
[[0 0 1 1 0 0]
 [0 0 0 0 1 0]
 [1 0 0 1 1 0]
 [1 0 1 0 1 0]
 [0 1 1 1 0 0]
 [0 0 0 0 0 0]]
修改概论为0.6进行测试：
[[0 0 1 1 0 0]
 [0 0 1 0 1 0]
 [1 1 0 1 1 0]
 [1 0 1 0 1 0]
 [0 1 1 1 0 0]
 [0 0 0 0 0 0]]
修改概论为0.7进行测试：
adjacency_matrix=
 [[0 0 1 1 0 0]
 [0 0 1 1 1 0]
 [1 1 0 1 1 0]
 [1 1 1 0 1 1]
 [0 1 1 1 0 1]
 [0 0 0 1 1 0]]
```

结论，如果有一个节点不与任何其他节点相连，则会优化失败，task为7时1个作为专用测试节点，剩余6个交换权重
至少连接概率要为0.7，才能保证每个节点至少会有一个连线。

### 切换为CVXOPT
conda install -c conda-forge cvxopt
prob.solve()    #求解问题，找到使得目标函数最小化的混合矩阵
prob.solve(solver=cp.CVXOPT)  # 尝试不同的求解器，例如 CVXOPT

同样是无解的

### 开始雷尼图连度概率为0.7的测试语句结果在2-15日完成
```
mixing_matrix = get_mixing_matrix(n=n_clients, p=0.5, seed=seed)
```
export n_components=1
export n_tasks=7
export alpha=0.5

## 2-15
完成data/imageNet_LT/conut_tain_class_num.py文件编写，可以输出不同数量的类别分组，并划分val集合：
```sh
group few count= 136
...
group medium count= 473
...
group many count= 391
...
for all = 1000
```
完成shells/m2d15_CIFAR10_alph05__component1_decentrolized_erdos-renyi-P07_ResNet34_IB10-50-100.sh

## 2-16
### 开始运行shells/m2d15_CIFAR10_alph05__component1_decentrolized_erdos-renyi-P07_ResNet34_IB10-50-100.sh
但是日志存在2-5日
### 16号服务开始训练imageNet centralized
export n_tasks=10
export alpha=0.5
--s_frac 1.0  --tr_frac 0.8
val为imagenet_LT的val.txt原版复制了10份


## 2-18 decentrolized CIFAR10 数据集运行完毕

### 开始运行centralized CIFAR10，同样将alpha调整到0.5
同时增加了fedavg的动量衰减为变量，为证明不同IB时，momentu影响了模型学习稀有尾部样本



## 2-22 CIFAR-10/100 所有实验结束，多余的实验结果移动到logs/logbackup
需要找个人统计数据！！
### 2-22日开始进行imagenet LT 的实验

设置了avg的动量衰减分别是0.1  0.5  0.9
测试curi从0.26-0.40


## 待做
fadavg，IB50时的精度似乎没有IB100高，需要重新泡一下IB50的avg
--use_RSloss还没有使用，可以加上(已完成)
### 需要补充检测


使用resnet 50在imageNet_LT上很好，还需要测试一下去中心化情况是否全都可行

2-13日解决凸优化雷尼图没有解的bug，但是cifar10的Decetralized实验是否需要重新跑一次

centrolize时，貌似很多方法精度不够高，后续需要验证一次

测试去中心化的CIFAR10训练
m2d5_CIFAR10_alph04_component1_decentrolized_ResNet34_IB10-50-100.sh

local step 作为变量，需要进一步测试

最后需要选中最优的curi、测试所有可能情况（IB）
将imagenet分为many-medium-few三个部分

（重要）验证在不适用动量时，few精度会高

## 8-30 存在两个关键问题，
imagenet上是否改变动量，精度不变
fagavg精度最高。
数据扰动有问题，
开始测试：
### data/imageNet_LT/generate_data.py，修改为只用train。测试decentrolized
### V27服务器，/home/yang/Desktop/code/FL_longtail 修噶为file_names = ['ImageNet_LT_train.txt', 'ImageNet_LT_test.txt', 'ImageNet_LT_val.txt']全部使用

## 无用的数据修改与网络修改，8-6开始测试换不打乱数据集，并且添加更多的mu数据
http://154.17.26.12:51860/api/v1/client/subscribe?token=30c8e358eca5d266f61fc677802f0505