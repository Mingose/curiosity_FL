from traceback import print_tb
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.optimizer import Optimizer, required
import numpy as np
from  learners.learner import  *
import math

from .Warmup import WarmupMultiStepLR

from utils_old.args import *

from collections import Counter

# # 在初始化优化器时设置滑动窗口大小
# from collections import deque

args = parse_args()

class ProxSGD(Optimizer):
    r"""Adaptation of  torch.optim.SGD to proximal stochastic gradient descent (optionally with momentum),
     presented in `Federated optimization in heterogeneous networks`__.

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Attributes
    ----------
    params (iterable): iterable of parameters to optimize or dicts defining parameter groups
    lr (float): learning rate
    mu (float, optional): parameter for proximal SGD
    momentum (float, optional): momentum factor (default: 0)
    weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    dampening (float, optional): dampening for momentum (default: 0)
    nesterov (bool, optional): enables Nesterov momentum (default: False)

    Methods
    ----------
    __init__
    __step__
    set_initial_params

    Example
    ----------
        >>> optimizer = ProxSGD(model.parameters(), lr=0.1, mu=0.01,momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input_), target_).backward()
        >>> optimizer.step()

    """

    def __init__(self, params, lr=required, mu=0., momentum=0., dampening=0.,
                 weight_decay=0., nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(ProxSGD, self).__init__(params, defaults)

        self.mu = mu

        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['initial_params'] = torch.clone(p.data)

    def __setstate__(self, state):
        super(ProxSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p = d_p.add(p.data, alpha=weight_decay)

                param_state = self.state[p]
                if momentum != 0:
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                # add proximal term
                d_p.add_(p.data - param_state['initial_params'], alpha=self.mu)

                p.data.add_(d_p, alpha=-group['lr'])

        return loss

    def set_initial_params(self, initial_params):
        r""".
            .. warning::
                Parameters need to be specified as collections that have a deterministic
                ordering that is consistent between runs. Examples of objects that don't
                satisfy those properties are sets and iterators over values of dictionaries.

            Arguments:
                initial_params (iterable): an iterable of :class:`torch.Tensor` s or
                    :class:`dict` s.
        """
        initial_param_groups = list(initial_params)
        if len(initial_param_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")
        if not isinstance(initial_param_groups[0], dict):
            initial_param_groups = [{'params': initial_param_groups}]

        for param_group, initial_param_group in zip(self.param_groups, initial_param_groups):
            for param, initial_param in zip(param_group['params'], initial_param_group['params']):
                param_state = self.state[param]
                param_state['initial_params'] = torch.clone(initial_param.data)





class LMSGD(Optimizer):
    def __init__(self, params, lr=0.001, momentum=0, dampening=0, weight_decay=0, curi=0, nesterov=False):
        """
        LMSGD 优化器，支持按需计算 total_y_value。
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, curi=curi, nesterov=nesterov, use_rare=False)
        super(LMSGD, self).__init__(params, defaults)
        self.current_epoch = 0

    def __setstate__(self, state):
        super(LMSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            group.setdefault('use_rare', False)

    def update_epoch(self, epoch):
        """更新当前的 epoch 值"""
        self.current_epoch = epoch

    @torch.no_grad()
    def step(self, closure=None, y=None):
        """
        执行一次优化更新。
        如果某些参数组需要计算稀有度得分（use_rare=True），则传递 y（类别标签）到 total_y_value。
        """
        loss = closure() if closure is not None else None

        # 仅在需要计算稀有度得分时调用 total_y_value
        if y is not None and any(group['use_rare'] for group in self.param_groups):
            self.total_y_value(y)

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            lr = group['lr']
            dampening = group['dampening']
            nesterov = group['nesterov']
            curi = group['curi']
            use_rare = group['use_rare']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad

                state = self.state[p]

                # 如果 use_rare 为 True，则获取稀有度得分
                rare_lr, rare_momentum = 0, 0
                if use_rare:
                    y_score_all = state.get('y_score_all', torch.tensor(1.0, device=d_p.device))
                    # print(y_score_all)
                    rare = state.get('y_sample_size_min', torch.tensor(1.0, device=d_p.device))

                    # 计算 rare_lr 和 rare_momentum
                    rare_lr = torch.tanh(1 / rare) * curi * (0.1 ** (self.current_epoch // 20))
                    rare_momentum = 1 * y_score_all * (0.1 ** (self.current_epoch // 10))

                # 初始化或更新 momentum buffer
                if 'momentum_buffer' not in state:
                    buf = state['momentum_buffer'] = torch.clone(d_p).detach()
                else:
                    buf = state['momentum_buffer']
                    if use_rare :#and y_score_all <= 0.69:
                        buf.mul_(momentum * (1 - rare_momentum)).add_(d_p, alpha=(1 - dampening))
                    else:
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)

                if momentum != 0:
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                # 更新参数
                p.add_(d_p, alpha=-lr * (rare_lr + 1))

        return loss

    def total_y_value(self, y, max_score=1.0):
        """
        仅在参数组的 use_rare 为 True 时计算稀有度得分。
        """
        if len(y) == 0:
            raise ValueError("Optimizer got an empty parameter list")

        # 将 y 转换为列表并统计当前 batch 中的类别出现次数
        list_y = y.tolist()
        y_counts = Counter(list_y)

        for param_group in self.param_groups:
            if not param_group.get('use_rare', False):
                continue

            if 'cumulative_y_classes' not in param_group:
                param_group['cumulative_y_classes'] = Counter()

            cumulative_y_classes = param_group['cumulative_y_classes']
            cumulative_y_classes.update(y_counts)

            rare_class_count_batch = min(y_counts.values(), default=0)
            common_class_count_batch = max(y_counts.values(), default=1)

            y_sample_size_min = rare_class_count_batch / common_class_count_batch

            num_classes = len(cumulative_y_classes)
            P_ideal = 1.0 / num_classes if num_classes > 0 else 0

            total_count_batch = sum(y_counts.values())
            P_actual = torch.zeros(num_classes, device='cuda')
            for cls, count in y_counts.items():
                if cls < num_classes:
                    P_actual[cls] = count / total_count_batch

            TVD = 0.5 * torch.sum(torch.abs(P_actual - P_ideal))

            y_score_all = max_score * (1 - TVD)
            score_tensor = torch.tensor(y_score_all, device='cuda')

            for param in param_group['params']:
                param_state = self.state[param]
                param_state['y_sample_size_min'] = torch.tensor(y_sample_size_min, device='cuda')
                param_state['y_score_all'] = score_tensor.clone().detach()

        return score_tensor





class LMSGD2(Optimizer):
    def __init__(self, params, lr=required, momentum=0, dampening=0, weight_decay=0, curi=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, curi=curi, nesterov=nesterov)
        super(LMSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(LMSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
    
    def update_epoch(self, epoch):
        """更新当前的 epoch 值"""
        self.current_epoch = epoch
    

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = closure() if closure is not None else None

        # if self.current_epoch == 1:
        #     for group in self.param_groups:
        #         for p in group['params']:
        #             # 计算稀有度得分，仅在第一 epoch 调用
        #             if p.grad is not None:
        #                 self.total_y_value(p.grad)

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            lr = group['lr']
            dampening = group['dampening']
            nesterov = group['nesterov']
            curi = group['curi']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad

                state = self.state[p]

                # 获取稀有度得分
                y_score_all = state.get('y_score_all', torch.tensor(1.0, device=d_p.device))  # Default score if not set
                rare = state['y_sample_size_min']   #用于计算学习率
                
                # print(y_score_all,rare)
                # rare_momentum = torch.log(1 + 10 * (0.9 - torch.tensor(y_score_all))) * curi
                # rare_momentum = curi * torch.log(5 * ((torch.tensor(y_score_all) - 0.4) /0.4 + 0.000001))
                # 调整后的 rare_momentum 计算公式
                # rare_momentum = 1*curi *(y_score_all) * torch.exp(torch.log1p(torch.tensor( -self.current_epoch, device="cuda"))) #对数减小rare的影响
                rare_momentum = 1 * y_score_all * (0.1 ** (self.current_epoch // 10))    #分多阶段减小,curi 太小导致1-curi_mu太大


                # rare_momentum = torch.clamp(0.1*y_score_all**curi , min=0.00001, max=0.99)    #y_score_all 与curi无关时
                rare_lr = torch.tanh(1/rare)*curi* (0.1 ** (self.current_epoch // 20))


                # 初始化或更新 momentum buffer
                if 'momentum_buffer' not in state:
                    buf = state['momentum_buffer'] = torch.clone(d_p).detach()
                    
                else:
                    buf = state['momentum_buffer']
                    # print('momentu:',momentum)
                    # momentum_new = max(0.6, momentum - rare_momentum)
                    if y_score_all<=0.69 :#and self.current_epoch >=90:# & self.current_epoch >1:
                        # buf.mul_(momentum).add_(d_p, alpha=(1 - dampening) * (1 + rare_momentum)) # 增加当前的
                        buf.mul_(momentum* (1 - rare_momentum)).add_(d_p, alpha=(1 - dampening))
                    else:
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    # print('momentum_new:',momentum_new)
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)

                if momentum != 0:
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                # 更新参数
                p.add_(d_p, alpha=-lr*(rare_lr+1))

        return loss



    def total_y_value(self, y, max_score=1.0):
        """
        计算当前批次数据的稀有度得分。
        - y_sample_size_min: 当前批次最稀有类别的样本数与最常见类别的样本数之比。
        - y_score_all: 基于总变差距离（TVD）计算当前批次与理想平衡分布的相似度得分。
        
        Args:
            y (Tensor): 当前批次的类别标签。
            max_score (float): 用于控制得分上限。
            
        Returns:
            score_tensor (Tensor): 当前批次的相似度得分 y_score_all，范围在 [0, max_score] 之间。
        """
        if len(y) == 0:
            raise ValueError("Optimizer got an empty parameter list")

        # 将 y 转换为列表并统计当前 batch 中的类别出现次数
        list_y = y.tolist()
        y_counts = Counter(list_y)

        # 初始化 cumulative_y_classes 并更新历史分布
        for param_group in self.param_groups:
            if 'cumulative_y_classes' not in param_group:
                param_group['cumulative_y_classes'] = Counter()
            
            cumulative_y_classes = param_group['cumulative_y_classes']
            cumulative_y_classes.update(y_counts)  # 更新历史类别分布

            # 获取当前批次最稀有和最常见类别的样本数
            rare_class_count_batch = min(y_counts.values(), default=0)
            common_class_count_batch = max(y_counts.values(), default=1)  # 默认为 1 以避免除以 0

            # 计算 y_sample_size_min
            y_sample_size_min = rare_class_count_batch / common_class_count_batch

            # 获取类别数量 K 和理想平衡分布 P_ideal
            num_classes = len(cumulative_y_classes)
            P_ideal = 1.0 / num_classes if num_classes > 0 else 0

            # 当前批次的实际分布 P_actual
            total_count_batch = sum(y_counts.values())
            P_actual = torch.zeros(num_classes, device='cuda')
            
            for cls, count in y_counts.items():
                if cls < num_classes:
                    P_actual[cls] = count / total_count_batch

            # 计算总变差距离 TVD
            TVD = 0.5 * torch.sum(torch.abs(P_actual - P_ideal))

            # 根据 TVD 计算相似度得分 y_score_all
            y_score_all = max_score * (1 - TVD)
            score_tensor = torch.tensor(y_score_all, device='cuda')
            # print(y_score_all)
            # 将 y_sample_size_min 和 y_score_all 保存到状态字典中
            for param in param_group['params']:
                param_state = self.state[param]
                param_state['y_sample_size_min'] = torch.tensor(y_sample_size_min, device='cuda')  # 当前批次稀有度比值
                param_state['y_score_all'] = score_tensor.clone().detach()  # 基于 TVD 的相似度得分

        return score_tensor














    ###########################原始的
    # def total_y_value(self, y):
    #     if len(y) == 0:
    #         raise ValueError("optimizer got an empty parameter list")
        
    #     # 将 y 转换为 Python 列表并统计类别出现次数
    #     list_y = [i.item() for i in y]
    #     y_counts = Counter(list_y)

    #     for param_group in self.param_groups:
    #         if 'y_classes_gruops' not in param_group:
    #             param_group['y_classes_gruops'] = Counter()
    #         y_classes_gruops = param_group['y_classes_gruops']
            
    #         # 更新历史统计
    #         y_classes_gruops.update(y_counts)
            
    #         # 当前 batch 的类别样本比率，直接在 GPU 上进行张量操作
    #         now_y_classes_param = torch.tensor([y_counts[key] for key in y_counts], dtype=torch.float, device='cuda')   #多U训练时，必须指定主U？下降了0.04%
    #         total_y_classes_param = torch.tensor([y_classes_gruops[key] for key in y_classes_gruops], dtype=torch.float, device='cuda')
            
    #         y_classes_sum = total_y_classes_param.sum()  # 类别样本总和
    #         y_classes_sample_size = now_y_classes_param / y_classes_sum  # 当前 batch 类别样本占比
            
    #         y_sample_size_min = y_classes_sample_size.min()#.item()  # 获取最小样本稀有度值

    #         # 将结果保存到 param 的状态字典中
    #         for param in param_group['params']:
    #             # self.state[param]['y_sample_size_min'] = torch.tensor(y_sample_size_min, device='cuda')  # 稀有度值放到 主GPU
    #             self.state[param]['y_sample_size_min'] = y_sample_size_min.clone().detach().to('cuda')  
    
    
    
    
    # def total_y_value(self, y):
    #     if len(y) == 0:
    #         raise ValueError("optimizer got an empty parameter list")
        
    #     # 将 y 转换为 Python 列表并统计类别出现次数
    #     list_y = [i.item() for i in y]
    #     y_counts = Counter(list_y)

    #     for param_group in self.param_groups:
    #         if 'y_classes_gruops' not in param_group:
    #             param_group['y_classes_gruops'] = Counter()
    #         y_classes_gruops = param_group['y_classes_gruops']
            
    #         # 更新历史统计
    #         y_classes_gruops.update(y_counts)
            
    #         # 当前 batch 的类别样本比率，直接在 GPU 上进行张量操作
    #         now_y_classes_param = torch.tensor([y_counts[key] for key in y_counts], dtype=torch.float, device='cuda')
    #         total_y_classes_param = torch.tensor([y_classes_gruops[key] for key in y_classes_gruops], dtype=torch.float, device='cuda')
            
    #         y_classes_sum = total_y_classes_param.sum()  # 类别样本总和
    #         y_classes_sample_size = now_y_classes_param / y_classes_sum  # 当前 batch 类别样本占比
            
    #         y_sample_size_min = y_classes_sample_size.min().item()  # 获取最小样本稀有度值

    #         # 将结果保存到 param 的状态字典中
    #         for param in param_group['params']:
    #             self.state[param]['y_sample_size_min'] = torch.tensor(y_sample_size_min, device='cuda')  # 稀有度值放到 GPU
    #             # self.state[param]['y_sample_size_min'] = y_sample_size_min.clone().detach().to('cuda') 

    # def total_y_value(self, y):
    #     y_classes_gruops = dict()
    #     now_y_classes = dict()

    #     y_classes_param_total = list()
    #     now_y_classes_param = list()

    #     if len(y) == 0:
    #         raise ValueError("optimizer got an empty parameter list")

    #     list_y = [i for i in y]  # 将 dtype 为 tensor 的转为 numpy 类型

    #     # print("list_y:".format([i for i in list(list_y)]))
    #       # 构造一个字典
    #     # y_classes_gruops_list={'y_classes_gruops':{}}
    #     # 将 'y_classes_gruops' 字典置在 网络接结构中的最顶层，目的减少 ‘y_classes_gruops'的次数
    #     # y_classes_gruops_flage = True
    #     for param_group in self.param_groups:

    #         #print("defore_param_group:{}".format(param_group))
    #         if isinstance(param_group,dict):
    #             if 'y_classes_gruops' in param_group.keys():
    #                 y_classes_gruops = param_group['y_classes_gruops'] # 获取历史batch y 类别和统计数值

    #       # 获取 在 list_y 内出现的类别，并统计
    #     for y_name in list_y:
    #         flag = True

    #         for key in now_y_classes.keys():
                
    #             if y_name == key:
    #                 now_y_classes[key] = now_y_classes[key] + 1
    #                 flag = False
    #                 break
    #         if flag:
    #             now_y_classes[y_name] = 1

    #     # 获取 在 list_y 内出现的类别，并和历史出现的进行统计
    #     for y_name in list_y:
    #         flag = True
    #         for key in y_classes_gruops.keys():       
    #             if y_name == key:
    #                 y_classes_gruops[key] = y_classes_gruops[key] + 1
    #                 flag = False
    #                 break
    #         if flag:
    #             y_classes_gruops[y_name] = 1


    #     # print("y_classes_gruops:{},now_y_classes:{}".format(y_classes_gruops, now_y_classes))

    #     # now_y_classes_param 获取  在这个 batc size 内含有 y 类别的出现数量，仅含在这个 batch size 内
    #     #  y_classes_param_total 获取 在这个 batch size 内有含有 y 类别的出现数量 ，含这个 batch size 和之前的

    #     for y_name in now_y_classes.keys():

    #         for key in y_classes_gruops.keys():
    #             if y_name == key:
    #                 now_y_classes_param.append(y_classes_gruops[key])  #将这一个batch y 类别历史统计的大小存入
    #                 break

    #     for y_name in y_classes_gruops.keys():

    #         y_classes_param_total.append(y_classes_gruops[y_name]) #将所有的类别统计的大小存入


    #     y_classes_param_total = torch.tensor(y_classes_param_total,
    #                                          dtype=torch.float)  # 将 dtype: numpy 转变为 tensor float 类型
    #     now_y_classes_param = torch.tensor(now_y_classes_param, dtype=torch.float)

    #     y_classes_sum = y_classes_param_total.sum()  # 求类别样本和
    #     # print("y_classes_sum:{}".format(y_classes_sum))

    #     y_classes_sample_size = now_y_classes_param.div(y_classes_sum)

    #     # print("y_classes_sample_size:{}".format(y_classes_sample_size))
    #     # 计算这一个batch类别样本占比的平均值

    #     y_sample_size_min = y_classes_sample_size.min(0)

    #     # print("y_sample_size_min:{}".format(y_sample_size_min[0]))

    #     for param_group in self.param_groups:
    #         for param in param_group['params']:
    #             param_state = self.state[param]
    #             param_state['y_sample_size_min'] = torch.clone(y_sample_size_min[0].data)  # 将当前batch y 类别的最小稀有度的值加入到 模型的参数里
              
    #     for param_group in self.param_groups:

    #         if isinstance(param_group,dict):
    #             # if 'y_classes_gruops' in param_group.keys():
    #             param_group['y_classes_gruops'] = y_classes_gruops



def get_optimizer(optimizer_name, model, lr_initial, mu=0.,momentum=args.momentum,curi=args.curi):
    """
    Gets torch.optim.Optimizer given an optimizer name, a model and learning rate

    :param optimizer_name: possible are adam and sgd
    :type optimizer_name: str
    :param model: model to be optimized
    :type optimizer_name: nn.Module
    :param lr_initial: initial learning used to build the optimizer
    :type lr_initial: float
    :param mu: proximal term weight; default=0.
    :type mu: float
    :return: torch.optim.Optimizer

    """

    if optimizer_name == "adam":
        return optim.Adam(
            [param for param in model.parameters() if param.requires_grad],
            lr=lr_initial,
            weight_decay=5e-4
        )

    elif optimizer_name == "sgd":
        return optim.SGD(
            [param for param in model.parameters() if param.requires_grad],
            lr=lr_initial,
            momentum=momentum,
            weight_decay=5e-4
        )

    elif optimizer_name == "prox_sgd":
        return ProxSGD(
            [param for param in model.parameters() if param.requires_grad],
            mu=mu,
            lr=lr_initial,
            momentum=momentum,
            weight_decay=5e-4,
            # curi=curi
        )

    elif optimizer_name == "lm_adam":
        return LMAdam(
            [param for param in model.parameters() if param.requires_grad],
            lr=lr_initial,
            weight_decay=5e-4,
            curi=curi
        )

    # elif optimizer_name == "lm_sgd":
    #     return  LMSGD(
    #         [param for param in model.parameters() if param.requires_grad],
    #         lr=lr_initial,
    #         momentum=momentum,
    #         weight_decay=5e-4,
    #         curi=curi
    #         )
    elif optimizer_name == "lm_sgd":
        # 假设模型包含 model.base 和 model.fc
        # 将全连接层 (fc) 的参数设置为 use_rare=True
        return LMSGD(
            [
                # 基础层，不使用稀有度得分
                {'params': [param for name, param in model.named_parameters() if param.requires_grad and "fc" not in name],
                'lr': lr_initial, 'use_rare': False},
                # 全连接层，使用稀有度得分
                {'params': [param for name, param in model.named_parameters() if param.requires_grad and "fc" in name],
                'lr': lr_initial*0.1 , 'use_rare': True}  # 通常全连接层可以使用更大学习率
            ],
            momentum=momentum,
            weight_decay=5e-4,
            curi=curi
        )


    else:
        raise NotImplementedError("Other optimizer are not implemented")


def get_lr_scheduler(optimizer, scheduler_name, n_rounds=None,lr_factor=None,warm_epoch=None):
    """
    Gets torch.optim.lr_scheduler given an lr_scheduler name and an optimizer

    :param optimizer:
    :type optimizer: torch.optim.Optimizer
    :param scheduler_name: possible are
    :type scheduler_name: str
    :param n_rounds: number of training rounds, only used if `scheduler_name == multi_step`
    :type n_rounds: int
    :return: torch.optim.lr_scheduler

    """

    if scheduler_name == "sqrt":
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1 / np.sqrt(x) if x > 0 else 1)

    elif scheduler_name == "linear":
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1 / x if x > 0 else 1)

    elif scheduler_name == "constant":
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1)

    elif scheduler_name == "cosine_annealing":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=0)

    elif scheduler_name == "multi_step":
        assert n_rounds is not None, "Number of rounds is needed for \"multi_step\" scheduler!"
        milestones = [n_rounds//2, 3*(n_rounds//4)]
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    elif scheduler_name== "WarmupMultiStepLR":
        print(" use WarmupMultiStepLR")
        assert n_rounds is not None, "Number of rounds is needed for \"multi_step\" scheduler!"
       # assert lr_factor is not None 
       #assert warm_epoch is not None
        print("warmup_epochs:{},lr_factor:{}".format(warm_epoch,lr_factor))
        milestones = [n_rounds//2, 3*(n_rounds//4)]
        return WarmupMultiStepLR(optimizer,milestones=milestones,gamma=lr_factor,warmup_epochs=warm_epoch)

    else:
        raise NotImplementedError("Other learning rate schedulers are not implemented")

