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


class LMAdam(Optimizer):
    r"""Implements Adam algorithm.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, curi=0.01,amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, curi=curi, amsgrad=amsgrad)
        super(LMAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(LMAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)


    @torch.no_grad()
    def step(self,closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        nn = 0 # 计数

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                rare = state['y_sample_size_min']

                # nn = nn + 1
                # if nn ==1:  # 控制 y_sample_size_min_param 打印输出
                #     # print("y_sample_size_min_param:{}".format(state['y_sample_size_min']))

                # print("state",state)

                # State initialization
                if len(state) == 1:                  # 0 改为 1 ，在 total_y_value 多了一个字典索引 'out_cosine_similarity'、 'y_classes_gruops'
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)  # size(c，n , h,w)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format) # size(c，n , h,w)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                #print("state_again",state)

                # print("cosine_similarity_value:{}".format(state['out_cosine_similarity']))

                # cosine_similarity_value = state['out_cosine_similarity']  # 获取余弦相似度的值

                
                # if cosine_similarity_value is None:
                #     raise ValueError("cosine_similarity_value is empty")
               
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                
                beta1, beta2 = group['betas']
                curi = group['curi']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1).add_(grad.mul_(curi/rare))                # 一阶动量   
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)   # 二阶动量
                
                # 一、二阶动量都除以 原始y和预测y 的相关性

               # print("exp_avg:{},exp_avg_sq:{},loss_:{}".format(exp_avg.shape,exp_avg_sq.shape,loss_vc.shape))
                
                # exp_avg = torch.div(exp_avg,cosine_similarity_value)     # 一阶动量除去  cosine_similarity_value
                # exp_avg_sq = torch.div(exp_avg_sq,cosine_similarity_value)

                # print("cosine_similarity_value{}".format(cosine_similarity_value))

                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)  
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1

                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

    def total_y_value(self, y):
        y_classes_gruops = dict()
        now_y_classes = dict()

        y_classes_param_total = list()
        now_y_classes_param = list()

        if len(y) == 0:
            raise ValueError("optimizer got an empty parameter list")

        list_y = [i for i in y]  # 将 dtype 为 tensor 的转为 numpy 类型

        # print("list_y:".format([i for i in list(list_y)]))
          # 构造一个字典
        y_classes_gruops_list={'y_classes_gruops':{}}
        # 将 'y_classes_gruops' 字典置在 网络接结构中的最顶层，目的减少 ‘y_classes_gruops'的次数
        # y_classes_gruops_flage = True
        for param_group in self.param_groups:

            #print("defore_param_group:{}".format(param_group))
            if isinstance(param_group,dict):
                if 'y_classes_gruops' in param_group.keys():
                    y_classes_gruops = param_group['y_classes_gruops'] # 获取历史batch y 类别和统计数值

          # 获取 在 list_y 内出现的类别，并统计
        for y_name in list_y:
            flag = True

            for key in now_y_classes.keys():
                
                if y_name == key:
                    now_y_classes[key] = now_y_classes[key] + 1
                    flag = False
                    break
            if flag:
                now_y_classes[y_name] = 1

        # 获取 在 list_y 内出现的类别，并和历史出现的进行统计
        for y_name in list_y:
            flag = True
            for key in y_classes_gruops.keys():       
                if y_name == key:
                    y_classes_gruops[key] = y_classes_gruops[key] + 1
                    flag = False
                    break
            if flag:
                y_classes_gruops[y_name] = 1


        # print("y_classes_gruops:{},now_y_classes:{}".format(y_classes_gruops, now_y_classes))

        # now_y_classes_param 获取  在这个 batc size 内含有 y 类别的出现数量，仅含在这个 batch size 内
        #  y_classes_param_total 获取 在这个 batch size 内有含有 y 类别的出现数量 ，含这个 batch size 和之前的

        for y_name in now_y_classes.keys():

            for key in y_classes_gruops.keys():
                if y_name == key:
                    now_y_classes_param.append(y_classes_gruops[key])  #将这一个batch y 类别历史统计的大小存入
                    break

        for y_name in y_classes_gruops.keys():

            y_classes_param_total.append(y_classes_gruops[y_name]) #将所有的类别统计的大小存入


        y_classes_param_total = torch.tensor(y_classes_param_total,
                                             dtype=torch.float)  # 将 dtype: numpy 转变为 tensor float 类型
        now_y_classes_param = torch.tensor(now_y_classes_param, dtype=torch.float)

        y_classes_sum = y_classes_param_total.sum()  # 求类别样本和
        # print("y_classes_sum:{}".format(y_classes_sum))

        y_classes_sample_size = now_y_classes_param.div(y_classes_sum)

        # print("y_classes_sample_size:{}".format(y_classes_sample_size))
        # 计算这一个batch类别样本占比的平均值

        y_sample_size_min = y_classes_sample_size.min(0)

        # print("y_sample_size_min:{}".format(y_sample_size_min[0]))

        for param_group in self.param_groups:
            for param in param_group['params']:
                param_state = self.state[param]
                param_state['y_sample_size_min'] = torch.clone(y_sample_size_min[0].data)  # 将当前batch y 类别的最小稀有度的值加入到 模型的参数里
              
        for param_group in self.param_groups:

            if isinstance(param_group,dict):
                # if 'y_classes_gruops' in param_group.keys():
                param_group['y_classes_gruops'] = y_classes_gruops


class LMSGD(Optimizer):
    r"""
      Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
        maximize (bool, optional): maximize the params based on the objective, instead of
            minimizing (default: False)
        foreach (bool, optional): whether foreach implementation of optimizer
            is used (default: None)

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + g_{t+1}, \\
                p_{t+1} & = p_{t} - \text{lr} * v_{t+1},
            \end{aligned}

        where :math:`p`, :math:`g`, :math:`v` and :math:`\mu` denote the
        parameters, gradient, velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + \text{lr} * g_{t+1}, \\
                p_{t+1} & = p_{t} - v_{t+1}.
            \end{aligned}

        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,weight_decay=0,curi=0, nesterov=False):

        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, curi=curi,nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(LMSGD, self).__init__(params, defaults)

        # # 设置滑动窗口大小
        # self.window_size = 10  # 可以根据需要调整窗口大小
        # self.recent_y_counts = deque(maxlen=self.window_size)  # 滑动窗口队列

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            weight_decay=group['weight_decay']
            momentum=group['momentum']
            lr=group['lr']
            dampening=group['dampening']
            nesterov=group['nesterov']
            curi=group['curi']
            

           

            for p in group['params']:
                if p.grad is not None:
                    d_p = p.grad

                    state = self.state[p]
                    rare = state['y_sample_size_min']
                    # 使用 torch.log1p 提高计算效率
                    # rare_momentum = (1 - torch.log1p(rare) / torch.log1p(torch.tensor(1/10))) ** curi
                    # 使用已知常数 log_base = 0.09531017980432493
                    log_base = 0.09531017980432493  # 预先计算的 log(1 + 1/10)
                    rare_momentum = (0.9 - torch.log1p(rare) / log_base) ** curi
                    
                    # rare_momentum = torch.tanh(1/rare)*curi
                    # rare_momentum = math.atan(1/rare)*curi
                    # rare_momentum = math.atan(curi/rare)
                    
                    if 'momentum_buffer' not in state:
                        momentum_buffer = None
                    else:
                        momentum_buffer = state['momentum_buffer']

                    if weight_decay != 0:
                        d_p = d_p.add(p, alpha=weight_decay)

                    if momentum != 0:
                        buf = momentum_buffer

                        if buf is None:
                            buf = torch.clone(d_p).detach()
                            state['momentum_buffer']= buf
                        else:
                            # buf.mul_(momentum).add_(d_p, alpha=1 - dampening)#.add_(d_p*(rare_momentum),alpha=1-dampening)
                            #################################################
                            momentum_new = (1-rare_momentum)*momentum       #M11 D12 修改这两句
                            # print("momentu_new:",momentum_new)
                            buf.mul_(momentum_new).add_(d_p, alpha=1 - dampening)
                            ###########################################################
                        if nesterov:
                            d_p = d_p.add(buf, alpha=momentum)
                        else:
                            d_p = buf

                    # alpha = lr if maximize else -lr
                    # p.data.add_(d_p, alpha=-lr)
                    if rare_momentum == 0:
                        p.data.add_(d_p, alpha=-lr)
                    else:
                        p.data.add_(d_p, alpha=-lr*rare_momentum)
        
        return loss
    
    ################################加权平均
# import torch
# from collections import Counter

    # def total_y_value(self, y, rare_threshold=0.05):
    #     """
    #     计算稀有度，基于稀有类别在当前 batch 中的相对比例。
        
    #     Args:
    #         y (Tensor): 当前 batch 的类别标签。
    #         rare_threshold (float): 稀有类别的相对比例阈值（例如 5%），
    #                                 只有当类别占比低于该阈值时才视为稀有。
    #     """
    #     if len(y) == 0:
    #         raise ValueError("Optimizer got an empty parameter list")
        
    #     # 将 y 转换为 Python 列表并统计类别出现次数
    #     list_y = [i.item() for i in y]
    #     y_counts = Counter(list_y)

    #     for param_group in self.param_groups:
    #         if 'y_classes_groups' not in param_group:
    #             param_group['y_classes_groups'] = Counter()
    #         y_classes_groups = param_group['y_classes_groups']
            
    #         # 更新历史统计
    #         y_classes_groups.update(y_counts)
            
    #         # 当前 batch 的类别样本比率
    #         batch_total = sum(y_counts.values())
    #         y_classes_sample_size = {key: count / batch_total for key, count in y_counts.items()}
            
    #         # 计算 batch 中稀有类别的总占比
    #         rare_sample_proportion = sum(value for value in y_classes_sample_size.values() if value < rare_threshold)
            
    #         # 如果没有稀有类别，则将比例设置为某个较小的值
    #         if rare_sample_proportion == 0:
    #             rare_sample_proportion = rare_threshold

    #         # 将结果保存到 param 的状态字典中
    #         y_sample_size_min = torch.tensor(rare_sample_proportion, device='cuda')
    #         for param in param_group['params']:
    #             self.state[param]['y_sample_size_min'] = y_sample_size_min.clone().detach()





    def total_y_value(self, y):
        if len(y) == 0:
            raise ValueError("optimizer got an empty parameter list")
        
        # 将 y 转换为 Python 列表并统计类别出现次数
        list_y = [i.item() for i in y]
        y_counts = Counter(list_y)

        for param_group in self.param_groups:
            if 'y_classes_gruops' not in param_group:
                param_group['y_classes_gruops'] = Counter()
            y_classes_gruops = param_group['y_classes_gruops']
            
            # 更新历史统计
            y_classes_gruops.update(y_counts)
            
            # 当前 batch 的类别样本比率，直接在 GPU 上进行张量操作
            now_y_classes_param = torch.tensor([y_counts[key] for key in y_counts], dtype=torch.float, device='cuda')   #多U训练时，必须指定主U？下降了0.04%
            total_y_classes_param = torch.tensor([y_classes_gruops[key] for key in y_classes_gruops], dtype=torch.float, device='cuda')
            
            y_classes_sum = total_y_classes_param.sum()  # 类别样本总和
            y_classes_sample_size = now_y_classes_param / y_classes_sum  # 当前 batch 类别样本占比
            
            y_sample_size_min = y_classes_sample_size.min()#.item()  # 获取最小样本稀有度值

            # 将结果保存到 param 的状态字典中
            for param in param_group['params']:
                # self.state[param]['y_sample_size_min'] = torch.tensor(y_sample_size_min, device='cuda')  # 稀有度值放到 主GPU
                self.state[param]['y_sample_size_min'] = y_sample_size_min.clone().detach().to('cuda')  
    
    
    
    
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

    elif optimizer_name == "lm_sgd":
        return  LMSGD(
            [param for param in model.parameters() if param.requires_grad],
            lr=lr_initial,
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

