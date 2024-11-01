import torch
from utils_old.args import *
import time

class Learner:
    """
    optimizer_step: 进行一轮优化迭代, 需要梯度已经被计算完毕
    fit_batch: 对一个批量进行一轮优化迭代
    fit_epoch: 单次遍历iterator中的得到的所有样本，进行一系列批量的迭代
    fit_epochs: 多次遍历将从iterator指向的训练集
    gather_losses:收集iterator迭代器所有样本的loss并拼接输出
    get_param_tensor: 获取获取一个flattened后的`model`的参数
    free_memory: 释放模型权重
    free_gradients: 释放模型梯度
    """

    def __init__(
            self, model,
            criterion,
            metric,
            device,
            optimizer,
            lr_scheduler=None,
            is_binary_classification=False
    ):

        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.metric = metric
        self.device = device
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.is_binary_classification = is_binary_classification

        self.model_dim = int(self.get_param_tensor().shape[0])

    def optimizer_step(self):
        """
         执行一轮优化迭代，调用之前需要反向传播先算好梯度（即已调用loss.backward())
        """
        self.optimizer.step()
        if self.lr_scheduler:
            self.lr_scheduler.step()

    def compute_gradients_and_loss(self, batch, weights=None):
        """
        计算一batch的梯度和损失

        :param batch: tuple of (x, y, indices)
        :param weights: tensor with the learners_weights of each sample or None
        :type weights: torch.tensor or None
        :return:
            loss

        """
        self.model.train()
     

        x, y, indices = batch
        x = x.to(self.device).type(torch.float32)
        y = y.to(self.device)
        # print(f'y:{y}')
        #将y添加到y.log文件中,用于训练时给优化器加上动量
        # with open('y.log', 'a') as f:
        #     f.write(str(y))
        # #如果./y.pt文件存在，则删除
        # if os.path.exists('./y.pt'):
        #     os.remove('./y.pt')

            
        if self.is_binary_classification:
            y = y.type(torch.float32).unsqueeze(1)

        self.optimizer.zero_grad()

        y_pred = self.model(x)
        loss_vec = self.criterion(y_pred, y)

        if weights is not None:
            weights = weights.to(self.device)
            loss = (loss_vec.T @ weights[indices]) / loss_vec.size(0)
            # loss = (loss_vec * weights[indices]).sum() / len(loss_vec)

            
        else:
            loss = loss_vec.mean()

        loss.backward()

        return loss.detach()

    def fit_batch(self, batch, weights=None):
        """
        基于来自`iterator`的一个batch的样本执行一轮优化迭代
        :参数 batch (元组(x, y, indices)):
        :参数 weights(tensor): 每个样本的权重，可为none
        :返回: loss.detach(), metric.detach()(训练数据)

        """
        self.model.train()
        print('fit_batch_model:',self.model)
        print(f"Optimizer type: {type(self.optimizer).__name__}")
        x, y, indices = batch
        x = x.to(self.device).type(torch.float32)
        y = y.to(self.device)
       # #将y添加到y.log文件中,用于训练时给优化器加上动量
       #  with open('y.log', 'a') as f:
       #      f.write(str(y))
       #  #如果./y.pt文件存在，则删除
       #  if os.path.exists('./y.pt'):
       #      os.remove('./y.pt')

        

        if self.is_binary_classification:
            y = y.type(torch.float32).unsqueeze(1)

        self.optimizer.zero_grad()

        y_pred = self.model(x)
        loss_vec = self.criterion(y_pred, y)
        metric = self.metric(y_pred, y) / len(y)

        if weights is not None:
            weights = weights.to(self.device)
            loss = (loss_vec.T @ weights[indices]) / loss_vec.size(0)
            # loss = (loss_vec * weights[indices]).sum() / len(loss_vec)
        else:
            loss = loss_vec.mean()

        loss.backward()

        self.optimizer.step()
        if self.lr_scheduler:
            self.lr_scheduler.step()

        return loss.detach(), metric.detach()

    def fit_epoch(self, iterator, weights=None):
        """
        将来自`iterator`的所有batches遍历一次，进行优化迭代
        :参数 iterator(torch.utils.data.DataLoader):
        :参数 weights(torch.tensor): 存储每个样本权重的向量，可为None
        :return: loss.detach(), metric.detach() (训练数据)

        """

        args = parse_args()
        self.model.train()

        global_loss = 0.
        global_metric = 0.
        n_samples = 0

        for x, y, indices in iterator:
            x = x.to(self.device).type(torch.float32)
            y = y.to(self.device)

            n_samples += y.size(0)

            if self.is_binary_classification:
                y = y.type(torch.float32).unsqueeze(1)

            self.optimizer.zero_grad()

            y_pred = self.model(x)

            loss_vec = self.criterion(y_pred, y)
            if weights is not None:
                weights = weights.to(self.device)
                # print('loss_vec.shape:',loss_vec.shape)
                # print('weights[indices].shape:',weights[indices].shape)
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

                loss = (loss_vec.T @ weights[indices]) / loss_vec.size(0)
                #这两种方法在数学上是等价的，但在实现上略有不同。使用元素乘法方法更直观，因为它明确地表示了每个损失值与其相应权重的乘积，然后对这些乘积求和。而矩阵乘法方法则使用了线性代数的点积概念。
                    #  如果调整为*后不启用CUDA断言无法训练FedEM
                # loss = (loss_vec * weights[indices]).sum() / len(loss_vec)

            else:
                loss = loss_vec.mean()
            loss.backward()


            
            if   args.optimizer =="lm_adam" or args.optimizer =="lm_sgd":#args.optimizer == "prox_sgd" or
                # print("prox_sgd")
                # print("prox_sgd")
                # start_time = time.time()  # 记录优化器开始时间
                self.optimizer.total_y_value(y=y)
                self.optimizer.step()

                # end_time = time.time()  # 记录优化器结束时间
                # time_sum = end_time - start_time
                # if  args.optimizer == "prox_sgd":
                #     print("prox_sgd time_sum:{}".format(time_sum))
                # elif args.optimizer == "lm_adam":
                #     print("lm_adam time_sum:{}".format(time_sum))
                # else:
                #     print("lm_sgd time_sum:{}".format(time_sum))
            else:

                self.optimizer.step()



            self.optimizer.step()

            global_loss += loss.detach() * loss_vec.size(0)
            global_metric += self.metric(y_pred, y).detach()

        return global_loss / n_samples, global_metric / n_samples

    def gather_losses(self, iterator):
        """
        收集迭代器所有元素的损失

        :param iterator:
        :type iterator: torch.utils.data.DataLoader
        :return
            tensor with losses of all elements of the iterator.dataset

        """
        self.model.eval()
        n_samples = len(iterator.dataset)
        all_losses = torch.zeros(n_samples, device=self.device)

        with torch.no_grad():
            for (x, y, indices) in iterator:
                x = x.to(self.device).type(torch.float32)
                y = y.to(self.device)

                if self.is_binary_classification:
                    y = y.type(torch.float32).unsqueeze(1)

                y_pred = self.model(x)
                all_losses[indices] = self.criterion(y_pred, y).squeeze()

        return all_losses

    def evaluate_iterator(self, iterator):
        """
        evaluate learner on `iterator`
        在“迭代器”上评估学习者
        :param iterator:
        :type iterator: torch.utils.data.DataLoader
        :return
            global_loss and  global_metric accumulated over the iterator

        """
        self.model.eval()

        global_loss = 0.
        global_metric = 0.
        n_samples = 0

        for x, y, _ in iterator:
            x = x.to(self.device).type(torch.float32)
            y = y.to(self.device)

            if self.is_binary_classification:
                y = y.type(torch.float32).unsqueeze(1)

            with torch.no_grad():
                y_pred = self.model(x)

                global_loss += self.criterion(y_pred, y).sum().detach()
                global_metric += self.metric(y_pred, y).detach()    #全局的metric为所有样本的metric的和

            n_samples += y.size(0)

        return global_loss / n_samples, global_metric / n_samples

    def get_y(self,iterator):
        for x, y, indices in iterator:
            # x = x.to(self.device).type(torch.float32)
            y = y.to(self.device)
            return y

    def fit_epochs(self, iterator, n_epochs, weights=None):
        """
        执行多个n_epochs的训练
        :参数 iterator(torch.utils.data.DataLoader):
        :参数 n_epochs(int):
        :参数 weights: 每个样本权重的向量，可为None
        :返回: None

        """
        for step in range(n_epochs):
            self.fit_epoch(iterator, weights)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

    def get_param_tensor(self):
        """
        get `model` parameters as a unique flattened tensor
        获取“模型”参数作为唯一的扁平张量
        :return: torch.tensor

        """
        param_list = []

        for param in self.model.parameters():
            param_list.append(param.data.view(-1, ))

        return torch.cat(param_list)

    def get_grad_tensor(self):
        """
        get `model` gradients as a unique flattened tensor
        获取“模型”梯度作为唯一的扁平张量
        :return: torch.tensor

        """
        grad_list = []

        for param in self.model.parameters():
            if param.grad is not None:
                grad_list.append(param.grad.data.view(-1, ))

        return torch.cat(grad_list)

    def free_memory(self):
        """
        free the memory allocated by the model weights

        """
        del self.optimizer
        del self.model

    def free_gradients(self):
        """
        free memory allocated by gradients

        """
        self.optimizer.zero_grad(set_to_none=True)


class LanguageModelingLearner(Learner):
    def fit_epoch(self, iterator, weights=None):

        self.model.train()

        global_loss = 0.
        global_metric = 0.
        n_samples = 0

        for x, y, indices in iterator:
            x = x.to(self.device)
            y = y.to(self.device)

            n_samples += y.size(0)

            chunk_len = y.size(1)

            self.optimizer.zero_grad()

            y_pred = self.model(x)
            loss_vec = self.criterion(y_pred, y)

            if weights is not None:
                weights = weights.to(self.device)
                loss = (loss_vec.T @ weights[indices]).mean() / loss_vec.size(0)
            else:
                loss = loss_vec.mean()

            loss.backward()

            self.optimizer.step()

            global_loss += loss.detach() * loss_vec.size(0) / chunk_len
            global_metric += self.metric(y_pred, y).detach() / chunk_len

        return global_loss / n_samples, global_metric / n_samples

    def fit_batch(self, batch, weights=None):

        self.model.train()

        x, y, indices = batch
        x = x.to(self.device)
        y = y.to(self.device)

        n_samples = y.size(0)
        chunk_len = y.size(1)

        self.optimizer.zero_grad()

        y_pred = self.model(x)
        loss_vec = self.criterion(y_pred, y)

        if weights is not None:
            weights = weights.to(self.device)
            loss = (loss_vec.T @ weights[indices]).mean() / loss_vec.size(0)
        else:
            loss = loss_vec.mean()

        loss.backward()

        self.optimizer.step()

        global_loss = loss.detach() * loss_vec.size(0) / chunk_len
        global_metric = self.metric(y_pred, y).detach() / chunk_len

        return global_loss / n_samples, global_metric / n_samples

    def compute_gradients_and_loss(self, batch, weights=None):
        """
        compute the gradients and loss over one batch.

        :param batch: tuple of (x, y, indices)
        :param weights: tensor with the learners_weights of each sample or None
        :type weights: torch.tensor or None
        :return:
            loss

        """
        raise NotImplementedError

    def gather_losses(self, iterator):
        """
        gathers losses for all elements of iterator

        :param iterator:
        :type iterator: torch.utils.data.DataLoader
        :return
            tensor with losses of all elements of the iterator.dataset

        """
        self.model.eval()
        n_samples = len(iterator.dataset)
        predictions = torch.zeros(n_samples, device=self.device)

        with torch.no_grad():
            for (x, y, indices) in iterator:
                x = x.to(self.device)
                y = y.to(self.device)

                y_pred = self.model(x)
                predictions[indices] = self.criterion(y_pred, y).mean(axis=1)

        return predictions

    def evaluate_iterator(self, iterator):
        """
        evaluate learner on `iterator`

        :param iterator:
        :type iterator: torch.utils.data.DataLoader
        :return
            global_loss and  global_metric accumulated over the iterator

        """
        self.model.eval()

        global_loss = 0.
        global_metric = 0.
        n_samples = 0

        with torch.no_grad():
            for x, y, _ in iterator:
                x = x.to(self.device)
                y = y.to(self.device)
                n_samples += y.size(0)

                chunk_len = y.size(1)

                y_pred = self.model(x)
                global_loss += self.criterion(y_pred, y).sum().detach() / chunk_len
                global_metric += self.metric(y_pred, y).detach() / chunk_len

        return global_loss / n_samples, global_metric / n_samples
