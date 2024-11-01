
from utils_old import *
from os import path
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option="A"):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == "A":
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(
                        x[:, :, ::2, ::2],
                        (0, 0, 0, 0, planes // 4, planes // 4),
                        "constant",
                        0,
                    )
                )
            elif option == "B":
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(self.expansion * planes),
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BBN_ResNet_Cifar(nn.Module):
    def __init__(self, block, num_blocks):
        super(BBN_ResNet_Cifar, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2] - 1, stride=2)
        self.cb_block = block(self.in_planes, self.in_planes, stride=1)
        self.rb_block = block(self.in_planes, self.in_planes, stride=1)

        self.apply(_weights_init)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def load_model(self, pretrain):
        print("Loading Backbone pretrain model from {}......".format(pretrain))
        model_dict = self.state_dict()
        pretrain_dict = torch.load(pretrain)["state_dict_best"]['feat_model']

        new_dict = OrderedDict()

        for k, v in pretrain_dict.items():
            if k.startswith("module"):
                k = k[7:]
            if "fc" not in k and "classifier" not in k:
                new_dict[k] = v

        model_dict.update(new_dict)
        self.load_state_dict(model_dict)
        print("Backbone model has been loaded......")

    def _make_layer(self, block, planes, num_blocks, stride, add_flag=True):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, **kwargs):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        if "feature_cb" in kwargs:
            out = self.cb_block(out)
            return out
        elif "feature_rb" in kwargs:
            out = self.rb_block(out)
            return out

        out1 = self.cb_block(out)
        out2 = self.rb_block(out)
        out = torch.cat((out1, out2), dim=1)
        
        out = self.avgpool(out)
        out = out.view(out.shape[0], -1)


        return out

#重新为imageNet写个resnet50

class BBN_ResNet_imageNet(nn.Module):
    def __init__(self, block, num_blocks, n_classes=1000):
        super(BBN_ResNet_imageNet, self).__init__()
        self.in_planes = 64

        # ImageNet需要更大的输入尺寸和更多的初始通道
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 添加最大池化层
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)  # 添加额外的层
        self.cb_block = block(self.in_planes, self.in_planes, stride=1)
        self.rb_block = block(self.in_planes, self.in_planes, stride=1)

        self.apply(_weights_init)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, n_classes)  # 添加全连接层
        self.fc = nn.Linear(1024, n_classes)  # 修改全连接层的输入特征数


    # def load_model(self, pretrain):
    #     print("Loading Backbone pretrain model from {}......".format(pretrain))
    #     model_dict = self.state_dict()
    #     pretrain_dict = torch.load(pretrain)["state_dict_best"]['feat_model']

    #     new_dict = OrderedDict()

    #     for k, v in pretrain_dict.items():
    #         if k.startswith("module"):
    #             k = k[7:]
    #         if "fc" not in k and "classifier" not in k:
    #             new_dict[k] = v

    #     model_dict.update(new_dict)
    #     self.load_state_dict(model_dict)
    #     print("Backbone model has been loaded......")
    def load_model(self, pretrain):
        print("Loading Backbone pretrain model from {}......".format(pretrain))
        model_dict = self.state_dict()
        pretrain_dict = torch.load(pretrain)["state_dict_best"]['feat_model']

        # 只保留那些在当前模型中存在并且大小匹配的权重
        pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}

        model_dict.update(pretrain_dict)
        self.load_state_dict(model_dict)
        print("Backbone model has been loaded......")


    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, **kwargs):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)  # 添加最大池化层
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)  # 添加额外的层
        ################################
        if "feature_cb" in kwargs:
            out = self.cb_block(out)
            return out
        elif "feature_rb" in kwargs:
            out = self.rb_block(out)
            return out

        out1 = self.cb_block(out)
        out2 = self.rb_block(out)
        out = torch.cat((out1, out2), dim=1)
        #####################################
        out = self.avgpool(out)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)  # 添加全连接层

        return out
        
def create_model(use_fc=False, pretrain=True, dropout=None, stage1_weights=False, dataset=None, log_dir=None, test=False, *args):
    
    print('Loading ResNet 32 Feature Model.')
    resnet32 = BBN_ResNet_Cifar(BasicBlock, [5, 5, 5])

    pretrained_model="weight/final_model_checkpoint.pth"
    if path.exists(pretrained_model) and pretrain:
        print('===> Load Initialization for ResNet32')
        resnet32.load_model(pretrain=pretrained_model)
    else:
        print('===> Train backbone from the scratch')

    return resnet32

#添加了一个最大池化层。
# 增加了一个额外的layer4，使其更像ResNet-50的结构。
# 在模型的最后添加了一个全连接层，用于分类。

def create_model_imageNet(use_fc=False, pretrain=True, dropout=None, stage1_weights=False, dataset=None, log_dir=None, test=False, *args):
    print('Loading ResNet for ImageNet.')
    resnet = BBN_ResNet_imageNet(BasicBlock, [3, 4, 6, 3], n_classes=1000)  # 使用ResNet-50的层数

    pretrained_model="weight/final_model_checkpoint.pth"
    if path.exists(pretrained_model) and pretrain:
        print('===> Load Initialization for ResNet for ImageNet')
        resnet.load_model(pretrain=pretrained_model)
    else:
        print('===> Train backbone from the scratch')

    return resnet

