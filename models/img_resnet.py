import torch
import torchvision
from torch import nn
from torch.nn import init
from models.utils import pooling


class ResNet50(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()

        resnet50 = torchvision.models.resnet50(pretrained=True)
        if config.MODEL.RES4_STRIDE == 1:
            resnet50.layer4[0].conv2.stride = (1, 1)
            resnet50.layer4[0].downsample[0].stride = (1, 1)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])

        if config.MODEL.POOLING.NAME == 'avg':
            self.globalpooling = nn.AdaptiveAvgPool2d(1)
        elif config.MODEL.POOLING.NAME == 'max':
            self.globalpooling = nn.AdaptiveMaxPool2d(1)
        elif config.MODEL.POOLING.NAME == 'gem':
            self.globalpooling = pooling.GeMPooling(p=config.MODEL.POOLING.P)
        elif config.MODEL.POOLING.NAME == 'maxavg':
            self.globalpooling = pooling.MaxAvgPooling()
        else:
            raise KeyError("Invalid pooling: '{}'".format(config.MODEL.POOLING.NAME))

        self.bn = nn.BatchNorm1d(2048)
        init.normal_(self.bn.weight.data, 1.0, 0.02)
        init.constant_(self.bn.bias.data, 0.0)

        self.part_quality_predictor = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=1),
            nn.BatchNorm2d(2048),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=(4, 12), stride=(4, 12)),
        )

        self.bn2048 = nn.BatchNorm2d(2048)
        # 使用平均池化将其变为[32, 4096, 1, 1]
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Linear(2048, 150)
        self.classifier1 = nn.Linear(2048, 150)
        init.normal_(self.classifier.weight.data, std=0.001)
        init.constant_(self.classifier.bias.data, 0.0)
        init.normal_(self.classifier1.weight.data, std=0.001)
        init.constant_(self.classifier1.bias.data, 0.0)

        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(1)
        self.threshold = 0.3

    def forward(self, x):

        # x1.shape torch.Size([64, 2048, 24, 12])
        x1 = self.base(x)
        x2 = self.pool1(x1)
        # x2.shape torch.Size([64, 4096, 1, 1])
        # f.shape torch.Size([64, 4096])
        f = x2.view(x2.size(0), -1)
        f = self.bn(f)

        # [64, 2048, 6, 12]
        split_tensors = torch.split(x1, 4, dim=2)
        split_tensors_1 = [self.part_quality_predictor(i) for i in split_tensors]
        split_tensors_2 = [i for i in split_tensors_1]

        if not self.training:
            # 对列表中的每个张量执行操作
            for i in range(len(split_tensors_1)):
                tensor = split_tensors_1[i]
                # 创建掩码，小于阈值的元素为True，否则为False
                mask = tensor < self.threshold
                # 将小于阈值的通道置为0
                tensor[mask] = 0
                # 将处理后的张量放回列表中
                split_tensors_2[i] = tensor

        split_tensors_3 = [split_tensors[i] * split_tensors_2[i] for i in range(len(split_tensors))]
        split_tensors_4 = [self.pool2(i) for i in split_tensors_3]
        # [64, 4096, 1, 1]
        # result = torch.mul(split_tensors_1_cat, b_4_cat)
        result = torch.cat(split_tensors_4, dim=2)
        # f1.shape torch.Size([64, 4096, 1, 1])
        f1 = self.bn2048(self.pool(result))

        # print("f1.shape", f1.shape)
        # f2.shape torch.Size([64, 4096, 1, 1])
        # print("f2.shape", f2.shape)

        if self.training:
            f1 = f1.view(f1.size(0), -1)
            f1 = self.classifier1(f1)
            split_tensors_5 = [i.view(i.size(0), -1) for i in split_tensors_4]
            print("*"*99)
            print(split_tensors_5[0].shape)
            # print("f3.shape", f3.shape)
            spt = [self.classifier(i) for i in split_tensors_5]
            # [64, 4096]
            return f, f1, spt[0], spt[1], spt[2], spt[3], spt[4], spt[5],
        else:

            f1 = f1.view(f1.size(0), -1)
            return f1