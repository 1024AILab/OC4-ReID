import torch
import torchvision
from torch import nn
from torch.nn import init
from models.utils import pooling


class ResNet50(nn.Module):
    def __init__(self):
        super().__init__()

        resnet50 = torchvision.models.resnet50(pretrained=True)
        resnet50.layer4[0].conv2.stride = (1, 1)
        resnet50.layer4[0].downsample[0].stride = (1, 1)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.globalpooling = pooling.MaxAvgPooling()

        self.bn = nn.BatchNorm1d(4096)
        init.normal_(self.bn.weight.data, 1.0, 0.02)
        init.constant_(self.bn.bias.data, 0.0)
        self.part_quality_predictor = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=4096, kernel_size=1),
            nn.BatchNorm2d(4096),
            nn.Sigmoid()
        )

        self.avg_pool = nn.AvgPool2d(kernel_size=(4, 12), stride=(4, 12))
        self.conv_layers = nn.ModuleList([nn.Conv2d(2048, 4096, kernel_size=1) for _ in range(6)])
        self.bn4096 = nn.BatchNorm2d(4096)
        # 使用平均池化将其变为[32, 4096, 1, 1]
        self.pool = nn.AvgPool2d(kernel_size=(6, 1))

        self.classifier = nn.Linear(4096, 150)
        init.normal_(self.classifier.weight.data, std=0.001)
        init.constant_(self.classifier.bias.data, 0.0)

    def forward(self, x):
        print(x.shape)
        # x1.shape torch.Size([64, 2048, 24, 12])
        x1 = self.base(x)
        # x2.shape torch.Size([64, 4096, 1, 1])
        x2 = self.globalpooling(x1)
        x = x2.view(x2.size(0), -1)
        # f.shape torch.Size([64, 4096])
        f = self.bn(x)

        split_tensors = torch.split(x1, 4, dim=2)
        split_tensors_1 = [self.part_quality_predictor(i) for i in split_tensors]
        split_tensors_2 = [self.avg_pool(i) for i in split_tensors_1]

        # result = torch.mul(split_tensors_1_cat, b_4_cat)
        result = torch.cat(split_tensors_2, dim=2)
        # f1.shape torch.Size([64, 4096, 1, 1])
        # print(result.shape)
        f1 = self.pool(result)
        # print("f1.shape", f1.shape)
        # f2.shape torch.Size([64, 4096, 1, 1])
        f2 = f1 * x2
        # print("f2.shape", f2.shape)
        if self.training:
            f3 = f2.view(f2.size(0), -1)
            # print("f3.shape", f3.shape)
            f4 = self.classifier(f3)
            # [64, 4096]
            return f, f4
        else:
            f2 = f2.view(f2.size(0), -1)
            return f2