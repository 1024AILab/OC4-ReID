# coding=utf-8
# @FileName:demo.py
# @Time:2024/7/13 
# @Author: CZH
import torch

tensor1 = torch.randn(1, 3, 24, 12)
tensor2 = torch.randn(1, 3, 1, 1)

tensor3 = tensor1 * tensor2

print("tensor1.shape", tensor1.shape, "tensor2.shape", tensor2.shape, "tensor3.shape", tensor3.shape)