# coding=utf-8
# @FileName:infer1.py
# @Time:2024/7/24 
# @Author: CZH
# coding=utf-8
# @FileName:infer.py
# @Time:2024/7/24
# @Author: CZH
from __future__ import print_function
import torch
import timm
from matplotlib import pyplot as plt
import numpy as np
import cv2
import os
import random
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad
from os import makedirs
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image
from tqdm import tqdm
from timm.models.vision_transformer import vit_small_patch16_224, vit_base_patch8_224, vit_large_patch16_224
from configs.infra_net import ResNet50
from torchvision import transforms


model = ResNet50()
weights_path = r"F:\acm\pythonProject\Simple-CCReID-main\best_model.pth.tar"

checkpoint = torch.load(weights_path)

res = model.load_state_dict(checkpoint['model_state_dict'])

print(res)

target_layers = [model.part_quality_predictor[0]]
use_cuda = False
if use_cuda:
    model = model.cuda()
model.eval()


cam = GradCAMPlusPlus(model=model, target_layers=target_layers)


# 读取图像并转换为RGB格式
rgb_img = cv2.imread(r"F:\acm\pythonProject\Simple-CCReID-main\B_cropped_rgb082.png", 1)[:, :, ::-1]

# 调整图像大小
rgb_img = cv2.resize(rgb_img, (384, 192))

# 转换为浮点数并归一化
rgb_img = np.float32(rgb_img) / 255

# 从HWC格式转换为CHW格式
chw_img = np.transpose(rgb_img, (2, 1, 0))

# 转换为torch.Tensor
tensor_img = torch.from_numpy(chw_img)

# 定义预处理操作
preprocess = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 预处理图像
input_tensor = preprocess(tensor_img).unsqueeze(0)
# input_tensor = input_tensor.unsqueeze(0)
if use_cuda:
    input_tensor = input_tensor.cuda()
for i in range(1, 4001, 10):
    targets = [ClassifierOutputTarget(i)]
    grayscale_cam = cam(input_tensor=input_tensor,
                        targets=targets)

    # grayscale_cam = cam(input_tensor=input_tensor)
    # print(grayscale_cam.shape)
    grayscale_cam = grayscale_cam[0, :]

    # grayscale_cam = cv2.resize(grayscale_cam, (32, 32))
    rgb_img = cv2.resize(rgb_img, (192, 384))
    # print(rgb_img.shape, grayscale_cam.shape)
    grayscale_cam = cv2.resize(grayscale_cam, (192, 384))
    cam_image = show_cam_on_image(rgb_img, grayscale_cam)
    # Convert BGR to RGB
    cam_image_rgb = cv2.cvtColor(cam_image, cv2.COLOR_BGR2RGB)
    # Display using matplotlib
    plt.title(str(i))
    plt.imshow(cam_image_rgb)
    plt.axis('off')  # Turn off axis

    plt.savefig(f"output2/{i}_heatmap.png")
    plt.show()
    # print(i)
