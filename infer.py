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


def get_random_image_paths(root_dir):
    image_paths = []
    for subdir, dirs, files in os.walk(root_dir):
        image_files = [f for f in files if f.endswith(".jpg") or f.endswith(".png")]
        if image_files:
            random_image = random.choice(image_files)
            image_paths.append(os.path.join(subdir, random_image))
    return image_paths


model = vit_small_patch16_224(img_size=(32, 32), depth=5, patch_size=8, mlp_ratio=2,
                                   num_heads=6, embed_dim=192, num_classes=43)
# model = lightViT(img_size=(224, 224), depth=5, patch_size=8, mlp_ratio=2, num_heads=6, embed_dim=192, num_classes=43)
# weights_path = r"E:\pythonProject\LVIT\lightViT1_train_on_german.pth"
weights_path = r"E:\pythonProject\LVIT\vit_small_patch16_224_train_on_German.pth"
checkpoint = torch.load(weights_path)
model.load_state_dict(checkpoint['model_state_dict'])
print(model.blocks[-1])

target_layers = [model.blocks[-1].norm1]
use_cuda = False
if use_cuda:
    model = model.cuda()
model.eval()


def reshape_transform(tensor):
    result = tensor[:, 1:, :]
    # print(result.shape)
    # print(tensor.shape)
    height = 32
    width = 32
    # print(tensor.shape)
    result = result.reshape(tensor.size(0), height, width, 3)
    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


cam = GradCAM(model=model, target_layers=target_layers,
              reshape_transform=reshape_transform)

test_dir = r"G:\dataset\Traffic-Sign\GTSRB_128x128\train"
random_image_paths = get_random_image_paths(test_dir)
# print(random_image_paths)
for j in tqdm(range(len(random_image_paths)), desc="Processing"):
    rgb_img = cv2.imread(random_image_paths[j], 1)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (32, 32))
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    # input_tensor = input_tensor.unsqueeze(0)
    if use_cuda:
        input_tensor = input_tensor.cuda()
    for i in range(43):
        targets = None
        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=targets)

        # grayscale_cam = cam(input_tensor=input_tensor)
        # print(grayscale_cam.shape)
        grayscale_cam = grayscale_cam[0, :]


        # grayscale_cam = cv2.resize(grayscale_cam, (32, 32))
        rgb_img = cv2.resize(rgb_img, (32, 32))
        # print(rgb_img.shape, grayscale_cam.shape)
        cam_image = show_cam_on_image(rgb_img, grayscale_cam)
        # Convert BGR to RGB
        cam_image_rgb = cv2.cvtColor(cam_image, cv2.COLOR_BGR2RGB)
        # Display using matplotlib
        plt.title(str(i))
        plt.imshow(cam_image_rgb)
        plt.axis('off')  # Turn off axis
        folder_path = f"./heatmap4/{j}"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path, exist_ok=True)
        plt.savefig(f"./heatmap4/{j}/{i}_heatmap.png")
        plt.show()
        # print(i)
