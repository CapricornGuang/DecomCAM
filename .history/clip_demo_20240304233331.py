from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad, DecomCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
import torchvision
import torch

from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize
import numpy as np
import os

import clip

def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    result = result.transpose(2, 3).transpose(1, 2)
    return result

# 读取图片
img_path = "./examples/both.png"
rgb_img = Image.open(img_path).convert('RGB')

model, preprocess = clip.load("RN50", device="cuda")
target_layers = [model.visual.layer4[-1]]
input_tensor = preprocess(rgb_img).unsqueeze(0).cuda()

# 声明 cam，对于 clip 模型 targets 可以为任意文本
cam = DecomCAM(model=model, target_layers=target_layers, n_components=10, p=500)
targets = [ClassifierOutputTarget("cat and dog")]

# 保存激活图
os.makedirs('./outputs/clip', exist_ok=True)

grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
grayscale_cam = grayscale_cam[0, :]

rgb_img = np.array(rgb_img, dtype=np.float32) / 255
visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
Image.fromarray(visualization, 'RGB').save(f'./outputs/clip/test.jpg')

# 保存前 10 主成分
pcas = cam.get_pca(input_tensor=input_tensor, targets=targets)
for i, grayscale_cam in enumerate(pcas):
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    Image.fromarray(visualization, 'RGB').save(f'./outputs/clip/componet_{i}.jpg')






