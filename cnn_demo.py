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

preprocess = Compose([
    Resize(224),
    ToTensor()
])

# 读取图片
img_path = "./examples/both.png"
rgb_img = Image.open(img_path).convert('RGB')
input_tensor = preprocess(rgb_img).unsqueeze(0)

# 指定模型与目标层
model = resnet50(pretrained=True).cuda().eval()
target_layers = [model.layer4[-1]]

# 指定 cam
# target 为类别，281: tabby tabby cat, 242: boxer dog
cam = GradCAM(model=model, target_layers=target_layers)
targets = targets = [ClassifierOutputTarget(281)]

# 生成激活图并保存
os.makedirs('./outputs/cnn', exist_ok=True)

grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
grayscale_cam = grayscale_cam[0, :]

rgb_img = np.array(rgb_img, dtype=np.float32) / 255
visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
Image.fromarray(visualization, 'RGB').save(f'./outputs/cnn/test.jpg')