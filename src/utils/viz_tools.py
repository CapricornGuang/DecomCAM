import torch, cv2
from torch import nn

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import filters

from PIL import Image
import os

def cal_gpu(module):
    if isinstance(module, torch.nn.DataParallel):
        module = module.module
    for submodule in module.children():
        if hasattr(submodule, "_parameters"):
            parameters = submodule._parameters
            if "weight" in parameters:
                return parameters["weight"].device

def normalize(x: np.ndarray) -> np.ndarray:
    # Normalize to [0, 1].
    x = x - x.min()
    if x.max() > 0:
        x = x / x.max()
    return x

# return (resized）image ndarray
def load_image(img_path, resize=None):     
    image = Image.open(img_path).convert('RGB')
    if resize is not None:
        image = image.resize((resize, resize))
    image_np = np.asarray(image, dtype=np.float32) / 255.
    return image_np

    #image = cv2.imread(img_path,  cv2.IMREAD_UNCHANGED)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #if resize is not None:
    #    image = cv2.resize(image, (resize, resize))
    #return np.asarray(image).astype(np.float32) / 255.

# Modified from: https://github.com/salesforce/ALBEF/blob/main/visualization.ipynb
def getAttMap(img, attn_map, blur=True):
    if blur:
        attn_map = filters.gaussian_filter(attn_map, 0.02*max(img.shape[:2]))
    attn_map = normalize(attn_map)
    cmap = plt.get_cmap('jet')
    attn_map_c = np.delete(cmap(attn_map), 3, 2)    # np.delete(array,obj,axis)
    attn_map = 1*(1-attn_map**0.7).reshape(attn_map.shape + (1,))*img + (attn_map**0.7).reshape(attn_map.shape+(1,)) * attn_map_c
    return attn_map


# visualize attention_map
def viz_attn(img, attn_map, file_name='CAM', blur=True, dir_path=''):
    image_pil = Image.fromarray(np.uint8(getAttMap(img, attn_map, blur)*255))
    image_pil.save(os.path.join(dir_path , '{}.jpg'.format(file_name)))

