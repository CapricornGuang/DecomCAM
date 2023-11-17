# CUDA_VISIBLE_DEVICES=0
import json, gc

import matplotlib.pyplot as plt
import numpy as np

import clip

from utils import api

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler    
import torchvision.datasets as datasets         
import torchvision.models as models             
import torchvision.transforms as transforms

from kornia.filters.gaussian import gaussian_blur2d
from tqdm import tqdm

from cam import *
from cam.base_cam import BaseCAM, FreezeGrad

__all__ = ['CausalMetric', 'auc']

import warnings
warnings.filterwarnings("ignore")

HW = 224 * 224
# img_label = json.load(open('./utils/resources/imagenet_class_index.json', 'r'))


# Plots image from tensor
def tensor_imshow(inp, title=None, **kwargs):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    # Mean and std for ImageNet
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp, **kwargs)
    if title is not None:
        plt.title(title)


def auc(arr):
    """Returns normalized Area Under Curve of the array."""
    return (arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1)


class CausalMetric(object):
    def __init__(self, model, mode, step, substrate_fn, names, device, softmax=True):
        """Create deletion/insertion metric instance.
        Args:
            model(nn.Module): Black-box model being explained.
            mode (str): 'del' or 'ins'.
            step (int): number of pixels modified per one iteration.
            substrate_fn (func): a mapping from old pixels to new pixels.
            names: all the class names
        """
        assert mode in ['del', 'ins']
        self.model = model.eval().to(device)
        self.device = device
        self.mode = mode
        self.step = step
        self.substrate_fn = substrate_fn
        self.names = names
        self.texts = []
        self.softmax = softmax
        
        for name in names:
            text = clip.tokenize([name]).to(device)
            text = model.encode_text(text).squeeze(dim=0)
            #to cpu
            text = text.detach().float().cpu()
            torch.cuda.empty_cache()
            #to gpu
            #text /= text.norm(p=2, dim=0)
            self.texts.append(text)
        self.texts = torch.stack(self.texts, dim=0)

    def sim(self, img):
        text = self.model.visual(img.to(self.device)).squeeze().detach()
        text /= text.norm(p=2, dim=0)
        #to cpu
        text = text.detach().float().cpu()
        
        #SoftMax
        sim = 100*torch.matmul(self.texts, text)
        if self.softmax:
          
          sim = F.softmax(sim, dim=-1)
        else:
          sim = sim/100
          pass
        #sim = (sim-torch.min(sim))/(torch.max(sim)-torch.min(sim))
        return sim
    
    def evaluate(self, img, mask, cls_idx):
        """
        Run metric on one image-saliency pair.
        Args:
            img (Tensor): normalized image tensor.
            mask (np.ndarray): saliency map.
        Return:
            scores (nd.array): Array containing scores at every step.
        """
        check = True
        forward = self.sim(img)
        if cls_idx is None:
            cls_idx = forward.max(0)[-1].item()
        elif cls_idx != forward.argmax().item():
            check = False


        n_steps = (HW + self.step - 1) // self.step
        if self.mode == 'del':
            title = 'Deletion Curve'
            ylabel = 'Pixels deleted'
            start = img.detach().clone()
            finish = self.substrate_fn(img).cpu()
        elif self.mode == 'ins':
            title = 'Insertion Curve'
            ylabel = 'Pixels inserted'
            start = self.substrate_fn(img)
            finish = img.detach().clone().cpu()
        
        scores = np.empty(n_steps + 1, dtype='float32')
        # Coordinates of pixels in order of decreasing saliency
        salient_order = np.flip(np.argsort(mask.reshape(-1, HW), axis=1), axis=-1)
        
        for i in range(n_steps + 1):
            score = self.sim(start)[cls_idx]
            scores[i] = score
            if i < n_steps:
                coords = salient_order[:, self.step * i:self.step * (i + 1)]
                
                start = start.cpu()
                start.numpy().reshape(1, 3, HW)[0, :, coords] = finish.numpy().reshape(1, 3, HW)[0, :, coords]
                start = start.to(self.device)
    
        return scores, check, torch.max(forward).item()
    def evaluate_in_wrong_condition(self, img, mask, cls_idx):
        """
        Run metric on one image-saliency pair.
        Args:
            img (Tensor): normalized image tensor.
            mask (np.ndarray): saliency map.
        Return:
            scores (nd.array): Array containing scores at every step.
        """
        check = True
        forward = self.sim(img)
        if cls_idx is None:
            cls_idx = forward.max(0)[-1].item()
        elif cls_idx != forward.argmax().item():
            check = False


        n_steps = (HW + self.step - 1) // self.step
        if self.mode == 'del':
            title = 'Deletion Curve'
            ylabel = 'Pixels deleted'
            start = img.detach().clone()
            finish = self.substrate_fn(img).cpu()
        elif self.mode == 'ins':
            title = 'Insertion Curve'
            ylabel = 'Pixels inserted'
            start = self.substrate_fn(img)
            finish = img.detach().clone().cpu()
        
        scores = np.empty(n_steps + 1, dtype='float32')
        # Coordinates of pixels in order of decreasing saliency
        salient_order = np.flip(np.argsort(mask.reshape(-1, HW), axis=1), axis=-1)
        
        for i in range(n_steps + 1):
            score = self.sim(start)[cls_idx]
            scores[i] = score
            if i < n_steps:
                coords = salient_order[:, self.step * i:self.step * (i + 1)]
                
                start = start.cpu()
                start.numpy().reshape(1, 3, HW)[0, :, coords] = finish.numpy().reshape(1, 3, HW)[0, :, coords]
                start = start.to(self.device)
    
        return scores, torch.max(forward).item()
    def evaluate_check(self, img, cls_idx):
        check = True
        forward = self.sim(img)
        if cls_idx is None:
            cls_idx = forward.max(0)[-1].item()
        elif cls_idx != forward.argmax().item():
            check = False
            pred_id = forward.argmax().item()
        elif cls_idx == forward.argmax().item():
            pred_id = cls_idx
        return check,pred_id

