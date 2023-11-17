import torch
from torch import nn
import torch.nn.functional as F 

import numpy as np
from PIL import Image
from scipy.ndimage import filters       
from utils import translate_tools       
from utils.viz_tools import cal_gpu     

class BaseCAM:
    '''
    The prototype (or templete) for all cam implement
    grad_cam, alpha, act, grad
    '''
    def __init__(self, model: nn.Module, preprocess, layer: nn.Module, type: str):
        self.model = model
        self.layer = layer
        self.preprocess = preprocess
        self.type = 'BaseCAM'

        #the attrs may vary among each Methods
        self.current_output = { 
                'attn_map': None,
                'alpha': None,      
                'act': None}
                
        if self.type in ['Grad-CAM', 'Grad-CAM++', 'Layer-CAM', 'Ablation-CAM']: 
            self.current_output['grad'] = None


    def get_attn_maps(self, input: torch.Tensor, target: torch.Tensor):     
        return self.current_output

    @staticmethod
    def accumulate_attn_map(attn_maps: list, weights: list):    
        pcam = None
        weights = weights*np.exp(np.array(weights))/np.sum(np.exp(np.array(weights))) #softmax
        #weights = np.exp(np.array(weights))/np.sum(np.exp(np.array(weights))) #softmax
        for attn_map_tmp, alpha in zip(attn_maps, weights):
            if pcam is None:
                pcam = alpha * attn_map_tmp
                continue
            pcam += alpha * attn_map_tmp
        pcam = np.maximum(pcam, 0)
        return pcam

    @staticmethod
    def get_image_mask(mask, image_np, image_blur):
        return np.maximum(image_np*mask[:,:,None], image_blur*(1-mask[:,:,None]))
        #return image_np*mask[:,:,None] + image_blur*(1-mask[:,:,None])
    #reference: https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.ndimage.filters.gaussian_filter.html
    @staticmethod
    def gaussian_blur_image(image_np, sigma=10, truncate=4):
        '''
        blur an image with gaussian filter
        '''
        assert image_np.shape[2] == 3, 'the image should be transformed into 3 channel'
        image_blur = np.zeros(image_np.shape)
        for i in range(image_np.shape[2]):
            image_blur[:,:,i] = filters.gaussian_filter(image_np[:,:,i], sigma, truncate=truncate)
        return image_blur

    @staticmethod
    def NpImage2Tensor(model, preprocess, image_np):
        if np.max(image_np) <= 1.:
            image_pil = Image.fromarray(np.uint8(image_np*255))
        else:
            image_pil = Image.fromarray(np.uint8(image_np))
        image_input = preprocess(image_pil).unsqueeze(0).to(cal_gpu(model))     
        return model(image_input)
        
    @staticmethod
    def getImageTextSim(image_embed, text_embed):      
        return F.cosine_similarity(image_embed, text_embed)



class FreezeGrad:
    '''
    The grad of the given module in this filed will be freezed
    '''
    def __init__(self, module: nn.Module):
        self.requires_grad = {}
        self.module = module

    def __enter__(self):            
        for name, param in self.module.named_parameters():
            self.requires_grad[name] = param.requires_grad
            param.requires_grad_(False)
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        for name, param in self.module.named_parameters():
            param.requires_grad_(self.requires_grad[name])
        return self

class Hook:
    """Attaches to a module and records its activations and gradients."""
    def __init__(self, module: nn.Module):
        self.data = None
        self.hook = module.register_forward_hook(self.save_grad)
        
    def save_grad(self, module, input, output):
        self.data = output
        output.requires_grad_(True)
        output.retain_grad()
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.hook.remove()
        
    @property
    def activation(self) -> torch.Tensor:
        return self.data
        
    @property
    def gradient(self) -> torch.Tensor:
        return self.data.grad
