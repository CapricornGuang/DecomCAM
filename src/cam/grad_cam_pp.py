import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import clip
from PIL import Image
from scipy.ndimage import filters
from torch import nn
from cam.base_cam import BaseCAM
from cam.base_cam import Hook
import time
from utils.viz_tools import normalize, cal_gpu

def exp(sim, hook):
    exp_similarity = torch.exp(sim)
    exp_similarity.backward(retain_graph=True)
    exp_grad = hook.gradient.float()
    act = hook.activation.float()
    sum_act = torch.sum(act, dim=(2,3))
    # calculate high order gradient of exponent
    hook.gradient.zero_()
    sim.backward()
    grad = hook.gradient.float()
    
    grad_power_2 = grad**2
    grad_power_3 = grad**3
    eps = 0.000001
    aij = grad_power_2 / (2 * grad_power_2 + sum_act[:,:,None,None] * grad_power_3 + eps)
    return exp_grad, act, aij

class GradCAM_PP(BaseCAM):

    def __init__(self, model: nn.Module, preprocess, layer: nn.Module, type: str):
        super().__init__(model, preprocess, layer, type)
        self.type = type
    
    #Overwrite 
    def get_attn_maps(self, input: torch.Tensor, target: torch.Tensor):
        # Zero out any gradients at the input.
        if input.grad is not None:
            input.grad.data.zero_()
            
        # Disable gradient settings.
        requires_grad = {}
        for name, param in self.model.named_parameters():
            requires_grad[name] = param.requires_grad
            param.requires_grad_(False)
        
        # Attach a hook to the model at the desired layer.
        assert isinstance(self.layer, nn.Module)
        with Hook(self.layer) as hook:
            # Do a forward and backward pass.
            
            output = self.model(input)
            
            #cosine_similarity = torch.mm(output.half(), target.half().T)
            cosine_similarity = torch.cosine_similarity(output, target, dim=1).unsqueeze(0)
            #cosine_similarity.backward()
            grad, act, aij = exp(cosine_similarity, hook)
            
            alpha = torch.sum(F.relu(grad) * aij, dim=(2,3), keepdim=True)
            # Weighted combination of activation maps over channel
            # dimension.
            gradcam = torch.sum(act * alpha, dim=1, keepdim=True)
            # We only want neurons with positive influence so we
            # clamp any negative ones.
            gradcam = torch.clamp(gradcam, min=0)
    
        # Resize gradcam to input resolution.
        gradcam = F.interpolate(
            gradcam,
            input.shape[2:],
            mode='bicubic',
            align_corners=False)
        
        # Restore gradient settings.
        for name, param in self.model.named_parameters():
            param.requires_grad_(requires_grad[name])
        torch.cuda.empty_cache()
        
        self.current_output = {
                'attn_map': gradcam, 
                'alpha': alpha,
                'aij': aij, 
                'grad': grad,
                'act': act}
            
        return self.current_output