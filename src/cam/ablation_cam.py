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

class Ablation_Hook(Hook):
    """Attaches to a module and records its activations and gradients."""
    def __init__(self, module, target_channel):
        self.data = None
        self.channel_nums = None
        self.target_channel = target_channel
        self.hook = module.register_forward_hook(self.save_grad)
        if type(self.target_channel) is int:
            self.ablation_hook = module.register_forward_hook(self.ablation_channel)
            
    #Overwrite    
    def save_grad(self, module, input, output):
        self.data = output
        self.channel_nums = output.size(1)
        output.requires_grad_(True)
        output.retain_grad()
        
    def ablation_channel(self, module, input, output):
        assert self.target_channel<self.channel_nums
        tmp = output.clone().detach()
        tmp[:, self.target_channel, :, :] = torch.zeros_like(tmp[:, self.target_channel, :, :])
        return tmp
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.hook.remove()
        if type(self.target_channel) is int:
            self.ablation_hook.remove()
        
    @property
    def activation(self) -> torch.Tensor:
        return self.data
    
    @property
    def gradient(self) -> torch.Tensor:
        return self.data.grad
    
    '''   
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.hook.remove()
        if type(self.target_channel) is int:
            self.ablation_hook.remove()
        
    @property
    def activation(self) -> torch.Tensor:
        return self.data
    
    @property
    def gradient(self) -> torch.Tensor:
        return self.data.grad
    ''' 

class AblationCAM(BaseCAM):

    def __init__(self, model: nn.Module, preprocess, layer: nn.Module, type: str):
        super().__init__(model, preprocess, layer, type)
        self.type = type
    
    #Overwrite 
    def get_attn_maps(self, input: torch.Tensor, target: torch.Tensor):
        # Zero out any gradients at the input.
        #print(time.time())
        if input.grad is not None:
            input.grad.data.zero_()
            
        # Disable gradient settings.
        requires_grad = {}
        for name, param in self.model.named_parameters():
            requires_grad[name] = param.requires_grad
            param.requires_grad_(False)
            
        # Attach a hook to the model at the desired layer.
        assert isinstance(self.layer, nn.Module)
        
        # Get the channel_nums of the desired layer
        channel_nums, act = None, None
        output = self.model(input)
        a=time.time()
        
        with Ablation_Hook(self.layer, None) as hook:
            output = self.model(input)
            act = hook.activation
            channel_nums = act.size(1)
    
        #Attach a channel-wise hook to the model at the desired layer
        global_semantic = self.model(input).float()
        alpha = []
        target = F.normalize(target, dim=1)
        global_semantic = F.normalize(global_semantic, dim=1)
        #print("channel_nums " + str(channel_nums))
    
        for i in range(channel_nums):
            #b=time.time()-a
            #print(b)
            with Ablation_Hook(self.layer, i) as hook:
                output = self.model(input).float()
                output = F.normalize(output, dim=1)
                local_semantic = (global_semantic-output)
                
                alpha.append(local_semantic.mm(target.T)/global_semantic.mm(target.T))
                
        alpha = torch.concat(alpha, dim=1).unsqueeze(2).unsqueeze(2)
        
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
        #print(time.time())
        torch.cuda.empty_cache()
        
        self.current_output = {
                'attn_map': gradcam, 
                'alpha': alpha, 
                'act': act}
        return self.current_output