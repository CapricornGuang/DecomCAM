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

class ScoreCAM_PP(BaseCAM):

    def __init__(self, model: nn.Module, preprocess, layer: nn.Module, type: str):
        super().__init__(model, preprocess, layer, type)
        self.device = cal_gpu(model)
        self.type = type
    
    #Overwrite 
    def get_attn_maps(self, input: torch.Tensor, target: torch.Tensor, image_np: np.ndarray):
        b,c,h,w = input.size()
    
        input.to(self.device)
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
            
            old_score = torch.cosine_similarity(output,target,dim=1)
            old_score.backward()
            old_score = old_score.item()
    #        print('old:',old_score)
            grad = hook.gradient.float()
    #        print('grad:','size:',grad.shape)
            act = hook.activation.float()
        
        b, k, u, v = act.size()
        score_saliency_map = torch.zeros((1, 1, h, w))#用来求和
        score_saliency_map = score_saliency_map.to(self.device)
        image_blur = BaseCAM.gaussian_blur_image(image_np)
        
        for i in range (k):
            ole_score = BaseCAM.getImageTextSim(
            BaseCAM.NpImage2Tensor(self.model, self.preprocess, image_blur), 
            target).item()
            map_grad = grad[:,i,:,:]
            saliency_map = act[:,i,:,:]
            saliency_map_dec = F.relu(map_grad * saliency_map)
            if saliency_map_dec.max() <= 0:
                continue
            
            # 将激活图上采样生成掩膜
            saliency_map = torch.unsqueeze(act[:,i,:,:],1)
            saliency_map = F.interpolate(saliency_map, size=(h,w),mode='bilinear',align_corners=False)
            
    #        map_grad = torch.unsqueeze(grad[:,i,:,:],1)
    #        map_grad = F.interpolate(map_grad, size=(h,w),mode='bilinear',align_corners=False)
    #        saliency_map_dec = saliency_map * map_grad#是否采用这个channel的判据，但是最后加权还是saliency_map
    #        saliency_map_dec = F.relu(saliency_map_dec)
            
            if saliency_map.max() == saliency_map.min():
                continue
            
            # 标准化到（0,1）
            norm_saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
    #        norm_saliency_map = norm_saliency_map.to(device)
            #计算得分
            norm_saliency_map = norm_saliency_map.detach().cpu().squeeze().squeeze().numpy()
            #output = self.model(input * norm_saliency_map) 
            #new_score = torch.cosine_similarity(output,target,dim=1).item()
            image_masked = BaseCAM.get_image_mask(norm_saliency_map, image_np, image_blur)
            new_score = BaseCAM.getImageTextSim(
                    BaseCAM.NpImage2Tensor(self.model, self.preprocess, image_masked), 
                    target).item()
            score = new_score#ole_score-
            
            #score =   new_score - old_score
    #        print("new:",new_score)
    #        saliency_map = saliency_map.to(device)
    #        score = score.to(device)
            
            score_saliency_map +=  score * saliency_map
            torch.cuda.empty_cache()
            
        score_saliency_map = F.relu(score_saliency_map)
        score_saliency_map_min, score_saliency_map_max = score_saliency_map.min(), score_saliency_map.max()
        
        self.current_output = {
                'attn_map': score_saliency_map, 
                #'alpha': alpha, 
                'act': act}
    
        return self.current_output