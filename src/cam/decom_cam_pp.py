import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
from PIL import Image
from sklearn.decomposition import PCA

from cam.grad_cam import GradCAM
from cam.base_cam import BaseCAM
from cam.grad_cam_pp import GradCAM_PP

from utils.viz_tools import normalize, cal_gpu
#
class DecomCAM_PP(GradCAM_PP):
    def __init__(self, model: nn.Module, preprocess, layer: nn.Module, type: str, K=100, n_components=10, save_fig={'option':False, 'path':None}):

        super().__init__(model, preprocess, layer, type)
        self.pca = PCA(n_components=n_components)

        self.p_attn_maps = []
        self.K = K
        self.n_components = n_components
        self.device = cal_gpu(model)
        self.save_fig = save_fig
    
    #Overwrite 
    def get_attn_maps(self, input: torch.Tensor, target: torch.Tensor, image_np: np.ndarray):

        #Get the output of GradCAM_PP
        self.current_output = super().get_attn_maps(input, target)
        act, alpha = self.current_output['act'], self.current_output['alpha']
        #alpha_grad = grad.mean(dim=(2, 3), keepdim=True)
        
        #Split channels with potential semantic activation out
        alpha = torch.flatten(alpha.detach().cpu())
        values = torch.sort(alpha, descending=True).values.numpy()
        indices = torch.sort(alpha, descending=True).indices.numpy()
        alpha = alpha.unsqueeze(0).unsqueeze(2).unsqueeze(3).to(self.device)
        
        #Execute PCA
        X = act[:,indices.tolist()[:self.K],:,:]*alpha[:,indices.tolist()[:self.K],:,:] #Note Here, 'act*alpha',  rather than 'act'
        X = X.flatten(start_dim=2).squeeze(0).T.detach().cpu()
        Y = self.pca.fit_transform(X)

        #Take each principle component as mask
        image_blur = BaseCAM.gaussian_blur_image(image_np)
        
        sim0 = BaseCAM.getImageTextSim(
            BaseCAM.NpImage2Tensor(self.model, self.preprocess, image_blur), 
            target).item()
        image_maskeds = []
        image_masked_alphas = []
        p_attn_maps = []
        act_map_weight, act_map_height = act.size(2), act.size(3)
        for i in range(self.n_components):

            #Calculate mask with each component
            y = Y[:, i].flatten()
            attn_map_i = torch.from_numpy(y.reshape(act_map_weight, act_map_height)
                ).unsqueeze(0).unsqueeze(0).to(self.device)
            
            attn_map_i = torch.clamp(attn_map_i, min=0)
            attn_map_i = F.interpolate(attn_map_i,
                input.shape[2:],
                mode='bicubic',
                align_corners=False)

            attn_map_i = attn_map_i.squeeze().detach().cpu().numpy()
            p_attn_maps.append(attn_map_i)

            mask = normalize(attn_map_i)
            image_masked = BaseCAM.get_image_mask(mask, image_np, image_blur) 

            #Calculate the contribution of each component
            sim = BaseCAM.getImageTextSim(
                    BaseCAM.NpImage2Tensor(self.model, self.preprocess, image_masked), 
                    target).item()

            image_maskeds.append(image_masked)

            image_masked_alphas.append(sim-sim0)
            #image_masked_alphas.append(sim0-sim)
            #Save Component Visulization
            if self.save_fig['option']:
                self.save_component_figure(image_masked, i)
        # Store internal variables
        self.p_attn_maps = p_attn_maps
        self.image_masked_alphas = image_masked_alphas
        self.image_maskeds = image_maskeds

        #Get the attn_map of Decom-Map
        self.current_output['attn_map'] = BaseCAM.accumulate_attn_map(p_attn_maps, image_masked_alphas)
        torch.cuda.empty_cache()
        return self.current_output

    def save_component_figure(self, image_masked, component_index):
        image_masked_pil = Image.fromarray(np.uint8(image_masked*255))
        image_masked_pil.save('{}/image_masked_component{}.jpg'.format(self.save_fig['path'],component_index))
        return None
