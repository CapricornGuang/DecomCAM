import torch
from torch import nn
import torch.nn.functional as F
from cam.base_cam import Hook
from cam.base_cam import BaseCAM, FreezeGrad
import numpy 

class WGradCAM(BaseCAM):

    def __init__(self, model: nn.Module, preprocess, layer: nn.Module, type: str):
        super().__init__(model, preprocess, layer, type)
        self.type = type

    #Overwrite 
    def get_attn_maps(self, input: torch.Tensor, target: torch.Tensor):
        threshold = 0

        if input.grad is not None:
            input.grad.data.zero_()
            
        with FreezeGrad(self.model) as freeze_grad_ctl:
            #Freeze the grad of the given model
            with Hook(self.layer) as hook:
                # Do a forward and backward pass.
                output = self.model(input)
                
                cosine_similarity = F.cosine_similarity(output.float(), target.float())
                cosine_similarity.backward()
                #print("look at here")
                
                grad = hook.gradient.float()
                
                act = hook.activation.float()
                alpha = (act*grad).mean(dim=(2, 3), keepdim=True)
                alpha = (alpha-alpha.min())/(alpha.max()-alpha.min())
                grad_cam = torch.sum(act * alpha, dim=1, keepdim=True)
                grad_cam = torch.clamp(grad_cam, min=threshold)
                
                
            grad_cam = F.interpolate(
                grad_cam,
                input.shape[2:],
                mode='bicubic',
                align_corners=False)
            

            torch.cuda.empty_cache()
            self.current_output = {
                'attn_map': grad_cam, 
                'alpha': alpha, 
                'act': act, 
                'grad': grad}
            
        return self.current_output
