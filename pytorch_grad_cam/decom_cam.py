import numpy as np
from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam.grad_cam_plusplus import GradCAMPlusPlus
from pytorch_grad_cam.score_cam import ScoreCAM
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients

from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
import torch.nn.functional as F
import torch
from PIL import Image
from scipy.ndimage import filters
from torch import nn

import clip


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



class DecomCAM(GradCAMPlusPlus):
    def __init__(self, model, target_layers,
                 reshape_transform=None, is_clip=False, is_transformer=False, n_components=10, p=500):
        super(DecomCAM, self).__init__(
            model,
            target_layers,
            reshape_transform,
        )

        self.n_components = n_components
        self.p = p
        self.reshape_transform =reshape_transform


    def get_cam_image(self,
                      input_tensor: torch.Tensor,
                      target_layer: torch.nn.Module,
                      targets: np.array,
                      activations: torch.Tensor,
                      grads: torch.Tensor,
                      eigen_smooth: bool = False,
                    ) -> np.ndarray:


        device = self.device
        image_np = input_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() 
        alpha = np.mean(grads, axis=(2, 3))
        
        # alpha = self.get_cam_weights(input_tensor, target_layer, targets, activations, grads)
        
        #with Hook(target_layer) as hook:
        #    img_embed = model.visual(img_tensor)
        #    text_emd = model.encode_text(text_tensor)
        #    sim = torch.cosine_similarity(img_embed, text_emd, dim=1).unsqueeze(0)
        #    grad, act, aij = exp(sim, hook)
        #    alpha = torch.sum(F.relu(grad) * aij, dim=(2,3), keepdim=True)

        #weights = copy.deepcopy(alpha)
        #weights = torch.from_numpy(weights).unsqueeze(2).unsqueeze(2)
        
        #grad_cam_pp = (weights * activations).float()
        #grad_cam_pp = torch.sum(grad_cam_pp, dim=1, keepdim=True)
        
        #grad_cam_pp = F.interpolate(grad_cam_pp,
        #                  img_tensor.shape[2:],
        #                  mode='bicubic',
        #                  align_corners=False)
        #grad_cam_pp = grad_cam_pp.squeeze(1)
        
        alpha = torch.from_numpy(alpha).to(device)
        act = torch.from_numpy(activations).to(device) 
        grad = torch.from_numpy(grads).to(device)
        
        p_attn_maps = []
        p = self.p
        n_components = self.n_components
        descending = True
        alpha = torch.flatten(alpha.detach().cpu())
        
        values = torch.sort(alpha, descending=descending).values.numpy()
        indices = torch.sort(alpha, descending=descending).indices.numpy()
        alpha = alpha.unsqueeze(0).unsqueeze(2).unsqueeze(3).to(device)
        
        
        X = act[:,indices.tolist()[:p],:,:] * alpha[:,indices.tolist()[:p],:,:]
        width, height = X.size(2), X.size(3)
        X = X.flatten(start_dim=2).squeeze(0).T.detach().cpu()
        X = torch.where(torch.isnan(X), torch.full_like(X, 0), X)


        # when p is too large, directly apply pca will cause progaram stuck so we apply incremental pca algorithm to get approximate result
        if p <= 500:
            pca = PCA(n_components=n_components)
            pca_result = pca.fit_transform(X)
        else:
            pca = IncrementalPCA(n_components)
            for i in range(200, p, 200):
                pca.partial_fit(X[:, i-200:i].T)
            pca.partial_fit(X[:,i:p].T)
            VT = pca.transform(X.T)
            pca_result = X.cpu().numpy() @ VT
            for i in range(n_components):
                pca_result[:, i] /=  pca.explained_variance_ratio_[i]


        for i in range(n_components):
            newX = pca_result[:,i].flatten()
        
            tmp_act = torch.from_numpy(newX.reshape(width,height)).unsqueeze(0).unsqueeze(0).to(device)
        
            grad_sam = tmp_act
            grad_sam = torch.clamp(grad_sam, min=0)
        
            grad_sam = F.interpolate(grad_sam,
                input_tensor.shape[2:],
                mode='bicubic',
                align_corners=False)
            attn_map_tmp = grad_sam.squeeze().detach().cpu().numpy()
            p_attn_maps.append(attn_map_tmp)
            
            
        
        image_maskeds = []
        image_masked_indexs = []
        image_masked_alphas = []
        image_blur = gaussian_blur_image(image_np, sigma=10)
        

        # text_emd = self.model.encode_text(text_tensor)
        # sim0 = getImageTextSim(NpImage2Tensor(model.visual, preprocess, image_blur, device), text_emd).item()

        
        # for index, attn_map in enumerate(p_attn_maps):
        #     mask = normalize(attn_map)
            
        #     image_masked = np.maximum(image_np * mask[:,:,None], image_blur * (1-mask[:,:,None]))

        #     image_maskeds.append(torch.tensor(image_masked).to(device))
            
        # masked_tensor = torch.stack(image_maskeds, dim=0)
        # img_emds = model.visual(masked_tensor)
        # image_masked_alphas = [getImageTextSim(img_emd.unsqueeze(0), text_emd).item() - sim0 for img_emd in img_emds] 
        
        target = targets[0]
        for attn_map in p_attn_maps:
            mask = normalize(attn_map)
            image_masked = np.maximum(image_np * mask[:,:,None], image_blur * (1-mask[:,:,None]))
            image_maskeds.append(torch.tensor(image_masked).to(device))
        
        masked_tensor = torch.stack(image_maskeds, dim=0).permute(0, 3, 1, 2).to(torch.float32)
        with torch.no_grad():
            image_masked_alphas = target(self.model(masked_tensor)).cpu().numpy()
        
        
        pcam = 0
        scaleNormalizer = ScaleNormalize()
        scaleNormalizer.fit(image_masked_alphas)
        
        for attn_map, alpha, p in zip(p_attn_maps, image_masked_alphas, pca.explained_variance_ratio_):
            alpha = 0. if alpha < 0.001 else scaleNormalizer.transformVal(alpha)

            pcam += attn_map * alpha
            
        pcam = pcam[np.newaxis, :].astype(np.float32) 
        
        return pcam 
        
   
    def get_pca(self,
                input_tensor: torch.tensor,
                targets,
            ):
                
        device = self.device
        p = self.p
        n_components = self.n_components
        
        if self.clip:
            for target in targets:
                if isinstance(target.category, str):
                    target.category = self.logit_scale * self.encode_text(clip.tokenize([target.category]).to(device))
        
        self.activations_and_grads = ActivationsAndGradients(self.model, self.target_layers, self.reshape_transform)
        outputs = self.activations_and_grads(input_tensor)
        
        self.model.zero_grad()
        loss = sum([target(output) for target, output in zip(targets, outputs)])
        
        
        loss.backward()
        act = self.activations_and_grads.activations[0].to(device)
        grads = self.activations_and_grads.gradients[0]
        
        alpha = torch.mean(grads, dim=(2, 3))
        
        p_attn_maps = []
        descending = True
        
        alpha = torch.flatten(alpha.detach().cpu())
        
        values = torch.sort(alpha, descending=descending).values.numpy()
        indices = torch.sort(alpha, descending=descending).indices.numpy()
        alpha = alpha.unsqueeze(0).unsqueeze(2).unsqueeze(3).to(device)
        
        
        X = act[:,indices.tolist()[:p],:,:] * alpha[:,indices.tolist()[:p],:,:]
        width, height = X.size(2), X.size(3)
        X = X.flatten(start_dim=2).squeeze(0).T.detach().cpu()
        X = torch.where(torch.isnan(X), torch.full_like(X, 0), X)
    
        pca = PCA(n_components=n_components)
    
        pca_result = pca.fit_transform(X)
        
        for i in range(n_components):
            newX = pca_result[:,i].flatten()
        
            tmp_act = torch.from_numpy(newX.reshape(width,height)).unsqueeze(0).unsqueeze(0).to(device)
        
            grad_sam = tmp_act
            grad_sam = torch.clamp(grad_sam, min=0)
            grad_sam = F.interpolate(grad_sam,
                input_tensor.shape[2:],
                mode='bicubic',
                align_corners=False)
            attn_map_tmp = grad_sam.squeeze().detach().cpu().numpy()
            
            attn_map_tmp = np.maximum(attn_map_tmp, 0)
            attn_map_tmp = attn_map_tmp - np.min(attn_map_tmp)
            attn_map_tmp = attn_map_tmp / (1e-7 + np.max(attn_map_tmp))
            
            p_attn_maps.append(attn_map_tmp)
        
        p_attn_maps = np.stack(p_attn_maps, axis=0)
        
        return p_attn_maps
      

class ScaleNormalize():
    def __init__(self, lst=None):
        if lst is not None:
            self.lst_min = min(lst)
            self.lst_max = max(lst)
        
    def fit(self, lst):
        self.lst_min = min(lst)
        self.lst_max = max(lst)
        return None
    
    def fit_transform(self, lst):
        self.lst_min = min(lst)
        self.lst_max = max(lst)
        return list(map(lambda x: x/max(lst), lst))

    def transformVal(self, val):
        return val/self.lst_max
        
def getFuzzySet(bol_set, val):
    fuzzy_set = (bol_set+0.)*val
    fuzzy_set = np.clip(fuzzy_set, 0., np.max(fuzzy_set))
    return fuzzy_set
    
def gaussian_blur_image(image_np, sigma=10, truncate=4):
    # assert image_np.shape[2] == 3, 'the image should be transformed into 3 channel'
    image_blur = np.zeros(image_np.shape)
    if image_np.ndim == 3:
        for i in range(3):
            image_blur[:,:,i] = filters.gaussian_filter(image_np[:,:,i], sigma, truncate=truncate)
    #else:
    #    image_blur[:,:,0] = filters.gaussian_filter(image_np[:,:,0], sigma, truncate=truncate)
    return image_blur
    
def NpImage2Tensor(clip, preprocess, image_np, device):
    image_pil = Image.fromarray(np.uint8(image_np*255))
    image_input = preprocess(image_pil).unsqueeze(0).to(device)
    return clip(image_input)

def getImageTextSim(image_embed, text_embed):
    return F.cosine_similarity(image_embed, text_embed)

def normalize(x: np.ndarray) -> np.ndarray:
    x = x - x.min()
    if x.max() > 0:
        x = x / x.max()
    return x

def _convert_image_to_rgb(image):
    return image.convert("RGB")
