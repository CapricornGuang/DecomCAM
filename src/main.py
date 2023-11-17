import torch
import clip
from PIL import Image
from utils.viz_tools import load_image, viz_attn
from utils import api
from utils.data_tools import load_data_coco, load_data_imagenet, load_data_imagenet2012
from cam.grad_cam import GradCAM
from cam.decom_cam import DecomCAM
from cam.decom_cam_pp import DecomCAM_PP
from cam.ablation_cam import AblationCAM
from cam.score_cam import ScoreCAM
from cam.grad_cam_pp import GradCAM_PP
from cam.score_cam_pp import ScoreCAM_PP
import matplotlib.pyplot as plt
import numpy as np
import ins_del_gc
from kornia.filters.gaussian import gaussian_blur2d
from ins_del_gc import CausalMetric, auc
from cam.base_cam import BaseCAM, FreezeGrad
import time
import argparse
import os 
from tqdm import tqdm
import json

from scipy.ndimage import filters
os.environ['CUDA_LAUNCH_BLOCKING']='12'


def generate_file_name(image_path, image_caption):
    '''
    image_path: ../x.jpg
    image_caption: cat
    ---output---
    return 'x_cat'
    '''
    return '{}_{}'.format(image_path.split('/')[-1].split('.')[0], image_caption)


def blur_fn(image_input, sigma=10, truncate=4):
    image_blur = torch.zeros_like(image_input).to(image_input.device)
    for i in range(image_np.shape[2]):
        image_blur[0, i, :,:] = torch.from_numpy(
                                  filters.gaussian_filter(
                                    image_input[0, i, :,:].detach().cpu().numpy(), 
                                    sigma, truncate=truncate)
                                ).to(image_input.device)
    return image_blur

def remove_files(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            remove_files(c_path)
        else:
            os.remove(c_path)
            print(c_path)
    print('The dinary dir {} has been initialized'.format(path))
    
def remove_dirs(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
          os.rmdir(c_path)
    print('The dinary dir {} has been initialized'.format(path))
            

device = "cuda:3" if torch.cuda.is_available() else "cpu"
clip_model = "RN50"        #@param ["RN50", "RN101", "RN50x4", "RN50x16"]
saliency_layer = "layer4"  #@param ["layer4", "layer3", "layer2", "layer1"]
blur = False               #@param {:"boolean"}
batch_size = 100
remove_files('./output_tmp')
#if os.path.exists('./output_tmp/grad_better'):
#    remove_dirs('./output_tmp/grad_better')
#if os.path.exists('./output_tmp/decom_better'):
#    remove_dirs('./output_tmp/decom_better')

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--val',type=str,default="json_1.json",help='json_1.json json_2.json json_3.json json_4.json json_5.json')
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    #Load CLIP Model
    #c=time.time()
    opt = parse_opt()
    print('Model Loading...')
    model, preprocess = clip.load(clip_model, device=device, jit=False)
    print(preprocess)
    #print("device is "+str(device))
    val_loader, all_cls = load_data_imagenet(is_val=True, batch_size=batch_size, start=0, end=100,val=opt.val)
    #val_loader, all_cls = load_data_imagenet2012(is_val=True, batch_size=batch_size, start=0, end=100)
    print('all_cls', all_cls)
    print(len(all_cls))
    print('testingc number:', len(val_loader))
    #generate CAM
    print("CAM generating...")
    grad_cam = GradCAM(model.visual, preprocess, getattr(model.visual, saliency_layer), 'Grad-CAM')
    decom_cam = DecomCAM(model.visual, preprocess, getattr(model.visual, saliency_layer), 'Decom-CAM', 
    save_fig={
         'option': False,
         'path':'../tmp'},
    n_components=10)
    decom_cam_pp = DecomCAM_PP(model.visual, preprocess, getattr(model.visual, saliency_layer), 'Decom-CAM-PP', 
    save_fig={
         'option': False,
         'path':'../tmp'},
    n_components=10)
    
    ablation_cam = AblationCAM(model.visual, preprocess, getattr(model.visual, saliency_layer),'Ablation-CAM')
    score_cam = ScoreCAM(model.visual, preprocess, getattr(model.visual, saliency_layer),'Score-CAM')
    grad_cam_pp = GradCAM_PP(model.visual, preprocess, getattr(model.visual, saliency_layer),'Grad-CAM-PP')
    score_cam_pp = ScoreCAM(model.visual, preprocess, getattr(model.visual, saliency_layer),'Score-CAM-PP')
    #print(time.time())
    image_path = './Ibizan_hound.jpg'
    image_caption = 'the Ibizan_hound in the picture'#  the Ibizan_hound in the picture
    image_input = preprocess(Image.open(image_path).convert('RGB')).unsqueeze(0).to(device)
    #image_input = preprocess(np.load('./Ibizan_hound.npy')).unsqueeze(0).to(device)
    #print(image_input.shape)
    #image_np = load_image(image_path, model.visual.input_resolution)
    image_np = np.load('./Ibizan_hound_grt_decomuse.npy')
    print(model.visual.input_resolution)
    text_input = clip.tokenize([image_caption]).to(device)
    text_input = model.encode_text(text_input)
    attn_map = api.get_singleImage_attnMap(
                    model = model,
                    layer = getattr(model.visual, saliency_layer),
                    cam = decom_cam_pp.get_attn_maps,
                    inputs = {
                        'image_input':image_input, 
                        'text_input':text_input,
                        'image_np': image_np   
                })
    #print(time.time()-c)
    file_name = generate_file_name(image_path, image_caption)
    viz_attn(image_np, attn_map, file_name, blur)
    print('success!')
    
    #blur_fn = lambda x: gaussian_blur2d(x, kernel_size=(51, 51), sigma=(50., 50.))
    all_cls_captions = ['the {} in the picture'.format(cls) for cls in all_cls]
    insertion = CausalMetric(model, 'ins', 224 * np.int8(224*0.1),substrate_fn=torch.zeros_like, names=all_cls_captions, device=device)#blur_fn
    deletion = CausalMetric(model, 'del', 224 * np.int8(224*0.1), substrate_fn=torch.zeros_like, names=all_cls_captions, device=device)
   
    scores = {'decom_pp':{'del': [], 'ins': []}, 'grad_pp':{'del': [], 'ins': []}}
    # info = {'Dataset':'COCO_val2017','decom':{}, 'grad':{}, 'images':[]}
    info = {'images':[], 'decom_pp':{}, 'grad_pp':{}, 'num':0}
    wrong= {'images':[]}
    for image_paths, image_ids in tqdm(val_loader, total=len(val_loader), desc='evaluating ({} per batch)'.format(batch_size)):
        image_captions = ['the {} in the picture'.format(cls) for cls in all_cls[image_ids]]

        if batch_size == 1:
            image_captions = [image_captions]
        
        for image_path, image_caption, cls_idx in zip(image_paths, image_captions, image_ids):
            print('check',cls_idx) 
            image_input = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
            image_np = load_image(image_path, model.visual.input_resolution)
            print(image_path)
            #image_np = np.load('.{}.npy'.format(image_path.split('.')[1]))
            
            print(image_caption, cls_idx)
            
            text_input = clip.tokenize([image_caption]).to(device)
            text_input = model.encode_text(text_input)

            image_input_ = image_input.detach().clone()
            text_input_ = text_input.detach().clone()

            # evaluating Decom-CAM
            decom_attn_map = api.get_singleImage_attnMap(
                    model = model,
                    layer = getattr(model.visual, saliency_layer),
                    cam = decom_cam_pp.get_attn_maps,
                    inputs = {
                        'image_input':image_input, 
                        'text_input':text_input,
                        'image_np': image_np   
                }
            )
            if not attn_map.any():
                continue
                
            del_score, check, cof = deletion.evaluate(img=image_input, mask=decom_attn_map, cls_idx=cls_idx)
            if check is False:
                wrong['images'].append(image_path)
                print('!!!'*10)
                continue

            ins_score, check, cof = insertion.evaluate(img=image_input, mask=decom_attn_map, cls_idx=cls_idx)
            

            scores['decom_pp']['del'].append(auc(del_score))
            scores['decom_pp']['ins'].append(auc(ins_score))
            
            # evaluating Grad-CAM
            grad_attn_map = api.get_singleImage_attnMap(
                    model = model,
                    layer = getattr(model.visual, saliency_layer),
                    cam = grad_cam_pp.get_attn_maps,
                    inputs = {
                        'image_input':image_input_, 
                        'text_input':text_input_,
                }
            )
            
            del_score, check, cof = deletion.evaluate(img=image_input_, mask=grad_attn_map, cls_idx=cls_idx)
            ins_score, check, cof = insertion.evaluate(img=image_input_, mask=grad_attn_map, cls_idx=cls_idx)
            scores['grad_pp']['del'].append(auc(del_score))
            scores['grad_pp']['ins'].append(auc(ins_score))

            
            info['images'].append({'image_path':image_path, 'categories':image_caption,
                                   'decom_pp':{'del':scores['decom_pp']['del'][-1], 'ins':scores['decom_pp']['ins'][-1]}, 
                                   'grad_pp':{'del':scores['grad_pp']['del'][-1], 'ins':scores['grad_pp']['ins'][-1]},
                                   'cls_idx': cls_idx.item(),
                                   'cof':cof
                                   })
            print('cof: ',cof)
            print('testing:{}'.format(image_path))
            print('caption:{}'.format(image_caption))
            print('decom_pp del:{} ins:{}'.format(scores['decom_pp']['del'][-1], scores['decom_pp']['ins'][-1]))
            print('grad_pp del:{} ins:{}\n'.format(scores['grad_pp']['del'][-1], scores['grad_pp']['ins'][-1]))
            
            
            if (scores['decom_pp']['ins'][-1]-scores['decom_pp']['del'][-1]) > (scores['grad_pp']['ins'][-1]-scores['grad_pp']['del'][-1] + 0.05):  
                dir_path = './output_tmp/decom_pp_better/' + image_caption + '_' + image_path.split('/')[-1].split('.')[0] 
                if os.path.exists(dir_path):
                    continue
                os.mkdir(dir_path)
                img = Image.open(image_path)
                img.save(os.path.join(dir_path, '{:.2f} origin.jpg'.format((scores['decom_pp']['ins'][-1]-scores['decom_pp']['del'][-1])-
                (scores['grad_pp']['ins'][-1]-scores['grad_pp']['del'][-1]))))
                #file_name = 'decomCAM' + '_' + generate_file_name(image_path, image_caption)
                viz_attn(image_np, decom_attn_map, file_name='decom_pp', blur=blur, dir_path=dir_path)
                #file_name = 'gradCAM' + '_' + generate_file_name(image_path, image_caption)
                viz_attn(image_np, grad_attn_map, file_name='grad_pp', blur=blur, dir_path=dir_path)
              
            if (scores['decom_pp']['ins'][-1]-scores['decom_pp']['del'][-1]+0.05) < (scores['grad_pp']['ins'][-1]-scores['grad_pp']['del'][-1]):  
                dir_path = './output_tmp/grad_pp_better/' + image_caption + '_' + image_path.split('/')[-1].split('.')[0]
                if os.path.exists(dir_path):
                    continue
                os.mkdir(dir_path)               
                img = Image.open(image_path)
                img.save(os.path.join(dir_path, '{:.2f} origin.jpg'.format(-(scores['decom_pp']['ins'][-1]-scores['decom_pp']['del'][-1]).item()+
                (scores['grad_pp']['ins'][-1]-scores['grad_pp']['del'][-1]).item())))
                #file_name = 'decomCAM' + '_' + generate_file_name(image_path, image_caption)
                viz_attn(image_np, decom_attn_map, file_name='decom_pp', blur=blur, dir_path=dir_path)
                #file_name = 'gradCAM' + '_' + generate_file_name(image_path, image_caption)
                viz_attn(image_np, grad_attn_map, file_name='grad_pp', blur=blur, dir_path=dir_path)
        #Code below, temporally annotated!!!!!!!!!!!!!       
        decom_del = np.mean(scores['decom_pp']['del']) 
        decom_ins = np.mean(scores['decom_pp']['ins'])
        
        grad_del = np.mean(scores['grad_pp']['del'])
        grad_ins = np.mean(scores['grad_pp']['ins'])
        
        info['decom_pp'].update({'ave_del': decom_del, 'ave_ins':decom_ins, 'ave_ins-del':decom_ins-decom_del})
        info['grad_pp'].update({'ave_del': grad_del, 'ave_ins':grad_ins, 'ave_ins-del':grad_ins-grad_del})
        info['num'] += 100
    
    with open('wrong100_{}_in_decom_pp_grad_cam_pp.json'.format(opt.val), 'w+', encoding='utf-8') as f:
        json.dump(wrong, f)

    with open('result100_{}_in_decom_pp_grad_cam_pp.json'.format(opt.val), 'w+', encoding='utf-8') as f:
        json.dump(info, f)
        
    print('----------------------------------------------------------------')
    print('Final:\nDecomCAM_PP:\nDeletion : {:.5f}\nInsertion : {:.5f}'.format(decom_del, decom_ins))
    print('Ins-Del: {:.5f}'.format(decom_ins-decom_del))
    print('GradCAM_PP:\nDeletion : {:.5f}\nInsertion : {:.5f}'.format(grad_del, grad_ins))
    print('Ins-Del: {:.5f}'.format(grad_ins-grad_del))
    
    print('success!')
    



