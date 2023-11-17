import numpy as np

import torch.nn.functional as F
from scipy.ndimage import filters

def generate_file_name(image_path, image_caption):
    '''
    image_path: ../x.jpg
    image_caption: cat
    ---output---
    return 'x_cat'
    '''
    return '{}_{}'.format(image_path.split('/')[-1].split('.')[0], image_caption)

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