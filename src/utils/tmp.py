import json
from data_tools import read_coco
import torch
import numpy as np

img_paths, img_ids, all_cls = read_coco(is_val=True)

with open('./backup.json', 'r', encoding='utf-8') as f:
    info = json.load(f)

sum, i = len(info['images']), 0
for image in info['images']:
    decom = image['decom']['ins'] - image['decom']['del']
    grad = image['grad']['ins'] - image['grad']['del']
    if decom > grad:
        i += 1

print('decom_better:{:.5f}'.format(1.0*i/sum))
print('grad_better: {:.5f} '.format(1-1.0*i/sum))

#with open('./result.json', 'w') as f:
#    json.dump(info, f)



