import torch
from collections import OrderedDict
import sys

src = sys.argv[1]
dst = sys.argv[2]
mmaction = torch.load(src, map_location='cpu')
output_dict = {}
output_dict['model'] = OrderedDict()
for key, val in mmaction['state_dict'].items():
    if 'backbone' in key:
        key = key.replace('backbone.', 'module.')
    if key.startswith('cls_head'):
        key = key.replace('cls_head.proj.', 'module.proj.')
    output_dict['model'][key] = val
torch.save(output_dict, dst)