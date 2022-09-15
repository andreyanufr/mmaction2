import torch
from collections import OrderedDict
import sys

src = sys.argv[1]
dst = sys.argv[2]
mmaction = torch.load(src, map_location='cpu')
output = OrderedDict()
for key, val in src['model'].items():
    if 'module' in key:
        key = key.replace('module.', '')
    if key.startswith('proj'):
        key = key.replace('proj.', 'cls_head.proj.')
    if 'decoder' in key:
        key = key.replace('decoder.', 'backbone.decoder.')
    output[key] = val
torch.save(output, dst)