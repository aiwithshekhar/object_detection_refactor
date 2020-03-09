import os
import numpy as np

def parse_model_cfg(path):
    with open(path) as file:
        text=file.read().split('\n')
    mod=[]
    text=[val for val in text if val and not val.startswith('#')]
    text=[x.rstrip().lstrip() for x in text]
    for loc in text:
        if loc.startswith('['):
            mod.append({})
            mod[-1]['type']=loc[1:-1].rstrip()
            if mod[-1]['type']=='convolutional':
                mod[-1]['batch_normalize']=0
        else:
            key, val =loc.split('=')
            key=key.rstrip()

            if key=='anchors':
                mod[-1][key]=np.array([float(x) for x in val.split(',')]).reshape(-1,2)
            else:
                mod[-1][key]=val
    return mod

def parse_data_cfg(path):
    with open(path) as fil:
        read=fil.read().split('\n')
    read=[x for x in read if x]
    data_cfg={}
    for x in read:
        data_cfg[x.split('=')[0].rstrip()]=x.split('=')[1].rstrip()
    return data_cfg