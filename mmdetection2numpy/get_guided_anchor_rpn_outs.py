from mmdet.models import build_detector
import torch.nn as nn
import torch
from PIL import Image
import numpy as np
from typing import Tuple,Sequence
import pickle as pkl


def get_img(img_path: str):
    img: Image.Image = Image.open(img_path).convert('RGB')
    img = img.resize((640, 320))
    img = np.asarray(img).transpose((2, 0, 1))
    return img


@torch.no_grad()
def get_out(model: nn.Module, img: np.ndarray, device: 'str'):
    def transpose_reshape(input: torch.Tensor, reshape: Tuple[int]):
        return input.permute((0, 2, 3, 1)).reshape(reshape)
    reshapes=[[1,-1,1],[1,-1,4],[1,-1,2],[1,-1,1]]
    rpn_outs = model.forward_dummy(torch.Tensor(img).unsqueeze(0).to(device))
    print(len(rpn_outs))
    rpn_outs_list=[[],[],[],[]]
    #cls_score, bbox_pred, shape_pred, loc_pred = [], [], [], []
    for i, out in enumerate(rpn_outs):
        print('i:{}|out:{}'.format(i, len(out)))
        for j, o in enumerate(out):
            o = transpose_reshape(o,reshapes[i])
            rpn_outs_list[i].append(o)
            print('j:{}|o:{}'.format(j, o.shape))
        rpn_outs_list[i]=torch.cat(rpn_outs_list[i],1).cpu().numpy()
    names=['cls_score','bbox_pred','shape_pred','loc_pred']
    outs_dict=dict(zip(names,rpn_outs_list))
    return outs_dict

def save(sequences:Sequence,out_files:str):
    with open(out_files,'wb') as f:
        pkl.dump(sequences,f,protocol=2)

if __name__ == '__main__':
    from importlib import import_module
    config = '.guided_anchoring.ga_rpn_r50_caffe_fpn_1x'
    config = import_module(config, 'configs')
    device = 'cuda'
    model = build_detector(config.model, train_cfg=config.train_cfg, test_cfg=config.test_cfg)
    model.to(device)
    weight = torch.load('../path/ga_rpn_r50_caffe_fpn_1x_20190513-95e91886.pth')
    state_dict = weight.get('state_dict')
    model.load_state_dict(state_dict)
    img_path = '../path/1.png'
    img = get_img(img_path)
    print(img.shape, type(model.state_dict()))
    # for i,param in enumerate(model.state_dict().items()):
    # name,data=list(param)
    # print("i:{}|name:{}|shape:{}".format(i,name,data.shape))
    # print(model.state_dict())
    outs_dict=get_out(model, img, device)
    save(outs_dict,'../path/torch_rpn_outs.pkl')

