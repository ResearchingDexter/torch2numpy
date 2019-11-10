import torch
import pickle as pkl
def torch2numpy(input_file='/Users/baidu/downloads/ga_rpn_r50_caffe_fpn_1x_20190513-95e91886.pth',
                out_file='ga.pkl'):
    param = torch.load(input_file)  # ['D']
    d = {}
    param = param.get('state_dict')
    ii = 0
    f = open(out_file, 'wb')
    for i, k in enumerate(param.keys()):

        if param[k].shape == torch.Size([]):
            continue
        d[k] = param[k].numpy()
        print('i:{}|k:{}|v:{}|numpy:{}|length:{}'.format(ii, k, param[k].shape, d[k].shape, len(d[k].shape)))
        ii += 1
    pkl.dump(d, f, protocol=2)
    f.close()
if __name__ == '__main__':
    input_file='../path/ga_rpn_r50_caffe_fpn_1x_20190513-95e91886.pth'
    out_file='../path/ga_rpn_r50_caffe_fpn_1x_numpy.pkl'
    torch2numpy(input_file,out_file)