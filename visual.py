
from email.policy import strict
from tkinter import Variable, font
from server import Server
import argparse
import torch.nn as nn
import torch

import os
import os.path as osp
import glob
import re
from collections import defaultdict
# import cv2 as cv
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms

from sklearn.manifold import TSNE

from utils import get_model

# install.packages("ggsci")
# install.packages("scales")

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--model_name',default='ft_ResNet50', type=str, help='output model name')
parser.add_argument('--project_dir',default='.', type=str, help='project path')
parser.add_argument('--data_dir',default='/data/plzhang/data',type=str, help='training dir path')
# parser.add_argument('--data_dir',default='data',type=str, help='training dir path')
# parser.add_argument('--datasets',default='market',type=str, help='datasets used')
parser.add_argument('--datasets',default='cuhk01,viper,prid,3dpes,ilids,market,duke,cuhk03-np,MSMT17',type=str, help='datasets used')
# parser.add_argument('--datasets',default='cuhk01,viper,prid,3dpes,duke,MSMT17,cuhk03-np,ilids,market',type=str, help='datasets used')
# parser.add_argument('--datasets',default='market,MSMT17,duke,cuhk01,viper,prid,cuhk03-np,3dpes,ilids',type=str, help='datasets used')
# parser.add_argument('--datasets',default='market,duke,MSMT17,cuhk03-np',type=str, help='datasets used')
# parser.add_argument('--datasets',default='Market,DukeMTMC-reID,cuhk03-np-detected,cuhk01,MSMT17,viper,prid,3dpes,ilids',type=str, help='datasets used')
parser.add_argument('--train_all', action='store_true', help='use all training data' )
parser.add_argument('--kd_train', default=True, type=bool, help='use all training data' )
parser.add_argument('--stride', default=2, type=int, help='stride')
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--drop_rate', default=0.5, type=float, help='drop rate')

# arguments for federated setting
# parser.add_argument('--local_epoch', default={'market':1, 'MSMT17':5, 'cuhk03-np':1}, type=int, help='number of local epochs')
parser.add_argument('--local_epoch', default=1, type=int, help='number of local epochs')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
# parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--num_of_clients', default=9, type=int, help='number of clients')
# parser.add_argument('--num_of_clients', default=9, type=int, help='number of clients')

# arguments for data transformation
parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training' )

# arguments for testing federated model
parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--multi', action='store_true', help='use multiple query' )
parser.add_argument('--multiple_scale',default='1', type=str,help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')
parser.add_argument('--test_dir',default='all',type=str, help='./test_data')
parser.add_argument('--test_per_round', default=10, type=int, help='federated model test per round')


# arguments for optimization  cdw_default false, kd_default false
parser.add_argument('--cdw', default=False, action='store_true', help='use cosine distance weight for model aggregation, default false' )
# parser.add_argument('--cdw', action='store_true', help='use cosine distance weight for model aggregation, default false' )
parser.add_argument('--kd', default=False, type=bool, help='apply knowledge distillation, default false' )
# parser.add_argument('--kd', action='store_true', help='apply knowledge distillation, default false' )
parser.add_argument('--regularization', default=True, type=bool, help='use regularization during distillation, default false' )
parser.add_argument('--feat_weight', default=0.05, type=float, help='feat similarity param')
parser.add_argument('--prox_weight', default=0.1, type=float, help='proximal param')
parser.add_argument('--local_model_reg', default=True, type=bool, help='local model regularity')
parser.add_argument('--feat_reg', default=False, type=bool, help='local model regularity')
parser.add_argument('--test_dataset', default='MSMT17', type=str, help='test dataset')


# arguments for vit_pytorch
parser.add_argument('--img_size', default=[256, 128], type=list, help='img_size')
parser.add_argument('--stride_size', default=[16, 16], type=list, help='patch_embed stride size')
parser.add_argument('--drop_path_rate', default=0.1, type=float, help='attn drop path rate')
parser.add_argument('--trans_drop_rate', default=0.0, type=float, help='model drop rate')
parser.add_argument('--attn_drop_rate', default=0.0, type=float, help='model attn rate')
parser.add_argument('--tranformer_type', default='vit_base_patch16_224_TransReID', type=str, help='transformer type')
parser.add_argument('--neck_feat', default='before', type=str, help='use feature whether is BN before or not ')

# arguments for loss 
parser.add_argument('--metric_type', default='triplet', type=str, help='use triplet')
parser.add_argument('--margin', default=True, type=bool, help='use margin')
parser.add_argument('--dataloader_sampler', default='softmax_triplet', type=str, help='use softmax_triplet')
parser.add_argument('--ID_LOSS_WEIGHT', default=0.5, type=float, help='id loss weight')
parser.add_argument('--TRI_LOSS_WEIGHT', default=0.5, type=float, help='triplet loss weight')


args = parser.parse_args()

# server = Server()

root = '/data/plzhang/data/visual_sample/'
# gallery_path = '/data/plzhang/data/visual_sample/market_gallery_2'
# query_path = '/data/plzhang/data/visual_sample/market_query_2'

gallery_path = '/data/plzhang/data/visual_sample/msmt_gallery'
query_path = '/data/plzhang/data/visual_sample/msmt_query'
data = os.walk(root)

gallery_set = []
query_set = []
def get_path(dir_path):
    tmp_set = []
    img_paths = glob.glob(osp.join(dir_path, '*jpg'))
    # for img_path in os.listdir(gallery_path):
    pattern = re.compile(r'([-\d]+)_c(\d)')
    for img_path in img_paths:
        pid, _ = map(int, pattern.search(img_path).groups())
        tmp_set.append((img_path, pid))
    return tmp_set


# def get_path(dir_path):
#     tmp_set = []
#     img_paths = glob.glob(osp.join(dir_path, '*jpg'))
#     # for img_path in os.listdir(gallery_path):
#     # pattern = re.compile(r'([-\d]+)_c(\d)')
#     for img_path in img_paths:
#         pid = img_path.split('_')[0]
#         tmp_set.append((img_path, pid))
#     return tmp_set


gallery_set = get_path(gallery_path)
query_set = get_path(query_path)

gallery_set = sorted(gallery_set, key=lambda x: x[1])
query_set = sorted(query_set, key=lambda x: x[1])
all_path_set = query_set + gallery_set

all_set = []
for path_pid in all_path_set:
    all_set.append(path_pid[0])
# print(all_set)


def transform_tensor(dataset):
    init_tensor = torch.randn((len(dataset), 3, 256, 128))
    all_tensor = []
    transforms_test = transforms.Compose([transforms.Resize(size=(256, 128), interpolation=3),
                       transforms.ToTensor(),
                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    for path in dataset:
        data = Image.open(path)
        data = transforms_test(data)
        data = data.unsqueeze(0)
        all_tensor.append(data)
    all_tensor = torch.cat(all_tensor, dim=0)
    return all_tensor
    # print(all_tensor.shape)
all_tensor = transform_tensor(all_set)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

model = get_model(args, 751, drop_rate=args.drop_rate, stride=args.stride).to(device)

ms = args.multiple_scale

def fliplr(img):
    """flip horizontal
    """
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip



def load_previous_model(model, model_path):
    assert model_path is not None
    ckpt = torch.load(model_path, map_location='cuda:0')
    state_dict = dict()
    for k, v in ckpt['state_dict'].items():
        if 'classifier' not in k:
            state_dict[k] = v
    model.load_state_dict(state_dict, strict=False)
    print('model size: {:.5f}M'.format(sum(p.numel() for p in model.parameters()) / 1e6))
    return model



def extract_feature(model, input, ms, device, model_path):
    print('dataset.shape', input.shape)
    model.load_param(model_path)
    features = torch.FloatTensor()
    n, c, h, w = input.size()
    img = input
    model = model.eval()
    model = model.to(device)
    img = img.to(device)
    # ff = torch.FloatTensor(n, 768).zero_().to(device)
    # for i in range(2):
    #         if(i==1):
    #             img = fliplr(img)
    #         input_img = Variable(img.to(device))
    #         for scale in ms: 
    #             if scale != 1:
    #                 # bicubic is only  available in pytorch>= 1.1
    #                 input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bicubic', align_corners=False)
    #             outputs = model(input_img)
    #             ff += outputs
    #     # norm feature
    ff = model(img)
    fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
    ff = ff.div(fnorm.expand_as(ff))

    features = torch.cat((features,ff.data.cpu()), 0)
    return features

# model_path = './logs_DA/market_debug_nocdw_0.5tri_0.5crosentropy/ft_ResNet50/federated_model.pth'
# model_path = './logs_DA/market_0.5tri_0.5crossentropy_0.1prox/ft_ResNet50/federated_model.pth'
# model_path = './logs_debug/market_debug_0.05featsimi_0.05prox/ft_ResNet50/federated_model.pth'
model_path = './logs_cdw/market_cdw_0.05prox_0.05simi_0.5tir_0.5id/ft_ResNet50/federated_model.pth'
feature = extract_feature(model, all_tensor, ms, device, model_path)
print('feature', feature.shape)

n_iter = 320

tsne0 = TSNE(n_components=2, n_iter=n_iter).fit_transform(feature)
plt.style.use('seaborn-whitegrid')
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 15})
fig = plt.figure()
ax = fig.add_subplot()
# for i in range(len(tsne0)):
# type1 = ax.scatter(tsne0[0:4, 0], tsne0[0:4, 1], label='query', marker='1', c='#3B49927F',  s=50, alpha=.5)
ax.scatter(tsne0[0:4, 0], tsne0[0:4, 1], label='query', marker='1', c='#3B49927F',  s=70, alpha=.5)
ax.scatter(tsne0[4:8, 0], tsne0[4:8, 1], marker='1', c='#EE00007F', s=70, alpha=.5)
ax.scatter(tsne0[8:12, 0], tsne0[8:12, 1], marker='1', c='#008B457F', s=70, alpha=.5)
ax.scatter(tsne0[12:16, 0], tsne0[12:16, 1], marker='1', c='#6318797F', s=70, alpha=.5)
ax.scatter(tsne0[16:20, 0], tsne0[16:20, 1], marker='1', c='#0082807F', s=70, alpha=.5)
ax.scatter(tsne0[20:24, 0], tsne0[20:24, 1], marker='1', c='#BB00217F', s=70, alpha=.5)
ax.scatter(tsne0[24:28, 0], tsne0[24:28, 1], marker='1', c='#5F559B7F', s=70, alpha=.5)
ax.scatter(tsne0[28:32, 0], tsne0[28:32, 1], marker='1', c='#A200567F', s=70, alpha=.5)
ax.scatter(tsne0[32:36, 0], tsne0[32:36, 1], marker='1', c='#8081807F', s=70, alpha=.5)
ax.scatter(tsne0[36:40, 0], tsne0[36:40, 1], marker='1', c='#1B19197F', s=70, alpha=.5)


# type2 = ax.scatter(tsne0[40:48, 0], tsne0[40:48, 1], label ='gallery', marker='h', c='#3B49927F', s=50, alpha=.5 )
ax.scatter(tsne0[40:48, 0], tsne0[40:48, 1], label ='gallery', marker='h', c='#3B49927F', s=70, alpha=.5 )
ax.scatter(tsne0[48:56, 0], tsne0[48:56, 1], marker='h', c='#EE00007F', s=70, alpha=.5)
ax.scatter(tsne0[56:64, 0], tsne0[56:64, 1], marker='h', c='#008B457F', s=70, alpha=.5)
ax.scatter(tsne0[64:72, 0], tsne0[64:72, 1], marker='h', c='#6318797F', s=70, alpha=.5)
ax.scatter(tsne0[72:80, 0], tsne0[72:80, 1], marker='h', c='#0082807F', s=70, alpha=.5)
ax.scatter(tsne0[80:88, 0], tsne0[80:88, 1], marker='h', c='#BB00217F', s=70, alpha=.5)
ax.scatter(tsne0[88:96, 0], tsne0[88:96, 1], marker='h', c='#5F559B7F', s=70, alpha=.5)
ax.scatter(tsne0[96:104, 0], tsne0[96:104, 1], marker='h', c='#A200567F', s=70, alpha=.5)
ax.scatter(tsne0[104:112, 0], tsne0[104:112, 1], marker='h', c='#8081807F', s=70, alpha=.5)
ax.scatter(tsne0[112:120, 0], tsne0[112:120, 1], marker='h', c='#1B19197F', s=70, alpha=.5)
ax.set_xlim(-20, 25)
ax.set_ylim(-20, 25)
# ax.legend((type1, type2), ('query', 'gallery'), loc=3)
ax.legend(loc='upper right')
# ax.set_ylabel('tsne_y')
# ax.set_title('baseline+prox_market')

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.grid(b=None)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ax.set_title('Baseline_MSMT17')
# ax.set_title('baseline+prox+feat_reg_market')
# ax.set_title('Baseline+prox+feat_reg+feat_cdw_MSMT17')

# plt.savefig('experiments/visual/msmt_raw' + str(n_iter), dpi = 200)
# plt.savefig('experiments/visual/prox', dpi = 200)
# plt.savefig('experiments/visual/feat_reg_prox', dpi = 200)
plt.savefig('experiments/visual/msmt_final' + str(n_iter), dpi = 200)
plt.show()
plt.close()














# ax.scatter(tsne0[0:4, 0], tsne0[0:4, 1], marker='1', c='sienna',  s=20, alpha=.5)
# ax.scatter(tsne0[4:8, 0], tsne0[4:8, 1], marker='1', c='darkorange', s=20, alpha=.5)
# ax.scatter(tsne0[8:12, 0], tsne0[8:12, 1], marker='1', c='yellowgreen', s=20, alpha=.5)
# ax.scatter(tsne0[12:16, 0], tsne0[12:16, 1], marker='1', c='brown', s=24, alpha=.5)
# ax.scatter(tsne0[16:20, 0], tsne0[16:20, 1], marker='1', c='dimgrey', s=20, alpha=.5)
# ax.scatter(tsne0[20:24, 0], tsne0[20:24, 1], marker='1', c='darkgoldenrod', s=20, alpha=.5)
# ax.scatter(tsne0[24:28, 0], tsne0[24:28, 1], marker='1', c='darkolivegreen', s=20, alpha=.5)
# ax.scatter(tsne0[28:32, 0], tsne0[28:32, 1], marker='1', c='palegreen', s=20, alpha=.5)
# ax.scatter(tsne0[32:36, 0], tsne0[32:36, 1], marker='1', c='peru', s=20, alpha=.5)
# ax.scatter(tsne0[36:40, 0], tsne0[36:40, 1], marker='1', c='y', s=20, alpha=.5)


# ax.scatter(tsne0[40:48, 0], tsne0[40:48, 1], marker='h', c='sienna', s=20, alpha=.5)
# ax.scatter(tsne0[48:56, 0], tsne0[48:56, 1], marker='h', c='darkorange', s=20, alpha=.5)
# ax.scatter(tsne0[56:64, 0], tsne0[56:64, 1], marker='h', c='yellowgreen', s=20, alpha=.5)
# ax.scatter(tsne0[64:72, 0], tsne0[64:72, 1], marker='h', c='brown', s=20, alpha=.5)
# ax.scatter(tsne0[72:80, 0], tsne0[72:80, 1], marker='h', c='dimgrey', s=20, alpha=.5)
# ax.scatter(tsne0[80:88, 0], tsne0[80:88, 1], marker='h', c='darkgoldenrod', s=20, alpha=.5)
# ax.scatter(tsne0[88:96, 0], tsne0[88:96, 1], marker='h', c='darkolivegreen', s=20, alpha=.5)
# ax.scatter(tsne0[96:104, 0], tsne0[96:104, 1], marker='h', c='palegreen', s=20, alpha=.5)
# ax.scatter(tsne0[104:112, 0], tsne0[104:112, 1], marker='h', c='peru', s=20, alpha=.5)
# ax.scatter(tsne0[112:120, 0], tsne0[112:120, 1], marker='h', c='y', s=20, alpha=.5)