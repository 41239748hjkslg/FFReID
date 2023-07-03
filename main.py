# -*- coding: utf-8 -*-
from __future__ import print_function, division
import argparse
from ast import arg
from tkinter.tix import Tree
import torch
import time
import os
import os.path as osp
import yaml
import random
import numpy as np
import scipy.io
import pathlib
import sys
import json
import copy
import multiprocessing as mp
import torch.nn.functional as F
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from PIL import Image
from client import Client
from server import Server
from utils import set_random_seed
from data_utils import Data
from util.logger import setup_logger

mp.set_start_method('spawn', force=True)
sys.setrecursionlimit(10000)
version =  torch.__version__

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--model_name',default='ft_ResNet50', type=str, help='output model name')
parser.add_argument('--project_dir',default='.', type=str, help='project path')
parser.add_argument('--data_dir',default='/data/plzhang/data',type=str, help='training dir path')
# parser.add_argument('--data_dir',default='data',type=str, help='training dir path')
# parser.add_argument('--datasets',default='market',type=str, help='datasets used')cuhk01,viper,prid,3dpes,duke,MSMT17,cuhk03-np,ilids,market
# parser.add_argument('--datasets',default='cuhk01,viper,prid,3dpes,ilids,market,duke,cuhk03-np,MSMT17',type=str, help='datasets used')
parser.add_argument('--datasets',default='cuhk01,viper,prid,3dpes,duke,MSMT17,cuhk03-np,ilids,market',type=str, help='datasets used')
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
parser.add_argument('--prox_weight', default=0.05, type=float, help='proximal param')
parser.add_argument('--local_model_reg', default=True, type=bool, help='local model regularity')
parser.add_argument('--feat_reg', default=False, type=bool, help='local model regularity')
parser.add_argument('--test_dataset', default='market', type=str, help='test dataset')


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

def train():
    args = parser.parse_args()
    print(args)
    log_dir = osp.join(args.project_dir, args.model_name)
    logger = setup_logger('federate_reid_train', log_dir, if_train=True)
    logger.info(args)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    set_random_seed(1)

    data = Data(args.datasets, args.data_dir, args.batch_size, args.erasing_p, args.color_jitter, args.train_all, args.kd_train)
    # data = Data(args.datasets, args.data_dir, args.batch_size, args.erasing_p, args.color_jitter, args.train_all)
    data.preprocess()
    
    clients = {}
    # client_list = data.client_list
    client_list = data.client_list[:8]
    # print('client_list', client_list)
    # for cid in data.client_list:
    for cid in client_list:
        clients[cid] = Client(
            cid, 
            data, 
            device,
            args, 
            args.project_dir, 
            args.model_name, 
            args.local_epoch, 
            args.lr, 
            args.batch_size, 
            args.drop_rate, 
            args.stride,
            logger) 

    server = Server(
        clients, 
        data, 
        device,
        args, 
        args.project_dir, 
        args.model_name, 
        args.num_of_clients, 
        args.lr, 
        args.drop_rate, 
        args.stride, 
        args.multiple_scale,
        logger)

    dir_name = os.path.join(args.project_dir, args.model_name)
    # dir_name = os.path.join(args.project_dir, 'model', args.model_name)
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
        # os.mkdir(dir_name)

    # print("=====training start!========")
    
        # log_dir = osp.join(self.project_dir, 'logs/federat', self.model_name)

    rounds = 800

    # for i in range(rounds):
    #     # print('='*10)
    #     # print("Round Number {}".format(i))
    #     # print('='*10)
    #     logger.info('=' * 10)
    #     logger.info("Round Number {}".format(i))
    #     logger.info('=' * 10)
    #     tmp_client_list = []
    #     # all_client = args.datasets.split(',')
    #     all_client = client_list
    #     # print('all_client', all_client)
    #     logger.info("all client: {}".format(client_list))
    #     for client_index in all_client:
    #         tmp_client_list.append(client_index)
    #     current_client_list = np.random.choice(tmp_client_list, 4, replace=False)

    #     server.train(i, args.cdw, use_cuda, current_client_list)
    #     save_path = os.path.join(dir_name, 'federated_model.pth')
    #     torch.save(server.federated_model.cpu().state_dict(), save_path)
        # if (i+1)%10 == 0:
        # if (i+1)%10 == 0:
    server.test(use_cuda)
    # if args.kd:
    #     # print('=======')
    #     # server.knowledge_distillation(args.regularization, data.kd_loader)
    #     server.knowledge_distillation(args.regularization, current_client_list)
    #     server.test_null()
    #     if i % 30 == 0:
    #         server.test(use_cuda)
    # server.draw_curve()
    # save_path = os.path.join(dir_name, 'federated_model.pth')







if __name__ == '__main__':
    train()




