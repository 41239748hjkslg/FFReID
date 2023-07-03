from asyncio.log import logger
import os
import os.path as osp
import numpy as np
import math
import json
import matplotlib.pyplot as plt
from utils import get_model, extract_feature
import torch.nn as nn
import torch
import scipy.io
import copy
from data_utils import ImageDataset
import random
import torch.optim as optim
from torchvision import datasets
import logging
from util.logger import setup_logger
from utils import create_generative_model
from evaluate import pratice_evaluate

def add_model(dst_model, src_model, dst_no_data, src_no_data):
    if dst_model is None:
        result = copy.deepcopy(src_model)
        return result
    params1 = src_model.named_parameters()
    params2 = dst_model.named_parameters()
    dict_params2 = dict(params2)
    with torch.no_grad():
        for name1, param1 in params1:
            if name1 in dict_params2:
                dict_params2[name1].set_(param1.data*src_no_data + dict_params2[name1].data*dst_no_data)
    return dst_model

def scale_model(model, scale):
    params = model.named_parameters()
    dict_params = dict(params)
    with torch.no_grad():
        for name, param in dict_params.items():
            dict_params[name].set_(dict_params[name].data * scale)
    return model

def aggregate_models(models, weights):
    """aggregate models based on weights
    params:
        models: model updates from clients
        weights: weights for each model, e.g. by data sizes or cosine distance of features
    """
    if models == []:
        return None
    model = add_model(None, models[0], 0, weights[0])
    total_no_data = weights[0]
    for i in range(1, len(models)):
        model = add_model(model, models[i], total_no_data, weights[i])
        model = scale_model(model, 1.0 / (total_no_data+weights[i]))
        total_no_data = total_no_data + weights[i]
    return model


class Server():
    def __init__(self, clients, data, device, args, project_dir, model_name, num_of_clients, lr, drop_rate, stride, multiple_scale, logger):
        self.project_dir = project_dir
        self.data = data
        self.device = device
        self.args = args
        self.model_name = model_name
        self.clients = clients
        self.client_list = self.data.client_list
        self.num_of_clients = num_of_clients
        self.lr = lr
        self.multiple_scale = multiple_scale
        self.drop_rate = drop_rate
        self.stride = stride

        self.multiple_scale = []
        for s in multiple_scale.split(','):
            self.multiple_scale.append(math.sqrt(float(s)))

        print('device', device)
        # device = torch.device('cuda:1')
        # self.full_model = get_model(750, drop_rate, stride).to(0)
        self.full_model = get_model(args, 751, drop_rate, stride).to(device)
        # self.full_model = get_model(args, 750, drop_rate, stride).to(device)
        self.full_model.classifier = nn.Sequential()
        self.full_model.bottleneck = nn.Sequential()
        # self.full_model.classifier.classifier = nn.Sequential()
        self.federated_model=self.full_model
        self.federated_model.eval()
        self.train_loss = []


        # self.log_dir = osp.join(args.project_dir, args.model_name)
        # self.logger = setup_logger('federate_reid_train', self.log_dir, if_train=True)
        self.logger = logger
        self.args = args
        # self.tmp_num_of_clients = np.random.randint(low=0, high=self.num_of_clients)

        # self.current_client_list = random.sample(self.client_list, self.tmp_num_of_clients)




        # self.generative_model = create_generative_model(self.data, 'fedgen', 'cnn', 0)

        # self.generative_optimizer = torch.optim.Adam(
        #     params=self.generative_model.parameters(),
        #     lr=self.ensemble_lr, betas=(0.9, 0.999),
        #     eps=1e-08, weight_decay=self.weight_decay, amsgrad=False)
        # self.generative_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        #     optimizer=self.generative_optimizer, gamma=0.98)
        # self.optimizer = torch.optim.Adam(
        #     params=self.model.parameters(),
        #     lr=self.ensemble_lr, betas=(0.9, 0.999),
        #     eps=1e-08, weight_decay=0, amsgrad=False)
        # self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.98)




    def train(self, epoch, cdw, use_cuda, current_client_list):
        models = []
        loss = []
        cos_distance_weights = []
        data_sizes = []
        # current_client_list = random.sample(self.client_list[:2], self.num_of_clients)        
        # current_client_list = random.sample(self.client_list, self.num_of_clients)

        # tmp_num_of_clients = np.random.randint(low=0, high=self.num_of_clients)

        # current_client_list = random.sample(self.client_list, tmp_num_of_clients)
        # m = max(9, 1)
        # current_client_list = np.random.choice(self.client_list, 4, replace=False)
        # current_client_list = np.random.choice(self.client_list, )

        # logger = logging.getLogger("server.train")
        # log_dir = osp.join(self.project_dir, self.model_name)
        # log_dir = osp.join(self.project_dir, 'logs/federat', self.model_name)
        # logger = setup_logger('federate_reid_train', log_dir, if_train=True)
        # logger = logging.getLogger('==================server.train===================')
        
        
        # for i in self.current_client_list:
        for i in current_client_list:
            self.clients[i].train(self.federated_model, use_cuda)
            
            cos_distance_weights.append(self.clients[i].get_cos_distance_weight())
            
            loss.append(self.clients[i].get_train_loss())
            models.append(self.clients[i].get_model())
            data_sizes.append(self.clients[i].get_data_sizes())

        if epoch==0:
            self.L0 = torch.Tensor(loss) 

        avg_loss = sum(loss) / self.num_of_clients

        # print("==============================")
        # print("number of clients used:", len(models))
        # print('Train Epoch: {}, AVG Train Loss among clients of lost epoch: {:.6f}'.format(epoch, avg_loss))
        # print()
        self.logger.info("==============================")
        self.logger.info("number of clients used:{}".format(len(models)))
        self.logger.info('Train Epoch: {}, AVG Train Loss among clients of lost epoch: {:.6f}'.format(epoch, avg_loss))
        self.logger.info('')


        self.train_loss.append(avg_loss)
        
        weights = data_sizes
        self.logger.info("data_size weights: %f, %f, %f, %f", weights[0], weights[1], weights[2], weights[3])
        print('cdw', cdw)
        if cdw:
            print("cos distance weights:", cos_distance_weights)
            self.logger.info("cos distance weights: %f, %f, %f, %f ", cos_distance_weights[0], cos_distance_weights[1], cos_distance_weights[2], cos_distance_weights[3])
            weights = cos_distance_weights
            # weights = [x/sum(weights) for x in weights]
            # print('weight', weights)

        
        # self.train_generator(self.args.batch_size,epoches=1, latent_layer_idx=1, verbose=True)

        self.federated_model = aggregate_models(models, weights)
        # for name, param in self.federated_model.named_parameters():
        #     print('name', name, '   ', param.shape)


    def draw_curve(self):
        plt.figure()
        x_epoch = list(range(len(self.train_loss)))
        plt.plot(x_epoch, self.train_loss, 'bo-', label='train')
        plt.legend()
        dir_name = os.path.join(self.project_dir, 'model', self.model_name)
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)
        plt.savefig(os.path.join(dir_name, 'train.png'))
        plt.close('all')
        
    def test(self, use_cuda):
        # print("="*10)
        # print("Start Tesing!")
        # print("="*10)
        # print('We use the scale: %s'%self.multiple_scale)
        # logger = logging.getLogger("="*10)
        self.logger.info("Start Testing!")
        self.logger.info("="*10)
        self.logger.info('We use the scale: %s'%self.multiple_scale)

        for dataset in self.data.datasets:
            # if dataset == self.args.test_dataset:
                # dataset = self.data.datasets[-1]
            print('dataset', dataset)
            self.federated_model = self.federated_model.eval()
            if use_cuda:
                self.federated_model = self.federated_model.to(self.device)
                # self.federated_model = self.federated_model.cuda()
            
            with torch.no_grad():
                gallery_feature = extract_feature(self.federated_model, self.data.test_loaders[dataset]['gallery'], self.multiple_scale, self.device)
                query_feature = extract_feature(self.federated_model, self.data.test_loaders[dataset]['query'], self.multiple_scale, self.device)

            result = {
                'gallery_f': gallery_feature.numpy(),
                'gallery_label': self.data.gallery_meta[dataset]['labels'],
                'gallery_cam': self.data.gallery_meta[dataset]['cameras'],
                'query_f': query_feature.numpy(),
                'query_label': self.data.query_meta[dataset]['labels'],
                'query_cam': self.data.query_meta[dataset]['cameras']}

            scipy.io.savemat(os.path.join(self.project_dir,
                        self.model_name,
                        'pytorch_result.mat'),
                        result)
                        
            # print(self.model_name)
            print(dataset)

            # os.system('python evaluate.py --result_dir {} --dataset {}'.format(os.path.join(self.project_dir, self.model_name), dataset))
            res_dir = os.path.join(self.project_dir, self.model_name)
            pratice_evaluate(res_dir, dataset, self.logger)
                # break
                # os.system('python evaluate.py --result_dir {} --dataset {}'.format(os.path.join(self.project_dir, 'model', self.model_name), dataset))
            # torch.cuda.emptssy_cache()




        # for dataset in self.data.datasets:
        #     self.federated_model = self.federated_model.eval()
        #     if use_cuda:
        #         self.federated_model = self.federated_model.to(1)
        #         # self.federated_model = self.federated_model.cuda()
            
        #     with torch.no_grad():
        #         gallery_feature = extract_feature(self.federated_model, self.data.test_loaders[dataset]['gallery'], self.multiple_scale)
        #         query_feature = extract_feature(self.federated_model, self.data.test_loaders[dataset]['query'], self.multiple_scale)

        #     result = {
        #         'gallery_f': gallery_feature.numpy(),
        #         'gallery_label': self.data.gallery_meta[dataset]['labels'],
        #         'gallery_cam': self.data.gallery_meta[dataset]['cameras'],
        #         'query_f': query_feature.numpy(),
        #         'query_label': self.data.query_meta[dataset]['labels'],
        #         'query_cam': self.data.query_meta[dataset]['cameras']}

        #     scipy.io.savemat(os.path.join(self.project_dir,
        #                 'model',
        #                 self.model_name,
        #                 'pytorch_result.mat'),
        #                 result)
                        
        #     print(self.model_name)
        #     print(dataset)

        #     os.system('python evaluate.py --result_dir {} --dataset {}'.format(os.path.join(self.project_dir, 'model', self.model_name), dataset))




    # def knowledge_distillation(self, regularization, kd_loader):
    def knowledge_distillation(self, regularization,current_client_list):
        MSEloss = nn.MSELoss().to(self.device)
        optimizer = optim.SGD(self.federated_model.parameters(), lr=self.lr*0.01, weight_decay=5e-4, momentum=0.9, nesterov=True)
        self.federated_model.train()
        
        print('len(self.data.kd_loader)', len(self.data.kd_loader))

        for _, data in enumerate(self.data.kd_loader): 
        # for _, (x, target) in enumerate(self.data.kd_loader): 
            x, target = data
            x, target = x.to(self.device), target.to(self.device)
            # target=target.long()

            optimizer.zero_grad()
            soft_target = torch.Tensor([[0]*768]*len(x)).to(self.device)
            # soft_target = torch.Tensor([[0]*512]*len(x)).to(self.device)
            proximal_term = 0
            for i in current_client_list:
            # for i in self.current_client_list:
            # for i in self.client_list:
                i_label = (self.clients[i].generate_soft_label(x, regularization))

                # for w, w_t in zip(self.client_list[i].model.parameters(), self.federated_model.parameters()):
                #     proximal_term += (w-w_t).norm(2) 
                soft_target += i_label
            # print('soft_target', soft_target.shape)
            soft_target /= len(self.client_list)

            fed_score, output = self.federated_model(x)
            # fed_score, output = self.federated_model(x)
            # print('output.shape', output.shape)

            # for w, w_t in zip(self.model.parameters(), self.federated_model.parameters()):
            #     proximal_term += (w-w_t).norm(2)  + (3/2) * proximal_term
            loss = MSEloss(output, soft_target)
            loss.backward()
            optimizer.step()
            # print("train_loss_fine_tuning", loss.data)
    
    def test_null(self):
        print('===== test_null====')
