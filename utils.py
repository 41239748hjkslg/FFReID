import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import torch.nn.functional as F
from model import ft_net
from model import build_transformer
from triplet_loss import TripletLoss

from torch.autograd import Variable

from torchvision import datasets, transforms
from vit_pytorch import vit_base_patch16_224_TransReID, vit_small_patch16_224_TransReID, deit_small_patch16_224_TransReID
from generator import Generator


__factory_T_type = {
    'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'deit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'vit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
    'deit_small_patch16_224_TransReID': deit_small_patch16_224_TransReID
}

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed**2)
    torch.manual_seed(seed**3)
    torch.cuda.manual_seed(seed**4)

def get_optimizer(model, lr):
    params = []
    base_lr = 0.008
    base_lr_factor = 2
    weight_decay_bias = 1e-4
    large_fc_lr = False
    ignored_params = list(map(id, model.classifier.parameters() ))
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    # # for key, value in model.named_parameters():
        
    # #     if not value.requires_grad:
    # #         continue
    # #     # print('key', key)
    # #     lr = base_lr
    # #     weight_decay = 1e-4
    # #     if "bias" in key:
    # #         lr = base_lr * base_lr_factor
    # #         weight_decay = weight_decay_bias
    # #     if large_fc_lr:
    # #         if "classifier" in key or "arcface" in key:
    # #             lr = base_lr * 2
    # #             print('Using two times learning rate for fc ')

    # #     params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    
    # # optimizer = getattr(torch.optim, 'SGD')(params, momentum=0.9)


    # optimizer_ft = optim.SGD([
    #         {'params': base_params, 'lr': lr},
    #         # {'params': base_params, 'lr': lr, "weight_decay": weight_decay_bias},
    #         {'params': model.classifier.parameters(), 'lr': lr}
    #     ], weight_decay=1e-4, momentum=0.9, nesterov=True)
        
    # return optimizer_ft


    optimizer_ft = optim.SGD([
            {'params': base_params, 'lr': 0.1*lr},
            {'params': model.classifier.parameters(), 'lr': lr}
        ], weight_decay=5e-4, momentum=0.9, nesterov=True)

    return optimizer_ft



def make_optimizer(cfg, model):
    params = []
    base_lr = 0.008
    base_lr_factor = 2
    weight_decay_bias = 1e-4
    large_fc_lr = False
    for key, value in model.named_parameters():
        
        if not value.requires_grad:
            continue
        # print('key', key)
        lr = base_lr
        weight_decay = 1e-4
        if "bias" in key:
            lr = base_lr * base_lr_factor
            weight_decay = weight_decay_bias
        if large_fc_lr:
            if "classifier" in key or "arcface" in key:
                lr = base_lr * 2
                print('Using two times learning rate for fc ')

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    
    optimizer = getattr(torch.optim, 'SGD')(params, momentum=0.9)

    # if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        
    #     optimizer = getattr(torch.optim, 'SGD')(params, momentum=0.9)
    #     # optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
    # elif cfg.SOLVER.OPTIMIZER_NAME == 'AdamW':
    #     optimizer = torch.optim.AdamW(params, lr=base_lr, weight_decay=weight_decay)
    # else:
    #     optimizer = getattr(torch.optim, 'SGD')(params)

    return optimizer






def save_network(network, cid, epoch_label, project_dir, name, gpu_ids):
    save_filename = 'net_%s.pth'% epoch_label
    dir_name = os.path.join(project_dir, 'model', name, cid)
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    save_path = os.path.join(project_dir, 'model', name, cid, save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available():
        network.cuda(gpu_ids[0])

def get_model(args, class_sizes, drop_rate, stride):
# def get_model(class_sizes, drop_rate, stride):
    model = build_transformer(args, class_sizes, __factory_T_type)
    # model = ft_net(args, class_sizes, drop_rate, stride)
    # model = ft_net(class_sizes, drop_rate, stride)
    return model

# functions for testing federated model
def fliplr(img):
    """flip horizontal
    """
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_feature(model, dataloaders, ms, device):
    # device = torch.device("cuda:0")

    features = torch.FloatTensor()
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        ff = torch.FloatTensor(n, 768).zero_().to(device)
        # ff = torch.FloatTensor(n, 512).zero_().cuda()

        for i in range(2):
            if(i==1):
                img = fliplr(img)
            input_img = Variable(img.to(device))
            # input_img = Variable(img.cuda())
            for scale in ms:
                if scale != 1:
                    # bicubic is only  available in pytorch>= 1.1
                    input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bicubic', align_corners=False)
                outputs = model(input_img)
                ff += outputs
        # norm feature
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))

        features = torch.cat((features,ff.data.cpu()), 0)
    return features


def make_loss(args, num_class):
    if 'triplet' in args.metric_type:
        if args.margin:
            triplet = TripletLoss()
            print("using soft triplet loss for training")
        else:
            # triplet = TripletLoss(cfg.SOLVER.MARGIN_1, cfg.SOLVER.MARGIN_2)  # triplet loss
            triplet = TripletLoss(0.3)  # triplet loss
    if args.dataloader_sampler == 'softmax':
        def loss_fn(score, feat, target):
            return F.cross_entropy(score, target)
    elif args.dataloader_sampler == 'softmax_triplet':
        def loss_fn(score, feat, target):
            # print('score', score.shape)
            # print('feat', feat.shape)
            # print('target', target.shape)
            ID_LOSS = F.cross_entropy(score, target)
            TRI_LOSS = triplet(feat, target)[0]
            # return ID_LOSS*0.5 + TRI_LOSS
            # return ID_LOSS*0.5 + TRI_LOSS*0.5
            return ID_LOSS * args.ID_LOSS_WEIGHT + TRI_LOSS * args.TRI_LOSS_WEIGHT
    return loss_fn
            


def create_generative_model(dataset, algorithm='', model='cnn', embedding=False):
    return Generator(dataset, model=model, embedding=embedding, latent_layer_idx=-1)