from audioop import bias
import torch

import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
# import pretrainedmodels
from resnet import ResNet, Bottleneck
from vit_pytorch import vit_base_patch16_224_TransReID, vit_small_patch16_224_TransReID, deit_small_patch16_224_TransReID

######################################################################
# def weights_init_kaiming(m):
#     classname = m.__class__.__name__
#     # print(classname)
#     if classname.find('Conv') != -1:
#         init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
#     elif classname.find('Linear') != -1:
#         init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
#         init.constant_(m.bias.data, 0.0)
#     elif classname.find('BatchNorm1d') != -1:
#         init.normal_(m.weight.data, 1.0, 0.02)
#         init.constant_(m.bias.data, 0.0)

# def weights_init_classifier(m):
#     classname = m.__class__.__name__
#     if classname.find('Linear') != -1:
#         init.normal_(m.weight.data, std=0.001)
#         init.constant_(m.bias.data, 0.0)

# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
# class ClassBlock(nn.Module):
#     def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True, return_f = False):
#         super(ClassBlock, self).__init__()
#         self.return_f = return_f
#         add_block = []
#         if linear:
#             add_block += [nn.Linear(input_dim, num_bottleneck)]
#         else:
#             num_bottleneck = input_dim
#         if bnorm:
#             add_block += [nn.BatchNorm1d(num_bottleneck)]
#         if relu:
#             add_block += [nn.LeakyReLU(0.1)]
#         if droprate>0:
#             add_block += [nn.Dropout(p=droprate)]
#         add_block = nn.Sequential(*add_block)
#         add_block.apply(weights_init_kaiming)

#         classifier = []
#         classifier += [nn.Linear(num_bottleneck, class_num)]
#         classifier = nn.Sequential(*classifier)
#         classifier.apply(weights_init_classifier)

#         self.add_block = add_block
#         self.classifier = classifier
#     def forward(self, x):
#         x = self.add_block(x)
#         if self.return_f:
#             f = x
#             x = self.classifier(x)
#             return x,f
#         else:
#             x = self.classifier(x)
#             return x

# Define the ResNet50-based Model
class ft_net(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2):
        super(ft_net, self).__init__()
       
        # model_ft = models.resnet50(pretrained=True)
        model_ft = ResNet(last_stride=1, block=Bottleneck, layers=[3, 4, 6, 3])
        model_path = ''
        model_ft.load_param('./.torch/models/resnet50-19c8e357.pth')
        # model_ft=torch.load('saved_res50.pkl')
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        self.classifier = ClassBlock(2048, class_num, droprate)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        # x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x








def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=768, linear=True, return_f = True):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []

        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        # if linear:
        #     add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim

        # if relu:
        #     add_block += [nn.LeakyReLU(0.1)]
        if droprate>0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)
        # add_block.bias.requires_grad_(False)

        classifier = []
        classifier += [nn.Linear(input_dim, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x, label):
        print('x.shape', x.shape)
        x = self.add_block(x)
        print('===x.shape', x.shape)
        if self.return_f:
            f = x
            print('====')
            x = self.classifier(x, label)
            print('*****')
            return x,f
        else:
            x = self.classifier(x)
            return x

class build_transformer(nn.Module):
    def __init__(self, cfg, num_classes, factory):
        super(build_transformer, self).__init__()
        # model_path = './.torch/models/jx_vit_base_p16_224-80ecf9dd.pth'
        model_path = './logs_DA/market_debug_nocdw_0.5tri_0.5crosentropy/ft_ResNet50/federated_model.pth'
        pretrain_choice = 'imagenet'
        # self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = 'bnneck'
        self.neck_feat = cfg.neck_feat
        # self.neck_feat = 'before'
        self.in_planes = 768

        # last_stride = cfg.MODEL.LAST_STRIDE
        # model_path = cfg.MODEL.PRETRAIN_PATH
        # model_name = cfg.MODEL.NAME
        # pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        # self.cos_layer = cfg.MODEL.COS_LAYER
        # self.neck = cfg.MODEL.NECK
        # self.neck_feat = cfg.TEST.NECK_FEAT
        # self.in_planes = 768

        # print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        # if cfg.MODEL.SIE_CAMERA:
        #     camera_num = camera_num
        # else:
        #     camera_num = 0
        # if cfg.MODEL.SIE_VIEW:
        #     view_num = view_num
        # else:
        #     view_num = 0

        self.base = factory[cfg.tranformer_type](img_size=cfg.img_size,
                                                        stride_size=cfg.stride_size, drop_path_rate=cfg.drop_path_rate,
                                                        drop_rate= cfg.trans_drop_rate,
                                                        attn_drop_rate=cfg.attn_drop_rate)
        if cfg.tranformer_type == 'deit_small_patch16_224_TransReID':
            self.in_planes = 384
        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.num_classes = num_classes
        # else:
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        # classifier = []
        # classifier += [nn.Linear(self.in_planes, self.num_classes, bias=False)]
        # classifier = nn.Sequential(*classifier)
        # classifier.apply(weights_init_classifier)

        # self.classifier = classifier
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        # self.classifier = ClassBlock(self.in_planes, self.num_classes, droprate=0.0)
        # print('classblock', self.classifier)
    def forward(self, x, label=None, cam_label= None, view_label=None):
        # print('x', x.shape)
        global_feat = self.base(x, cam_label=cam_label, view_label=view_label)
        # print('global_feat.shape', global_feat.shape)
        # feat = self.classifier(x)
        # return feat
        feat = self.bottleneck(global_feat)

        # if self.training:
        #     if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
        #         cls_score = self.classifier(feat, label)
        #     else:
        if self.training:
            # print('***********')
            cls_score = self.classifier(feat)
            # print('cls_score', cls_score.shape)
            # print('global_feat', global_feat.shape)
            return cls_score, global_feat  # global feature for triplet loss
        # else:
        # print('+++++++++++++')
        if self.neck_feat == 'after':
            # print("Test with feature after BN")
            return feat
        else:
            # print("Test with feature before BN")
            return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            # print('i', i)
            if 'classifier' in i:
                print('===')
                continue
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


__factory_T_type = {
    'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'deit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'vit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
    'deit_small_patch16_224_TransReID': deit_small_patch16_224_TransReID
}


'''
# debug model structure
# Run this code with:
python model.py
'''
if __name__ == '__main__':
# Here I left a simple forward function.
# Test the model, before you train it. 
    net = ft_net(751, stride=1)
    model = build_transformer(num_class, camera_num, cfg, __factory_T_type)
    print('===========building transformer===========')
    # model = build_transformer(num_class, camera_num, view_num, cfg, __factory_T_type)
    # print('===========building transformer===========')
    net.classifier = nn.Sequential()
    print(net)
    input = Variable(torch.FloatTensor(8, 3, 256, 128))
    output = net(input)
    print('net output size:')
    print(output.shape)
