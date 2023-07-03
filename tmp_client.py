


import time
import torch
import logging
from utils import get_optimizer, get_model
from utils import make_loss
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.autograd import Variable
import copy
from optimization import Optimization
from cosine_lr import CosineLRScheduler


def create_scheduler(optimizer):
    num_epochs = 100
    # type 1
    # lr_min = 0.01 * cfg.SOLVER.BASE_LR
    # warmup_lr_init = 0.001 * cfg.SOLVER.BASE_LR
    # type 2
    lr_min = 0.002 * 0.008
    warmup_lr_init = 0.01 * 0.008
    # warmup_lr_init = 0.01 * cfg.SOLVER.BASE_LR
    # type 3
    # lr_min = 0.001 * cfg.SOLVER.BASE_LR
    # warmup_lr_init = 0.01 * cfg.SOLVER.BASE_LR

    warmup_t = 5
    # warmup_t = cfg.SOLVER.WARMUP_EPOCHS
    noise_range = None

    lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_epochs,
            lr_min=lr_min,
            t_mul= 1.,
            decay_rate=0.1,
            warmup_lr_init=warmup_lr_init,
            warmup_t=warmup_t,
            cycle_limit=1,
            t_in_epochs=True,
            noise_range_t=noise_range,
            noise_pct= 0.67,
            noise_std= 1.,
            noise_seed=42,
        )

    return lr_scheduler


class Client():
    def __init__(self, cid, data, device, args, project_dir, model_name, local_epoch, lr, batch_size, drop_rate, stride):
        self.cid = cid
        self.project_dir = project_dir
        self.model_name = model_name
        self.data = data
        self.device = device
        self.local_epoch = local_epoch
        self.lr = lr
        self.batch_size = batch_size
        
        self.dataset_sizes = self.data.train_dataset_sizes[cid]
        self.train_loader = self.data.train_loaders[cid]

        self.full_model = get_model(args, self.data.train_class_sizes[cid], drop_rate, stride)
        # self.classifier = self.full_model.classifier.classifier
        # self.full_model.classifier.classifier = nn.Sequential()
        self.classifier = self.full_model.classifier
        self.full_model.classifier = nn.Sequential()
        # self.full_model.classifier = nn.Linear(768, )
        self.model = self.full_model
        self.distance=0
        self.optimization = Optimization(self.train_loader, self.device)
        # print("class name size",class_names_size[cid])

        self.criterion = make_loss(args, self.dataset_sizes)
        self.loss = 0
        self.running_loss = 0
        self.running_corrects = 0

    def train(self, federated_model, use_cuda):
        self.y_err = []
        self.y_loss = []

        # self.model.load_state_dict(federated_model.state_dict())
        # self.model.classifier.classifier = self.classifier
        # print('self.model', self.model)
        # print(federated_model.state_dict().keys())
        self.model.load_state_dict(federated_model.state_dict())
        self.model.classifier = self.classifier
        self.old_classifier = copy.deepcopy(self.classifier)

        self.model = self.model.to(self.device)

        self.optimizer = get_optimizer(self.model, self.lr)
        # scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
        self.scheduler = create_scheduler(self.optimizer)
        # criterion = nn.CrossEntropyLoss()
        
        logger = logging.getLogger("local.train")
        # train_logger.info('start training')
        since = time.time()

        # print('Client', self.cid, 'start training')
        logger.info("Client: {}, start training".format(self.cid))
        for epoch in range(self.local_epoch):
            # print('Epoch {}/{}'.format(epoch, self.local_epoch - 1))
            # print('-' * 10)
            logger.info("Epoch {}/{}".format(epoch, self.local_epoch-1))
            logger.info('-' * 10)

            self.scheduler.step(epoch)
            # scheduler.step()
            self.model.train(True)
            running_loss = 0.0
            running_corrects = 0.0
            
            for data in self.train_loader:
                inputs, labels = data
                # img_per_pid = _ogranize_data(inputs, labels)
                # print('img_per_pid', img_per_pid)
                b, c, h, w = inputs.shape
                if b < self.batch_size:
                    continue
                if use_cuda:
                    # inputs = Variable(inputs.cuda().detach())
                    # labels = Variable(labels.cuda().detach())
                    inputs = Variable(inputs.to(self.device))
                    labels = Variable(labels.to(self.device))
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
                
                self.optimizer.zero_grad()

                score, outputs = self.model(inputs)
                # outputs = self.model(inputs)
        return score, outputs
                                





        #         _, preds = torch.max(outputs.data, 1)
        #         # loss = criterion(outputs, labels)
        #         loss = self.criterion(score, outputs, labels)
        #         loss.backward()

        #         optimizer.step()

        #         running_loss += loss.item() * b
        #         running_corrects += float(torch.sum(preds == labels.data))

        #     used_data_sizes = (self.dataset_sizes - self.dataset_sizes % self.batch_size)
        #     epoch_loss = running_loss / used_data_sizes
        #     epoch_acc = running_corrects / used_data_sizes

        #     # print('{} Loss: {:.4f} Acc: {:.4f}'.format(
        #     #     'train', epoch_loss, epoch_acc))
        #     logger.info("{} Loss: {:.4f} Acc: {:.4f}".format("train", epoch_loss, epoch_acc))

        #     self.y_loss.append(epoch_loss)
        #     self.y_err.append(1.0-epoch_acc)

        #     time_elapsed = time.time() - since
        #     # print('Client', self.cid, ' Training complete in {:.0f}m {:.0f}s'.format(
        #     #     time_elapsed // 60, time_elapsed % 60))
        #     logger.info('Client', self.cid, ' Training complete in {:.0f}m {:.0f}s'.format(
        #         time_elapsed // 60, time_elapsed % 60))

        # time_elapsed = time.time() - since
        # # print('Client', self.cid, 'Training complete in {:.0f}m {:.0f}s'.format(
        # #     time_elapsed // 60, time_elapsed % 60))
        # logger.info('Client', self.cid, 'Training complete in {:.0f}m {:.0f}s'.format(
        #     time_elapsed // 60, time_elapsed % 60))

        # # save_network(self.model, self.cid, 'last', self.project_dir, self.model_name, gpu_ids)
        
        # self.classifier = self.model.classifier
        # # self.classifier = self.model.classifier.classifier
        # self.distance = self.optimization.cdw_feature_distance(federated_model, self.old_classifier, self.model)
        # self.model.classifier = nn.Sequential()
        # # self.model.classifier.classifier = nn.Sequential()







    def generate_soft_label(self, x, regularization):
        return self.optimization.kd_generate_soft_label(self.model, x, regularization)

    def get_model(self):
        return self.model

    def get_data_sizes(self):
        return self.dataset_sizes

    def get_train_loss(self):
        return self.y_loss[-1]

    def get_cos_distance_weight(self):
        return self.distance