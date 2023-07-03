from sklearn.utils import shuffle
from torch.utils.data import Dataset
from PIL import Image
from torchvision import datasets, transforms
import os
import json
import torch
from random_erasing import RandomErasing
from sampler import RandomIdentitySampler

class ImageDataset(Dataset):
    def __init__(self, imgs,  transform = None):
        self.imgs = imgs
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        data,label = self.imgs[index]
        return self.transform(Image.open(data)), label


class Data():
    def __init__(self, datasets, data_dir, batch_size, erasing_p, color_jitter, train_all, kd_train):
    # def __init__(self, datasets, data_dir, batch_size, erasing_p, color_jitter, train_all):
        self.datasets = datasets.split(',')
        self.batch_size = batch_size
        self.erasing_p = erasing_p
        self.color_jitter = color_jitter
        self.data_dir = data_dir
        self.train_all = '_all' if train_all else ''
        self.kd_train = '_all' if kd_train else ''
        
    def transform(self):
        transform_train = [
                transforms.Resize((256,128), interpolation=3),
                transforms.Pad(10),
                transforms.RandomCrop((256,128)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]

        transform_val = [
                transforms.Resize(size=(256,128),interpolation=3),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]

        if self.erasing_p > 0:
            transform_train = transform_train + [RandomErasing(probability=self.erasing_p, mode='pixel', max_count=1, device='cpu')]
            # transform_train = transform_train + [RandomErasing(probability=self.erasing_p, mean=[0.0, 0.0, 0.0])]

        if self.color_jitter:
            transform_train = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)] + transform_train

        self.data_transforms = {
            'train': transforms.Compose(transform_train),
            'val': transforms.Compose(transform_val),
        }        

    def preprocess_kd_data(self, dataset):
        loader, image_dataset = self.kd_train_dataset(dataset)
        # loader, image_dataset = self.preprocess_one_train_dataset(dataset)
        self.kd_loader = loader


    def kd_train_dataset(self, dataset):
        data_path = os.path.join(self.data_dir, dataset, 'pytorch')
        data_path = os.path.join(data_path, 'train' + self.kd_train)
        print('data_path', data_path)
        image_dataset = datasets.ImageFolder(data_path)
        # print('image_dataset', image_dataset.imgs)
        img_per_batch = 64
        num_instance = 4

        loader = torch.utils.data.DataLoader(
            ImageDataset(image_dataset.imgs, self.data_transforms['train']), 
            batch_size=self.batch_size,
            shuffle=True,
            # sampler=RandomIdentitySampler(image_dataset.imgs, img_per_batch, num_instance),
            # shuffle=True, 
            # num_workers=0, 
            num_workers=2, 
            pin_memory=True,
            # pin_memory=False,
            collate_fn = train_collate_fn)

        return loader, image_dataset


    def preprocess_one_train_dataset(self, dataset):
        """preprocess a training dataset, construct a data loader.
        """
        data_path = os.path.join(self.data_dir, dataset, 'pytorch')
        data_path = os.path.join(data_path, 'train' + self.train_all)
        image_dataset = datasets.ImageFolder(data_path)
        img_per_batch = 64
        num_instance = 4

        loader = torch.utils.data.DataLoader(
            ImageDataset(image_dataset.imgs, self.data_transforms['train']), 
            batch_size=self.batch_size,
            sampler=RandomIdentitySampler(image_dataset.imgs, img_per_batch, num_instance),
            # shuffle=True, 
            # num_workers=0, 
            num_workers=2, 
            pin_memory=True,
            # pin_memory=False,
            collate_fn = train_collate_fn)

        return loader, image_dataset

    def preprocess_train(self):
        """preprocess training data, constructing train loaders
        """
        self.train_loaders = {}
        self.train_dataset_sizes = {}
        self.train_class_sizes = {}
        self.client_list = []
        
        # for dataset in self.datasets[:3]:
        for dataset in self.datasets:
            self.client_list.append(dataset)
          
            loader, image_dataset = self.preprocess_one_train_dataset(dataset)

            self.train_dataset_sizes[dataset] = len(image_dataset)
            self.train_class_sizes[dataset] = len(image_dataset.classes)
            self.train_loaders[dataset] = loader
            
        print('Train dataset sizes:', self.train_dataset_sizes)
        print('Train class sizes:', self.train_class_sizes)
        
    def preprocess_test(self):
        """preprocess testing data, constructing test loaders
        """
        self.test_loaders = {}
        self.gallery_meta = {}
        self.query_meta = {}

        for test_dir in self.datasets:
        # test_dir = self.datasets[-1]
            dataset_name = test_dir
            print('test_dir', test_dir)
            test_dir = '/data/plzhang/data/'+test_dir+'/pytorch'
            # test_dir = 'data/'+test_dir+'/pytorch'

            dataset = test_dir.split('/')[1]
            gallery_dataset = datasets.ImageFolder(os.path.join(test_dir, 'gallery'))
            query_dataset = datasets.ImageFolder(os.path.join(test_dir, 'query'))

            gallery_dataset = ImageDataset(gallery_dataset.imgs, self.data_transforms['val'])
            query_dataset = ImageDataset(query_dataset.imgs, self.data_transforms['val'])
            # print('dataset', dataset)
            # print('test_dir', test_dir)
            # self.test_loaders[dataset] = {key: torch.utils.data.DataLoader(
            self.test_loaders[dataset_name] = {key: torch.utils.data.DataLoader(
                                                dataset, 
                                                batch_size=self.batch_size,
                                                shuffle=False, 
                                                # num_workers=0, 
                                                num_workers=2, 
                                                pin_memory=True) for key, dataset in {'gallery': gallery_dataset, 'query': query_dataset}.items()}
        

            gallery_cameras, gallery_labels = get_camera_ids(gallery_dataset.imgs)
            self.gallery_meta[dataset_name] = {
            # self.gallery_meta[dataset] = {
                'sizes':  len(gallery_dataset),
                'cameras': gallery_cameras,
                'labels': gallery_labels
            }

            query_cameras, query_labels = get_camera_ids(query_dataset.imgs)
            self.query_meta[dataset_name] = {
            # self.query_meta[dataset] = {
                'sizes':  len(query_dataset),
                'cameras': query_cameras,
                'labels': query_labels
            }

        print('Query Sizes:', self.query_meta[dataset_name]['sizes'])
        print('Gallery Sizes:', self.gallery_meta[dataset_name]['sizes'])
        # print('Query Sizes:', self.query_meta[dataset]['sizes'])
        # print('Gallery Sizes:', self.gallery_meta[dataset]['sizes'])

    def preprocess(self):
        self.transform()
        self.preprocess_train()
        self.preprocess_test()
        self.preprocess_kd_data('cuhk02')

def get_camera_ids(img_paths):
    """get camera id and labels by image path
    """
    camera_ids = []
    labels = []
    for path, v in img_paths:
        filename = os.path.basename(path)
        # print('filename', filename)
        
        # if filename[:3]!='cam':
        if 'c' not in filename:
            label = filename[0:4]
            camera = filename.split('_')[2]
        elif filename[:3]!='cam':
            # print('filename', filename[:3])
            label = filename[0:4]
            camera = filename.split('c')[1]
            camera = camera.split('s')[0]
        
        else:
            label = filename.split('_')[2]
            camera = filename.split('_')[1]
        
        if label[0:2]=='-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_ids.append(int(camera[0]))
    return camera_ids, labels

def train_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """

    imgs, pids = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    # viewids = torch.tensor(viewids, dtype=torch.int64)
    # camids = torch.tensor(camids, dtype=torch.int64)
    # return torch.stack(imgs, dim=0), pids, camids, viewids,
    return torch.stack(imgs, dim=0), pids













# from random import shuffle
# from torch.utils.data import Dataset
# from PIL import Image
# from torchvision import datasets, transforms
# import os
# import json
# import torch
# import sys
# sys.path.append('./datasets')
# sys.path.append('./datasets/market1501')
# from random_erasing import RandomErasing

# from datasets.market1501 import Market1501
# from datasets.dukemtmcreid import DukeMTMCreID
# from datasets.msmt17 import MSMT17


# factory = {
#     'market': Market1501,
#     'duke': DukeMTMCreID,
#     'MSMT17_V1': MSMT17,
#     # 'cuhk03-np':,
#     # 'occ_duke': OCC_DukeMTMCreID,
#     # 'veri': VeRi,
#     # 'VehicleID': VehicleID,
# }


# class ImageDataset(Dataset):
#     def __init__(self, imgs,  transform = None):
#         self.imgs = imgs
#         self.transform = transform

#     def __len__(self):
#         return len(self.imgs)

#     def __getitem__(self, index):
#         data,label, camid, trackid = self.imgs[index]
#         return self.transform(Image.open(data)), label, camid


# class Data():
#     def __init__(self, datasets, data_dir, batch_size, erasing_p, color_jitter, train_all):
#         self.datasets = datasets.split(',')
#         self.batch_size = batch_size
#         self.erasing_p = erasing_p
#         self.color_jitter = color_jitter
#         self.data_dir = data_dir
#         self.train_all = '_all' if train_all else ''



#     def transform(self):
#         transform_train = [
#                 transforms.Resize((256,128), interpolation=3),
#                 transforms.Pad(10),
#                 transforms.RandomCrop((256,128)),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ToTensor(),
#                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#                 ]

#         transform_val = [
#                 transforms.Resize(size=(256,128),interpolation=3),
#                 transforms.ToTensor(),
#                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#                 ]

#         if self.erasing_p > 0:
#             transform_train = transform_train + [RandomErasing(probability=self.erasing_p, mean=[0.0, 0.0, 0.0])]

#         if self.color_jitter:
#             transform_train = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)] + transform_train

#         self.data_transforms = {
#             'train': transforms.Compose(transform_train),
#             'val': transforms.Compose(transform_val),
#         }        

#     def preprocess_kd_data(self, dataset):
#         loader, image_dataset = self.preprocess_one_train_dataset(dataset)
#         self.kd_loader = loader


#     def preprocess_one_train_dataset(self, dataset):
#         """preprocess a training dataset, construct a data loader.
#         """
#         print('dataset', dataset)
#         # data_path = os.path.join(self.data_dir, dataset, 'pytorch')
#         data_path = os.path.join(self.data_dir, dataset)
#         # data_path = os.path.join(self.data_dir, dataset, 'bounding_box_train')
#         # data_path = os.path.join(data_path, 'train' + self.train_all)
#         print('data_path', data_path)
        
#         image_dataset = factory[dataset](root=self.data_dir)
#         image_train_set = ImageDataset(image_dataset.train, self.data_transforms['train'])
#         num_classes = image_dataset.num_train_pids
#         # image_dataset = datasets.ImageFolder(data_path)

#         loader = torch.utils.data.DataLoader(
#             # ImageDataset(image_dataset.train, self.data_transforms['train']), 
#             # ImageDataset(image_dataset.imgs, self.data_transforms['train']), 
#             image_train_set,
#             batch_size=self.batch_size,
#             shuffle=True, 
#             num_workers=2, 
#             pin_memory=False)

#         return loader, image_train_set, num_classes
#         # return loader, image_dataset

#     def preprocess_train(self):
#         """preprocess training data, constructing train loaders
#         """
#         self.train_loaders = {}
#         self.train_dataset_sizes = {}
#         self.train_class_sizes = {}
#         self.client_list = []
        
#         for dataset in self.datasets:
#             self.client_list.append(dataset)
          
#             loader, image_dataset, num_classes = self.preprocess_one_train_dataset(dataset)

#             self.train_dataset_sizes[dataset] = len(image_dataset)
#             self.train_class_sizes[dataset] = num_classes
#             # self.train_class_sizes[dataset] = len(image_dataset.classes)
#             self.train_loaders[dataset] = loader
            
#         print('Train dataset sizes:', self.train_dataset_sizes)
#         print('Train class sizes:', self.train_class_sizes)
        
#     def preprocess_test(self):
#         """preprocess testing data, constructing test loaders
#         """
#         self.test_loaders = {}
#         self.gallery_meta = {}
#         self.query_meta = {}

#         for test_dir in self.datasets:
#             # test_dir = '/mnt/dataset1/plzhang/data/'+test_dir+'/pytorch'
#             dir = '/mnt/dataset1/plzhang/data/'
#             test_set_dir = '/mnt/dataset1/plzhang/data/' + test_dir
#             # test_dir = 'data/'+test_dir+'/pytorch'

#             dataset = test_set_dir.split('/')[1]
#             # gallery_dataset = datasets.ImageFolder(os.path.join(test_dir, 'gallery'))
#             # query_dataset = datasets.ImageFolder(os.path.join(test_dir, 'query'))
#             test_dataset = factory[test_dir](root=dir)
#             gallery_dataset = ImageDataset(test_dataset.gallery, self.data_transforms['val'])
#             query_dataset = ImageDataset(test_dataset.query, self.data_transforms['val'])
#             # gallery_dataset = ImageDataset(gallery_dataset.imgs, self.data_transforms['val'])
#             # query_dataset = ImageDataset(query_dataset.imgs, self.data_transforms['val'])

#             # self.test_loaders[dataset] = {key: torch.utils.data.DataLoader(
#             #                                     dataset, 
#             #                                     batch_size=self.batch_size,
#             #                                     shuffle=False, 
#             #                                     num_workers=8, 
#             #                                     pin_memory=True) for key, dataset in {'gallery': gallery_dataset, 'query': query_dataset}.items()}
#             self.test_loaders[dataset] = {key: torch.utils.data.DataLoader(
#                                                 dataset,
#                                                 batch_size=self.batch_size,
#                                                 shuffle=False,
#                                                 num_workers=8,
#                                                 pin_memory=True) for key, dataset in {'gallery': gallery_dataset, 'query': query_dataset}.items()}

#             # print(gallery_dataset)

#             gallery_cameras, gallery_labels = get_cam_label(gallery_dataset)
#             query_cameras, query_labels = get_cam_label(query_dataset)
#             self.gallery_meta[dataset] = {
#                 'sizes': len(gallery_dataset),
#                 'cameras': gallery_cameras,
#                 'labels': gallery_labels
#             }
#             self.query_meta[dataset] = {
#                 'sizes': len(query_dataset),
#                 'cameras': query_cameras,
#                 'labels': query_labels
#             }

#             # gallery_cameras, gallery_labels = get_camera_ids(gallery_dataset.imgs)
#             # self.gallery_meta[dataset] = {
#             #     'sizes':  len(gallery_dataset),
#             #     'cameras': gallery_cameras,
#             #     'labels': gallery_labels
#             # }

#             # query_cameras, query_labels = get_camera_ids(query_dataset.imgs)
#             # self.query_meta[dataset] = {
#             #     'sizes':  len(query_dataset),
#             #     'cameras': query_cameras,
#             #     'labels': query_labels
#             # }

#         print('Query Sizes:', self.query_meta[dataset]['sizes'])
#         print('Gallery Sizes:', self.gallery_meta[dataset]['sizes'])

#     def preprocess(self):
#         self.transform()
#         self.preprocess_train()
#         self.preprocess_test()
#         # self.preprocess_kd_data('cuhk02')

# def get_camera_ids(img_paths):
#     """get camera id and labels by image path
#     """
#     camera_ids = []
#     labels = []
#     for path, v in img_paths:
#         filename = os.path.basename(path)
#         if filename[:3]!='cam':
#             label = filename[0:4]
#             print('filename', filename)
#             camera = filename.split('c')[1]
#             camera = camera.split('s')[0]
#         else:
#             label = filename.split('_')[2]
#             camera = filename.split('_')[1]
#         if label[0:2]=='-1':
#             labels.append(-1)
#         else:
#             labels.append(int(label))
#         camera_ids.append(int(camera[0]))
#     return camera_ids, labels

# def get_cam_label(data_set):
#     cam_labels = []
#     pids = []
#     for img_path, pid, cam_label in data_set:
#         cam_labels.append(cam_label)
#         pids.append(pid)
#     return cam_labels, pids
