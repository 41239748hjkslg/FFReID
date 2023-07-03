from PIL import Image, ImageFile

from torch.utils.data import Dataset
from torchvision import transforms as T
import os.path as osp
import random
import torch
import math
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
# from skimage.metrics import compare_ssim as ssim
from skimage.metrics import structural_similarity as ssim

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img

def rand_bbox(size, lam):
    W = size[1]
    H = size[2]

    # print('lam', lam)
    cut_rat = np.sqrt(1. - lam)
    # print('cut_rat', cut_rat)
    # print('W', W)
    # print('H', H)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)
    # cx = np.random.randint(W//8, W*7//8)
    # cy = np.random.randint(H//8, H*7//8)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    # bbx1, bby1, bbx2, bby2  = random.choice([(0, 0, int(W/3), int(W/2)), (int(W*2/3), 0, W, W), (0, int(W/3), W, H), (0, 0, int(W), int(H/3))])

    # W = size[1]
    # H = size[2]
    # area = W * H
    # min_aspect = 0.3
    # max_aspect = None or 1 / 0.3
    # for attempt in range(10):
    #     target_area = random.uniform(0.15, 0.25) * area
    #     aspect_ratio = math.exp(random.uniform(math.log(min_aspect), math.log(max_aspect)))
    #     h = int(round(math.sqrt(target_area * aspect_ratio)))
    #     w = int(round(math.sqrt(target_area / aspect_ratio)))
    #     if w < W and h < H:
    #         bbx1 = random.randint(0, H - h)
    #         bby1 = random.randint(0, W - w)
    #         bbx2 = bbx1+h
    #         bby2 = bby1+w
    #         # img[:, top:top + h, left:left + w] = _get_pixels(
    #         #     self.per_pixel, self.rand_color, (chan, h, w),
    #         #     dtype=dtype, device=self.device)
    #         break

    return bbx1, bby1, bbx2, bby2

def take_pid_img(camid, data_idx, img):
    img = T.Resize((256, 128))(img)
    # img = T.Resize((256, 128), interpolation=3)(img)
    raw_img = np.array(img.convert('L'))
    img_frames = []
    # imgs_path = data_idx[tmp_mix_camid_1] + data_idx[tmp_mix_camid_2]
    all_data_idx = sum(data_idx.values(), [])
    all_img = set(all_data_idx)
    same_camid_img = data_idx[camid]
    defferent_camid_img = list(all_img.difference(same_camid_img))
    if len(defferent_camid_img) == 0:
        defferent_camid_img = same_camid_img
    for img_path in defferent_camid_img:
        compare_img = Image.open(img_path)
        compare_img = T.Resize((256, 128))(compare_img)
        # compare_img = T.Resize((256, 128), interpolation=3)(compare_img)
        compare_img = np.array(compare_img.convert('L'))
        img_ssim = ssim(raw_img, compare_img)
        img_frames.append(img_ssim)
    max_frame_diff = max(img_frames)
    min_frame_diff = min(img_frames)
    max_frame_diff_index = img_frames.index(max_frame_diff)
    min_frame_diff_index = img_frames.index(min_frame_diff)
    return defferent_camid_img[max_frame_diff_index], defferent_camid_img[min_frame_diff_index]



class BaseDataset(object):
    """
    Base class of reid dataset
    """

    def get_imagedata_info(self, data):
        pids, cams, tracks = [], [], []

        for _, pid, camid, trackid in data:
            pids += [pid]
            cams += [camid]
            tracks += [trackid]
        pids = set(pids)
        cams = set(cams)
        tracks = set(tracks)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        num_views = len(tracks)
        return num_pids, num_imgs, num_cams, num_views

    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams, num_train_views = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams, num_train_views = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams, num_train_views = self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        print("  ----------------------------------------")


class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
    # def __init__(self, dataset, transform=None, pid_index_dic=None, is_mix_up=None):
    #     self.dataset = dataset
    #     self.transform = transform
    #     self.dataset_index_idc = pid_index_dic
    #     self.is_mix_up = is_mix_up

    def __len__(self):
        return len(self.dataset)

    # def __getitem__(self, index):
    #     img_path, pid, camid, trackid = self.dataset[index]
    #     img = read_image(img_path)
    #     # img = msrcp(img_path)
    #     raw_img = img
                
        # if self.transform is not None and self.is_mix_up:
        #     img = self.transform(img)
        #     mix_up_camid = camid            
        #     # if index % 2 == 0:
        #     if random.random() <= 0.5:
        #         # return
        #         # lam = np.random.beta(1.0, 1.0)  # 均匀分布
        #         lam = np.random.uniform(0, 1.0)
        #         current_pid_all_img = self.dataset_index_idc[pid]
        #         difference_camid_max_frame_path, difference_camid_min_frame_path = take_pid_img(camid, current_pid_all_img, raw_img)
        #         # mixup_max_img = read_image(difference_camid_max_frame_path)
        #         mixup_max_img = read_image(difference_camid_max_frame_path)
        #         # tmp_pid, tmp_camid = map(int, pattern.search(difference_camid_max_frame_path).groups())
        #         # mixup_max_img = msrcp(difference_camid_max_frame_path)                
        #         if self.transform:
        #             mixup_max_img = self.transform(mixup_max_img)
        #         # img = lam * img + (1 - lam) * mixup_max_img
        #         bbx1, bby1, bbx2, bby2 = rand_bbox(img.shape, lam)

        #         # img[:, bbx1:bbx2, bby1:bby2] = mixup_max_img[:, bbx1:bbx2, bby1:bby2]
        #         # lam =  ((bbx2 - bbx1) * (bby2 - bby1) / (img.shape[-1] * img.shape[-2]))
        #         # if lam >= 0.05:
        #         #     mix_up_camid = camid + 6

        #         lam = ((bbx2 - bbx1) * (bby2 - bby1) / (img.shape[-1] * img.shape[-2]))
        #         if lam >=0.3 and lam<=0.4:
        #         # if (1-lam) <=0.2 and (1-lam) >=0:
        #             img[:, bbx1:bbx2, bby1:bby2] = mixup_max_img[:, bbx1:bbx2, bby1:bby2]
        #             # img = lam * img + (1 - lam) * mixup_max_img
        #             mix_up_camid = camid + 6

        #     return img, pid, camid, trackid, round(mix_up_camid), img_path.split('/')[-1]
        # else:
        #     img = self.transform(img)
        #     return img, pid, camid, trackid, img_path.split('/')[-1]


    def __getitem__(self, index):
        img_path, pid, camid, trackid = self.dataset[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid, trackid,img_path.split('/')[-1]