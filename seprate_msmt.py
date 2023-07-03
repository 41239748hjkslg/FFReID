# from distutils.file_util import copy_file
import os.path as osp
from shutil import copyfile
from shutil import copy
# from tkinter import Image
import matplotlib.image as image
import matplotlib.pyplot as plt

def _process_dir( dir_path, list_path, dist_path):
    with open(list_path, 'r') as txt:
        lines = txt.readlines()
    dataset = []
    pid_begin = 0
    pid_container = set()
    cam_container = set()
    for img_idx, img_info in enumerate(lines):
        img_path, pid = img_info.split(' ')
        pid = int(pid)  # no need to relabel
        camid = int(img_path.split('_')[2])
        img_path = osp.join(dir_path, img_path)
        # # print('img_path', img_path)
        # a = image.imread(img_path)
        # plt.imshow(a)
        copy(img_path, dist_path)
        dataset.append((img_path, pid_begin +pid, camid-1, 1))
        pid_container.add(pid)
        cam_container.add(camid)
    print(cam_container, 'cam_container')
    # check if pid starts from 0 and increments with 1
    for idx, pid in enumerate(pid_container):
        assert idx == pid, "See code comment for explanation"
    return dataset

train_dir_path = '/data/plzhang/data/MSMT17/train'
query_dir_path = '/data/plzhang/data/MSMT17/test'
gallery_dir_path = '/data/plzhang/data/MSMT17/test'
list_train_path = "/data/plzhang/data/MSMT17/list_train.txt"
list_query_path = '/data/plzhang/data/MSMT17/list_query.txt'
list_gallery_path = '/data/plzhang/data/MSMT17/list_gallery.txt'

dist_train_path = '/data/plzhang/data/MSMT17/bounding_box_train'
dist_query_path = '/data/plzhang/data/MSMT17/query'
dist_gallery_path = '/data/plzhang/data/MSMT17/bounding_box_test'

# train_set = _process_dir(train_dir_path, list_train_path, dist_train_path)
_process_dir(train_dir_path, list_train_path, dist_train_path)
# _process_dir(query_dir_path, list_query_path, dist_query_path)
# _process_dir(gallery_dir_path, list_gallery_path, dist_gallery_path)
# print(train_set)