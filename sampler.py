
# import enum
# from torch.utils.data.sampler import Sampler
# from collections import defaultdict
# import copy
# import random
# import numpy as np
# import torch

# from functools import reduce
# import itertools




# class RandomIdentitySampler(Sampler):
#     """
#     Randomly sample N identities, then for each identity,
#     randomly sample K instances, therefore batch size is N*K.
#     Args:
#     - data_source (list): list of (img_path, pid, camid).
#     - num_instances (int): number of instances per identity in a batch.
#     - batch_size (int): number of examples in a batch.
#     """

#     def __init__(self, data_source, batch_size, num_instances):
#         self.data_source = data_source
#         self.batch_size = batch_size
#         self.num_instances = num_instances
#         self.num_camid_per_batch = 4
#         self.num_pid_per_camid = 4
#         self.camid_instances = 2
#         self.num_pids_per_batch = self.batch_size // self.num_instances
#         self.index_dic = defaultdict(list) #dict with list value
#         # self.camid_dic = defaultdict(list)
#         #{783: [0, 5, 116, 876, 1554, 2041],...,}
        
#         # print(len(self.data_source))
#         for index, (_, pid) in enumerate(self.data_source):
#             self.index_dic[pid].append(index)
#         self.pids = list(self.index_dic.keys())
#         # for index, (_, _, camid, _)in enumerate(self.data_source):
#         #     self.camid_dic[camid].append(index)
#         # self.camids = list(self.camid_dic.keys())


#         # estimate number of examples in an epoch
#         self.length = 0
#         for pid in self.pids:
#             idxs = self.index_dic[pid]
#             num = len(idxs)
#             if num < self.num_instances:
#                 num = self.num_instances
#             self.length += num - num % self.num_instances

#     def __iter__(self):
#         batch_idxs_dict = defaultdict(list)
#         for pid in self.pids:
#             idxs = copy.deepcopy(self.index_dic[pid])
#             if len(idxs) < self.num_instances:
#                 idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
#             random.shuffle(idxs)
#             batch_idxs = []
#             for idx in idxs:
#                 batch_idxs.append(idx)
#                 if len(batch_idxs) == self.num_instances:
#                     batch_idxs_dict[pid].append(batch_idxs)
#                     batch_idxs = []

#         avai_pids = copy.deepcopy(self.pids)
#         final_idxs = []

#         while len(avai_pids) >= self.num_pids_per_batch:
#             selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
#             for pid in selected_pids:
#                 batch_idxs = batch_idxs_dict[pid].pop(0)
#                 final_idxs.extend(batch_idxs)
#                 if len(batch_idxs_dict[pid]) == 0:
#                     avai_pids.remove(pid)
#         # print('final_idxs', len(final_idxs))
#         # file = open('./logs/market/transreid/sampler/sample_debug_9/raw_img_idx.txt', 'w')
#         # record_final_idxs = final_idxs
#         # file.write(str(record_final_idxs))
#         # file.close()
#         return iter(final_idxs)



#     def __len__(self):
#         return self.length





from torch.utils.data.sampler import Sampler
from collections import defaultdict
import copy
import random
import numpy as np

class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list) #dict with list value
        #{783: [0, 5, 116, 876, 1554, 2041],...,}
        for index, (_, pid) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)
        # print('self.pids', len(self.pids))
        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length