import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Plotting')
parser.add_argument('--file_name',default='FedPAV-20200722_233845.log', type=str,help='path-name of the log file to be read')
parser.add_argument('--fig_name',default='local-epoch-1-curve.pdf', type=str, help='output figure path-name')
parser.add_argument('--num_epochs', default=110, type=int, help='number of epochs we want to plot, we chose 300 in the paper')
parser.add_argument('--dataset', default='market,duke,cuhk03-np,MSMT17,cuhk01,viper,prid,3dpes,ilids',type=str, help='The datasets we want to plot')

args = parser.parse_args()

with open(args.file_name, 'r') as f:
    f = f.readlines()

datasets = args.dataset.split(',')
acc = {x:[] for x in datasets}
mAP = {x:[] for x in datasets}

epoch_count = 0
local_count = 0
for line in f:
    # print('line', line)
    # if epoch_count==int(args.num_epochs//10)*len(datasets):
    if epoch_count==int(args.num_epochs):
        break
    # print('============')
    # print(line.split(' ')[4])
    if ('Rank' in line) and (line.split(' ')[4] in datasets):
        name = line.split(' ')[4]

        # print('name', name)
        index = line.index('Rank@1')
        # print('index', index)
        # print('line[index+1:index+9]', line[index+7:index+15])
        acc[name].append(float(line[index+7:index+15]))      # index+1:index+9 数值
        mAP[name].append(float(line[-8:]))

        local_count+=1
        # epoch_count+=1
        if local_count==len(datasets):
            local_count = 0
            epoch_count+=1


length = len(list(acc.values())[0])

# a = args.fig_name.split('/')
# print('a', a)

plt.figure()
for name in acc.keys():
    plt.plot(np.arange(1,length+1), acc[name], label = name)
    # plt.plot(np.arange(1,length+1)*10, acc[name], label = name)
plt.xlabel('Epochs')
plt.ylabel('Rank-1 Accuracy (%)')
plt.xlim(0,(length+1))
# plt.xlim(0,(length+1)*10)
plt.legend(loc=3)
# plt.savefig('_ACC'.join(args.fig_name), dpi = 300)
plt.savefig(args.fig_name + '_ACC' + str(epoch_count), dpi = 800)
# plt.savefig('_ACC'.join(args.fig_name.split('/')), dpi = 800)
plt.show()
plt.close()

plt.figure()
for name in mAP.keys():
    plt.plot(np.arange(1,length+1), mAP[name], label = name)
    # plt.plot(np.arange(1,length+1)*10, mAP[name], label = name)
plt.xlabel('Epochs')
plt.ylabel('mAP (%)')
plt.xlim(0,(length+1))
# plt.xlim(0,(length+1)*10)
plt.legend(loc=3)
plt.savefig(args.fig_name + '_mAP' + str(epoch_count), dpi = 800)
# plt.savefig('_mAP.'.join(args.fig_name.split('/')), dpi = 800)
# plt.savefig('_mAP.'.join(args.fig_name.split('.')), dpi = 300)
plt.show()

plt.close()