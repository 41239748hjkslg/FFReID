# from cProfile import label
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Plotting')
parser.add_argument('--file_name',default='FedPAV-20200722_233845.log', type=str,help='path-name of the log file to be read')
parser.add_argument('--raw_file_name',default='FedPAV-20200722_233845.log', type=str, help='output figure path-name')
parser.add_argument('--fig_name',default='local-epoch-1-curve.pdf', type=str, help='output figure path-name')
parser.add_argument('--num_epochs', default=30, type=int, help='number of epochs we want to plot, we chose 300 in the paper')
parser.add_argument('--dataset', default='market,duke,cuhk03-np,MSMT17,cuhk01,viper,prid,3dpes,ilids',type=str, help='The datasets we want to plot')

args = parser.parse_args()

datasets = args.dataset.split(',')

# all_filename = args.file_name.split(',')

with open(args.raw_file_name, 'r') as raw_f:
    raw_f = raw_f.readlines()

raw_acc = {'raw_'+x:[] for x in datasets}
raw_mAP = {'raw_'+x:[] for x in datasets}


with open(args.file_name, 'r') as f:
    f = f.readlines()

acc = {x:[] for x in datasets}
mAP = {x:[] for x in datasets}



epoch_count = 0
local_count = 0
for line in f:

    if epoch_count==int(args.num_epochs):
        break

    if ('Rank' in line) and (line.split(' ')[4] in datasets):
        name = line.split(' ')[4]
        index = line.index('Rank@1')
        acc[name].append(float(line[index+7:index+15]))      # index+1:index+9 数值
        mAP[name].append(float(line[-8:]))
    # if ('Rank' in raw_line) and (raw_line.split(' ')[4] in datasets):
    #     raw_name = raw_line.split(' ')[4]
    #     raw_index = raw_line.index('Rank@1')
    #     raw_acc['raw_'+raw_name].append(float(raw_line[raw_index+7:raw_index+15]))
    #     raw_mAP['raw_'+raw_name].append(float(raw_line[-8:]))

        local_count+=1
        if local_count==len(datasets):
            local_count = 0
            epoch_count+=1


epoch_count = 0
local_count = 0

for raw_line in  raw_f:

    if epoch_count==int(args.num_epochs):
        break

    # if ('Rank' in line) and (line.split(' ')[4] in datasets):
    #     name = line.split(' ')[4]
    #     index = line.index('Rank@1')
    #     acc[name].append(float(line[index+7:index+15]))      # index+1:index+9 数值
    #     mAP[name].append(float(line[-8:]))
    # print('raw_line', raw_line)
    # raw_line = raw_line.decode("utf8", 'ignore')
    # print('raw_line', raw_line)
    if ('Rank' in raw_line) and (raw_line.split(' ')[4] in datasets):
        raw_name = raw_line.split(' ')[4]
        raw_index = raw_line.index('Rank@1')
        raw_acc['raw_'+raw_name].append(float(raw_line[raw_index+7:raw_index+15]))
        raw_mAP['raw_'+raw_name].append(float(raw_line[-8:]))

        local_count+=1
        if local_count==len(datasets):
            local_count = 0
            epoch_count+=1

# print(raw_mAP)
# print(mAP)
# print(acc)


# length = len(list(raw_acc.values())[0])
length = args.num_epochs
print('length', length)

for name, raw_name in zip(acc.keys(), raw_acc.keys()):
    if name == 'cuhk03-np' and raw_name == 'raw_cuhk03-np':
        plt.figure()
        plt.plot(np.arange(1,length+1), acc[name][:length], label = name, marker='>')
        plt.plot(np.arange(1, length+1), raw_acc[raw_name][:length], label=raw_name, marker='>')
        plt.xlabel('Epochs')
        plt.ylabel('Rank-1 Accuracy (%)')
        plt.xlim(0,(length+1))
        # plt.ylim(0, 1)
        plt.legend(loc=3)
        plt.savefig(args.fig_name + name + '_ACC' + str(epoch_count), dpi = 800)

        plt.show()
        plt.close()

        plt.figure()
        plt.plot(np.arange(1,length+1), mAP[name][:length], label = name, marker='>')
        plt.plot(np.arange(1, length+1), raw_mAP[raw_name][:length], label=raw_name, marker='>')
        plt.xlabel('Epochs')
        plt.ylabel('mAP (%)')
        plt.xlim(0,(length+1))
        # plt.ylim(0, 1)
        plt.legend(loc=3)
        plt.savefig(args.fig_name + name + '_mAP' + str(epoch_count), dpi = 800)

        plt.show()
        plt.close()

    elif name == 'market' and raw_name == 'raw_market':
        plt.figure()
        plt.plot(np.arange(1,length+1), acc[name][:length], label = name, marker='>')
        plt.plot(np.arange(1, length+1), raw_acc[raw_name][:length], label=raw_name, marker='>')
        plt.xlabel('Epochs')
        plt.ylabel('Rank-1 Accuracy (%)')
        plt.xlim(0,(length+1))
        # plt.ylim(0, 1)
        plt.legend(loc=3)
        plt.savefig(args.fig_name + name + '_ACC' + str(epoch_count), dpi = 800)

        plt.show()
        plt.close()

        plt.figure()
        plt.plot(np.arange(1,length+1), mAP[name][:length], label = name, marker='>')
        plt.plot(np.arange(1, length+1), raw_mAP[raw_name][:length], label=raw_name, marker='>')
        plt.xlabel('Epochs')
        plt.ylabel('mAP (%)')
        plt.xlim(0,(length+1))
        # plt.ylim(0, 1)
        plt.legend(loc=3)
        plt.savefig(args.fig_name + name + '_mAP' + str(epoch_count), dpi = 800)

        plt.show()
        plt.close()


    elif name == 'MSMT17' and raw_name == 'raw_MSMT17':
        plt.figure()
        plt.plot(np.arange(1,length+1), acc[name][:length], label = name, marker='>')
        plt.plot(np.arange(1, length+1), raw_acc[raw_name][:length], label=raw_name, marker='>')
        plt.xlabel('Epochs')
        plt.ylabel('Rank-1 Accuracy (%)')
        plt.xlim(0,(length+1))
        # plt.ylim(0, 1)
        plt.legend(loc=3)
        plt.savefig(args.fig_name + name + '_ACC' + str(epoch_count), dpi = 800)

        plt.show()
        plt.close()

        plt.figure()
        plt.plot(np.arange(1,length+1), mAP[name][:length], label = name, marker='>')
        plt.plot(np.arange(1, length+1), raw_mAP[raw_name][:length], label=raw_name, marker='>')
        plt.xlabel('Epochs')
        plt.ylabel('mAP (%)')
        plt.xlim(0,(length+1))
        # plt.ylim(0, 1)
        plt.legend(loc=3)
        plt.savefig(args.fig_name + name + '_mAP' + str(epoch_count), dpi = 800)

        plt.show()
        plt.close()


    elif name == 'duke' and raw_name == 'raw_duke':
        plt.figure()
        plt.plot(np.arange(1,length+1), acc[name][:length], label = name, marker='>')
        plt.plot(np.arange(1, length+1), raw_acc[raw_name][:length], label=raw_name, marker='>')
        plt.xlabel('Epochs')
        plt.ylabel('Rank-1 Accuracy (%)')
        plt.xlim(0,(length+1))
        # plt.ylim(0, 1)
        plt.legend(loc=3)
        plt.savefig(args.fig_name + name + '_ACC' + str(epoch_count), dpi = 800)

        plt.show()
        plt.close()

        plt.figure()
        plt.plot(np.arange(1,length+1), mAP[name][:length], label = name, marker='>')
        plt.plot(np.arange(1, length+1), raw_mAP[raw_name][:length], label=raw_name, marker='>')
        plt.xlabel('Epochs')
        plt.ylabel('mAP (%)')
        plt.xlim(0,(length+1))
        # plt.ylim(0, 1)
        plt.legend(loc=3)
        plt.savefig(args.fig_name + name + 'mAP' + str(epoch_count), dpi = 800)

        plt.show()
        plt.close()

    elif name == 'cuhk01' and raw_name == 'raw_cuhk01':
        plt.figure()
        plt.plot(np.arange(1,length+1), acc[name][:length], label = name, marker='>')
        plt.plot(np.arange(1, length+1), raw_acc[raw_name][:length], label=raw_name, marker='>')
        plt.xlabel('Epochs')
        plt.ylabel('Rank-1 Accuracy (%)')
        plt.xlim(0,(length+1))
        # plt.ylim(0, 1)
        plt.legend(loc=3)
        plt.savefig(args.fig_name + name + '_ACC' + str(epoch_count), dpi = 800)

        plt.show()
        plt.close()

        plt.figure()
        plt.plot(np.arange(1,length+1), mAP[name][:length], label = name, marker='>')
        plt.plot(np.arange(1, length+1), raw_mAP[raw_name][:length], label=raw_name, marker='>')
        plt.xlabel('Epochs')
        plt.ylabel('mAP (%)')
        plt.xlim(0,(length+1))
        # plt.ylim(0, 1)
        plt.legend(loc=3)
        plt.savefig(args.fig_name + name + '_mAP' + str(epoch_count), dpi = 800)

        plt.show()
        plt.close()

    elif name == 'viper' and raw_name == 'raw_viper':
        plt.figure()
        plt.plot(np.arange(1,length+1), acc[name][:length], label = name, marker='>')
        plt.plot(np.arange(1, length+1), raw_acc[raw_name][:length], label=raw_name, marker='>')
        plt.xlabel('Epochs')
        plt.ylabel('Rank-1 Accuracy (%)')
        plt.xlim(0,(length+1))
        # plt.ylim(0, 1)
        plt.legend(loc=3)
        plt.savefig(args.fig_name + name + '_ACC' + str(epoch_count), dpi = 800)

        plt.show()
        plt.close()

        plt.figure()
        plt.plot(np.arange(1,length+1), mAP[name][:length], label = name, marker='>')
        plt.plot(np.arange(1, length+1), raw_mAP[raw_name][:length], label=raw_name, marker='>')
        plt.xlabel('Epochs')
        plt.ylabel('mAP (%)')
        plt.xlim(0,(length+1))
        # plt.ylim(0, 1)
        plt.legend(loc=3)
        plt.savefig(args.fig_name + name + '_mAP' + str(epoch_count), dpi = 800)

        plt.show()
        plt.close()

    elif name == 'prid' and raw_name == 'raw_prid':
        plt.figure()
        plt.plot(np.arange(1,length+1), acc[name][:length], label = name, marker='>')
        plt.plot(np.arange(1, length+1), raw_acc[raw_name][:length], label=raw_name, marker='>')
        plt.xlabel('Epochs')
        plt.ylabel('Rank-1 Accuracy (%)')
        plt.xlim(0,(length+1))
        # plt.ylim(0, 1)
        plt.legend(loc=3)
        plt.savefig(args.fig_name + name + '_ACC' + str(epoch_count), dpi = 800)

        plt.show()
        plt.close()

        plt.figure()
        plt.plot(np.arange(1,length+1), mAP[name][:length], label = name, marker='>')
        plt.plot(np.arange(1, length+1), raw_mAP[raw_name][:length], label=raw_name, marker='>')
        plt.xlabel('Epochs')
        plt.ylabel('mAP (%)')
        plt.xlim(0,(length+1))
        # plt.ylim(0, 1)
        plt.legend(loc=3)
        plt.savefig(args.fig_name + name + '_mAP' + str(epoch_count), dpi = 800)

        plt.show()
        plt.close()


    elif name == '3dpes' and raw_name == 'raw_3dpes':
        plt.figure()
        plt.plot(np.arange(1,length+1), acc[name][:length], label = name, marker='>')
        plt.plot(np.arange(1, length+1), raw_acc[raw_name][:length], label=raw_name, marker='>')
        plt.xlabel('Epochs')
        plt.ylabel('Rank-1 Accuracy (%)')
        plt.xlim(0,(length+1))
        # plt.ylim(0, 1)
        plt.legend(loc=3)
        plt.savefig(args.fig_name + name + '_ACC' + str(epoch_count), dpi = 800)

        plt.show()
        plt.close()

        plt.figure()
        plt.plot(np.arange(1,length+1), mAP[name][:length], label = name, marker='>')
        plt.plot(np.arange(1, length+1), raw_mAP[raw_name][:length], label=raw_name, marker='>')
        plt.xlabel('Epochs')
        plt.ylabel('mAP (%)')
        plt.xlim(0,(length+1))
        # plt.ylim(0, 1)
        plt.legend(loc=3)
        plt.savefig(args.fig_name + name + '_mAP' + str(epoch_count), dpi = 800)

        plt.show()
        plt.close()

    else:
        plt.figure()
        plt.plot(np.arange(1,length+1), acc[name][:length], label = name, marker='>')
        plt.plot(np.arange(1, length+1), raw_acc[raw_name][:length], label=raw_name, marker='>')
        plt.xlabel('Epochs')
        plt.ylabel('Rank-1 Accuracy (%)')
        plt.xlim(0,(length+1))
        # plt.ylim(0, 1)
        plt.legend(loc=3)
        plt.savefig(args.fig_name + name + '_ACC' + str(epoch_count), dpi = 800)

        plt.show()
        plt.close()


        plt.figure()
        plt.plot(np.arange(1,length+1), mAP[name][:length], label = name, marker='>')
        plt.plot(np.arange(1, length+1), raw_mAP[raw_name][:length], label=raw_name, marker='>')
        plt.xlabel('Epochs')
        plt.ylabel('mAP (%)')
        plt.xlim(0,(length+1))
        # plt.ylim(0,1)
        plt.legend(loc=3)
        plt.savefig(args.fig_name + name + '_mAP' + str(epoch_count), dpi = 800)

        plt.show()
        plt.close()


# plt.figure()

# for name in mAP.keys():
#     plt.plot(np.arange(1,length+1), mAP[name], label = name)
# plt.xlabel('Epochs')
# plt.ylabel('mAP (%)')
# plt.xlim(0,(length+1))

# plt.legend(loc=3)
# plt.savefig(args.fig_name + '_mAP' + str(epoch_count), dpi = 800)
# plt.show()

# plt.close()