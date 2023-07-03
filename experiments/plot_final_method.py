from cProfile import label
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Plotting')
parser.add_argument('--final_file_name',default='FedPAV-20200722_233845.log', type=str,help='path-name of the log file to be read')
parser.add_argument('--logit_cdw_file_name',default='FedPAV-20200722_233845.log', type=str, help='output figure path-name')
parser.add_argument('--no_cdw_file_name',default='FedPAV-20200722_233845.log', type=str, help='output figure path-name')
parser.add_argument('--simi_file_name',default='FedPAV-20200722_233845.log', type=str, help='output figure path-name')
parser.add_argument('--raw_file_name',default='FedPAV-20200722_233845.log', type=str, help='output figure path-name')

parser.add_argument('--fig_name',default='local-epoch-1-curve.pdf', type=str, help='output figure path-name')
parser.add_argument('--num_epochs', default=30, type=int, help='number of epochs we want to plot, we chose 300 in the paper')
parser.add_argument('--dataset', default='market,duke,cuhk03-np,MSMT17,cuhk01,viper,prid,3dpes,ilids',type=str, help='The datasets we want to plot')

args = parser.parse_args()

datasets = args.dataset.split(',')



with open(args.final_file_name, 'r') as final:
    final = final.readlines()

with open(args.logit_cdw_file_name, 'r') as logit_cdw:
    logit_cdw = logit_cdw.readlines()

with open(args.no_cdw_file_name, 'r') as no_cdw:
    no_cdw = no_cdw.readlines()

with open(args.simi_file_name, 'r') as simi:
    simi = simi.readlines()

with open(args.raw_file_name, 'r') as raw_f:
    raw_f = raw_f.readlines()

final_filename = 'final_'
logit_cdw_file_name = 'logit_cdw_'
no_cdw_file_name = 'no_cdw_'
simi_file_name = 'simi_'
raw_file_name = 'raw_'

# print('final_filename', final_filename)
# print('logit_cdw_file_name', logit_cdw_file_name)
# print('no_cdw_file_name', no_cdw_file_name)
# print('simi_file_name', simi_file_name)
# print('raw_file_name', raw_file_name)


final_acc = {final_filename + x:[] for x in datasets}
final_mAP = {final_filename + x:[] for x in datasets}

logit_cdw_acc = {logit_cdw_file_name + x:[] for x in datasets}
logit_cdw_mAP = {logit_cdw_file_name + x:[] for x in datasets}

no_cdw_acc = {no_cdw_file_name + x:[] for x in datasets}
no_cdw_mAP = {no_cdw_file_name + x:[] for x in datasets}

simi_acc = {simi_file_name + x:[] for x in datasets}
simi_mAP = {simi_file_name + x:[] for x in datasets}

raw_acc = {raw_file_name+x:[] for x in datasets}
raw_mAP = {raw_file_name+x:[] for x in datasets}


# final_acc = {x:[] for x in datasets}
# final_mAP = {x:[] for x in datasets}

# logit_cdw_acc = {x:[] for x in datasets}
# logit_cdw_mAP = {x:[] for x in datasets}

# no_cdw_acc = {x:[] for x in datasets}
# no_cdw_mAP = {x:[] for x in datasets}

# simi_acc = {x:[] for x in datasets}
# simi_mAP = {x:[] for x in datasets}

# raw_acc = {x:[] for x in datasets}
# raw_mAP = {x:[] for x in datasets}




epoch_count = 0
local_count = 0
for line in final:

    if epoch_count==int(args.num_epochs):
        break

    if ('Rank' in line) and (line.split(' ')[4] in datasets):
        name = line.split(' ')[4]
        index = line.index('Rank@1')
        final_acc[final_filename + name].append(float(line[index+7:index+15]))      # index+1:index+9 数值
        final_mAP[final_filename + name].append(float(line[-8:]))

        local_count+=1
        if local_count==len(datasets):
            local_count = 0
            epoch_count+=1

epoch_count = 0
local_count = 0

for line in logit_cdw:

    if epoch_count==int(args.num_epochs):
        break

    if ('Rank' in line) and (line.split(' ')[4] in datasets):

        name = line.split(' ')[4]

        index = line.index('Rank@1')
        logit_cdw_acc[logit_cdw_file_name + name].append(float(line[index+7:index+15]))      # index+1:index+9 数值
        logit_cdw_mAP[logit_cdw_file_name + name].append(float(line[-8:]))

        local_count+=1
        if local_count==len(datasets):
            local_count = 0
            epoch_count+=1

epoch_count = 0
local_count = 0

for line in no_cdw:

    if epoch_count==int(args.num_epochs):
        break

    if ('Rank' in line) and (line.split(' ')[4] in datasets):
        name = line.split(' ')[4]
        index = line.index('Rank@1')
        no_cdw_acc[no_cdw_file_name + name].append(float(line[index+7:index+15]))      # index+1:index+9 数值
        no_cdw_mAP[no_cdw_file_name + name].append(float(line[-8:]))

        local_count+=1
        if local_count==len(datasets):
            local_count = 0
            epoch_count+=1

epoch_count = 0
local_count = 0

for line in simi:

    if epoch_count==int(args.num_epochs):
        break

    if ('Rank' in line) and (line.split(' ')[4] in datasets):
        name = line.split(' ')[4]
        index = line.index('Rank@1')
        simi_acc[simi_file_name + name].append(float(line[index+7:index+15]))      # index+1:index+9 数值
        simi_mAP[simi_file_name + name].append(float(line[-8:]))

        local_count+=1
        if local_count==len(datasets):
            local_count = 0
            epoch_count+=1


epoch_count = 0
local_count = 0

for raw_line in  raw_f:

    if epoch_count==int(args.num_epochs):
        break

    if ('Rank' in raw_line) and (raw_line.split(' ')[4] in datasets):
        raw_name = raw_line.split(' ')[4]
        raw_index = raw_line.index('Rank@1')
        raw_acc[raw_file_name+raw_name].append(float(raw_line[raw_index+7:raw_index+15]))
        raw_mAP[raw_file_name+raw_name].append(float(raw_line[-8:]))

        local_count+=1
        if local_count==len(datasets):
            local_count = 0
            epoch_count+=1


# length = len(list(raw_acc.values())[0])
length = args.num_epochs
print('length', length)

# print(no_cdw_acc)

for final_name, logit_cdw_name, no_cdw_name, simi_name, raw_name in zip(final_acc.keys(), logit_cdw_acc.keys(), no_cdw_acc.keys(), simi_acc.keys(), raw_acc.keys()):
    if final_name == 'final_market' and raw_name == 'raw_market' and logit_cdw_name == 'logit_cdw_market' and no_cdw_name == 'no_cdw_market' and simi_name == 'simi_market':
        name = final_name.split('_')[1]
        plt.figure()
        plt.plot(np.arange(1,length+1), final_acc[final_name][:length], label = final_name, marker='>')
        plt.plot(np.arange(1,length+1), logit_cdw_acc[logit_cdw_name][:length], label = logit_cdw_name, marker='>')
        plt.plot(np.arange(1,length+1), no_cdw_acc[no_cdw_name][:length], label = no_cdw_name, marker='>')
        plt.plot(np.arange(1,length+1), simi_acc[simi_name][:length], label = simi_name, marker='>')
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
        plt.plot(np.arange(1,length+1), final_mAP[final_name][:length], label = final_name, marker='>')
        plt.plot(np.arange(1,length+1), logit_cdw_mAP[logit_cdw_name][:length], label = logit_cdw_name, marker='>')
        plt.plot(np.arange(1,length+1), no_cdw_mAP[no_cdw_name][:length], label = no_cdw_name, marker='>')
        plt.plot(np.arange(1,length+1), simi_mAP[simi_name][:length], label = simi_name, marker='>')
        plt.plot(np.arange(1, length+1), raw_mAP[raw_name][:length], label=raw_name, marker='>')
        plt.xlabel('Epochs')
        plt.ylabel('mAP (%)')
        plt.xlim(0,(length+1))
        # plt.ylim(0, 1)
        plt.legend(loc=3)
        plt.savefig(args.fig_name + name + '_mAP' + str(epoch_count), dpi = 800)

        plt.show()
        plt.close()

    # elif final_name == 'final_duke' and raw_name == 'raw_duke' and logit_cdw_name == 'logit_cdw_duke' and no_cdw_name == 'no_cdw_duke' and simi_name == 'simi_duke':
    #     name = final_name.split('_')[1]
    #     plt.figure()
    #     plt.plot(np.arange(1,length+1), final_acc[final_name][:length], label = final_name, marker='>')
    #     plt.plot(np.arange(1,length+1), logit_cdw_acc[logit_cdw_name][:length], label = logit_cdw_name, marker='>')
    #     plt.plot(np.arange(1,length+1), no_cdw_acc[no_cdw_name][:length], label = no_cdw_name, marker='>')
    #     plt.plot(np.arange(1,length+1), simi_acc[simi_name][:length], label = simi_name, marker='>')
    #     plt.plot(np.arange(1, length+1), raw_acc[raw_name][:length], label=raw_name, marker='>')
    #     plt.xlabel('Epochs')
    #     plt.ylabel('Rank-1 Accuracy (%)')
    #     plt.xlim(0,(length+1))
    #     # plt.ylim(0, 1)
    #     plt.legend(loc=3)
    #     plt.savefig(args.fig_name + name + '_ACC' + str(epoch_count), dpi = 800)

    #     plt.show()
    #     plt.close()

    #     plt.figure()
    #     plt.plot(np.arange(1,length+1), final_mAP[final_name][:length], label = final_name, marker='>')
    #     plt.plot(np.arange(1,length+1), logit_cdw_mAP[logit_cdw_name][:length], label = logit_cdw_name, marker='>')
    #     plt.plot(np.arange(1,length+1), no_cdw_mAP[no_cdw_name][:length], label = no_cdw_name, marker='>')
    #     plt.plot(np.arange(1,length+1), simi_mAP[simi_name][:length], label = simi_name, marker='>')
    #     plt.plot(np.arange(1, length+1), raw_mAP[raw_name][:length], label=raw_name, marker='>')
    #     plt.xlabel('Epochs')
    #     plt.ylabel('mAP (%)')
    #     plt.xlim(0,(length+1))
    #     # plt.ylim(0, 1)
    #     plt.legend(loc=3)
    #     plt.savefig(args.fig_name + name + '_mAP' + str(epoch_count), dpi = 800)

    #     plt.show()
    #     plt.close()

    # elif final_name == 'final_MSMT17' and raw_name == 'raw_MSMT17' and logit_cdw_name == 'logit_cdw_MSMT17' and no_cdw_name == 'no_cdw_MSMT17' and simi_name == 'simi_MSMT17':
    #     name = final_name.split('_')[1]
    #     plt.figure()
    #     plt.plot(np.arange(1,length+1), final_acc[final_name][:length], label = final_name, marker='>')
    #     plt.plot(np.arange(1,length+1), logit_cdw_acc[logit_cdw_name][:length], label = logit_cdw_name, marker='>')
    #     plt.plot(np.arange(1,length+1), no_cdw_acc[no_cdw_name][:length], label = no_cdw_name, marker='>')
    #     plt.plot(np.arange(1,length+1), simi_acc[simi_name][:length], label = simi_name, marker='>')
    #     plt.plot(np.arange(1, length+1), raw_acc[raw_name][:length], label=raw_name, marker='>')
    #     plt.xlabel('Epochs')
    #     plt.ylabel('Rank-1 Accuracy (%)')
    #     plt.xlim(0,(length+1))
    #     # plt.ylim(0, 1)
    #     plt.legend(loc=3)
    #     plt.savefig(args.fig_name + name + '_ACC' + str(epoch_count), dpi = 800)

    #     plt.show()
    #     plt.close()

    #     plt.figure()
    #     plt.plot(np.arange(1,length+1), final_mAP[final_name][:length], label = final_name, marker='>')
    #     plt.plot(np.arange(1,length+1), logit_cdw_mAP[logit_cdw_name][:length], label = logit_cdw_name, marker='>')
    #     plt.plot(np.arange(1,length+1), no_cdw_mAP[no_cdw_name][:length], label = no_cdw_name, marker='>')
    #     plt.plot(np.arange(1,length+1), simi_mAP[simi_name][:length], label = simi_name, marker='>')
    #     plt.plot(np.arange(1, length+1), raw_mAP[raw_name][:length], label=raw_name, marker='>')
    #     plt.xlabel('Epochs')
    #     plt.ylabel('mAP (%)')
    #     plt.xlim(0,(length+1))
    #     # plt.ylim(0, 1)
    #     plt.legend(loc=3)
    #     plt.savefig(args.fig_name + name + '_mAP' + str(epoch_count), dpi = 800)

    #     plt.show()
    #     plt.close()


    # elif final_name == 'final_cuhk03-np' and raw_name == 'raw_cuhk03-np' and logit_cdw_name == 'logit_cdw_cuhk03-np' and no_cdw_name == 'no_cdw_cuhk03-np' and simi_name == 'simi_cuhk03-np':
    #     name = final_name.split('_')[1]
    #     plt.figure()
    #     plt.plot(np.arange(1,length+1), final_acc[final_name][:length], label = final_name, marker='>')
    #     plt.plot(np.arange(1,length+1), logit_cdw_acc[logit_cdw_name][:length], label = logit_cdw_name, marker='>')
    #     plt.plot(np.arange(1,length+1), no_cdw_acc[no_cdw_name][:length], label = no_cdw_name, marker='>')
    #     plt.plot(np.arange(1,length+1), simi_acc[simi_name][:length], label = simi_name, marker='>')
    #     plt.plot(np.arange(1, length+1), raw_acc[raw_name][:length], label=raw_name, marker='>')
    #     plt.xlabel('Epochs')
    #     plt.ylabel('Rank-1 Accuracy (%)')
    #     plt.xlim(0,(length+1))
    #     # plt.ylim(0, 1)
    #     plt.legend(loc=3)
    #     plt.savefig(args.fig_name + name + '_ACC' + str(epoch_count), dpi = 800)

    #     plt.show()
    #     plt.close()

    #     plt.figure()
    #     plt.plot(np.arange(1,length+1), final_mAP[final_name][:length], label = final_name, marker='>')
    #     plt.plot(np.arange(1,length+1), logit_cdw_mAP[logit_cdw_name][:length], label = logit_cdw_name, marker='>')
    #     plt.plot(np.arange(1,length+1), no_cdw_mAP[no_cdw_name][:length], label = no_cdw_name, marker='>')
    #     plt.plot(np.arange(1,length+1), simi_mAP[simi_name][:length], label = simi_name, marker='>')
    #     plt.plot(np.arange(1, length+1), raw_mAP[raw_name][:length], label=raw_name, marker='>')
    #     plt.xlabel('Epochs')
    #     plt.ylabel('mAP (%)')
    #     plt.xlim(0,(length+1))
    #     # plt.ylim(0, 1)
    #     plt.legend(loc=3)
    #     plt.savefig(args.fig_name + name + '_mAP' + str(epoch_count), dpi = 800)

    #     plt.show()
    #     plt.close()

    # elif final_name == 'final_cuhk01' and raw_name == 'raw_cuhk01' and logit_cdw_name == 'logit_cdw_cuhk01' and no_cdw_name == 'no_cdw_cuhk01' and simi_name == 'simi_cuhk01':
    #     name = final_name.split('_')[1]
    #     plt.figure()
    #     plt.plot(np.arange(1,length+1), final_acc[final_name][:length], label = final_name, marker='>')
    #     plt.plot(np.arange(1,length+1), logit_cdw_acc[logit_cdw_name][:length], label = logit_cdw_name, marker='>')
    #     plt.plot(np.arange(1,length+1), no_cdw_acc[no_cdw_name][:length], label = no_cdw_name, marker='>')
    #     plt.plot(np.arange(1,length+1), simi_acc[simi_name][:length], label = simi_name, marker='>')
    #     plt.plot(np.arange(1, length+1), raw_acc[raw_name][:length], label=raw_name, marker='>')
    #     plt.xlabel('Epochs')
    #     plt.ylabel('Rank-1 Accuracy (%)')
    #     plt.xlim(0,(length+1))
    #     # plt.ylim(0, 1)
    #     plt.legend(loc=3)
    #     plt.savefig(args.fig_name + name + '_ACC' + str(epoch_count), dpi = 800)

    #     plt.show()
    #     plt.close()

    #     plt.figure()
    #     plt.plot(np.arange(1,length+1), final_mAP[final_name][:length], label = final_name, marker='>')
    #     plt.plot(np.arange(1,length+1), logit_cdw_mAP[logit_cdw_name][:length], label = logit_cdw_name, marker='>')
    #     plt.plot(np.arange(1,length+1), no_cdw_mAP[no_cdw_name][:length], label = no_cdw_name, marker='>')
    #     plt.plot(np.arange(1,length+1), simi_mAP[simi_name][:length], label = simi_name, marker='>')
    #     plt.plot(np.arange(1, length+1), raw_mAP[raw_name][:length], label=raw_name, marker='>')
    #     plt.xlabel('Epochs')
    #     plt.ylabel('mAP (%)')
    #     plt.xlim(0,(length+1))
    #     # plt.ylim(0, 1)
    #     plt.legend(loc=3)
    #     plt.savefig(args.fig_name + name + '_mAP' + str(epoch_count), dpi = 800)

    #     plt.show()
    #     plt.close()

    # elif final_name == 'final_viper' and raw_name == 'raw_viper' and logit_cdw_name == 'logit_cdw_viper' and no_cdw_name == 'no_cdw_viper' and simi_name == 'simi_viper':
    #     name = final_name.split('_')[1]
    #     plt.figure()
    #     plt.plot(np.arange(1,length+1), final_acc[final_name][:length], label = final_name, marker='>')
    #     plt.plot(np.arange(1,length+1), logit_cdw_acc[logit_cdw_name][:length], label = logit_cdw_name, marker='>')
    #     plt.plot(np.arange(1,length+1), no_cdw_acc[no_cdw_name][:length], label = no_cdw_name, marker='>')
    #     plt.plot(np.arange(1,length+1), simi_acc[simi_name][:length], label = simi_name, marker='>')
    #     plt.plot(np.arange(1, length+1), raw_acc[raw_name][:length], label=raw_name, marker='>')
    #     plt.xlabel('Epochs')
    #     plt.ylabel('Rank-1 Accuracy (%)')
    #     plt.xlim(0,(length+1))
    #     # plt.ylim(0, 1)
    #     plt.legend(loc=3)
    #     plt.savefig(args.fig_name + name + '_ACC' + str(epoch_count), dpi = 800)

    #     plt.show()
    #     plt.close()

    #     plt.figure()
    #     plt.plot(np.arange(1,length+1), final_mAP[final_name][:length], label = final_name, marker='>')
    #     plt.plot(np.arange(1,length+1), logit_cdw_mAP[logit_cdw_name][:length], label = logit_cdw_name, marker='>')
    #     plt.plot(np.arange(1,length+1), no_cdw_mAP[no_cdw_name][:length], label = no_cdw_name, marker='>')
    #     plt.plot(np.arange(1,length+1), simi_mAP[simi_name][:length], label = simi_name, marker='>')
    #     plt.plot(np.arange(1, length+1), raw_mAP[raw_name][:length], label=raw_name, marker='>')
    #     plt.xlabel('Epochs')
    #     plt.ylabel('mAP (%)')
    #     plt.xlim(0,(length+1))
    #     # plt.ylim(0, 1)
    #     plt.legend(loc=3)
    #     plt.savefig(args.fig_name + name + '_mAP' + str(epoch_count), dpi = 800)

    #     plt.show()
    #     plt.close()

    # elif final_name == 'final_prid' and raw_name == 'raw_prid' and logit_cdw_name == 'logit_cdw_prid' and no_cdw_name == 'no_cdw_prid' and simi_name == 'simi_prid':
    #     name = final_name.split('_')[1]
    #     plt.figure()
    #     plt.plot(np.arange(1,length+1), final_acc[final_name][:length], label = final_name, marker='>')
    #     plt.plot(np.arange(1,length+1), logit_cdw_acc[logit_cdw_name][:length], label = logit_cdw_name, marker='>')
    #     plt.plot(np.arange(1,length+1), no_cdw_acc[no_cdw_name][:length], label = no_cdw_name, marker='>')
    #     plt.plot(np.arange(1,length+1), simi_acc[simi_name][:length], label = simi_name, marker='>')
    #     plt.plot(np.arange(1, length+1), raw_acc[raw_name][:length], label=raw_name, marker='>')
    #     plt.xlabel('Epochs')
    #     plt.ylabel('Rank-1 Accuracy (%)')
    #     plt.xlim(0,(length+1))
    #     # plt.ylim(0, 1)
    #     plt.legend(loc=3)
    #     plt.savefig(args.fig_name + name + '_ACC' + str(epoch_count), dpi = 800)

    #     plt.show()
    #     plt.close()

    #     plt.figure()
    #     plt.plot(np.arange(1,length+1), final_mAP[final_name][:length], label = final_name, marker='>')
    #     plt.plot(np.arange(1,length+1), logit_cdw_mAP[logit_cdw_name][:length], label = logit_cdw_name, marker='>')
    #     plt.plot(np.arange(1,length+1), no_cdw_mAP[no_cdw_name][:length], label = no_cdw_name, marker='>')
    #     plt.plot(np.arange(1,length+1), simi_mAP[simi_name][:length], label = simi_name, marker='>')
    #     plt.plot(np.arange(1, length+1), raw_mAP[raw_name][:length], label=raw_name, marker='>')
    #     plt.xlabel('Epochs')
    #     plt.ylabel('mAP (%)')
    #     plt.xlim(0,(length+1))
    #     # plt.ylim(0, 1)
    #     plt.legend(loc=3)
    #     plt.savefig(args.fig_name + name + '_mAP' + str(epoch_count), dpi = 800)

    #     plt.show()
    #     plt.close()


    # elif final_name == 'final_3dpes' and raw_name == 'raw_3dpes' and logit_cdw_name == 'logit_cdw_3dpes' and no_cdw_name == 'no_cdw_3dpes' and simi_name == 'simi_3dpes':
    #     name = final_name.split('_')[1]
    #     plt.figure()
    #     plt.plot(np.arange(1,length+1), final_acc[final_name][:length], label = final_name, marker='>')
    #     plt.plot(np.arange(1,length+1), logit_cdw_acc[logit_cdw_name][:length], label = logit_cdw_name, marker='>')
    #     plt.plot(np.arange(1,length+1), no_cdw_acc[no_cdw_name][:length], label = no_cdw_name, marker='>')
    #     plt.plot(np.arange(1,length+1), simi_acc[simi_name][:length], label = simi_name, marker='>')
    #     plt.plot(np.arange(1, length+1), raw_acc[raw_name][:length], label=raw_name, marker='>')
    #     plt.xlabel('Epochs')
    #     plt.ylabel('Rank-1 Accuracy (%)')
    #     plt.xlim(0,(length+1))
    #     # plt.ylim(0, 1)
    #     plt.legend(loc=3)
    #     plt.savefig(args.fig_name + name + '_ACC' + str(epoch_count), dpi = 800)

    #     plt.show()
    #     plt.close()

    #     plt.figure()
    #     plt.plot(np.arange(1,length+1), final_mAP[final_name][:length], label = final_name, marker='>')
    #     plt.plot(np.arange(1,length+1), logit_cdw_mAP[logit_cdw_name][:length], label = logit_cdw_name, marker='>')
    #     plt.plot(np.arange(1,length+1), no_cdw_mAP[no_cdw_name][:length], label = no_cdw_name, marker='>')
    #     plt.plot(np.arange(1,length+1), simi_mAP[simi_name][:length], label = simi_name, marker='>')
    #     plt.plot(np.arange(1, length+1), raw_mAP[raw_name][:length], label=raw_name, marker='>')
    #     plt.xlabel('Epochs')
    #     plt.ylabel('mAP (%)')
    #     plt.xlim(0,(length+1))
    #     # plt.ylim(0, 1)
    #     plt.legend(loc=3)
    #     plt.savefig(args.fig_name + name + '_mAP' + str(epoch_count), dpi = 800)

    #     plt.show()
    #     plt.close()

    # else:
    #     name = final_name.split('_')[1]
    #     plt.figure()
    #     plt.plot(np.arange(1,length+1), final_acc[final_name][:length], label = final_name, marker='>')
    #     plt.plot(np.arange(1,length+1), logit_cdw_acc[logit_cdw_name][:length], label = logit_cdw_name, marker='>')
    #     plt.plot(np.arange(1,length+1), no_cdw_acc[no_cdw_name][:length], label = no_cdw_name, marker='>')
    #     plt.plot(np.arange(1,length+1), simi_acc[simi_name][:length], label = simi_name, marker='>')
    #     plt.plot(np.arange(1, length+1), raw_acc[raw_name][:length], label=raw_name, marker='>')
    #     plt.xlabel('Epochs')
    #     plt.ylabel('Rank-1 Accuracy (%)')
    #     plt.xlim(0,(length+1))
    #     # plt.ylim(0, 1)
    #     plt.legend(loc=3)
    #     plt.savefig(args.fig_name + name + '_ACC' + str(epoch_count), dpi = 800)

    #     plt.show()
    #     plt.close()

    #     plt.figure()
    #     plt.plot(np.arange(1,length+1), final_mAP[final_name][:length], label = final_name, marker='>')
    #     plt.plot(np.arange(1,length+1), logit_cdw_mAP[logit_cdw_name][:length], label = logit_cdw_name, marker='>')
    #     plt.plot(np.arange(1,length+1), no_cdw_mAP[no_cdw_name][:length], label = no_cdw_name, marker='>')
    #     plt.plot(np.arange(1,length+1), simi_mAP[simi_name][:length], label = simi_name, marker='>')
    #     plt.plot(np.arange(1, length+1), raw_mAP[raw_name][:length], label=raw_name, marker='>')
    #     plt.xlabel('Epochs')
    #     plt.ylabel('mAP (%)')
    #     plt.xlim(0,(length+1))
    #     # plt.ylim(0, 1)
    #     plt.legend(loc=3)
    #     plt.savefig(args.fig_name + name + '_mAP' + str(epoch_count), dpi = 800)

    #     plt.show()
    #     plt.close()

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