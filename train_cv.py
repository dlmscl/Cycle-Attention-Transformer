import os
import logging
import numpy as np
import random
import pickle
from networks import *
import torch
import torch.backends.cudnn as cudnn

# Env
from data_loaders import *
from options import parse_args, print_options
from train_test import train, test
from Save import print_root
from utils import makeAUROCPlot, get_checkpoint_pathgraphomic, GradMetrics


### 1. Initializes parser and device
opt, parser = parse_args()
file, time = print_root(opt)
print_options(parser, opt, file=file)

device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
print("Using device:", device, file=file)
if not os.path.exists(opt.checkpoints_dir): os.makedirs(opt.checkpoints_dir)
if not os.path.exists(os.path.join(opt.checkpoints_dir, opt.exp_name)): os.makedirs(os.path.join(opt.checkpoints_dir, opt.exp_name))
if not os.path.exists(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name)): os.makedirs(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name))

### 2. Initializes Data
ignore_missing_histype = 1 if 'grad' in opt.task else 0
ignore_missing_moltype = 1 if 'omic' in opt.mode else 0
use_patch, roi_dir = ('_patch_', 'all_st_patches_512') if opt.use_vgg_features else ('_', 'all_st')
use_rnaseq = '_rnaseq' if opt.use_rnaseq else ''

if opt.dataroot == './data/TCGA_GBMLGG':
    data_cv_path = '%s/splits/gbmlgg15cv_%s_%d_%d_%d%s.pkl' % (opt.dataroot, roi_dir, ignore_missing_moltype, ignore_missing_histype, opt.use_vgg_features, use_rnaseq)
else:
    if not opt.use_vgg_features:
        data_cv_path = './data/TCGA_KIRC/splits/KIRC_st_0.pkl'
    else:
        data_cv_path = './data/TCGA_KIRC/splits/KIRC_st_1_ag.pkl'


print("Loading %s" % data_cv_path, file=file)
data_cv = pickle.load(open(data_cv_path, 'rb'))

if opt.dataroot == './data/TCGA_KIRC' and opt.task == 'surv' and (opt.mode == 'graphomic' or opt.mode == 'graph' or opt.mode == 'pathgraphomic'):
    for i in range(1,16):
        name = [[614, 732, 448, 682, 8, 328, 552, 465, 704, 759, 212],[154, 160],
                [558, 626, 688, 762, 480, 610, 589, 221, 710, 331],[2, 94, 183],
                [659, 346, 489, 472, 780, 734, 715, 221],[2, 105, 112, 115, 168],
                [468, 741, 11, 448, 331, 558, 707, 604, 224],[148, 152, 172, 186],
                [450, 726, 574, 549, 586, 750, 8, 602, 301, 692],[68, 127, 190],
                [614, 462, 313, 11, 595, 744, 713, 771, 540, 574, 445],[65, 172],
                [209, 613, 322, 448, 768, 592, 564, 11, 632, 741, 465, 710],[163],
                [466, 209, 732, 322, 567, 701, 598, 682, 610, 8, 759],[99, 149],
                [469, 641, 576, 777, 11, 703, 625, 218, 753, 337, 725],[90, 121],
                [688, 313, 629, 713, 447, 212, 768, 555, 427, 589, 14],[145, 177],
                [592, 328, 448, 747, 771, 691, 626, 719, 555, 462, 610],[2, 50],
                [670, 483, 580, 611, 215, 692, 592, 558, 313, 750, 463, 14, 726],[],
                [716, 322, 567, 777, 750, 11, 595, 632, 480, 610, 212],[91, 163],
                [215, 451, 623, 691, 558, 768, 322, 462, 610, 589, 744, 14],[173],
                [759, 698, 571, 729, 586, 543, 602, 307, 14],[59, 124, 129, 184]]

        data_cv['split'][i]['train']['x_path'] = np.delete(data_cv['split'][i]['train']['x_path'],name[2*(i-1)])
        data_cv['split'][i]['train']['x_omic'] = np.delete(data_cv['split'][i]['train']['x_omic'],name[2*(i-1)],axis = 0)
        data_cv['split'][i]['train']['x_grph'] = np.delete(data_cv['split'][i]['train']['x_grph'],name[2*(i-1)])
        data_cv['split'][i]['train']['e'] = np.delete(data_cv['split'][i]['train']['e'],name[2*(i-1)])
        data_cv['split'][i]['train']['t'] = np.delete(data_cv['split'][i]['train']['t'],name[2*(i-1)])
        data_cv['split'][i]['train']['g'] = np.delete(data_cv['split'][i]['train']['g'],name[2*(i-1)])
        data_cv['split'][i]['test']['x_path'] = np.delete(data_cv['split'][i]['test']['x_path'],name[2*i-1])
        data_cv['split'][i]['test']['x_omic'] = np.delete(data_cv['split'][i]['test']['x_omic'],name[2*i-1],axis = 0)
        data_cv['split'][i]['test']['x_grph'] = np.delete(data_cv['split'][i]['test']['x_grph'],name[2*i-1])
        data_cv['split'][i]['test']['e'] = np.delete(data_cv['split'][i]['test']['e'],name[2*i-1])
        data_cv['split'][i]['test']['t'] = np.delete(data_cv['split'][i]['test']['t'],name[2*i-1])
        data_cv['split'][i]['test']['g'] = np.delete(data_cv['split'][i]['test']['g'],name[2*i-1])


if opt.dataroot == './data/TCGA_GBMLGG':
    data_cv_splits = data_cv['cv_splits']
else:
    data_cv_splits = data_cv['split']

print(len(data_cv_splits[1]['train']['x_path']) + len(data_cv_splits[1]['test']['x_path']), file=file)
print(len(data_cv_splits[1]['train']['x_grph']) + len(data_cv_splits[1]['test']['x_grph']), file=file)
print(len(data_cv_splits[1]['train']['x_omic']) + len(data_cv_splits[1]['test']['x_omic']), file=file)
      
results = []

for k, data in data_cv_splits.items():
    if k >=opt.begin_k:
        print("*******************************************", file=file)
        print("************** SPLIT (%d/%d) **************" % (k, len(data_cv_splits.items())), file=file)
        print("*******************************************", file=file)
        
        model, optimizer, metric_logger = train(opt, data, device, k, file, time)
        print('Train finish, will test soon', file=file)
        
    #### 3.2 Evalutes Train + Test Error, and Saves Model
    
        loss_train, cindex_train, pvalue_train, surv_acc_train, grad_acc_train, pred_train, auc_train, ap_train, f1_train, f1_IV_train = test(opt, model, data, 'train', device)
        loss_test, cindex_test, pvalue_test, surv_acc_test, grad_acc_test, pred_test, auc_test, ap_test, f1_test, f1_IV_test = test(opt, model, data, 'test', device)

        if opt.task == 'surv':
            print("[Final] Apply model to training set: C-Index: %.10f, P-Value: %.10e" % (cindex_train, pvalue_train), file=file)
            logging.info("[Final] Apply model to training set: C-Index: %.10f, P-Value: %.10e" % (cindex_train, pvalue_train))
            print("[Final] Apply model to testing set: C-Index: %.10f, P-Value: %.10e" % (cindex_test, pvalue_test), file=file)
            logging.info("[Final] Apply model to testing set: cC-Index: %.10f, P-Value: %.10e" % (cindex_test, pvalue_test))
            results.append(cindex_test)
        elif opt.task == 'grad':
            print("[Final] Apply model to training set: Loss: %.10f, Acc: %.4f" % (loss_train, grad_acc_train), file=file)
            logging.info("[Final] Apply model to training set: Loss: %.10f, Acc: %.4f" % (loss_train, grad_acc_train))
            print("[Final] Apply model to testing set: Loss: %.10f, Acc: %.4f" % (loss_test, grad_acc_test), file=file)
            logging.info("[Final] Apply model to testing set: Loss: %.10f, Acc: %.4f" % (loss_test, grad_acc_test))
            results.append(grad_acc_test)
        
print('Split Results:', results, file=file)
print("Average:", np.array(results).mean(), file=file)

