import random
from tqdm import tqdm
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import RandomSampler
import io
import sys
import time
import matplotlib.pyplot as plt
from data_loaders import PathgraphomicDatasetLoader, PathgraphomicFastDatasetLoader,  PathgraphomicDatasetLoader_Augmentation
from networks import define_net, define_reg, define_optimizer, define_scheduler
from utils import unfreeze_unimodal, CoxLoss, CIndex_lifeline, cox_log_rank, accuracy_cox, mixed_collate, count_parameters, makeAUROCPlot
from Save import model_save, best_save, picture
from utils import GradMetrics,CosineSimilarityLoss

#from GPUtil import showUtilization as gpu_usage
import pdb
import pickle
import os

def train(opt, data, device, k, file, time):
    cudnn.deterministic = True
    torch.cuda.manual_seed_all(2019)
    torch.manual_seed(2019)
    np.random.seed(2019)
    random.seed(2019)
   
    model = define_net(opt, k)
    optimizer = define_optimizer(opt, model)
    scheduler = define_scheduler(opt, optimizer)
    cos_loss = CosineSimilarityLoss()

    print("Number of Trainable Parameters: %d" % count_parameters(model), file=file)
    print("Activation Type:", opt.act_type, file=file)
    print("Optimizer Type:", opt.optimizer_type, file=file)
    print("Regularization Type:", opt.reg_type, file=file)

    use_patch, roi_dir = ('_patch_', 'all_st_patches_512') if opt.use_vgg_features else ('_', 'all_st')

    if opt.data_Augmentation:
        custom_data_loader = PathgraphomicDatasetLoader_Augmentation(opt, data, split='train', mode=opt.mode)
    else:
        custom_data_loader = PathgraphomicFastDatasetLoader(opt, data, split='train', mode=opt.mode) if opt.use_vgg_features else PathgraphomicDatasetLoader(opt, data, split='train', mode=opt.mode)
    
    
    train_loader = torch.utils.data.DataLoader(dataset=custom_data_loader, batch_size=opt.batch_size, shuffle=True, collate_fn=mixed_collate)

    
    metric_logger = {'train':{'loss':[], 'pvalue':[], 'cindex':[], 'surv_acc':[], 'grad_acc':[], 'auc':[], 'ap':[], 'f1':[], 'f1_IV':[]},
                      'test':{'loss':[], 'pvalue':[], 'cindex':[], 'surv_acc':[], 'grad_acc':[], 'auc':[], 'ap':[], 'f1':[], 'f1_IV':[]}}
    
    
    epoch_list=[]
    for epoch in tqdm(range(opt.epoch_count, opt.niter+opt.niter_decay+1), file=file):
        
        epoch_list.append(epoch)
        if opt.finetune == 1:
            unfreeze_unimodal(opt, model, epoch, file)

        model.train()
        risk_pred_all, censor_all, survtime_all = np.array([]), np.array([]), np.array([])    # Used for calculating the C-Index
        loss_epoch, grad_acc_epoch = 0, 0
        
        for batch_idx, (x_path, x_grph, x_omic, censor, survtime, grade) in enumerate(train_loader):

            censor = censor.to(device) if "surv" in opt.task else censor
            grade = grade.to(device) if "grad" in opt.task else grade
            
            feature, pred = model(x_path=x_path.to(device), x_grph=x_grph.to(device), x_omic=x_omic.to(device))

            loss_cox = CoxLoss(survtime, censor, pred, device) if opt.task == "surv" else 0
            loss_reg = define_reg(opt, model)
            loss_nll = F.nll_loss(pred, grade) if opt.task == "grad" else 0
            loss = opt.lambda_cox*loss_cox + opt.lambda_nll*loss_nll + opt.lambda_reg*loss_reg
            loss_epoch += loss.data.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(' ', file=file)
        print("current opt is:", file=file)
        print(optimizer.param_groups[0]['lr'], file=file)
        print("__________________________", file=file)
        scheduler.step()

        if opt.measure or epoch == (opt.niter+opt.niter_decay - 1):
            loss_epoch, cindex_epoch, pvalue_epoch, surv_acc_epoch, grad_acc_epoch, pred_epoch, auc_epoch, ap_epoch, f1_epoch, f1_IV_epoch = test(opt, model, data, 'train', device)
            loss_test, cindex_test, pvalue_test, surv_acc_test, grad_acc_test, pred_test, auc_test, ap_test, f1_test, f1_IV_test = test(opt, model, data, 'test', device)

            metric_logger['train']['loss'].append(loss_epoch)
            metric_logger['train']['cindex'].append(cindex_epoch)
            metric_logger['train']['pvalue'].append(pvalue_epoch)
            metric_logger['train']['surv_acc'].append(surv_acc_epoch)
            metric_logger['train']['grad_acc'].append(grad_acc_epoch)
            metric_logger['train']['auc'].append(auc_epoch)
            metric_logger['train']['ap'].append(ap_epoch)
            metric_logger['train']['f1'].append(f1_epoch)
            metric_logger['train']['f1_IV'].append(f1_IV_epoch)

            metric_logger['test']['loss'].append(loss_test)
            metric_logger['test']['cindex'].append(cindex_test)
            metric_logger['test']['pvalue'].append(pvalue_test)
            metric_logger['test']['surv_acc'].append(surv_acc_test)
            metric_logger['test']['grad_acc'].append(grad_acc_test)
            metric_logger['test']['auc'].append(auc_test)
            metric_logger['test']['ap'].append(ap_test)
            metric_logger['test']['f1'].append(f1_test)
            metric_logger['test']['f1_IV'].append(f1_IV_test)

            if opt.verbose > 0:
                if opt.task == 'surv':
                    print('[{:s}]\t\tLoss: {:.4f}, {:s}: {:.4f}'.format('Train', loss_epoch, 'C-Index', cindex_epoch), file=file)
                    print('[{:s}]\t\tLoss: {:.4f}, {:s}: {:.4f}\n'.format('Test', loss_test, 'C-Index', cindex_test), file=file)
                elif opt.task == 'grad':
                    print('[{:s}]\t\tLoss: {:.4f}, {:s}: {:.4f}, {:s}: {:.4f}, {:s}: {:.4f}, {:s}: {:.4f}, {:s}: {:.4f}'.format('Train', loss_epoch, 'Accuracy', grad_acc_epoch, 'AUC', auc_epoch, 'AP', ap_epoch, 'f1', f1_epoch, 'f1_IV', f1_IV_epoch), file=file)
                    print('[{:s}]\t\tLoss: {:.4f}, {:s}: {:.4f}, {:s}: {:.4f}, {:s}: {:.4f}, {:s}: {:.4f}, {:s}: {:.4f}\n'.format('Test', loss_test, 'Accuracy', grad_acc_test, 'AUC', auc_test, 'AP', ap_test, 'f1', f1_test, 'f1_IV', f1_IV_test), file=file)

        
        if opt.is_picture:        
            picture(opt,k,epoch_list,metric_logger,time)
            
    
        if opt.save_best_function==1 and opt.task == 'grad':
            if epoch<opt.begin_select_epoch:
                if epoch % opt.save_epoch == 0:
                    model_save(opt, model, optimizer, epoch, k, time, pred_test)
            elif epoch==opt.begin_select_epoch:
                best_test_acc=auc_test
                best_epoch=epoch
                best_save(opt, model, optimizer, epoch, k, time, pred_test)
            elif epoch>opt.begin_select_epoch and auc_test-best_test_acc>=opt.update_gate:
                best_test_acc=auc_test
                best_epoch=epoch
                best_save(opt, model, optimizer, epoch, k, time, pred_test)
            elif epoch>opt.begin_select_epoch and auc_test-best_test_acc > 0 and epoch - best_epoch >=opt.save_across_epoch:
                best_test_acc=auc_test
                best_epoch=epoch
                best_save(opt, model, optimizer, epoch, k, time, pred_test)
        elif opt.save_best_function==1 and opt.task == 'surv':
            if epoch<opt.begin_select_epoch:
                if epoch % opt.save_epoch == 0:
                    model_save(opt, model, optimizer, epoch, k, time, pred_test)
            elif epoch==opt.begin_select_epoch or (opt.pretrained_root != 'none' and epoch == checkpoint['epoch']):
                best_test_acc=cindex_test
                best_epoch=epoch
                best_save(opt, model, optimizer, epoch, k, time, pred_test)
            elif epoch>opt.begin_select_epoch and cindex_test-best_test_acc>=opt.update_gate:
                best_test_acc=cindex_test
                best_epoch=epoch
                best_save(opt, model, optimizer, epoch, k, time, pred_test)
            elif epoch>opt.begin_select_epoch and cindex_test-best_test_acc > 0 and epoch - best_epoch >=opt.save_across_epoch:
                best_test_acc=cindex_test
                best_epoch=epoch
                best_save(opt, model, optimizer, epoch, k, time, pred_test)
        else:
            if epoch % opt.save_epoch == 0:
                model_save(opt, model, optimizer, epoch, k, time, pred_test)
        
    return model, optimizer, metric_logger


def test(opt, model, data, split, device):
    model.eval()

    if opt.data_Augmentation:
        custom_data_loader = PathgraphomicDatasetLoader_Augmentation(opt, data, split, mode=opt.mode)
    else:
        custom_data_loader = PathgraphomicFastDatasetLoader(opt, data, split, mode=opt.mode) if opt.use_vgg_features else PathgraphomicDatasetLoader(opt, data, split=split, mode=opt.mode)

    test_loader = torch.utils.data.DataLoader(dataset=custom_data_loader, batch_size=opt.batch_size, shuffle=False, collate_fn=mixed_collate)
    
    risk_pred_all, censor_all, survtime_all = np.array([]), np.array([]), np.array([])
    probs_all, gt_all = None, np.array([])
    loss_test, grad_acc_test = 0, 0
    
    with torch.no_grad():
        for batch_idx, (x_path, x_grph, x_omic, censor, survtime, grade) in enumerate(test_loader):

            censor = censor.to(device) if "surv" in opt.task else censor
            grade = grade.to(device) if "grad" in opt.task else grade
            feature, pred = model(x_path=x_path.to(device), x_grph=x_grph.to(device), x_omic=x_omic.to(device))
             

            loss_cox = CoxLoss(survtime, censor, pred, device) if opt.task == "surv" else 0
            loss_reg = define_reg(opt, model)
            loss_nll = F.nll_loss(pred, grade) if opt.task == "grad" else 0
            loss = opt.lambda_cox*loss_cox + opt.lambda_nll*loss_nll + opt.lambda_reg*loss_reg
            loss_test += loss.data.item()

            gt_all = np.concatenate((gt_all, grade.detach().cpu().numpy().reshape(-1)))   # Logging Information

            if opt.task == "surv":
                risk_pred_all = np.concatenate((risk_pred_all, pred.detach().cpu().numpy().reshape(-1)))   # Logging Information
                censor_all = np.concatenate((censor_all, censor.detach().cpu().numpy().reshape(-1)))   # Logging Information
                survtime_all = np.concatenate((survtime_all, survtime.detach().cpu().numpy().reshape(-1)))   # Logging Information
            elif opt.task == "grad":
                grade_pred = pred.argmax(dim=1, keepdim=True)
                grad_acc_test += grade_pred.eq(grade.view_as(grade_pred)).sum().item()
                probs_np = pred.detach().cpu().numpy()
                probs_all = probs_np if probs_all is None else np.concatenate((probs_all, probs_np), axis=0)   # Logging Information

            del x_path
            del x_grph
            del x_omic
        ################################################### 
        # ==== Measuring Test Loss, C-Index, P-Value ==== #
        ###################################################
        loss_test /= len(test_loader)
        cindex_test = CIndex_lifeline(risk_pred_all, censor_all, survtime_all) if opt.task == 'surv' else None
        pvalue_test = cox_log_rank(risk_pred_all, censor_all, survtime_all) if opt.task == 'surv' else None
        surv_acc_test = accuracy_cox(risk_pred_all, censor_all) if opt.task == 'surv' else None
        grad_acc_test = grad_acc_test / len(test_loader.dataset) if opt.task == 'grad' else None
        pred_test = [risk_pred_all, survtime_all, censor_all, probs_all, gt_all]

        indicator = GradMetrics(pred_test, model='pathomic_aug', split='test', avg='micro') if opt.task == 'grad' else None
        auc_test = float(indicator[0][0:6]) if opt.task == 'grad' else None
        ap_test = float(indicator[1][0:6]) if opt.task == 'grad' else None
        f1_test = float(indicator[2][0:6]) if opt.task == 'grad' else None
        f1_IV_test = float(indicator[3][0:6]) if opt.task == 'grad' else None
        
        return loss_test, cindex_test, pvalue_test, surv_acc_test, grad_acc_test, pred_test, auc_test, ap_test, f1_test, f1_IV_test

        