from options import *
import os
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import torch
import csv
import pandas as pd


def print_root(args):
    # get the time
    t = time.strftime('%m_%d-%H_%M', time.localtime(time.time()))
    if args.log_root != 'none':
        os.makedirs(args.log_root + '_(' + t + ')', exist_ok=True)
        f = open(args.log_root + '_(' + t + ')/'+ args.log_name + '_(' + t + ')' + '.txt', 'a')
    else:
        f = sys.stdout
    return f, t


def picture(opt,k,e,m,time):
    if opt.task == "surv":
        plt.figure(k)
        plt.subplot(2, 1, 1)
        train_loss=plt.plot(e, m['train']['loss'], color='red', linestyle='-.')
        test_loss=plt.plot(e, m['test']['loss'], color='blue', linestyle='--')
        plt.title('Loss vs. epoches(train:red)')

        plt.subplot(2, 1, 2)
        train_acc=plt.plot(e, m['train']['cindex'], color='red', linestyle='-.')
        test_acc=plt.plot(e, m['test']['cindex'], color='blue', linestyle='--')
        plt.title('Acc vs. epoches(train:red)')
        
        plt.suptitle(opt.picture_save + "_split" + str(k))
        plt.savefig(opt.log_root +'_(' + time + ')/' + opt.log_name + "_k" + str(k) + '_(' + time + ')' + ".png")
    
    elif opt.task == 'grad':
        plt.figure(k)
        plt.subplot(2, 1, 1)
        train_loss=plt.plot(e, m['train']['loss'], color='red', linestyle='-.')
        test_loss=plt.plot(e, m['test']['loss'], color='blue', linestyle='--')
        plt.title('Loss vs. epoches(train:red)')
        
        plt.subplot(2, 1, 2)
        train_acc=plt.plot(e, m['train']['grad_acc'], color='red', linestyle='-.')
        test_acc=plt.plot(e, m['test']['grad_acc'], color='blue', linestyle='--')
        plt.title('Acc vs. epoches(train:red)')
        
        plt.suptitle(opt.picture_save + "_split" + str(k))
        plt.savefig(opt.log_root +'_(' + time + ')/' + opt.log_name + "_k" + str(k) + '_loss_acc' + '_(' + time + ')' + ".png")
        
        plt.figure(k+15)
        plt.subplot(2, 1, 1)
        train_acc=plt.plot(e, m['train']['auc'], color='red', linestyle='-.')
        test_acc=plt.plot(e, m['test']['auc'], color='blue', linestyle='--')
        plt.title('auc vs. epoches(train:red)')
        
        plt.subplot(2, 1, 2)
        train_acc=plt.plot(e, m['train']['ap'], color='red', linestyle='-.')
        test_acc=plt.plot(e, m['test']['ap'], color='blue', linestyle='--')
        plt.title('ap vs. epoches(train:red)')
        
        plt.suptitle(opt.picture_save + "_split" + str(k))
        plt.savefig(opt.log_root +'_(' + time + ')/' + opt.log_name + "_k" + str(k) + '_auc_ap' + '_(' + time + ')' + ".png")
        
        plt.figure(k+30)
        plt.subplot(2, 1, 1)
        train_acc=plt.plot(e, m['train']['f1'], color='red', linestyle='-.')
        test_acc=plt.plot(e, m['test']['f1'], color='blue', linestyle='--')
        plt.title('f1 vs. epoches(train:red)')
        
        plt.subplot(2, 1, 2)
        train_acc=plt.plot(e, m['train']['f1_IV'], color='red', linestyle='-.')
        test_acc=plt.plot(e, m['test']['f1_IV'], color='blue', linestyle='--')
        plt.title('f1_IV vs. epoches(train:red)')
        
        plt.suptitle(opt.picture_save + "_split" + str(k))
        plt.savefig(opt.log_root +'_(' + time + ')/' + opt.log_name + "_k" + str(k) + '_f1' + '_(' + time + ')' + ".png")


def model_save(args,model,optimizer, epoch, k, time, pred_test):
    if args.log_root != 'none':
        os.makedirs(args.log_root + '_(' + time + ')/checkpoint', exist_ok=True)
        pred = pred_test if args.save_pred == 1 else None
        torch.save({
            'k': k,
            'epoch': epoch,  
            'model_state_dict': model.state_dict(),  
            'optimizer_state_dict': optimizer.state_dict(),  
            'task': args.task,  
            'pred': pred
        }, args.log_root + '_(' + time + ')/checkpoint/'+ "_k" + str(k) +'_epoch' + str(epoch) + ".pt")

        
def value_save(result, args, time):
    if args.log_root != 'none':
        fileName = args.log_root + '_(' + time + ')/' + args.log_name + '_(' + time + ')' + ".xlsx"
        df = pd.DataFrame()
        df['epoch']=result['epoch']
        df['train_acc']=result['train']['acc']
        df['test_acc']=result['test']['acc']
        df['val_acc']=result['val']['acc']
        df.to_excel(fileName)

def best_save(args,model,optimizer,epoch,k,time, pred_test):
    if args.log_root != 'none':
        os.makedirs(args.log_root + '_(' + time + ')/best_checkpoint', exist_ok=True)
        pred = pred_test if args.save_pred == 1 else None
        torch.save({
            'k':k,
            'epoch': epoch,  
            'model_state_dict': model.state_dict(),  
            'optimizer_state_dict': optimizer.state_dict(),  
            'task': args.task, 
            'pred': pred
        }, args.log_root + '_(' + time + ')/best_checkpoint/'+ "_k" + str(k) +'_epoch' + str(epoch) + ".pt")





