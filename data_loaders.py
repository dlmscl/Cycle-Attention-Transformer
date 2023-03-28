### data_loaders.py
import os

import numpy as np
import pandas as pd
from PIL import Image
from sklearn import preprocessing

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset  # For custom datasets
from torchvision import datasets, transforms
from torch_geometric.data import Data  # lsc 7.28 18:00
from utils import random_noise, white_noise, part_noise

################
# Dataset Loader
################
class PathgraphomicDatasetLoader(Dataset):
    def __init__(self, opt, data, split, mode='omic'):
        """
        Args:
            X = data
            e = overall survival event
            t = overall survival in months
        """

        self.X_path = data[split]['x_path']
        self.X_grph = data[split]['x_grph']
        self.X_omic = data[split]['x_omic']
        self.e = data[split]['e']
        self.t = data[split]['t']
        self.g = data[split]['g']
        
        self.fail = []
        self.mode = mode
        self.data_name = opt.dataroot
        
        self.transforms = transforms.Compose([
                            transforms.RandomHorizontalFlip(0.5),
                            transforms.RandomVerticalFlip(0.5),
                            transforms.RandomCrop(opt.input_size_path),
                            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.01),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __getitem__(self, index):
        single_e = torch.tensor(self.e[index]).type(torch.FloatTensor)
        single_t = torch.tensor(self.t[index]).type(torch.FloatTensor)
        single_g = torch.tensor(self.g[index]).type(torch.LongTensor)

        if self.mode == "path" or self.mode == 'path_effnet':
            single_X_path = Image.open(self.X_path[index]).convert('RGB')
            return (self.transforms(single_X_path), 0, 0, single_e, single_t, single_g)
        elif self.mode == "graph":
            if self.data_name = './data/TCGA_GBMLGG':
                single_X_grph = torch.load('../pathomicFusion_data/graph/'+ self.X_grph[index][30:])
            elif self.data_name = './data/TCGA_KIRC':
                single_X_grph = torch.load('../pathomicFusion_data/graph_KIRC/'+ self.X_grph[index][35:])
            return (0, single_X_grph, 0, single_e, single_t, single_g)
        elif self.mode == "omic":
            single_X_omic = torch.tensor(self.X_omic[index]).type(torch.FloatTensor)
            return (0, 0, single_X_omic, single_e, single_t, single_g)
        elif self.mode == "pathomic":
            single_X_path = Image.open(self.X_path[index]).convert('RGB')
            single_X_omic = torch.tensor(self.X_omic[index]).type(torch.FloatTensor)
            return (self.transforms(single_X_path), 0, single_X_omic, single_e, single_t, single_g)
        elif self.mode == "graphomic":
            if self.data_name = './data/TCGA_GBMLGG':
                single_X_grph = torch.load('../pathomicFusion_data/graph/'+ self.X_grph[index][30:])
            elif self.data_name = './data/TCGA_KIRC':
                single_X_grph = torch.load('../pathomicFusion_data/graph_KIRC/'+ self.X_grph[index][35:])
            single_X_omic = torch.tensor(self.X_omic[index]).type(torch.FloatTensor)
            return (0, single_X_grph, single_X_omic, single_e, single_t, single_g)
        elif self.mode == "pathgraph":
            single_X_path = Image.open(self.X_path[index]).convert('RGB')
            if self.data_name = './data/TCGA_GBMLGG':
                single_X_grph = torch.load('../pathomicFusion_data/graph/'+ self.X_grph[index][30:])
            elif self.data_name = './data/TCGA_KIRC':
                single_X_grph = torch.load('../pathomicFusion_data/graph_KIRC/'+ self.X_grph[index][35:])
            return (self.transforms(single_X_path), single_X_grph, 0, single_e, single_t, single_g)
        elif self.mode == "pathgraphomic":
            single_X_path = Image.open(self.X_path[index]).convert('RGB')
            if self.data_name = './data/TCGA_GBMLGG':
                single_X_grph = torch.load('../pathomicFusion_data/graph/'+ self.X_grph[index][30:])
            elif self.data_name = './data/TCGA_KIRC':
                single_X_grph = torch.load('../pathomicFusion_data/graph_KIRC/'+ self.X_grph[index][35:])
            single_X_omic = torch.tensor(self.X_omic[index]).type(torch.FloatTensor)
            return (self.transforms(single_X_path), single_X_grph, single_X_omic, single_e, single_t, single_g)

    def __len__(self):
        return len(self.X_grph)


class PathgraphomicFastDatasetLoader(Dataset):
    def __init__(self, opt, data, split, mode='omic'):
        """
        Args:
            X = data
            e = overall survival event
            t = overall survival in months
        """
        self.X_path = data[split]['x_path']
        self.X_grph = data[split]['x_grph']
        self.X_omic = data[split]['x_omic']
        self.e = data[split]['e']
        self.t = data[split]['t']
        self.g = data[split]['g']
        self.mode = mode

    def __getitem__(self, index):
        single_e = torch.tensor(self.e[index]).type(torch.FloatTensor)
        single_t = torch.tensor(self.t[index]).type(torch.FloatTensor)
        single_g = torch.tensor(self.g[index]).type(torch.LongTensor)

        if self.mode == "path" or self.mode == 'pathpath' or self.mode == 'path_Unet':
            single_X_path = torch.tensor(self.X_path[index]).type(torch.FloatTensor).squeeze(0)
            return (single_X_path, 0, 0, single_e, single_t, single_g)
        elif self.mode == "graph" or self.mode == 'graphgraph':
            key1 = self.X_grph[index][-4:-3]
            if key1 == '0':
                key2 = self.X_grph[index][-6:-5]
                if key2 == '0':
                    end = -7
                else:
                    end = -9
            else:
                key2 = self.X_grph[index][-8:-7]
                if key2 == '0':
                    end = -9
                else:
                    end = -11
            single_X_grph = torch.load('../pathomicFusion_data/graph/'+ self.X_grph[index][42:end] + '.pt')
            return (0, single_X_grph, 0, single_e, single_t, single_g)
        elif self.mode == "omic" or self.mode == 'omicomic':
            single_X_omic = torch.tensor(self.X_omic[index]).type(torch.FloatTensor)
            return (0, 0, single_X_omic, single_e, single_t, single_g)
        elif self.mode == "pathomic":
            single_X_path = torch.tensor(self.X_path[index]).type(torch.FloatTensor).squeeze(0)
            single_X_omic = torch.tensor(self.X_omic[index]).type(torch.FloatTensor)
            return (single_X_path, 0, single_X_omic, single_e, single_t, single_g)
        elif self.mode == "graphomic":
            key1 = self.X_grph[index][-4:-3]
            if key1 == '0':
                key2 = self.X_grph[index][-6:-5]
                if key2 == '0':
                    end = -7
                else:
                    end = -9
            else:
                key2 = self.X_grph[index][-8:-7]
                if key2 == '0':
                    end = -9
                else:
                    end = -11
            single_X_grph = torch.load('../pathomicFusion_data/graph/'+ self.X_grph[index][42:end] + '.pt')
            single_X_omic = torch.tensor(self.X_omic[index]).type(torch.FloatTensor)
            return (0, single_X_grph, single_X_omic, single_e, single_t, single_g)
        elif self.mode == "pathgraph":
            single_X_path = torch.tensor(self.X_path[index]).type(torch.FloatTensor).squeeze(0)
            single_X_grph = torch.load(self.X_grph[index])
            return (single_X_path, single_X_grph, 0, single_e, single_t, single_g)
        elif self.mode == "pathgraphomic":
            single_X_path = torch.tensor(self.X_path[index]).type(torch.FloatTensor).squeeze(0)
            key1 = self.X_grph[index][-4:-3]
            if key1 == '0':
                key2 = self.X_grph[index][-6:-5]
                if key2 == '0':
                    end = -7
                else:
                    end = -9
            else:
                key2 = self.X_grph[index][-8:-7]
                if key2 == '0':
                    end = -9
                else:
                    end = -11
            single_X_grph = torch.load('../pathomicFusion_data/graph/'+ self.X_grph[index][42:end] + '.pt')
            single_X_omic = torch.tensor(self.X_omic[index]).type(torch.FloatTensor)
            return (single_X_path, single_X_grph, single_X_omic, single_e, single_t, single_g)

    def __len__(self):
        return len(self.X_path)

    
class PathgraphomicDatasetLoader_Augmentation(Dataset):
    def __init__(self, opt, data, split, mode='omic'):
        """
        Args:
            X = data
            e = overall survival event
            t = overall survival in months
        """
        self.X_path = data[split]['x_path']
        self.X_grph = data[split]['x_grph']
        self.X_omic = data[split]['x_omic']
        self.e = data[split]['e']
        self.t = data[split]['t']
        self.g = data[split]['g']
        
        self.mode = mode
    
        self.transforms=[]
        
        
        self.transforms = transforms.Compose([
                            transforms.RandomHorizontalFlip(0.5),
                            transforms.RandomVerticalFlip(0.5),
#                             transforms.RandomCrop(opt.input_size_path),
                            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.01),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __getitem__(self, index):
        step=(index+1)//(len(self.e))+1
        if (index+1)/(len(self.e))<25:
            index=index%(len(self.e)-1)
        else:
            raise StopIteration
        single_e = torch.tensor(self.e[index]).type(torch.FloatTensor)
        single_t = torch.tensor(self.t[index]).type(torch.FloatTensor)
        single_g = torch.tensor(self.g[index]).type(torch.LongTensor)
        
        if self.mode == "path" or self.mode == 'path_effnet':
            single_X_path = Image.open(self.X_path[index]).convert('RGB')
            w=step%5
            h=step//5
            single_X_path=torchvision.transforms.functional.crop(single_X_path,w*160,h*160,224,224)
            return (self.transforms(single_X_path), 0, 0, single_e, single_t, single_g)
        
        elif self.mode == "graph":
            if self.data_name = './data/TCGA_GBMLGG':
                single_X_grph = torch.load('../pathomicFusion_data/graph/'+ self.X_grph[index][30:])
            elif self.data_name = './data/TCGA_KIRC':
                single_X_grph = torch.load('../pathomicFusion_data/graph_KIRC/'+ self.X_grph[index][35:])
            return (0, single_X_grph, 0, single_e, single_t, single_g)
           
        elif self.mode == "omic":
            single_X_omic = torch.tensor(self.X_omic[index]).type(torch.FloatTensor)
            return (0, 0, single_X_omic, single_e, single_t, single_g)
        elif self.mode == "pathomic":
            single_X_path = Image.open(self.X_path[index]).convert('RGB')
            w=step%5
            h=step//5
            single_X_path=torchvision.transforms.functional.crop(single_X_path,w*160,h*160,224,224)
            single_X_omic = torch.tensor(self.X_omic[index]).type(torch.FloatTensor)
            return (self.transforms(single_X_path), 0, single_X_omic, single_e, single_t, single_g)
        elif self.mode == "graphomic":
            single_X_grph = torch.load(self.X_grph[index])
            single_X_omic = torch.tensor(self.X_omic[index]).type(torch.FloatTensor)
            return (0, single_X_grph, single_X_omic, single_e, single_t, single_g)
        elif self.mode == "pathgraph":
            single_X_path = Image.open(self.X_path[index]).convert('RGB')
            single_X_grph = torch.load(self.X_grph[index])
            return (self.transforms(single_X_path), single_X_grph, 0, single_e, single_t, single_g)
        elif self.mode == "pathgraphomic":
            single_X_path = Image.open(self.X_path[index]).convert('RGB')
            w=step%5
            h=step//5
            single_X_path=torchvision.transforms.functional.crop(single_X_path,w*160,h*160,224,224)
            if self.data_name = './data/TCGA_GBMLGG':
                single_X_grph = torch.load('../pathomicFusion_data/graph/'+ self.X_grph[index][30:])
            elif self.data_name = './data/TCGA_KIRC':
                single_X_grph = torch.load('../pathomicFusion_data/graph_KIRC/'+ self.X_grph[index][35:])
            single_X_omic = torch.tensor(self.X_omic[index]).type(torch.FloatTensor)
            return (self.transforms(single_X_path), single_X_grph, single_X_omic, single_e, single_t, single_g)

    def __len__(self):
        return len(self.X_path)*25