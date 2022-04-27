import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from nlb_tools.nwb_interface import NWBDataset
from nlb_tools.make_tensors import make_train_input_tensors, make_eval_input_tensors



class nlb_data(Dataset):
    def __init__(self, fpath, name, split='train'):
        super(nlb_data, self).__init__()
        
        # curr_path = os.getcwd()
        # fpath = curr_path + '/000140/sub-Jenkins/'
        self.dataset = NWBDataset(fpath=fpath)
        self.dataset.resample(5)
        self.train_dict = make_train_input_tensors(dataset=self.dataset, 
                                      dataset_name=name, 
                                      trial_split=split, # trial_split=['train', 'val'], for Test phase
                                      save_file=False, 
                                      include_forward_pred=True)
        print("nlb dataset loaded...!")

    def __len__(self):
        return self.train_dict['train_spikes_heldin'].shape[0]
    
    def __getitem__(self, idx):
        
        a = torch.tensor(
                        np.concatenate([
                            self.train_dict['train_spikes_heldin'][idx], 
                            np.zeros(self.train_dict['train_spikes_heldin_forward'][idx].shape), # zeroed inputs for forecasting
                        ], axis=0), dtype=torch.float32)
        b = torch.tensor(
                        np.concatenate([
                            np.concatenate([
                                self.train_dict['train_spikes_heldin'][idx],
                                self.train_dict['train_spikes_heldin_forward'][idx],
                            ], axis=0),
                            np.concatenate([
                                self.train_dict['train_spikes_heldout'][idx],
                                self.train_dict['train_spikes_heldout_forward'][idx],
                            ], axis=0),
                        ], axis=1), dtype=torch.float32)
        # b = torch.tensor(self.train_dict['train_spikes_heldout'][idx], dtype=torch.float32)
        # c = torch.tensor(self.train_dict['train_spikes_heldin_forward'][idx], dtype=torch.float32)
        # d = torch.tensor(self.train_dict['train_spikes_heldout_forward'][idx], dtype=torch.float32)
        
        # e = torch.cat((torch.cat((a,b), dim=1), torch.cat((c,d), dim=1)), dim=0)


        return a, b


        






class fashionMNIST():
    def __init__(self, dir, download, train):
        #self.transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x/255.0)])#transforms.Normalize((0.5,),(0.5,))])
        self.transform = transforms.ToTensor()
        self.dataset = datasets.FashionMNIST(dir, download=download, train=train, transform=self.transform)

    def load_data(self, batch_size, shuffle=True):
        loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle)
        
        return loader

class MNIST():
    def __init__(self, dir, download, train):
        self.transform = transforms.ToTensor()
        self.dataset = datasets.MNIST(dir, download=download, train=train, transform=self.transform)

    def load_data(self, batch_size, shuffle=True):
        loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle)
        
        return loader

class CelebA():
    def __init__(self, dir, download, split):
        self.transform = transforms.ToTensor()
        self.dataset = datasets.CelebA(dir, download=download, split=split)

    def load_data(self, batch_size, shuffle=True):
        loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle, 
        collate_fn=lambda x: self.pre_process(x))

        return loader

    def pre_process(self, data):
        images = []
        labels = []
        for x, y in data:
            x = self.transform(x)
            src = x.numpy().transpose(1,2,0)
            im = cv2.resize(src, (128,128))
            im = self.transform(im)
            
            images.append(im)
        
            labels.append(y)
        batch_images = torch.stack(images)
        batch_labels = torch.stack(labels)
        return batch_images, batch_labels
        
        
