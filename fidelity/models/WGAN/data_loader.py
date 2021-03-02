#!/usr/bin/env python
# coding: utf-8
import torch
from torch.utils import data
import h5py
import numpy as np
from torch.utils.data import Dataset



class HDF5Dataset(data.Dataset):
    def __init__(self, file_path, train_size, transform=None):
        super().__init__()
        self.file_path = file_path
        self.transform = transform
        self.hdf5file = h5py.File(self.file_path, 'r')
        
        if train_size > self.hdf5file['hcal_only']['layers'].shape[0]-1:
            self.train_size = self.hdf5file['hcal_only']['layers'].shape[0]-1
        else:
            self.train_size = train_size
        
        #print('layers shape')
        #print(self.hdf5file['30x30']['layers'].shape)
        #print('energy shape')
        #print(self.hdf5file['30x30']['energy'].shape)
        
        # Search for all h5 files
        
    def __len__(self):
        return self.train_size
        #return self.hdf5file['hcal_only']['layers'][0:self.train_size].shape[0]
             
    def __getitem__(self, index):
        # get data
        x = self.get_data(index)
        if self.transform:
            x = torch.from_numpy(self.transform(x)).float()
        else:
            x = torch.from_numpy(x).float()
        e = torch.from_numpy(self.get_energy(index))
        if torch.sum(x) != torch.sum(x): #checks for NANs
            return sefl.__getitem__(int(np.rand()*self.__len__()))
        else:
            return x, e

            
    def get_data(self, i):
        return self.hdf5file['hcal_only']['layers'][i]
    
    def get_energy(self, i):
        return self.hdf5file['hcal_only']['energy'][i]

    def get_data_range(self, i, j):
        return self.hdf5file['hcal_only']['layers'][i:j]
    
    def get_energy_range(self, i, j):
        return self.hdf5file['hcal_only']['energy'][i:j]

    def get_data_range_tf(self, i, j):
        x = self.get_data_range(i,j)
        print(x.shape)
        #x = np.transpose(x, (1, 2, 3, 0))
        x =self.transform(x)
        print(x.shape)
        return x #np.transpose(x, (2, 0, 1))
    
    
    

#class PionsDataset(data.Dataset):
#    
#    def __init__(self, path_list):
#        self.path_list = path_list
#        part = np.array([])
#        index = np.array([])
#        for i,path in enumerate(path_list):
#            file = h5py.File(path, 'r')['hcal_only/energy']
#            part = np.append(part, np.ones(len(file))*i)
#            index = np.append(index, np.arange(len(file)))
#            
#                
#        self.keys = pd.DataFrame({'part' : part.astype(int),
#                                  'index' : index.astype(int)
#                                 })

#    def __len__(self):
#        return len(self.keys)
    
#    def __getitem__(self, idx):
#        part = self.keys['part'][idx]
#        index = self.keys['index'][idx]
#        path = self.path_list[part]
        
#        file = h5py.File(path, 'r')['hcal_only']
#        energy = file['energy'][index]
#        shower = file['layers'][index]
        
#        energy = energy.reshape(1,1,1,1)
#        shower = np.expand_dims(shower, 0)

#       return {'energy' : energy,
#                'shower' : shower}