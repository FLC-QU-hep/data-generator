import pkbar
import numpy as np
import argparse
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import array as arr
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from models.WGAN.data_loader import HDF5Dataset
import scipy.spatial.distance as dist
from scipy import stats


def coreCutAna(x):
    return x[:, :, 19:32, 17:30]


def tf_lin_cut_F_coreCut(x):
    x = coreCutAna(x)
    x[x < global_thresh] = 0.0
    return x


def getTotE(data, xbins=13, ybins=13, layers=48):
    data = np.reshape(data,[-1, layers*xbins*ybins])
    etot_arr = np.sum(data, axis=(1))
    return etot_arr

def getHitE(data, xbins=13, ybins=13, layers=48):
    ehit_arr = np.reshape(data,[data.shape[0]*xbins*ybins*layers])
    #etot_arr = np.trim_zeros(etot_arr)
    ehit_arr = ehit_arr[ehit_arr != 0.0]
    return ehit_arr


# Valid for pion showers-core ---> 48 x 13 x 13
def getRealImagesCore(filepath, number):
    dataset_physeval = HDF5Dataset(filepath, transform=tf_lin_cut_F_coreCut, train_size=number)
    data = dataset_physeval.get_data_range_tf(0, number)
    ener = dataset_physeval.get_energy_range(0, number)
    return [data, ener]

def JSDsingle_E(data_real, data_fake, nbins, minE, maxE):
    
    figSE = plt.figure(figsize=(6,6*0.77/0.67))
    axSE = figSE.add_subplot(1,1,1)

    pSEb = axSE.hist(data_real, bins=nbins, range=[minE, maxE], density=None, 
                       weights=np.ones_like(data_real)/(float(len(data_real))) )

    pSEa = axSE.hist(data_fake, bins=nbins, range=None, density=None, 
                       weights=np.ones_like(data_fake)/(float(len(data_fake))))
    frq1 = pSEa[0]
    frq2 = pSEb[0]

    # Jensen Shannon Divergence (JSD)
    if len(frq1) != len(frq2):
        print('ERROR JSD: Histogram bins are not matching!!')
    return dist.jensenshannon(frq1, frq2)

    


def lat_opt_ngd(G,D,z, energy, batch_size, device, alpha=500, beta=0.1, norm=1000):
    
    z.requires_grad_(True)
    x_hat = G(z.cuda(), energy)
    x_hat = x_hat.unsqueeze(1) 
    
    f_z = D(x_hat, energy)

    fz_dz = torch.autograd.grad(outputs=f_z,
                                inputs= z,
                                grad_outputs=torch.ones(f_z.size()).to(device),
                                retain_graph=True,
                                create_graph= True,
                                   )[0]
    
    delta_z = torch.ones_like(fz_dz)
    delta_z = (alpha * fz_dz) / (beta +  torch.norm(delta_z, p=2, dim=0) / norm)
    with torch.no_grad():
        z_prime = torch.clamp(z + delta_z, min=-1, max=1) 
        
    return z_prime


def wGAN(model, number, E_max, E_min, batchsize, fixed_noise, input_energy, device):


    pbar = pkbar.Pbar(name='Generating {} photon showers with energies [{},{}]'.format(number, E_max,E_min), target=number)
    
    fake_list=[]
    energy_list = []
    

    for i in np.arange(batchsize, number+1, batchsize):
        with torch.no_grad():
            fixed_noise.uniform_(-1,1)
            input_energy.uniform_(E_min,E_max)            
            fake = model(fixed_noise, input_energy)
            fake = fake.data.cpu().numpy()
            
            fake_list.append(fake)
            energy_list.append(input_energy.data.cpu().numpy())

            pbar.update(i)
            
            

    energy_full = np.vstack(energy_list)
    fake_full = np.vstack(fake_list)
    fake_full = fake_full.reshape(len(fake_full), 48, 13, 13)

    

    return fake_full, energy_full


def wGAN_LO(model, modelC, number, E_max, E_min, batchsize, latent_dim, device, mip_cut=0.25):


    pbar = pkbar.Pbar(name='Generating {} photon showers with energies [{},{}]'.format(number, E_max,E_min), target=number)
    
    fake_list=[]
    energy_list = []
    

    for i in np.arange(batchsize, number+1, batchsize):
    
        fixed_noise = torch.FloatTensor(batchsize, latent_dim).uniform_(-1, 1)
        fixed_noise = fixed_noise.view(-1, latent_dim, 1,1,1)
        fixed_noise = fixed_noise.to(device)

        input_energy = torch.FloatTensor(batchsize ,1).to(device) 
        input_energy.resize_(batchsize,1,1,1,1).uniform_(E_min, E_max)
           
        z_prime = lat_opt_ngd(model, modelC, fixed_noise, input_energy, batchsize, device)

            
        with torch.no_grad():
            
            fake = model(z_prime, input_energy)
            fake = fake.data.cpu().numpy()
            fake[fake < mip_cut] = 0.0
        
            fake_list.append(fake)
            energy_list.append(input_energy.data.cpu().numpy())

            pbar.update(i)
            
            

    energy_full = np.vstack(energy_list)
    fake_full = np.vstack(fake_list)
    fake_full = fake_full.reshape(len(fake_full), 48, 13, 13)

    print ("\n")
    return fake_full, energy_full

