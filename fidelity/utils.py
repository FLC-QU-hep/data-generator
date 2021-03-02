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
    x[x < 0.25] = 0.0
    return x

def getOcc(data, xbins=13, ybins=13, layers=48):
    data = np.reshape(data,[-1, layers*xbins*ybins])
    occ_arr = (data > 0.0).sum(axis=(1))
    return occ_arr


def getTotE(data, xbins=13, ybins=13, layers=48):
    data = np.reshape(data,[-1, layers*xbins*ybins])
    etot_arr = np.sum(data, axis=(1))
    return etot_arr

def getHitE(data, xbins=13, ybins=13, layers=48):
    ehit_arr = np.reshape(data,[data.shape[0]*xbins*ybins*layers])
    #etot_arr = np.trim_zeros(etot_arr)
    ehit_arr = ehit_arr[ehit_arr != 0.0]
    return ehit_arr

# calculates the center of gravity (as in CALICE) (0st moment)
def get0Moment(x):
    n, l = x.shape
    tiles = np.tile(np.arange(l), (n,1))
    y = x * tiles
    y = y.sum(1)
    y = y/x.sum(1)
    return y

def getSpinalProfile(data, xbins=x_size, ybins=y_size, layers=z_size):
    data = np.reshape(data,[-1, layers, xbins*ybins])
    etot_arr = np.sum(data, axis=(2))
    return etot_arr

def getRadialDistribution(data, xbins, ybins, layers):
    current = np.reshape(data,[-1, layers, xbins,ybins])
    current_sum = np.sum(current, axis=(0,1))
 
    r_list=[]
    phi_list=[]
    e_list=[]
    n_cent_x = (xbins-1)/2.0
    n_cent_y = (ybins-1)/2.0

    for n_x in np.arange(0, xbins):
        for n_y in np.arange(0, ybins):
            if current_sum[n_x,n_y] != 0.0:
                r = np.sqrt((n_x - n_cent_x)**2 + (n_y - n_cent_y)**2)
                r_list.append(r)
                phi = np.arctan((n_x - n_cent_x)/(n_y - n_cent_y))
                phi_list.append(phi)
                e_list.append(current_sum[n_x,n_y])
                
    r_arr = np.asarray(r_list)
    phi_arr = np.asarray(phi_list)
    e_arr = np.asarray(e_list)

    return r_arr, phi_arr, e_arr


# Valid for pion showers-core ---> 48 x 13 x 13
def getRealImagesCore(filepath, number):
    dataset_physeval = HDF5Dataset(filepath, transform=tf_lin_cut_F_coreCut, train_size=number)
    data = dataset_physeval.get_data_range_tf(0, number)
    ener = dataset_physeval.get_energy_range(0, number)
    return [data, ener]

def jsdHist(data_real, data_fake, nbins, minE, maxE):
    
    figSE = plt.figure(figsize=(6,6*0.77/0.67))
    axSE = figSE.add_subplot(1,1,1)

    
    pSEa = axSE.hist(data_real, bins=nbins, range=[minE, maxE])

    pSEb = axSE.hist(data_fake, bins=nbins, range=None)

    frq1 = pSEa[0]
    frq2 = pSEb[0]

    # Jensen Shannon Divergence (JSD)
    if len(frq1) != len(frq2):
        print('ERROR JSD: Histogram bins are not matching!!')
    return dist.jensenshannon(frq1, frq2)

def jsdHist_radial(data_real, data_fake, real_enr, fake_enr, nbins, minE, maxE):
    
    figSE = plt.figure(figsize=(6,6*0.77/0.67))
    axSE = figSE.add_subplot(1,1,1)

    
    pSEa = axSE.hist(data_real, bins=nbins, range=[minE, maxE], weights=real_enr/(float(data_real.shape[0])))

    pSEb = axSE.hist(data_fake, bins=nbins, range=None, weights=fake_enr/(float(fake_enr.shape[0])))

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

