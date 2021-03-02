import pkbar
import numpy as np
import argparse
import torch
import array as arr
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F

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


    return fake_full, energy_full

