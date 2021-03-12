import numpy as np
import argparse
import torch
import array as arr
import torch.utils.data
#from models.WGAN.data_loader import HDF5Dataset
from torch import nn, optim
from torch.nn import functional as F
import models.WGAN.dcgan3Dcore as wgan
import models.WGAN.dcganPhoton as wgan_photon 
#import models.WGAN.dataUtilsCore as DataLoader
import yaml

import time
import os 
import pickle
import random
import json

import matplotlib.pyplot as plt
import matplotlib as mpl
import utils as UTIL

no_cuda=False
cuda = not no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

save_locations = {
    "Full" : '/beegfs/desy/user/eren/data_generator/pion/hcal_only/uniform/pionP6.hdf5',
    "Single50" : '/beegfs/desy/user/eren/data_generator/pion/hcal_only/single/pion50GeV.hdf5',
    "Single20" : '/beegfs/desy/user/eren/data_generator/pion/hcal_only/single/pion20GeV.hdf5',
    "Single80" : '/beegfs/desy/user/eren/data_generator/pion/hcal_only/single/pion80GeV.hdf5'
}

save_locations_photons = {
    "single": '/beegfs/desy/user/eren/WassersteinGAN/data/singleEnergies-corr.hdf5'
}


if __name__ == "__main__":
    
    
    ### Check these parameters very carefully ### 
    baseModelKey = 'wganLO'
    baseModelNN = wgan
    mip_thrs = 0.25
    xbins = 13
    ybins = 13
    layers = 48
    #######################################

    with open(r'model_conf.yaml') as file:
        documents = yaml.full_load(file)
        ngf = documents[baseModelKey][0]['ndf']
        ndf = documents[baseModelKey][1]['ngf']
        LATENT_DIM = documents[baseModelKey][2]['latent_dim']
        prefix = documents[baseModelKey][3]['eph_folders']
        nepoch = documents[baseModelKey][4]['nepochs']
        nshowers = documents[baseModelKey][5]['nshowers']
        particle = documents[baseModelKey][6]['type']

    
    model_WGANLO = baseModelNN.DCGAN_G(ngf,LATENT_DIM).to(device)
    model_WGANLO_aD = baseModelNN.DCGAN_D(ndf).to(device)

    model_WGANLO = nn.DataParallel(model_WGANLO)
    model_WGANLO_aD = nn.DataParallel(model_WGANLO_aD)

    save_Totalfide = []
    save_EnrFide = []
    save_EnrFide20 = []
    save_EnrFide80  = []
    save_nhits = []
    save_eph = []
    save_longt = []
    save_CoGz = []
    save_radial = []

     ## Getting real data
    if particle == 'pion':
        [real_data, real_ener] = UTIL.getRealImagesCore(save_locations['Single50'], nshowers)
        [real_data20, real_ener20] = UTIL.getRealImagesCore(save_locations['Single20'], nshowers)
        [real_data80, real_ener80] = UTIL.getRealImagesCore(save_locations['Single80'], nshowers)
    elif particle == 'photon':
        [real_data, real_ener] = UTIL.getRealImagesPhotons(save_locations_photons['single'], nshowers + 5000, 50.0)
        [real_data20, real_ener20] = UTIL.getRealImagesPhotons(save_locations_photons['single'], nshowers + 5000, 20.0)
        [real_data80, real_ener80] = UTIL.getRealImagesPhotons(save_locations_photons['single'], nshowers + 5000, 80.0)


    esum_real = UTIL.getTotE(real_data, xbins, ybins, layers)
    esum_real20 = UTIL.getTotE(real_data20, xbins, ybins, layers)
    esum_real80 = UTIL.getTotE(real_data80, xbins, ybins, layers)


    hitE_real = UTIL.getHitE(real_data, xbins, ybins, layers)
    hitN_real = UTIL.getOcc(real_data, xbins, ybins, layers)
    cogz_real = UTIL.get0Moment(np.sum(real_data, axis=(2,3)))
    r_real, phi_real, e_real = UTIL.getRadialDistribution(real_data, xbins, ybins, layers)
    spinalE_real = UTIL.getSpinalProfile(real_data, xbins, ybins, layers)

    fidelityRecord = {
        1: {
          'radial': 1.0, 
          'Esum' : [1.0, 1.0, 1.0],
          'CoG': 1.0,
          'Nhit': 1.0,
          'hitE': 1.0,
          'longt': 1.0
        }


    }

    for eph in range(1000,nepoch+500, 500):
        print("Iterations #", eph)
        try:
            weightsGANLO = prefix + '/wgan_itrs_'+ str(eph) + '.pth' 
            checkpointLO = torch.load(weightsGANLO)
        except:
            pass


        model_WGANLO.load_state_dict(checkpointLO['Generator'])
        model_WGANLO_aD.load_state_dict(checkpointLO['Critic'])
        

        showers, energy = UTIL.wGAN_LO(model_WGANLO, model_WGANLO_aD, nshowers, 50, 50, 100, LATENT_DIM, device, mip_thrs, particle)
        showers20, energy20 = UTIL.wGAN_LO(model_WGANLO, model_WGANLO_aD, nshowers, 20, 20, 100, LATENT_DIM, device, mip_thrs, particle)
        showers80, energy80 = UTIL.wGAN_LO(model_WGANLO, model_WGANLO_aD, nshowers, 80, 80, 100, LATENT_DIM, device, mip_thrs, particle)
        
        
        ## Esum
        esum_fake = UTIL.getTotE(showers, xbins, ybins, layers)
        esum_fake20 = UTIL.getTotE(showers20, xbins, ybins, layers)
        esum_fake80 = UTIL.getTotE(showers80, xbins, ybins, layers)

        
        ## HitE
        hitE_fake = UTIL.getHitE(showers, xbins, ybins, layers)
        
        ## Nhits
        hitN_fake = UTIL.getOcc(showers, xbins, ybins, layers)
       
        ## CoGz
        cogz_fake = UTIL.get0Moment(np.sum(showers, axis=(2,3)))

        ## radial Energy
        r_fake, phi_fake, e_fake = UTIL.getRadialDistribution(showers, xbins, ybins, layers)

        ## Longt. Energy
        spinalE_fake = UTIL.getSpinalProfile(showers, xbins, ybins, layers)        


        JSD_singleE = UTIL.jsdHist(esum_real, esum_fake, 50, 100, 2000)
        JSD_singleE20 = UTIL.jsdHist(esum_real20, esum_fake20, 50, 100, 2000)
        JSD_singleE80 = UTIL.jsdHist(esum_real80, esum_fake80, 50, 100, 2000)


        JSD_hitE = UTIL.jsdHist(hitE_real, hitE_fake, 200, 0.2, 500) 
        JSD_nhits = UTIL.jsdHist(hitN_real, hitN_fake, 50, 200, 1000)
        JSD_cogz  = UTIL.jsdHist(cogz_real, cogz_fake, 30, 0, 30)
        JSD_radial = UTIL.jsdHist_radial(r_real, r_fake, e_real, e_fake, 30, 0, 30)
        JSD_spinal = UTIL.jsdHist_spinal(spinalE_real, spinalE_fake, 48)

        fidelityRecord[eph] = {
          'radial': JSD_radial, 
          'Esum' : [JSD_singleE20, JSD_singleE, JSD_singleE80],
          'CoG': JSD_cogz,
          'Nhit': JSD_nhits,
          'hitE': JSD_hitE,
          'longt': JSD_spinal
        }

        #print("Epoch #", eph)
        #print ("SingleE:", JSD_singleE, "\t hitE:", JSD_hitE, "\t Nhits: ", JSD_nhits, "\t Radial:", JSD_radial, 
        #      "\t CoGz: ", JSD_cogz,
        #      "\t Longt. Profile: ", JSD_spinal
        #  )
        totalFid = (JSD_singleE + JSD_nhits + JSD_cogz + JSD_singleE + JSD_radial + JSD_spinal) / 6 
        #print("Total Fidelity: ", totalFid)
        save_Totalfide.append(totalFid)
        save_EnrFide.append(JSD_singleE)
        save_EnrFide20.append(JSD_singleE20)
        save_EnrFide80.append(JSD_singleE80)

        save_nhits.append(JSD_nhits)
        save_longt.append(JSD_spinal)
        save_CoGz.append(JSD_cogz)
        save_radial.append(JSD_radial)
        save_eph.append(eph)

    
    with open('recordJSD'+ particle + '.txt', 'w') as file:
        json.dump(fidelityRecord, file)


    plt.figure(figsize=(12,4), facecolor='none', dpi=200)

  

    #plt.scatter(save_eph, save_Totalfide, color='blue', label='Total Fidelity (Averaged)')
    plt.scatter(save_eph, save_EnrFide, color='red', label='Energy-Sum')
    plt.scatter(save_eph, save_nhits, color='black', label='Number of hits')
    plt.scatter(save_eph, save_radial , color='green', label='Radial Energy')
    plt.scatter(save_eph, save_CoGz , color='magenta', label='CoGz')
    
    plt.legend(loc='upper right', fontsize=10)
    plt.xlabel('iterations', fontsize=14)
    plt.ylabel('fidelity JSD', fontsize=16)
    plt.ylim(0.0,1.1)
    plt.savefig("plots/4fidelity_iterations_"+particle+".png")

    plt.figure(figsize=(12,4), facecolor='none', dpi=200)
    plt.scatter(save_eph, save_EnrFide20, color='salmon', label='Energy-Sum20')
    plt.scatter(save_eph, save_EnrFide, color='red', label='Energy-Sum50')
    plt.scatter(save_eph, save_EnrFide80 , color='peru', label='Energy-Sum80')

    plt.legend(loc='upper right', fontsize=10)
    plt.xlabel('iterations', fontsize=14)
    plt.ylabel('fidelity JSD', fontsize=16)
    plt.ylim(0.0,1.1)
    plt.savefig("plots/3ofKindEsum_iterations_"+particle+".png")

   




