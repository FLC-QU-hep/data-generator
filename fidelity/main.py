import numpy as np
import argparse
import torch
import array as arr
import torch.utils.data
from models.WGAN.data_loader import HDF5Dataset
from torch import nn, optim
from torch.nn import functional as F
import models.WGAN.dcgan3Dcore as wgan
#import models.WGAN.dataUtilsCore as DataLoader
import yaml

import time
import os 
import pickle
import random

import matplotlib.pyplot as plt
import matplotlib as mpl
import utils as UTIL

no_cuda=False
cuda = not no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

save_locations = {
    "Full" : '/beegfs/desy/user/eren/data_generator/pion/hcal_only/uniform/pionP6.hdf5',
    "Single50" : '/beegfs/desy/user/eren/data_generator/pion/hcal_only/single/pion50GeV.hdf5',
}


def get_parser():
    parser = argparse.ArgumentParser(
        description='Generation and Fidelity Score',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--nbsize', action='store',
                        type=int, default=1000,
                        help='Batch size for generation')

    parser.add_argument('--model', action='store',
                        type=str, default="wgan",
                        help='type of model (bib-ae , wgan or gan)')

    parser.add_argument('--output', action='store',
                        type=str, help='Name of the output file')

    parser.add_argument('--ephPrefix', action='store',
                        type=str, help='')


    return parser



#####

if __name__ == "__main__":

    parser = get_parser()
    parse_args = parser.parse_args() 
    path_prefix = parse_args.ephPrefix


    ## Getting real data
    [real_data, real_ener] = UTIL.getRealImagesCore(save_locations['Single50'], 5000)
    esum_real = UTIL.getTotE(real_data)
    hitE_real = UTIL.getHitE(real_data)
    hitN_real = UTIL.getOcc(real_data)
    cogz_real = UTIL.get0Moment(np.sum(real_data, axis=(2,3)))
    r_real, phi_real, e_real = UTIL.getRadialDistribution(real_data, xbins=13, ybins=13, layers=48)


    ### Testing Fidelity for MODEL ----> WGAN-LO ### 
    
    with open(r'model_conf.yaml') as file:
        documents = yaml.full_load(file)
        ngf = documents['wganLO'][0]['ndf']
        ndf = documents['wganLO'][1]['ngf']
        LATENT_DIM = documents['wganLO'][2]['latent_dim']
        prefix = documents['wganLO'][3]['eph_folders']
        nepoch = documents['wganLO'][4]['nepochs']
        nshowers = documents['wganLO'][5]['nshowers']

    model_WGANLO = wgan.DCGAN_G(ngf,LATENT_DIM).to(device)
    model_WGANLO_aD = wgan.DCGAN_D(ndf).to(device)
    model_WGANLO = nn.DataParallel(model_WGANLO)
    model_WGANLO_aD = nn.DataParallel(model_WGANLO_aD)


    for eph in range(1,nepoch+1):
        weightsGANLO = prefix + '/wgan_'+ str(eph) + '.pth' 
        checkpointLO = torch.load(weightsGANLO)
        model_WGANLO.load_state_dict(checkpointLO['Generator'])
        model_WGANLO_aD.load_state_dict(checkpointLO['Critic'])
        showers, energy = UTIL.wGAN_LO(model_WGANLO, model_WGANLO_aD, nshowers, 50, 50, 100, LATENT_DIM, device, mip_cut=0.25)
        
        ## Esum
        esum_fake = UTIL.getTotE(showers) 
        
        ## HitE
        hitE_fake = UTIL.getHitE(showers)
        
        ## Nhits
        hitN_fake = UTIL.getOcc(showers)
       
        ## CoGz
        cogz_fake = UTIL.get0Moment(np.sum(showers, axis=(2,3)))

        ## radial Energy
        r_fake, phi_fake, e_fake = UTIL.getRadialDistribution(showers, xbins=13, ybins=13, layers=48)

        ## Longt. Energy
                


        JSD_singleE = UTIL.jsdHist(esum_real, esum_fake, 100, 300, 1200)
        JSD_hitE = UTIL.jsdHist(hitE_real, hitE_fake, 200, 0.2, 500) 
        JSD_nhits = UTIL.jsdHist(hitN_real, hitN_fake, 100, 10, 500)
        JSD_cogz  = UTIL.jsdHist(cogz_real, cogz_fake, 50, 0, 50)
        JSD_radial = UTIL.jsdHist_radial(r_real, r_fake, e_real, e_fake, 30, 0, 15)


        print("Epoch #", eph)
        print ("SingleE:", JSD_singleE, "\t hitE:", JSD_hitE, "\t Nhits: ", JSD_nhits, "\t Radial:", JSD_radial, "\t CoGz: ", JSD_cogz)
        totalFid = (JSD_singleE + JSD_nhits + JSD_cogz + JSD_singleE + JSD_radial) / 5 
        print("Total Fidelity: ", totalFid)



