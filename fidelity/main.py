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
    "Single" : '/beegfs/desy/user/eren/data_generator/pion/hcal_only/pion40part1.hdf5',
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





