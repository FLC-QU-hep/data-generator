#!/bin/bash

#SBATCH --partition=cms-desy
#SBATCH --time=12:00:00     
#SBATCH --chdir   /beegfs/desy/user/eren          # directory must already exist!
#SBATCH --job-name  fidelityRun
#SBATCH --output    fidelityRun-%N.out            # File to which STDOUT will be written
#SBATCH --error     fidelityRun-%N.err            # File to which STDERR will be written
#SBATCH --mail-type END 

## go to the target directory
cd /beegfs/desy/user/eren/synthetic-data-generator/fidelity
python main.py


exit 0;


