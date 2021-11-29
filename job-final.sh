#!/bin/bash                                                                                                                                     
#SBATCH --time=24:00:00
#SBATCH --qos=regular
#SBATCH --nodes=32
#SBATCH --constraint="knl"
#SBATCH --output=split-sample-%j.out
#SBATCH --mem-per-cpu=MaxMemPerNode

python3 test.py "CCCC_3s2DH"
