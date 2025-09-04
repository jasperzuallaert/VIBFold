#!/bin/bash
#PBS -N Boltz_test
#PBS -l nodes=1:ppn=8
#PBS -l mem=32gb
#PBS -l walltime=6:0:0

# input file MUST have the file extension added here
input_file=RBD_VHHE.yaml
out_dir=results/RBD_VHHE_boltz_pred

ml Boltz-1/0.4.1-foss-2023a-CUDA-12.1.1
cd $PBS_O_WORKDIR
boltz predict $input_file --write_full_pae --cache $VSC_DATA --out_dir $out_dir --use_msa_server
