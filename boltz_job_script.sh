#!/bin/bash
#PBS -N Boltz_test
#PBS -l nodes=1:ppn=8:gpus=1
#PBS -l mem=64gb
#PBS -l walltime=24:0:0

# input file MUST have the file extension added here
input_file=fastas/RBD_VHHE.fasta

ml Boltz-1/0.4.1-foss-2023a-CUDA-12.1.1
cd $PBS_O_WORKDIR
boltz predict $input_file --cache /data/gent/vo/001/gvo00133/vsc43898 --use_msa_server
