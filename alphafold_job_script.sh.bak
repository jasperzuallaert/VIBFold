#!/bin/bash
#PBS -N AlphaFold_tutorial_script
#PBS -l nodes=1:ppn=8:gpus=1
#PBS -l mem=64gb
#PBS -l walltime=24:0:0

PROTEIN=rbd

module load AlphaFold/2.1.1-fosscuda-2020b
export ALPHAFOLD_DATA_DIR=/arcanine/scratch/gent/apps/AlphaFold/20211201

WORKDIR=$VSC_DATA/alphafold/runs/$PBS_JOBID-$PROTEIN
mkdir -p $WORKDIR
cp -a $PBS_O_WORKDIR/fastas/$PROTEIN.fasta $WORKDIR/
cd $WORKDIR

echo Running $PROTEIN.fasta, output found at $WORKDIR
alphafold --fasta_paths=$PROTEIN.fasta \
          --max_template_date=2020-05-14 \
          --db_preset=full_dbs \
          --output_dir=$PWD \
          --model_preset=monomer_ptm



