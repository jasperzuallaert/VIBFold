#!/bin/bash
#PBS -N AlphaFold_multimer_script
#PBS -l nodes=1:ppn=8:gpus=1
#PBS -l mem=64gb
#PBS -l walltime=24:0:0

PROTEIN=SARSCOV2_VHH72

module load AlphaFold/2.3.1-foss-2022a-CUDA-11.7.0
export ALPHAFOLD_DATA_DIR=/arcanine/scratch/gent/apps/AlphaFold/20230310

WORKDIR=$VSC_DATA/alphafold/runs/$PBS_JOBID-$PROTEIN
mkdir -p $WORKDIR
cp -a $PBS_O_WORKDIR/fastas/$PROTEIN.fasta $WORKDIR/
cd $WORKDIR

echo Running $PROTEIN.fasta, output found at $WORKDIR
alphafold --fasta_paths=$PROTEIN.fasta \
          --max_template_date=2999-01-01 \
          --db_preset=full_dbs \
          --output_dir=$PWD \
          --model_preset=multimer \
          --num_multimer_predictions_per_model=1