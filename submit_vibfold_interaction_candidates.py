MAIN_FASTA = 'fastas/test_main.fasta' # list of proteins, to interact with each of the proteins in CANDIDATES_FASTA
CANDIDATES_FASTA = 'fastas/test_candidates.fasta' # list of proteins, to interact with each of the proteins in MAIN_FASTA

MSA_MODE = 'mmseqs2_server'                # 'alphafold_default' or 'mmseqs2_server'
SAVE_DIR = 'results/test'                # location of results directory between '' - abs or rel path possible
DO_RELAX = 'none'                             # 'all', 'best' or 'none'
USE_TEMPLATES = True                          # True, False
MAX_RECYCLES = 3                              # default == 3
NUM_RUNS_PER_MODEL = 5                  # number of runs per model, with different random seed

# IS_COMPLEX = True # Fixed value, this is no longer considered

import subprocess
import os
import time

timestamp = time.strftime('%Y%m%d_%H%M%S') + '_'

def submit(MAIN_FASTA, CANDIDATES_FASTA, MSA_MODE, SAVE_DIR, DO_RELAX, USE_TEMPLATES, MAX_RECYCLES, NUM_RUNS_PER_MODEL):
    assert ' ' not in MAIN_FASTA, 'The name of your FASTA files cannot contain any spaces'
    assert ' ' not in CANDIDATES_FASTA, 'The name of your FASTA files cannot contain any spaces'
    # automatically select accelgor/joltik based on output of 'ml'
    module_info = subprocess.check_output('ml',shell=True).decode('utf-8')
    cluster = 'accelgor' if 'accelgor' in module_info else \
              'joltik' if 'joltik' in module_info else ''
    if not cluster:
        raise NotImplementedError('Cluster joltik/accelgor not found in "ml" output. Did you use "module swap cluster/joltik" (or other)?')

    def read_fasta(file):
        fasta_d = {}
        seq = ''
        ctr = 1
        for line in open(file):
            if line.startswith('>'):
                if seq:
                    fasta_d[prot_id] = seq
                    seq = ''
                prot_id = f'{ctr}_{line.rstrip().lstrip(">").replace(" ", "_").replace(":", "_").replace("(", "").replace(")", "")}'
                if '|' in prot_id: prot_id = prot_id.split('|')[1]
                ctr+=1
            elif line:
                seq += line.rstrip()
        if seq:
            fasta_d[prot_id] = seq
        return fasta_d

    proteins_1 = read_fasta(MAIN_FASTA)
    proteins_2 = read_fasta(CANDIDATES_FASTA)

    all_interactions = {}
    for prot1_id, seq1 in proteins_1.items():
        for prot2_id, seq2 in proteins_2.items():
            prot1_id = prot1_id.replace(' ', '_').replace(':', '_').replace('(', '').replace(')', '')
            prot2_id = prot2_id.replace(' ', '_').replace(':', '_').replace('(', '').replace(')', '')
            prot_names = f'{prot1_id}:{prot2_id}'
            all_interactions[prot_names] = f'{seq1}:{seq2}'

    if not SAVE_DIR.startswith('/'):
        SAVE_DIR = f'$PBS_O_WORKDIR/{SAVE_DIR}'

        for prot_names, seq in all_interactions.items():
            run_save_dir = f'{SAVE_DIR}/{timestamp}_{prot_names.replace(":", "_")}'
            script_content = f'''#!/bin/bash
#PBS -N interactions_VIBFold_{prot_names.replace(":", "_")}
#PBS -l nodes=1:ppn={12 if cluster=='accelgor' else 8},gpus=1
#PBS -l mem={125 if cluster=='accelgor' else 64}g
#PBS -l walltime=48:00:00

module load Python/3.11.3-GCCcore-12.3.0

module load tqdm/4.66.1-GCCcore-12.3.0 
module load matplotlib/3.7.2-gfbf-2023a
module load AlphaFold/2.3.2-foss-2023a-CUDA-12.1.1
export ALPHAFOLD_DATA_DIR=/arcanine/scratch/gent/apps/AlphaFold/20230310

PROTEIN={prot_names.replace(":", "_")}

jobname="$PROTEIN"_"$PBS_JOBID"

SAVEDIR={run_save_dir}
mkdir -p $SAVEDIR

cd $PBS_O_WORKDIR
python VIBFold.py \
 --seq {seq} \
 --prot_names "{prot_names}" \
 --jobname $jobname \
 --save_dir $SAVEDIR \
 --do_relax {DO_RELAX} \
 {"--no_templates" if not USE_TEMPLATES else ""} \
 --msa_mode {MSA_MODE} \
 --max_recycles {MAX_RECYCLES} \
 --num_runs_per_model {NUM_RUNS_PER_MODEL} \
 --do_gather_best
'''

            scriptname = 'submit_new.sh'
            f = open(scriptname,'w')
            print(script_content,file=f)
            f.close()

            print()
            print(f'############# submitting {prot_names.replace(":", "_")} #############')
            subprocess.Popen(['echo',f'{prot_names.replace(":", "_")}'],shell=False)
            subprocess.Popen(['qsub',f'{scriptname}'],shell=False).wait()
            subprocess.Popen(['rm',f'{scriptname}'],shell=False).wait()
            print()

if __name__ == "__main__":
    submit(MAIN_FASTA, CANDIDATES_FASTA, MSA_MODE, SAVE_DIR, DO_RELAX, USE_TEMPLATES, MAX_RECYCLES, NUM_RUNS_PER_MODEL)
