
FASTA_FILE = 'fastas/NAME.fasta'        # location of fasta file between '' - absolute or relative path possible
IS_COMPLEX = True                       # True or False
MSA_MODE = 'mmseqs2_server'             # 'alphafold_default' or 'mmseqs2_server'
SAVE_DIR = 'results/naa10_practical'               # location of results directory between '' - abs or rel path possible
DO_RELAX = 'none'                       # 'all', 'best' or 'none'
NUM_RUNS_PER_MODEL = 1                  # number of runs per model, with different random seed
USE_TEMPLATES = True                   # True, False
MAX_RECYCLES = 3                        # default == 3


import subprocess
import os
import time

timestamp = time.strftime('%Y%m%d_%H%M%S') + '_'

def submit(FASTA_FILE, IS_COMPLEX, MSA_MODE, SAVE_DIR, DO_RELAX, USE_TEMPLATES, MAX_RECYCLES):
    assert ' ' not in FASTA_FILE, 'The name of your FASTA file cannot contain any spaces'
    # automatically select accelgor/joltik based on output of 'ml'
    module_info = subprocess.check_output('ml',shell=True).decode('utf-8')
    cluster = 'accelgor' if 'accelgor' in module_info else \
              'joltik' if 'joltik' in module_info else \
              'donphan' if 'donphan' in module_info else ''
    if not cluster:
        raise NotImplementedError('Cluster joltik/accelgor/donphan not found in "ml" output. Did you use "module swap cluster/joltik" (or other)?')

    fasta_d = {}
    seq = ''
    ctr = 1
    for line in open(FASTA_FILE):
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

    all_seqs = {}
    all_protnames = {}
    if not IS_COMPLEX: # create one new FASTA file per entry
        for prot_id, seq in fasta_d.items():
            all_seqs[prot_id] = seq
            all_protnames[prot_id] = prot_id
    else: # create a copy of the multi-entry FASTA file
        fasta_id = os.path.basename(FASTA_FILE).split('.')[0]
        seqs = fasta_d.values()
        all_protnames[fasta_id] = ':'.join(fasta_d.keys())
        all_seqs[fasta_id] = ':'.join(seqs)

    if not SAVE_DIR.startswith('/'):
        SAVE_DIR = f'$PBS_O_WORKDIR/{SAVE_DIR}'

        for prot_id, seq in all_seqs.items():
            prot_names = all_protnames[prot_id]
            run_save_dir = f'{SAVE_DIR}/{timestamp}_{prot_id}'
            script_content = f'''#!/bin/bash
#PBS -N VIBFold_{prot_id}
#PBS -l nodes=1:ppn={12 if cluster=='accelgor' else 8}{",gpus=1" if cluster in ['accelgor','joltik'] else ""}
#PBS -l mem={125 if cluster=='accelgor' else 64 if cluster=='joltik' else 20}g
#PBS -l walltime=48:00:00

module load Python/3.10.4-GCCcore-11.3.0

module load tqdm/4.64.0-GCCcore-11.3.0
module load matplotlib/3.5.2-foss-2022a
module load AlphaFold/2.3.1-foss-2022a{"-CUDA-11.7.0" if cluster in ['accelgor','joltik'] else ""}
export ALPHAFOLD_DATA_DIR=/arcanine/scratch/gent/apps/AlphaFold/20230310
PROTEIN={prot_id}

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
 --num_runs_per_model {NUM_RUNS_PER_MODEL} \
 --max_recycles {MAX_RECYCLES} \
 --do_gather_best
'''

            scriptname = 'submit_new.sh'
            f = open(scriptname,'w')
            print(script_content,file=f)
            f.close()

            print()
            print(f'############# submitting {prot_id} #############')
            subprocess.Popen(['echo',f'{prot_id}'],shell=False)
            subprocess.Popen(['qsub',f'{scriptname}', '--pass=reservation=D013637_2023'],shell=False).wait()
            subprocess.Popen(['rm',f'{scriptname}'],shell=False).wait()
            print()

if __name__ == "__main__":
    submit(FASTA_FILE, IS_COMPLEX, MSA_MODE, SAVE_DIR, DO_RELAX, USE_TEMPLATES, MAX_RECYCLES)