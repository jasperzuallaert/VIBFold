
MAIN_FASTA = 'fastas/for_yehudi.fasta' # list of proteins, to interact with each of the proteins in CANDIDATES_FASTA
CANDIDATES_FASTA = 'fastas/for_bert.fasta' # list of proteins, to interact with each of the proteins in MAIN_FASTA

MSA_MODE = 'mmseqs2_server'                   # 'alphafold_default' or 'mmseqs2_server'
SAVE_DIR = 'results/debugging'                # location of results directory between '' - abs or rel path possible
DO_RELAX = 'best'                             # 'all', 'best' or 'none'
USE_TEMPLATES = True                          # True, False
MAX_RECYCLES = 3                              # default == 3

# IS_COMPLEX = True # Fixed value, this is no longer considered

import subprocess
import os

def submit(MAIN_FASTA, CANDIDATES_FASTA, MSA_MODE, SAVE_DIR, DO_RELAX, USE_TEMPLATES, MAX_RECYCLES):
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
                prot_id = f'{ctr}_{line.rstrip().lstrip(">")}'
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
            prot1_id = prot1_id.replace(' ', '_')
            prot2_id = prot2_id.replace(' ', '_')
            all_interactions[f'{prot1_id}_{prot2_id}'] = f'{seq1}:{seq2}'

    if not SAVE_DIR.startswith('/'):
        SAVE_DIR = f'$PBS_O_WORKDIR/{SAVE_DIR}'

        for prot_id, seq in all_interactions.items():
            script_content = f'''#!/bin/bash
#PBS -N test_VIBFold_{prot_id}
#PBS -l nodes=1:ppn={12 if cluster=='accelgor' else 8},gpus=1
#PBS -l mem={125 if cluster=='accelgor' else 64}g
#PBS -l walltime=48:00:00

module load Python/3.8.6-GCCcore-10.2.0

{'module load tqdm/4.56.2-GCCcore-10.2.0' if cluster == 'accelgor' else 
'module load tqdm/4.60.0-GCCcore-10.2.0'}
module load matplotlib/3.3.3-fosscuda-2020b
module load AlphaFold/2.1.1-fosscuda-2020b
export ALPHAFOLD_DATA_DIR=/arcanine/scratch/gent/apps/AlphaFold/20211201

PROTEIN={prot_id}

jobname="$PROTEIN"_"$PBS_JOBID"

SAVEDIR={SAVE_DIR}
mkdir -p $SAVEDIR

cd $PBS_O_WORKDIR
python VIBFold.py \
 --seq {seq} \
 --jobname $jobname \
 --save_dir $SAVEDIR \
 --do_relax {DO_RELAX} \
 {"--no_templates" if not USE_TEMPLATES else ""} \
 --msa_mode {MSA_MODE} \
 --max_recycles {MAX_RECYCLES}
'''

            scriptname = 'submit_new.sh'
            f = open(scriptname,'w')
            print(script_content,file=f)
            f.close()

            print()
            print(f'############# submitting {prot_id} #############')
            subprocess.Popen(['echo',f'{prot_id}'],shell=False)
            subprocess.Popen(['qsub',f'{scriptname}'],shell=False).wait()
            subprocess.Popen(['rm',f'{scriptname}'],shell=False).wait()
            print()

if __name__ == "__main__":
    submit(MAIN_FASTA, CANDIDATES_FASTA, MSA_MODE, SAVE_DIR, DO_RELAX, USE_TEMPLATES, MAX_RECYCLES)