#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 4 ]; then
  echo "Usage: $0 output_directory fasta_location_1 fasta_location_2 number_of_predictions_per_model"
  exit 1
fi

# Create the output directory if it doesn't exist
mkdir -p "$1"

# Assign arguments to variables
output_directory=`realpath $1`
fasta_location_1=`realpath $2`
fasta_location_2=`realpath $3`
pred_per_model=$4

# Function to count the number of entries in a fasta file
count_entries() {
  grep -c '^>' "$1"
}

# Function to create a protein list from a fasta file
create_protein_list() {
  grep '^>' "$1" | sed 's/^>//' | sed 's/[|= #;]/_/g' > "$2"
}

# Count entries in each fasta file
count1=$(count_entries "$fasta_location_1")
count2=$(count_entries "$fasta_location_2")
total_count=$((count1 + count2))
total_preds=$((count1 * count2))

# Create protein lists
protein_list_1="${output_directory}/protein_list_1.txt"
protein_list_2="${output_directory}/protein_list_2.txt"

create_protein_list "$fasta_location_1" "$protein_list_1"
create_protein_list "$fasta_location_2" "$protein_list_2"

# Create MSA job script
msa_job_script="${output_directory}/msa_job_script.sh"
cat <<EOT > "$msa_job_script"
#PBS -N msa_pulldown
#PBS -l nodes=1:ppn=4
#PBS -l walltime=12:00:00
#PBS -l mem=80gb

module load AlphaPulldown/2.0.3-foss-2023a
export ALPHAFOLD_DATA_DIR=/arcanine/scratch/gent/apps/AlphaFold/20230310
cd $output_directory

create_individual_features.py \
  --fasta_paths=$fasta_location_1,$fasta_location_2 \
  --data_dir=\$ALPHAFOLD_DATA_DIR \
  --save_msa_files=False \
  --output_dir=$output_directory/msas \
  --use_precomputed_msas=False \
  --max_template_date=2050-01-01 \
  --skip_existing=True \
  --uniref30_database_path=/arcanine/scratch/gent/apps/AlphaFold/20230310/uniref30/UniRef30_2021_03 \
  --seq_index \${PBS_ARRAYID}
EOT

instr_file=$output_directory/instructions.txt
echo "" > $instr_file
# Print MSA job script command
echo "######################################" >> $instr_file
echo "#####     STEP 1. MSA search     #####" >> $instr_file
echo "######################################" >> $instr_file
echo ">>> To launch the MSA search, run: " >> $instr_file
echo "          module swap cluster/doduo" >> $instr_file
echo "          qsub -m ae $msa_job_script -t1-$total_count -o $output_directory/output_msa.log -e $output_directory/error_msa.log" >> $instr_file
echo ">>> Note that you can change the doduo cluster to any CPU cluster you wish" >> $instr_file
echo ">>> Wait until all jobs have been successfully completed - you will receive an email when finished." >> $instr_file
echo " " >> $instr_file

# Create the GPU job script
predict_job_script="${output_directory}/predict_job_script.sh"
cat <<EOT > "$predict_job_script"
#PBS -N predict_pulldown
#PBS -l nodes=1:ppn=8,gpus=1
#PBS -l mem=64g
#PBS -l walltime=12:00:00

module load AlphaPulldown/2.0.3-foss-2023a-CUDA-12.1.1

cd $output_directory
export EBROOTCOLABFOLD=\$PBS_O_WORKDIR/$output_directory
export ALPHAFOLD_DATA_DIR=/arcanine/scratch/gent/apps/AlphaFold/20230310

run_multimer_jobs.py --mode=pulldown \
--num_cycle=3 \
--num_predictions_per_model=$pred_per_model \
--output_path=$output_directory/preds/ \
--data_dir=\$ALPHAFOLD_DATA_DIR \
--protein_lists=$protein_list_1,$protein_list_2 \
--monomer_objects_dir=$output_directory/msas/ \
--compress_result_pickles=True \
--remove_result_pickles=True \
--job_index=\${PBS_ARRAYID}
EOT

echo "######################################" >> $instr_file
echo "#####     STEP 2. Prediction     #####" >> $instr_file
echo "######################################" >> $instr_file
echo ">>> To launch the prediction search, run: " >> $instr_file
echo "          module swap cluster/joltik" >> $instr_file
echo "          qsub -m ae $predict_job_script -t1-$total_preds -o $output_directory/output_predict.log -e $output_directory/output_predict.log" >> $instr_file
echo ">>> Note that you can change the joltik cluster to any GPU cluster you wish" >> $instr_file
echo ">>> You will receive an email when finished." >> $instr_file
echo " " >> $instr_file


gather_results_script="${output_directory}/gather_results_script.sh"
cat <<EOT > "$gather_results_script"
#PBS -N gather_results_pulldown
#PBS -l nodes=1:ppn=4
#PBS -l walltime=12:00:00
#PBS -l mem=16gb

cd $output_directory

apptainer exec /arcanine/scratch/gent/apps/AlphaPulldown/alpha-analysis_jax_0.4.sif run_get_good_pae.sh --cutoff=10 --output_dir=$output_directory/preds 
EOT

echo "######################################" >> $instr_file
echo "#####      STEP 3. Results       #####" >> $instr_file
echo "######################################" >> $instr_file
echo ">>> Finally, you can gather results in a csv by running" >> $instr_file
echo "          module swap cluster/doduo" >> $instr_file
echo "          qsub -m ae $gather_results_script -o $output_directory/output_gather.log -e $output_directory/output_gather.log" >> $instr_file
echo ">>> Note that you can adjust the cutoff parameter in the file to include lower quality interactions as well" >> $instr_file
echo ">>> Note that you can change the doduo cluster to any CPU cluster you wish" >> $instr_file
echo " " >> $instr_file

echo ">>> Outputs can be found in $output_directory/outputs." >> $instr_file

echo "Instructions can be found at $instr_file as well"
cat $instr_file

