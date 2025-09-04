 
import glob
import math
import os
import numpy as np
from matplotlib import pyplot as plt
import argparse

def get_pae_plddt(input_dir):
    out = {}
    
    # Look for PAE and pLDDT files recursively in predictions directory
    pae_files = sorted(glob.glob(f'{input_dir}/predictions/**/pae_*.npz', recursive=True))
    plddt_files = sorted(glob.glob(f'{input_dir}/predictions/**/plddt_*.npz', recursive=True))
    
    if not pae_files or not plddt_files:
        print(f"Warning: No PAE files found: {len(pae_files)}, No pLDDT files found: {len(plddt_files)}")
        print(f"Looking in: {input_dir}/predictions/**/")
        return out
    
    # For Boltz-1, there should be only one of each
    pae_file = pae_files[0]
    plddt_file = plddt_files[0]
    
    print(f"Loading PAE from: {pae_file}")
    print(f"Loading pLDDT from: {plddt_file}")
    
    pae_data = np.load(pae_file)['pae']
    plddt_data = np.load(plddt_file)['plddt']
    
    basename = os.path.basename(pae_file).replace('pae_', '').replace('.npz', '')
    out[f'boltz_model_{basename}'] = {'plddt': plddt_data, 'pae': pae_data}
    
    return out

def generate_output_images(out_dir, name, pae_plddt_per_model):
    ##################################################################
    plt.figure(figsize=(14, 4), dpi=100)
    ##################################################################
    plt.subplot(1, 1, 1)
    plt.title("Predicted LDDT per position")
    for model_name, value in pae_plddt_per_model.items():
        plt.plot(value["plddt"], label=model_name)
    plt.ylim(0, 1)
    plt.ylabel("Predicted LDDT")
    plt.xlabel("Positions")
    plt.legend()
    plt.savefig(f"{out_dir}/{name+('_' if name else '')}LDDT.png")
    plt.close()
    ##################################################################

    ##################################################################
    num_models = 1 # columns
    num_runs_per_model = math.ceil(len(pae_plddt_per_model)/num_models)
    fig = plt.figure(figsize=(3 * num_models, 2 * num_runs_per_model), dpi=100)
    for n, (model_name, value) in enumerate(pae_plddt_per_model.items()):
        plt.subplot(num_runs_per_model, num_models, n + 1)
        plt.imshow(value["pae"], label=model_name, cmap="bwr", vmin=0, vmax=30)
        plt.colorbar()
    fig.tight_layout()
    plt.savefig(f"{out_dir}/{name+('_' if name else '')}PAE.png")
    plt.close()
    ##################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir',dest='input_dir',required=True)
parser.add_argument('--name',dest='name')
parser.set_defaults(name='')
parser.add_argument('--output_dir',dest='output_dir')
parser.set_defaults(output_dir='')
args = parser.parse_args()

pae_plddt_per_model = get_pae_plddt(args.input_dir)
generate_output_images(args.output_dir if args.output_dir else args.input_dir, args.name, pae_plddt_per_model)