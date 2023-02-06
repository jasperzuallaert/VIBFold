import argparse
import itertools
import re
import logging
import dataclasses
import json
import copy
from alphafold.model import data as alpha_data, config as alpha_config
from alphafold.common import protein as alpha_protein
from alphafold.relax import relax
from alphafold.model import model as alpha_model
from random import randint
from alphafold.data import templates as alpha_templates
from alphafold.data import feature_processing, msa_pairing, msa_identifiers
from alphafold.data.tools import hhsearch
from alphafold.data.tools import hmmsearch
from alphafold.data import pipeline as alpha_pipeline
from alphafold.data import templates
from alphafold.data import parsers
from alphafold.common import residue_constants
from alphafold.data import pipeline_multimer
import numpy as np
from matplotlib import pyplot as plt
import alphafold
import os
import shutil
import VIBFold_adapted_functions as adapted
import pickle

r = randint(0,100000)

ALPHAFOLD_DATA_DIR = os.environ['ALPHAFOLD_DATA_DIR']
MAX_TEMPLATE_HITS = 20 # default alphafold setting
MAX_UNIREF_HITS = 10000
MAX_UNIPROT_HITS = 50000 # for multimer
MAX_MGNIFY_HITS = 501
jackhmmer_binary_path = shutil.which('jackhmmer')
hhblits_binary_path=shutil.which('hhblits')
hmmsearch_binary_path=shutil.which('hmmsearch')
hmmbuild_binary_path=shutil.which('hmmbuild')
hhsearch_binary_path=shutil.which('hhsearch')
kalign_binary_path=shutil.which('kalign')
uniref90_database_path=ALPHAFOLD_DATA_DIR + '/uniref90/uniref90.fasta'
mgnify_database_path=ALPHAFOLD_DATA_DIR + '/mgnify/mgy_clusters_2018_12.fa'
bfd_database_path=ALPHAFOLD_DATA_DIR + '/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt'
template_mmcif_dir=ALPHAFOLD_DATA_DIR + '/pdb_mmcif/mmcif_files'
obsolete_pdbs_path=ALPHAFOLD_DATA_DIR + '/pdb_mmcif/obsolete.dat'
uniclust30_database_path=ALPHAFOLD_DATA_DIR + '/uniclust30/uniclust30_2018_08/uniclust30_2018_08'
pdb70_database_path=ALPHAFOLD_DATA_DIR + '/pdb70/pdb70'
pdb_seqres_database_path=ALPHAFOLD_DATA_DIR + '/pdb_seqres/pdb_seqres.txt'
uniprot_database_path = ALPHAFOLD_DATA_DIR + '/uniprot/uniprot.fasta' # for multimer

def run_alphafold_advanced_complex(seq, jobname, save_dir, use_templates, do_relax, max_recycles=3, tolerance=0, msa_mode='alphafold_default'):
    # process input parameters
    sequences = seq.split(':')
    jobname = "".join(jobname.split())
    jobname = re.sub(r'\W+', '', jobname)
    use_env = True if msa_mode.startswith('mmseqs2') else False
    model_used = 'monomer_ptm' if len(sequences) == 1 else 'multimer'
    # Load model parameters, same for all permutations
    model_params, model_runner_12, model_runner_345 = prepare_models(model_used, max_recycles, tolerance)

    seq_to_msa_d = {}
    # if multiple sequences are present in the query, a single run is done for each of the permutations
    # Protein complexes with 'first A then B', will yield different results than 'first B then A'
    for perm_idx, query_sequences in enumerate(set(itertools.permutations(sequences))):
        # the output dir gets a permutation number if applicable
        out_dir = f'{save_dir}/{jobname}' if len(sequences) == 1 else f'{save_dir}/{jobname}_permutation{perm_idx}'
        out_dir = out_dir.rstrip('/')
        os.makedirs(out_dir,exist_ok=True)

        fasta_file = f'{out_dir}/query.fasta'
        write_to_fasta = open(fasta_file, 'w')
        for i, query_sequence in enumerate(query_sequences):
            print(f'>chain_{i}',file=write_to_fasta)
            print(query_sequence,file=write_to_fasta)
        write_to_fasta.close()
        print(open(fasta_file).read())

        # set logging
        logger = logging.Logger(f'perm{perm_idx}', level = logging.INFO)
        fh = logging.FileHandler(filename=f'{out_dir}/info.log', mode='w')
        fh.setFormatter(logging.Formatter(datefmt='%Y-%m-%d %H:%M:%S', fmt='%(asctime)s - %(message)s'))
        logger.addHandler(fh)
        logger.info(f'Parameters: num_sequences = {len(query_sequences)}')
        logger.info(f'Parameters: model used = {model_used}')
        logger.info(f'Parameters: use_amber = {do_relax}')
        logger.info(f'Parameters: msa_mode = {msa_mode}')
        logger.info(f'Parameters: use_templates = {use_templates}')
        logger.info(f'Parameters: max_recycles = {max_recycles}')
        logger.info(f'Parameters: recycling_tolerance = {tolerance}')
        logger.info(f'Started permutation # {perm_idx}')
        print(f'Parameters: num_sequences = {len(query_sequences)}')
        print(f'Parameters: model used = {model_used}')
        print(f'Parameters: use_amber = {do_relax}')
        print(f'Parameters: msa_mode = {msa_mode}')
        print(f'Parameters: use_templates = {use_templates}')
        print(f'Parameters: max_recycles = {max_recycles}')
        print(f'Parameters: recycling_tolerance = {tolerance}')
        print(f'Started permutation # {perm_idx}')

        # Do MSA search
        logger.info('Starting MSA search (+ templates if req)')
        feature_dict = run_msa_search(msa_mode, query_sequences, fasta_file, seq_to_msa_d, use_templates, out_dir, jobname, logger)

        #pickle.dump(feature_dict,file=open(out_dir+'/features_for_debugging.pkl','wb'))
        
        if use_templates:
            try:
                log_template_info(feature_dict, logger)
            except Exception as exx:
                logger.info('Exception encountered during template logging. Feel free to contact jasper.zuallaert@vib-ugent.be about this. Error message:')
                logger.info(exx)
        else:
            logger.info('No templates used.')
        # Predict
        logger.info('Starting predictions...')
        unrelaxed_pdb_lines, unrelaxed_proteins, paes, plddts, ptms, iptms = predict_structures(logger,
                                                                                                  model_params,
                                                                                                  model_runner_12,
                                                                                                  model_runner_345,
                                                                                                  feature_dict,
                                                                                                  model_used == 'multimer')
        # rank models, relax, write pdb files
        pae_plddt_per_model = rank_relax_write(logger, unrelaxed_pdb_lines, unrelaxed_proteins, plddts, paes, ptms, iptms, out_dir, jobname, do_relax, model_used=='multimer')
        # generate output images

        logger.info('Generating output images...')
        generate_output_images(query_sequences, pae_plddt_per_model, feature_dict['msa'], out_dir, jobname)
        # remove intermediate directories
        if msa_mode.startswith('mmseqs2'):
            os.popen(f'rm -rf {out_dir}/{jobname}_*{"env" if use_env else ""}/')
        else:
            os.popen(f'rm -rf {out_dir}/{jobname}_*{"env" if use_env else "all"}/')
        logger.info(f'Permutation {perm_idx} finished!')

# Loads the models, and compiles them. Weights are loaded, but only at prediction time added to the model.
# There is distinction between model_1 and model_3, because 1-2 and 3-4-5 have a different number of parameters
def prepare_models(model_used, max_recycles, tolerance):
    model_params = {}
    model_runner_12, model_runner_345 = None, None
    model_extension = 'ptm' if model_used == 'monomer_ptm' else 'multimer_v2'
    for model_num in range(1,6):
        model_name = f'model_{model_num}_{model_extension}'
        model_params[model_name] = alpha_data.get_model_haiku_params(model_name=model_name, data_dir=ALPHAFOLD_DATA_DIR)
        if model_num in (1,3):
            model_config = alpha_config.model_config(model_name)
            model_config.model.recycle_tol = tolerance
            if model_extension == "ptm":
                model_config.data.eval.num_ensemble = 1
                model_config.data.common.num_recycle = max_recycles
                model_config.model.num_recycle = max_recycles
            else: # multimer
                model_config.model.num_ensemble_eval = 1
                model_config.model.num_recycle = max_recycles
            if model_num == 1:
                model_runner_12 = alpha_model.RunModel(model_config, model_params[model_name])
            elif model_num == 3:
                model_runner_345 = alpha_model.RunModel(model_config, model_params[model_name])
    return model_params, model_runner_12, model_runner_345

def run_msa_search(msa_type, query_sequences, query_fasta, seq_to_msa_d, use_templates, out_dir, jobname, logger):
    run_multimer_system = len(query_sequences) > 1
    max_template_hits = MAX_TEMPLATE_HITS if use_templates else 0
    if msa_type == 'alphafold_default':
        if run_multimer_system:
            template_searcher = hmmsearch.Hmmsearch(binary_path=hmmsearch_binary_path, hmmbuild_binary_path=hmmbuild_binary_path, database_path=pdb_seqres_database_path)
            template_featurizer = templates.HmmsearchHitFeaturizer(mmcif_dir=template_mmcif_dir, max_template_date='2100-10-28', max_hits=max_template_hits, kalign_binary_path=kalign_binary_path, release_dates_path=None, obsolete_pdbs_path=obsolete_pdbs_path)
        else:
            template_searcher = hhsearch.HHSearch(binary_path=hhsearch_binary_path, databases=[pdb70_database_path])
            template_featurizer = templates.HhsearchHitFeaturizer(mmcif_dir=template_mmcif_dir, max_template_date='2100-10-28', max_hits=max_template_hits, kalign_binary_path=kalign_binary_path, release_dates_path=None, obsolete_pdbs_path=obsolete_pdbs_path)
        logger.info('setting up data pipeline')
        monomer_data_pipeline = alpha_pipeline.DataPipeline(
            jackhmmer_binary_path=jackhmmer_binary_path,
            hhblits_binary_path=hhblits_binary_path,
            uniref90_database_path=uniref90_database_path,
            mgnify_database_path=mgnify_database_path,
            bfd_database_path=bfd_database_path,
            uniclust30_database_path=uniclust30_database_path,
            small_bfd_database_path=None,
            template_searcher=template_searcher,
            template_featurizer=template_featurizer,
            use_small_bfd=False,
            use_precomputed_msas=False)
        if run_multimer_system:
            data_pipeline = adapted.Cached_DataPipeline(
                monomer_data_pipeline=monomer_data_pipeline,
                jackhmmer_binary_path=jackhmmer_binary_path,
                uniprot_database_path=uniprot_database_path,
                use_precomputed_msas=False,
                seq_to_features_cache=seq_to_msa_d
            )
        else:
            data_pipeline = monomer_data_pipeline
        msa_dir = f'{out_dir}/{jobname}_seq_all'
        if not os.path.exists(msa_dir): os.mkdir(msa_dir)
        feature_dict = data_pipeline.process(input_fasta_path=query_fasta, msa_output_dir=msa_dir)
        # if not use_templates:
        #     for key in list(feature_dict.keys()):
        #         if key.startswith('template_'):
        #             del feature_dict[key]
        #     feature_dict.update(mk_placeholder_template(1, sum(len(x) for x in query_sequences)))
        return feature_dict
    elif msa_type == 'mmseqs2_server':
        if run_multimer_system:
            # special stuff
            if use_templates:
                unpaired_a3m_lines, template_paths = adapted.run_mmseqs2(query_sequences, f'{out_dir}/{jobname}', True, use_templates=True, use_pairing=False)
                paired_a3m_lines = adapted.run_mmseqs2(query_sequences, f'{out_dir}/{jobname}', True, use_templates=False, use_pairing=True)
                template_features = []
                for query_sequence, a3m_lines_unpaired, template_path in zip(query_sequences, unpaired_a3m_lines, template_paths):
                    tt = None
                    if template_path:
                        try:
                            tt = mk_template(query_sequence, a3m_lines_unpaired, template_path, logger)
                        except RuntimeError:
                            print(f'Error in template construction for {template_path}')
                            logger.info(f'Error in template construction for {template_path}')
                    if tt == None:
                        tt = mk_placeholder_template(1,len(query_sequence))
                    template_features.append(tt)
            else:
                unpaired_a3m_lines = adapted.run_mmseqs2(query_sequences, f'{out_dir}/{jobname}', True, use_templates=False, use_pairing=False)
                paired_a3m_lines = adapted.run_mmseqs2(query_sequences, f'{out_dir}/{jobname}', True, use_templates=False, use_pairing=True)
                template_features = [mk_placeholder_template(1,len(seq)) for seq in query_sequences]

            features_for_chain = {}
            chain_cnt = 0
            for chain_seq, chain_unp_msa, chain_p_msa, chain_temp_feat in zip(query_sequences, unpaired_a3m_lines, paired_a3m_lines, template_features):
                msa = alpha_pipeline.parsers.parse_a3m(chain_unp_msa)
                feature_dict = {
                    **alpha_pipeline.make_sequence_features(sequence=chain_seq, description="none", num_res=len(chain_seq)),
                    **alpha_pipeline.make_msa_features([msa]),
                    **chain_temp_feat
                }

                parsed_paired_msa = alpha_pipeline.parsers.parse_a3m(chain_p_msa)
                paired_feature_dict = {
                    f"{k}_all_seq": v for k, v in adapted.my_make_msa_features_keep_duplicates([parsed_paired_msa]).items()
                }
                feature_dict.update(paired_feature_dict)

                features_for_chain[chr(ord('A')+chain_cnt)] = feature_dict
                chain_cnt += 1

            return adapted.process_multimer_features(features_for_chain)

        else:
            query_sequence = query_sequences[0]
            if use_templates:
                a3m_lines_unpaired, template_paths = adapted.run_mmseqs2(query_sequence, f'{out_dir}/{jobname}', True, use_templates=True)
                if template_paths[0] is not None:
                    template_features = mk_template(query_sequence, a3m_lines_unpaired[0], template_paths[0], logger)
                else:
                    template_features = mk_placeholder_template(1, len(query_sequence))
            else:
                a3m_lines_unpaired = adapted.run_mmseqs2(query_sequence, f'{out_dir}/{jobname}', True, use_templates=False)
                template_features = mk_placeholder_template(1, len(query_sequence))
                print('bbb template_aatype',template_features['template_aatype'].shape if 'template_aatype' in template_features else 'not present')

            msas = [alpha_pipeline.parsers.parse_a3m(a3m_lines_unpaired[0])]
            feature_dict = {
                **alpha_pipeline.make_sequence_features(sequence=query_sequence, description="none", num_res=len(query_sequence)),
                **alpha_pipeline.make_msa_features(msas=msas),
                **template_features
            }

            return feature_dict
    else:
        raise NotImplementedError(msa_type)

# returns a filled in template dict
# the dict contains:
#   - 'template_aatype' - shape (num_templates, num_residues, 22)
#   - 'template_all_atom_masks' - shape (num_templates, num_residues, 37)
#   - 'template_all_atom_positions' - shape (num_templates, num_residues, 37, 3)
#   - 'template_domain_names' - shape (num_templates, )
#   - 'template_sum_probs' - shape (num_templates, 1)
# Eventually, this dict is processed by AlphaFold code and turned into input features for AlphaFold
def mk_template(query_sequence, a3m_lines, template_paths, logger):
    # default settings
    template_featurizer = alpha_templates.HhsearchHitFeaturizer(
        mmcif_dir=template_paths,
        max_template_date="2100-01-01",
        max_hits=20,
        kalign_binary_path="kalign",
        release_dates_path=None,
        obsolete_pdbs_path=None)
    hhsearch_pdb70_runner = hhsearch.HHSearch(binary_path="hhsearch", databases=[f"{template_paths}/pdb70"])
    hhsearch_result = hhsearch_pdb70_runner.query(a3m_lines)
    hhsearch_hits = alphafold.data.parsers.parse_hhr(hhsearch_result)
    templates_result = template_featurizer.get_templates(query_sequence,
                                                         hhsearch_hits)

    logger.info('Templates found:')
    logger.info('\t'.join(['hit_id','query_start_idx', 'query_stop_idx', 'hit_start_idx', 'hit_stop_idx', 'hit_info']))
    for hit in hhsearch_hits:
        name = hit.name
        id = name.split(' ')[0]
        info = ' '.join(name.split(' ')[1:])
        query_start, query_stop = min(x for x in hit.indices_query if x != -1), max(hit.indices_query)
        hit_start, hit_stop = min(x for x in hit.indices_hit if x != -1), max(hit.indices_hit)
        logger.info('\t'.join(str(x) for x in [id, query_start, query_stop, hit_start, hit_stop, info]))

    # check for empty template - otherwise problems with shapes
    if len(templates_result.features['template_aatype']) == 0:
        return mk_placeholder_template(1, len(query_sequence))
    else:
        return templates_result.features

# returns a template placeholder, with size None in the first dimension
def mk_placeholder_template(num_templates_, num_res_):
    print(f'Dummy template created ({num_templates_})') ### TMP
    return {
        'template_aatype': np.zeros([num_templates_, num_res_, 22], np.float32),
        # 'template_all_atom_mask': np.zeros([num_templates_, num_res_, 37], np.float32),
        'template_all_atom_masks': np.zeros([num_templates_, num_res_, 37], np.float32),
        'template_all_atom_positions': np.zeros([num_templates_, num_res_, 37, 3], np.float32),  # 3d coords
        'template_sequence': np.zeros(20),
        'template_domain_names': np.zeros([num_templates_], np.float32),
        'template_sum_probs': np.zeros([num_templates_, 1], np.float32),
    }


# For each model to run, collect:
# - predicted structure (unrelaxed) in pdb lines
# - predicted structure (unrelaxed) as an object
# - predicted alignment error (PAE)
# - predicted local distance difference test (LDDT)
def predict_structures(logger, model_params, model_runner_12, model_runner_345, feature_dict, is_multimer):
    plddts, paes, ptms, iptms = [], [], [], []
    unrelaxed_pdb_lines = []
    unrelaxed_proteins = []
    for model_name, params in model_params.items():
        logger.info(f'running {model_name}')
        # swap params to avoid recompiling
        # note: models 1,2 have diff number of params compared to models 3,4,5
        if any(str(m) in model_name for m in [1, 2]): model_runner = model_runner_12
        if any(str(m) in model_name for m in [3, 4, 5]): model_runner = model_runner_345
        model_runner.params = params
        processed_feature_dict = model_runner.process_features(feature_dict, random_seed=r)
        prediction_result = model_runner.predict(processed_feature_dict,random_seed=r)
        if 'predicted_aligned_error' in prediction_result: paes.append(prediction_result['predicted_aligned_error'])
        if 'plddt' in prediction_result: plddts.append(prediction_result['plddt'])
        if 'ptm' in prediction_result: ptms.append(prediction_result['ptm'])
        if 'iptm' in prediction_result: iptms.append(prediction_result['iptm'])
        final_atom_mask = prediction_result["structure_module"]["final_atom_mask"]
        b_factors = prediction_result["plddt"][:, None] * final_atom_mask

        unrelaxed_protein = alpha_protein.from_prediction(processed_feature_dict, prediction_result, b_factors, remove_leading_feature_dimension = not is_multimer)
        unrelaxed_pdb_lines.append(alpha_protein.to_pdb(unrelaxed_protein))
        unrelaxed_proteins.append(unrelaxed_protein)
    return unrelaxed_pdb_lines, unrelaxed_proteins, paes, plddts, ptms, iptms

# Rerank based on pLDDT, relax best model if desired, and write pdb files
def rank_relax_write(logger, unrelaxed_pdb_lines, unrelaxed_proteins, plddts, paes, ptms, iptms, out_dir, jobname, do_relax, is_multimer):
    logger.info(f'Reranking based on {"pLDDT" if not is_multimer else "0.2*PTM+0.8*iPTM"}...')
    if iptms: ptms = [0.2*_ptm+0.8*_iptm for _ptm, _iptm in zip(ptms, iptms)]
    if is_multimer:
        rank = np.asarray(ptms).argsort()[::-1]
    else:
        rank = np.mean(plddts, -1).argsort()[::-1]
    plddt_pae_per_model = {}
    for n, r in enumerate(rank):
        logger.info(f"model_{n + 1} plddt={np.mean(plddts[r]):2.3f}, ptm={ptms[r]:2.3f}")
        unrelaxed_pdb_path = f'{out_dir}/{jobname}_unrelaxed_model_{n + 1}.pdb'
        with open(unrelaxed_pdb_path, 'w') as f:
            f.write(unrelaxed_pdb_lines[r])
        if (do_relax == 'best' and n == 0) or (do_relax == 'all'):
            try:
                amber_relaxer = relax.AmberRelaxation(max_iterations=0, tolerance=2.39, stiffness=10.0, exclude_residues=[], max_outer_iterations=20, use_gpu=True)
                relaxed_pdb_str, _, _ = amber_relaxer.process(prot=unrelaxed_proteins[r])
                relaxed_pdb_path = f'{out_dir}/{jobname}_relaxed_model_{n + 1}.pdb'
                with open(relaxed_pdb_path, 'w') as f:
                    f.write(relaxed_pdb_str)
            except ValueError:
                print('Error during relaxation - skipped!')
                logger.info('Error during relaxation - skipped!')
        plddt_pae_per_model[f"model_{n + 1}"] = {"plddt": plddts[r], "pae": paes[r]}
    return plddt_pae_per_model

# Generates images, with beautiful visualizations of:
# - the MSA
# - the pLDDT per model
# - the PAE per model
def generate_output_images(seqs, pae_plddt_per_model, msa, out_dir, jobname):
    # gather MSA info
    if len(seqs) > 1:
        seq = ''.join(seqs)
        map_d = {c: i for i, c in enumerate(residue_constants.restypes)}
    else:
        seq = seqs[0]
        map_d = residue_constants.HHBLITS_AA_TO_ID
    seq_by_indices = [map_d[c] for c in seq]
    seqid = (np.array(seq_by_indices == msa).mean(-1))
    seqid_sort = seqid.argsort()
    non_gaps = (msa != 21).astype(float)
    non_gaps[non_gaps == 0] = np.nan
    final = non_gaps[seqid_sort] * seqid[seqid_sort, None]

    ##################################################################
    plt.figure(figsize=(14, 4), dpi=100)
    ##################################################################
    plt.subplot(1, 2, 1)
    plt.title("Sequence coverage")
    plt.imshow(final,
               interpolation='nearest', aspect='auto',
               cmap="rainbow_r", vmin=0, vmax=1, origin='lower')
    plt.plot((msa != 21).sum(0), color='black')
    plt.xlim(-0.5, msa.shape[1] - 0.5)
    plt.ylim(-0.5, msa.shape[0] - 0.5)
    plt.colorbar(label="Sequence identity to query", )
    plt.xlabel("Positions")
    plt.ylabel("Sequences")

    ##################################################################
    plt.subplot(1, 2, 2)
    plt.title("Predicted LDDT per position")
    for model_name, value in pae_plddt_per_model.items():
        plt.plot(value["plddt"], label=model_name)
    plt.legend()
    plt.ylim(0, 100)
    plt.ylabel("Predicted LDDT")
    plt.xlabel("Positions")
    plt.savefig(f"{out_dir}/{jobname}_coverage_LDDT.png")
    ##################################################################

    ##################################################################
    num_models = 5
    plt.figure(figsize=(3 * num_models, 2), dpi=100)
    for n, (model_name, value) in enumerate(pae_plddt_per_model.items()):
        plt.subplot(1, num_models, n + 1)
        plt.title(model_name)
        plt.imshow(value["pae"], label=model_name, cmap="bwr", vmin=0, vmax=30)
        plt.colorbar()
    plt.savefig(f"{out_dir}/{jobname}_PAE.png")
    ##################################################################

# gets a list of sequences from the fasta file (one for monomer, multiple for multimer)
def get_sequences_from_fasta(fasta_file):
    seqs = []
    seq = ''
    for line in open(fasta_file):
        if line.startswith('>'):
            if seq:
                seqs.append(seq)
                seq = ''
        elif line:
            seq += line.rstrip()
    if seq:
        seqs.append(seq)
    return seqs

def log_template_info(feature_dict, logger):
    if 'template_domain_names' in feature_dict: # monomer
        logger.info('Templates shown here are reduced to 4 later on in the pipeline.')

    residue_idx = feature_dict['residue_index']
    template_aatype = feature_dict['template_aatype']
    if len(template_aatype.shape) == 3: # extra last dimension, one-hot encoded
        template_aatype = template_aatype.argmax(axis=-1)
    chain_starts = [idx for idx in range(len(residue_idx)) if residue_idx[idx] == 0]
    chain_ids = np.zeros_like(template_aatype[0])
    for chain_start_idx in chain_starts:
        chain_ids[chain_start_idx:]+=1

    logger.info('Templates found:')
    for k, single_template_aatype in enumerate(template_aatype):
        fros, tos = [], []
        idx = 0
        while True:
            while idx < len(single_template_aatype) and single_template_aatype[idx] == 21:
                idx += 1
            if idx == len(single_template_aatype): # if this is the case, we are at the end of the sequence
                break
            chain_id = chain_ids[idx]
            fros.append(idx)
            while idx < len(single_template_aatype) and single_template_aatype[idx] != 21 and chain_id == chain_ids[idx]:
                idx += 1
            tos.append(idx-1)
            if idx == len(single_template_aatype): # if this is the case, we are at the end of the sequence
                break
        for fro, to in zip(fros, tos):
            chain_id = chain_ids[fro]
            assert chain_id == chain_ids[to]
            logger.info(f' ({k+1}) At chain #{chain_id}, positions [{residue_idx[fro]}, {residue_idx[to]}]')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq',help='the input sequence or sequences (in case of multiple, separate with ":"')
    parser.add_argument('--jobname',help='the name for the job, used for file naming')
    parser.add_argument('--save_dir',help='the location where new directories (per permutation) will be created')
    parser.add_argument('--max_recycles',type=int,default=3,help='the maximum number of recycles',required=False)
    parser.add_argument('--tolerance',type=int,default=0,help='the tolerance level, decides when recycling stops',required=False)
    parser.add_argument('--do_relax',dest='do_relax',required=True)
    parser.add_argument('--no_templates',dest='use_templates',action='store_false')
    parser.set_defaults(use_templates=True)
    parser.add_argument('--msa_mode',type=str,default='mmseqs2_server',help='choose one of "mmseqs2_server", "mmseqs2_local", "alphafold_default", "single_sequence"',required=False)
    args = parser.parse_args()
    run_alphafold_advanced_complex(seq=args.seq,
                                   jobname=args.jobname,
                                   save_dir=args.save_dir,
                                   max_recycles=args.max_recycles,
                                   tolerance=args.tolerance,
                                   use_templates=args.use_templates,
                                   do_relax=args.do_relax,
                                   msa_mode=args.msa_mode)
    print('Finished run!')
