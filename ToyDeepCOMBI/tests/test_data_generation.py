import os
import h5py
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import chi2
from tensorflow.python.client import device_lib
from tqdm import tqdm
import random
from helpers import generate_syn_genotypes, generate_syn_phenotypes, check_genotype_unique_allels, genomic_to_featmat
from parameters_complete import (
    
    SYN_DATA_DIR, noise_snps, inform_snps, n_total_snps, syn_n_subjects, ttbr as ttbr, disease_IDs,
    FINAL_RESULTS_DIR, REAL_DATA_DIR)

class TestDataGeneration(object):
    """
    Generates all synthetic and real data and labels
    """
    def test_synthetic_genotypes_generation(self, rep):
        data_path = generate_syn_genotypes(root_path=SYN_DATA_DIR, n_subjects=syn_n_subjects,
                                          n_info_snps=inform_snps, n_noise_snps=noise_snps,
                                           quantity=rep)
        with h5py.File(data_path, 'r') as file:
            print("Verifying the generated phenotypes...")
            genotype = file['0'][:] # Synthetish genotype n_syn_indiv X (n_noise snp + n_inform_snp)
            print(genotype)
            n_indiv, n_snps, _ = genotype.shape
            assert n_indiv == syn_n_subjects
            assert n_snps == inform_snps + noise_snps
            # Check that at most 3 unique allels exists
            check_genotype_unique_allels(genotype)
            
    def test_feature_map_generation(self):
        """
        From synthetic data in h5py format generate a corresponding feature matrix
        a written featmat at data/synthetic/2d_syn_fm.h5py
        and a written featmat at data/synthetic/3d_syn_fm.h5py
        """
        m2 = genomic_to_featmat(embedding_type='2d', overwrite=True)