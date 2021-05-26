import os
import h5py
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import chi2
from tensorflow.python.client import device_lib
from tqdm import tqdm
import random
from helpers import generate_syn_genotypes, generate_syn_phenotypes
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
        print(data_path)
        with h5py.File(data_path, 'r') as file:
            print("Verifying the generated phenotypes...")
            genotype = file[0][:]
            n_indiv, n_snps, _ = genotype.shape
            assert n_indiv == syn_n_subjects