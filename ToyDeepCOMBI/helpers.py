import os
import h5py
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import chi2
from tensorflow.python.client import device_lib
from tqdm import tqdm
import random

from parameters_complete import (
    
    SYN_DATA_DIR, noise_snps, inform_snps, n_total_snps, syn_n_subjects, ttbr as ttbr, disease_IDs,
    FINAL_RESULTS_DIR, REAL_DATA_DIR
)

def generate_syn_genotypes(root_path = SYN_DATA_DIR, n_subjects=syn_n_subjects, n_info_snps=20,
                           n_noise_snps=10000, quantity=1):
        """ Generate synthetic genotypes and labels by removing all minor allels with low frequency,
        and missing SNPs.
        First step of data preprocessing, has to be followed by string_to_featmat()
        > Checks that that each SNP in each chromosome has at most 2 unique values in the whole
        dataset.
        """
        print("Starting synthetic genotypes generation..")
        try:
            print(root_path)
            os.remove(os.path.join(root_path, "genomic.h5py"))
        except FileNotFoundError:
            pass

def generate_syn_phenotypes():
    return "hello"