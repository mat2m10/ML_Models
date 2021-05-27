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

# First Method

def remove_small_frequencies(chrom):
    """
    This returns a chromosom with only minor allel freq > 0.15
    This chromosom can be safely used to generate synthetic genotypes/
    This returned Value can contain unmapped SNP's!
    """
    chrom[chrom == 48] = 255 # I dont understand what is happening here
    n_indiv = chrom.shape[0]
    lex_min = np.tile(np.min(chrom, axis=(0, 2)), [n_indiv, 1]) # Make a matrix of MAF?
    allel1 = chrom[:, :, 0]
    allel2 = chrom[:, :, 1]
    lexmin_mask_1 = (allel1 == lex_min) # True and False matrix
    lexmin_mask_2 = (allel2 == lex_min)
    maf = (lexmin_mask_1.sum(0) + lexmin_mask_2.sum(0))/(2*n_indiv) # Array of [n_indiv X maf]?
    maf = np.minimum(maf, 1-maf) # Array of [n_indiv X maf]? all < 0.5
    chrom[chrom == 255] = 48
    print(chrom)
    
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
            os.remove(os.path.join(root_path, "genomic.h5py"))
        except FileNotFoundError:
            pass
        with h5py.File(os.path.join(REAL_DATA_DIR,'AZ','chromo_2.mat'),'r') as f2:
            chrom2_full = np.array(f2.get('X')).T
            chrom2_full = chrom2_full.reshape(chrom2_full.shape[0],-1,3)[:,:,:2]
            chrom2_full = remove_small_frequencies(chrom2_full)
def generate_syn_phenotypes():
    return "hello"