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

# Second Method
def check_genotype_unique_allels(genotype):
    """
    Check if we dont have unique allels (maybe)
    """
    assert(max([len(np.unique(genotype[:, i, :]))
                for i in range(genotype.shape[1])]) <= 3)
# First Method
def remove_small_frequencies(chrom):
    """
    This returns a chromosom with only minor allel freq > 0.15
    This chromosom can be safely used to generate synthetic genotypes/
    This returned Value can contain unmapped SNP's!
    """
    chrom[chrom == 48] = 255 # I think passing from int(48) to int(255)
    n_indiv = chrom.shape[0]
    lex_min = np.tile(np.min(chrom, axis=(0, 2)), [n_indiv, 1]) # Make a matrix of MAF?
    allel1 = chrom[:, :, 0]
    allel2 = chrom[:, :, 1]
    lexmin_mask_1 = (allel1 == lex_min) # True and False matrix
    lexmin_mask_2 = (allel2 == lex_min)
    maf = (lexmin_mask_1.sum(0) + lexmin_mask_2.sum(0))/(2*n_indiv) # Array of [n_indiv X maf]?
    maf = np.minimum(maf, 1-maf) # Array of [n_indiv X maf]? all < 0.5
    chrom[chrom == 255] = 48
    chrom_low_f_removed = chrom[:, maf > 0.15, :] # Remove elements with their minor allele < 0.15 
    chrom_low_f_removed.sort()
    check_genotype_unique_allels(chrom_low_f_removed) # Second Method
    return chrom_low_f_removed
    
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
            chrom2_full = remove_small_frequencies(chrom2_full) # First Method
            chrom2_full = chrom2_full[:, :n_noise_snps]
            assert chrom2_full.shape[0] > n_subjects # We want to keep only n synthetisch people
            chrom2 = chrom2_full[:n_subjects]
        with h5py.File(os.path.join(REAL_DATA_DIR,'AZ', 'chromo_1.mat'), 'r') as f:
            chrom1_full = np.array(f.get('X')).T
            chrom1_full = chrom1_full.reshape(chrom1_full.shape[0], -1, 3)[:, :, :2]
            chrom1_full = remove_small_frequencies(chrom1_full)
            assert chrom1_full.shape[0] > n_subjects 
            chrom1 = chrom1_full[:n_subjects]  # Keep only 300 people
        """
        
        """
        half_noise_size = int(n_noise_snps/2)
        with h5py.File(os.path.join(root_path, 'genomic.h5py'), 'w') as file:
            for i in tqdm(range(quantity)):
                print(range(quantity))
                print(np.arange(len(chrom1[0,:])-n_info_snps))
                # random starting position
                start_info = random.choice(np.arange(len(chrom1[0,:])-n_info_snps))
                #chrom1_subset = chrom1[:, i*n_info_snps:(i+1)*n_info_snps]
                chrom1_subset = chrom1[:, start_info:(start_info+n_info_snps)]
                data = np.concatenate((chrom2[:, :half_noise_size], chrom1_subset, chrom2[:,
                                                                                          half_noise_size:half_noise_size*2]), axis=1)
                ## If the number of encoded SNPs is insufficient
                print(data.shape[1])
                print(n_info_snps + n_noise_snps)
                if data.shape[1] != n_info_snps + n_noise_snps:
                    raise Exception("Not enough SNPs")
                # Write everything!
                file.create_dataset(str(i), data=data)
        return os.path.join(root_path, 'genomic.h5py')
    
def generate_syn_phenotypes():
    return "hello"