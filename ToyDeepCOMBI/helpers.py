import os
import h5py
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import chi2
from tensorflow.python.client import device_lib
from tqdm import tqdm
import random

from parameters_complete import (
    
    SYN_DATA_DIR, 
    noise_snps, 
    inform_snps, 
    n_total_snps, 
    syn_n_subjects, 
    ttbr as ttbr, 
    disease_IDs,
    seed,
    FINAL_RESULTS_DIR, 
    REAL_DATA_DIR, 
    pnorm_feature_scaling
)
# Sixth Method
def generate_syn_phenotypes(root_path=SYN_DATA_DIR, tower_to_base_ratio=ttbr, n_info_snps=2, n_noise_snps=10, quantity=1):
    """
    > Assumes that each SNP has at most 3 unique values in the whole dataset (Two allels and possibly unmapped values)
    IMPORTANT: DOES NOT LOAD FROM FILE
    returns: dict(key, labels)
    """
    print("Starting synthetic phenotypes generation...")
    info_snp_idx = int(n_noise_snps/2) + int(n_info_snps/2)
    labels_dict = {}
    def f(genotype, key):
        n_indiv = genotype.shape[0]
        print(info_snp_idx)
        info_snp = genotype[:,  info_snp_idx]  # (n,2)
        major_allel = np.max(info_snp)
        major_mask_1 = (info_snp[:, 0] == major_allel)  # n
        major_mask_2 = (info_snp[:, 1] == major_allel)  # n
        invalid_mask = (info_snp[:, 0] == 48) | (info_snp[:, 1] == 48)
        nb_major_allels = np.sum(
            [major_mask_1, major_mask_2, invalid_mask], axis=0) - 1.0  # n
        probabilities = 1 / \
            (1 + np.exp(-tower_to_base_ratio * (nb_major_allels - np.median(nb_major_allels))))
        random_vector = np.random.RandomState(seed).uniform(low=0.0, high=1.0, size=n_indiv)
        labels = np.where(probabilities > random_vector, 1, -1)
        assert genotype.shape[0] == labels.shape[0]
        labels_dict[key] = labels   
        del genotype
    with h5py.File(os.path.join(root_path, 'genomic.h5py'), 'r') as h5py_file:
        
        Parallel(n_jobs=-1, require='sharedmem')(delayed(f)(h5py_file[str(i)][:],str(i)) for i in tqdm(range(quantity)))
    return labels_dict

    # Fifth Method
def char_matrix_to_featmat(data, embedding_type='2d', norm_feature_scaling=pnorm_feature_scaling):
    """
    transforms AA AT TT,  NOOO == [1 1] [1 20] [20 20]
    into 1 0 0; 0 1 0; 0 0 1
    """
    ###  Global Parameters   ###
    (n_subjects, num_snp3, _) = data.shape

    # Computes lexicographically highest and lowest nucleotides for each position of each strand
    lexmax_overall_per_snp = np.max(data, axis=(0, 2))
    #data_now = data.copy()

    data[data == 48] = 255
    lexmin_overall_per_snp = np.min(data, axis=(0, 2))
    # Masks showing valid or invalid indices
    # SNPs being unchanged amongst the whole dataset, hold no information

    lexmin_mask_per_snp = np.tile(lexmin_overall_per_snp, [n_subjects, 1])
    lexmax_mask_per_snp = np.tile(lexmax_overall_per_snp, [n_subjects, 1])

    invalid_bool_mask = (lexmin_mask_per_snp == lexmax_mask_per_snp)

    allele1 = data[:, :, 0]
    allele2 = data[:, :, 1]

    # indices where allel1 equals the lowest value
    allele1_lexminor_mask = (allele1 == lexmin_mask_per_snp)
    # indices where allel1 equals the highest value
    allele1_lexmajor_mask = (allele1 == lexmax_mask_per_snp)
    # indices where allel2 equals the lowest value
    allele2_lexminor_mask = (allele2 == lexmin_mask_per_snp)
    # indices where allel2 equals the highest value
    allele2_lexmajor_mask = (allele2 == lexmax_mask_per_snp)

    f_m = np.zeros((n_subjects, num_snp3), dtype=(int, 3))
    f_m[allele1_lexminor_mask & allele2_lexminor_mask] = [1, 0, 0]
    f_m[(allele1_lexmajor_mask & allele2_lexminor_mask) |
        (allele1_lexminor_mask & allele2_lexmajor_mask)] = [0, 1, 0]
    f_m[allele1_lexmajor_mask & allele2_lexmajor_mask] = [0, 0, 1]
    f_m[invalid_bool_mask] = [0, 0, 0]
    f_m = np.reshape(f_m, (n_subjects, 3*num_snp3))
    f_m = f_m.astype(np.double)
    # Rescale feature matrix
    f_m -= np.mean(f_m, dtype=np.float64, axis=0) # centering    
    stddev = ((np.abs(f_m)**norm_feature_scaling).mean(axis=0) * f_m.shape[1])**(1.0/norm_feature_scaling)
    
    # Safe division
    f_m = np.divide(f_m, stddev, out=np.zeros_like(f_m), where=stddev!=0)
    # Reshape Feature matrix
    if embedding_type == '2d':
        pass
    elif embedding_type == '3d':
        f_m = np.reshape(f_m, (n_subjects, num_snp3, 3))
    return f_m.astype(float)

# Fourth Method
def genomic_to_featmat(embedding_type="2d", overwrite=False):
    """
    Transforms a h5py dictionary of genomic matrix of chars to a tensor of features {'0': genomic_mat_0, ... 'rep': genomic_mat_rep}
    3d [n_subjects, 3*n_snps] 2d [n_subj[n_snps[0 0 1]]]
    """
    
    data_path = os.path.join(SYN_DATA_DIR, 'genomic.h5py')
    fm_path = os.path.join(SYN_DATA_DIR, embedding_type + '_fm.h5py')
    if overwrite:
        try:
            os.remove(fm_path)
        except (FileNotFoundError, OSError):
            pass

    if not overwrite:
        try:
            return h5py.File(fm_path, 'r')
        except (FileNotFoundError, OSError):
            print('Featmat not found: generating new one...')
    
    with h5py.File(fm_path, 'w') as feature_file:
        with h5py.File(data_path, 'r') as data_file:
            for key in tqdm(list(data_file.keys())):
                data = data_file[key][:]
                f_m = char_matrix_to_featmat(data, embedding_type)
                feature_file.create_dataset(key, data=f_m)
                del data

    return h5py.File(fm_path, 'r')

# Third Method
def check_genotype_unique_allels(genotype):
    """
    Check if we dont have unique allels (maybe)
    """
    assert(max([len(np.unique(genotype[:, i, :]))
                for i in range(genotype.shape[1])]) <= 3)
# Second Method
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

# First Method
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
        Create #rep different datasets, each with a different set of informative SNPs - no they only
        need to be random, not all different. otherwise we couldnt have 10000 datasets.
        """
        half_noise_size = int(n_noise_snps/2)
        with h5py.File(os.path.join(root_path, 'genomic.h5py'), 'w') as file:
            for i in tqdm(range(quantity)):
                # random starting position
                start_info = random.choice(np.arange(len(chrom1[0,:])-n_info_snps))
                #chrom1_subset = chrom1[:, i*n_info_snps:(i+1)*n_info_snps]
                chrom1_subset = chrom1[:, start_info:(start_info+n_info_snps)]
                data = np.concatenate((chrom2[:, :half_noise_size], chrom1_subset, chrom2[:,
                                                                                          half_noise_size:half_noise_size*2]), axis=1)
                ## If the number of encoded SNPs is insufficient
                if data.shape[1] != n_info_snps + n_noise_snps:
                    raise Exception("Not enough SNPs")
                # Write everything!
                file.create_dataset(str(i), data=data)
        return os.path.join(root_path, 'genomic.h5py')