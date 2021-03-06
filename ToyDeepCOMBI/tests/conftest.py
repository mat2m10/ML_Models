import os
import pdb
import h5py
import numpy as np
import pytest
import scipy
import tensorflow as tf
from sklearn.model_selection import StratifiedShuffleSplit

from parameters_complete import (
    SYN_DATA_DIR, ttbr as default_ttbr, 
    syn_n_subjects, 
    n_total_snps, 
    noise_snps, 
    inform_snps, 
    FINAL_RESULTS_DIR, 
    REAL_DATA_DIR,
    random_state
)
features_path = os.path.join(SYN_DATA_DIR, 'genomic.h5py')

from helpers import genomic_to_featmat, generate_syn_phenotypes
TRAIN_PERCENTAGE = 0.5
TEST_PERCENTAGE = 0.5
VAL_PERCENTAGE = 1 - TRAIN_PERCENTAGE - TEST_PERCENTAGE

from Indices import Indices

def pytest_addoption(parser):
    parser.addoption("--rep", 
                     action="store", 
                     default=2)
    parser.addoption("--ttbr", 
                     action="store", 
                     default=default_ttbr)
    
@pytest.fixture(scope='function')
def syn_true_pvalues(rep):
    """
    An array of zeroes except for where the loci are informative, in case 1.
    """
    pvalues = np.zeros((rep, n_total_snps), dtype=bool)
    pvalues[:, int(noise_snps/2):int(noise_snps/2)+inform_snps] = True
    return pvalues

@pytest.fixture
def rep(request):
    return int(request.config.getoption("--rep"))
@pytest.fixture(scope="module")
def syn_genomic_data():
    """
    Provides the 3D genomic data corresponding to all synthetic datasets
    {
        '0': matrix(N,n_snps,2),
        '1': matrix(N,n_snps,2),
        'rep': matrix(N,n_snps,2)
    }
    where 2 coressponds to the number of allels.    
    """
    return h5py.File(features_path,'r')
@pytest.fixture(scope="module")
def syn_fm(syn_genomic_data):
    """
    Returns a dictionary of feature matrices associated
    to the synthetic dataset
    """
    def fm_(dimensions):
        return genomic_to_featmat(embedding_type = dimensions)
    return fm_
@pytest.fixture(scope="function")
def syn_idx(syn_labels_0based):
    """ Gets indices splitting our datasets into  train and test sets
    :return
    {
        '0':Index,
        ...,
        'rep':Index
    }
    Ex: indices['0'].train gets the indices for dataset 0, train set.
    """
    assert VAL_PERCENTAGE <=0.00001
    indices_ = {}
    splitter =  StratifiedShuffleSplit(n_splits=1, test_size = TEST_PERCENTAGE, random_state=random_state)
    for key, labels in syn_labels_0based.items():

        train_indices, test_indices = next(splitter.split(np.zeros(syn_n_subjects), labels))
        indices_[key] = Indices(train_indices, test_indices, None)
    
    print('Dataset sizes: Train: {}; Test: {}; Validation: ERROR'.format(len(train_indices),len(test_indices)))
    return indices_

    
@pytest.fixture(scope='function')
def syn_labels_0based(syn_labels):
    """ Same as syn_labels but labels are 2-hot-encoded
    """
    labels_0based = {}
    for key, l in syn_labels.items():
        labels_0based[key] = ((l+1)/2).astype(int)
    return labels_0based

@pytest.fixture(scope='function')
def syn_labels(rep, ttbr):
    """
    Loads synthetic labels for all datasets
    :return:
    {
        '0':[-1,-1,...,1],
        ...,
        'rep':[1,1,-1,...,-1]
    }

    """
    return generate_syn_phenotypes(root_path=SYN_DATA_DIR, 
                                   tower_to_base_ratio=ttbr, 
                                   n_info_snps=inform_snps, 
                                   n_noise_snps=noise_snps, 
                                   quantity=rep)

@pytest.fixture(scope='function')
def ttbr(request):
    """
    Returns the tower to base ratio

    """
    return int(request.config.getoption("--ttbr"))
