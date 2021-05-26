import os
import pdb
import h5py
import numpy as np
import pytest
import scipy
import tensorflow as tf

from parameters_complete import SYN_DATA_DIR, ttbr as default_ttbr, syn_n_subjects, n_total_snps, noise_snps, inform_snps, FINAL_RESULTS_DIR, REAL_DATA_DIR

def pytest_addoption(parser):
    parser.addoption("--rep", action="store", default=2)
    
@pytest.fixture
def rep(request):
    return int(request.config.getoption("--rep"))