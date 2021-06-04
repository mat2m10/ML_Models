import os
import numpy as np
from time import time
#  Use of os.environ to get access of environment variables
if 'ROOT_DIR' not in os.environ:
    os.environ['ROOT_DIR'] = "/home/Desktop/ML_Models/ToyDeepCOMBI"
    
if 'PREFIX' not in os.environ:
    os.environ['PREFIX'] = 'default'
    
disease_IDs = ['AZ']
disease = ['Alzheimer']
ROOT_DIR = os.environ['ROOT_DIR']
SYN_DATA_DIR = os.path.join(ROOT_DIR, 'data', 'synthetic')
REAL_DATA_DIR = os.path.join(ROOT_DIR,'data','UKB')
TEST_DIR = os.path.join(ROOT_DIR,'tests')
IMG_DIR = os.path.join(ROOT_DIR,'img')
TALOS_OUTPUT_DIR = os.path.join(TEST_DIR,'talos_output')
PARAMETERS_DIR = os.path.join(TEST_DIR,'parameters')
SAVED_MODELS_DIR = os.path.join(TEST_DIR,'exported_models')
TB_DIR = os.path.join(TEST_DIR,'exported_models')
NUMPY_ARRAYS = os.path.join(ROOT_DIR,'numpy_arrays')
FINAL_RESULTS_DIR = os.path.join(ROOT_DIR,'experiments','MONTAEZ_final')

pvalue_threshold = 1e-2 #1.1 #1e-4

###
ttbr = 6 #Tower to base ratio
syn_n_subjects = 4 # number of subjects for synthetisch data
inform_snps = 40 #?
noise_snps = 20
n_total_snps = inform_snps + noise_snps
top_k = 30 #?
real_top_k = 100 #?

# USED in char_matrix_to_featmat to scale the resulting feature matrix
pnorm_feature_scaling = 2 
real_pnorm_feature_scaling = 6

seed = 6 # Satan helps us
random_state = np.random.RandomState(seed)
