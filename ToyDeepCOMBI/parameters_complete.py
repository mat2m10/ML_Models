import os
import numpy as np
from time import time
#  Use of os.environ to get access of environment variables
print(os.environ)
if 'ROOT_DIR' not in os.environ:
    os.environ['ROOT_DIR'] = "/home/Desktop/ML_Models/ToyDeepCOMBI"
    
if 'PREFIX' not in os.environ:
    os.environ['PREFIX'] = 'default'
    
disease_IDs = ['AZ']
disease = ['Alzheimer']
ROOT_DIR = os.environ['ROOT_DIR']
SYN_DATA_DIR = os.path.join(ROOT_DIR, 'data', 'synthetic')
REAL_DATA_DIR = os.path.join(ROOT_DIR,'data','WTCCC')
TEST_DIR = os.path.join(ROOT_DIR,'tests')
IMG_DIR = os.path.join(ROOT_DIR,'img')
TALOS_OUTPUT_DIR = os.path.join(TEST_DIR,'talos_output')
PARAMETERS_DIR = os.path.join(TEST_DIR,'parameters')
SAVED_MODELS_DIR = os.path.join(TEST_DIR,'exported_models')
TB_DIR = os.path.join(TEST_DIR,'exported_models')
NUMPY_ARRAYS = os.path.join(ROOT_DIR,'numpy_arrays')
FINAL_RESULTS_DIR = os.path.join(ROOT_DIR,'experiments','MONTAEZ_final')