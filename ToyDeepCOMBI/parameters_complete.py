import os
import numpy as np
from time import time
#  Use of os.environ to get access of environment variables
print(os.environ)
if 'ROOT_DIR' not in os.environ:
    os.environ['ROOT_DIR'] = "/home/Desktop/ML_Models/ToyDeepCOMBI"
    
if 'PREFIX' not in os.environ:
    os.environ['PREFIX'] = 'default'