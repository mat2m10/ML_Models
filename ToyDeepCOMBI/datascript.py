import pickle
import pandas as pd
import numpy as np

from pandas_plink import read_plink
(bim, fam, bed) = read_plink("./data_ignored/toy")
X = bed.compute()
file = open("./data_ignored/bed.txt", "w")
for row in bed:
    np.savetxt(file, row)
file.close()
