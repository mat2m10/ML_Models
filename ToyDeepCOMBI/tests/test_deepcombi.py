import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow

from helpers import generate_syn_phenotypes
from models import best_params_montaez

class TestDeepCOMBI(object):
    def test_lrp_svm(self, syn_genomic_data, 
                     syn_fm, syn_idx, rep, tmp_path, 
                     syn_true_pvalues):
        """ Compares efficiency of the combi method with several TTBR
        """
        rep_to_plot = 0
        ttbrs = [0.5,1,1.5]
        idx = syn_idx[str(rep_to_plot)]
        fig, axes = plt.subplots(len(ttbrs), 5, figsize=[30,15])
        x_3d = syn_fm("3d")[str(rep_to_plot)][:]
        x_2d = syn_fm("2d")[str(rep_to_plot)][:]
        indices_true= [inds_true for inds_true, 
                       x in enumerate(syn_true_pvalues[0].flatten()) if x]
        for i, ttbr in enumerate(ttbrs):
            print('Using tbrr={}'.format(ttbr))
            labels = generate_syn_phenotypes(tower_to_base_ratio=ttbr, quantity=rep)
            labels_cat = {}
            for key, l in labels.items():
                labels_cat[key] = tensorflow.keras.utils.to_categorical((l+1)/2)
            best_params_montaez['n_snps']= x_3d.shape[1]