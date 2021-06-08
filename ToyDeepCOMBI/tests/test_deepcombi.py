import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow
import numpy as np
from helpers import generate_syn_phenotypes
from models import best_params_montaez
from models import create_montaez_dense_model
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import ReduceLROnPlateau
#from keras.callbacks import ReduceLROnPlateau
import innvestigate.utils as iutils

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
            l_0b=labels_cat[str(rep_to_plot)]
            """
            The good stuff starts here
            """
            model = create_montaez_dense_model(best_params_montaez)
            y_integers = np.argmax(l_0b[idx.train], axis=1)
            class_weights = class_weight.compute_class_weight('balanced', np.unique(y_integers), y_integers)
            d_class_weights = dict(enumerate(class_weights))
            model.fit(x=x_3d[idx.train], y=l_0b[idx.train], validation_data=(x_3d[idx.test], l_0b[idx.test]), epochs=best_params_montaez['epochs'], class_weight=d_class_weights, callbacks=[ ReduceLROnPlateau(monitor='val_loss', factor=best_params_montaez['factor'], patience=best_params_montaez['patience'], mode='min'),],)

            model = iutils.keras.graph.model_wo_softmax(model)
            analyzer = innvestigate.analyzer.LRPAlpha1Beta0(model)
            weights = analyzer.analyze(x_3d).sum(0)