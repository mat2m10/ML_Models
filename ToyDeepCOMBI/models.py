import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from keras.regularizers import l1_l2
from helpers import get_available_gpus
from tensorflow.keras import optimizers
best_params_montaez = {
    'epochs': 500,
#    'batch_size': 32,   
    'l1_reg': 1e-4,
    'l2_reg': 1e-6,
    'lr' : 0.01,
    'dropout_rate':0.3,
    'factor':0.7125,
    'patience':50,
    'hidden_neurons':64
}

def create_montaez_dense_model(params):
    model=Sequential()
    model.add(Flatten(input_shape=(int(params['n_snps']), 3)))

    model.add(Dense(activation='relu', units=int(params['hidden_neurons']), kernel_regularizer=l1_l2(l1=params['l1_reg'], l2=params['l2_reg'])))

    model.add(Dropout(params['dropout_rate']))
    model.add(Dense(activation='relu', units=int(params['hidden_neurons']), kernel_regularizer=l1_l2(l1=params['l1_reg'], l2=params['l2_reg'])))
	
    model.add(Dropout(params['dropout_rate']))
    model.add(Dense(activation='softmax', units=2, kernel_regularizer=l1_l2(l1=params['l1_reg'], l2=params['l2_reg'])))
	
    nb_gpus = get_available_gpus()
    if nb_gpus ==1:
        parallel_gpu_jobs(0.5)
    if nb_gpus >= 2:
        model = multi_gpu(model, gpus=get_available_gpus())
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=params['lr']),  weighted_metrics=['categorical_accuracy'], metrics=['categorical_accuracy'])
    #tf.keras.metrics.AUC(), tf.keras.metrics.AUC(curve='PR')
    return model