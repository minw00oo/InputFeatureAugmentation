import numpy as np
import matplotlib.pyplot as plt
from math import *

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import *
from tensorflow.keras.models import *

# set seed
tf.random.set_seed(2)

# dataset size
# 250, 500, 750, and 1000
for N in [500, 750, 1000]:
    # load data
    vf = np.load('vol_fraction.npy')[:N]
    
    im = np.load('section.npy')[:N]       # input feature 1: cross-sectional image
    li = np.load('linear.npy')[:N]        # input feature 2: normalized stress field
    mori = np.load('Moritanaka.npy')[:N]  # input feature 3: mori-tanaka effective elastic modulus
    ss = np.load('stress.npy')[:N]        # output 1: stress
    ef = np.load('fracture_strain.npy')[:N]  # output 2: fracture strain
    
    # dataset split
    p = np.where(vf < 79)   # train: lower than VF = 50%
    q = np.where(vf >= 79)  # test : higher than VF = 50%
    
    train_im = im[p]
    train_ss = ss[p]
    train_ef = ef[p]
    train_li = li[p]
    train_mori = mori[p] * 1e-8
    
    test_im = im[q]
    test_ss = ss[q]
    test_ef = ef[q]
    test_li = li[q]
    test_mori = mori[q] * 1e-8
    
    # Auotoencoder?
    def encoder():
        inputs = Input(41,)
        x = Dense(500, activation='relu', input_shape=(41,))(inputs)
        x = Dense(300, activation='relu')(x)
        outputs = Dense(6, activation=None)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    def decoder():
        inputs = Input(6,)
        x = Dense(300, activation='relu', input_shape=(2,))(inputs)
        x = Dense(500, activation='relu')(x)
        outputs = Dense(41, activation=None)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    Encoder = encoder()
    Decoder = decoder()
    
    def autoencoder():
        inputs = Input(41,)
        x = Encoder(inputs)
        outputs = Decoder(x)
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    AE = autoencoder()
    AE.summary()
    
    # train
    # step-wise decaying learning rate
    for lr in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
        adam = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        
        AE.compile(optimizer=adam,
                      loss='mse',
                      metrics=['mse'])
        
        history = AE.fit(train_ss, train_ss, batch_size=50, epochs=100)
    
    pred_ss = AE.predict(test_ss)
    
    #
    Encoder.save('encoder_' + str(N) + '.h5')
    Decoder.save('decoder_' + str(N) + '.h5')
    AE.save('autoencoder_' + str(N) + '.h5')
