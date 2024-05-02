import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import *
from tensorflow.keras.models import *
from tensorflow.keras.callbacks import *

# set seed
tf.random.set_seed(2)

# dataset size
# 250, 500, 750, 1000
# for N in [500, 750, 1000]:

# load data
vf = np.load('vol_fraction.npy')[:N]

im = np.load('section.npy')[:N]
ss = np.load('stress.npy')[:N]
ef = np.load('fracture_strain.npy')[:N]
li = np.load('linear.npy')[:N]
mori = np.load('Moritanaka.npy')[:N] * 1e-8

# dataset split
p = np.where(vf < 79)   # train
q = np.where(vf >= 79)  # test

train_im = im[p]
train_ss = ss[p]
train_ef = ef[p]
train_li = li[p]
train_mori = mori[p]

test_im = im[q]
test_ss = ss[q]
test_ef = ef[q]
test_li = li[q]
test_mori = mori[q]

encoder = load_model('encoder_' + str(N) + '.h5')
decoder = load_model('decoder_' + str(N) + '.h5')

train_lv = encoder.predict(train_ss, verbose=0)

#
latent_dim = 2

def conv_block(inputs, filters):
    x = Conv2D(filters, 3, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(2, 2, padding='same')(x)
    return x

# conventional CNN model
def CNN1():
    x_inputs = Input((400, 400, 3))
    
    x = conv_block(x_inputs, 2)
    x = conv_block(x, 4)
    x = conv_block(x, 8)
    x = conv_block(x, 16)
    x = conv_block(x, 32)
    x = conv_block(x, 64)
    x = conv_block(x, 128)
    x = conv_block(x, 256)
    x = conv_block(x, 512)
    
    f = Flatten()(x)
    
    d = Dense(256, activation='relu')(f)
    outputs = Dense(latent_dim, activation=None)(d)
    
    model = Model(inputs=x_inputs, outputs=outputs)
    return model

# CNN model equipped with MT feature
def CNN2():
    x_inputs = Input((400, 400, 3))
    y_inputs = Input((1,))
    
    x = conv_block(x_inputs, 2)
    x = conv_block(x, 4)
    x = conv_block(x, 8)
    x = conv_block(x, 16)
    x = conv_block(x, 32)
    x = conv_block(x, 64)
    x = conv_block(x, 128)
    x = conv_block(x, 256)
    x = conv_block(x, 512)
    
    f = Flatten()(x)
    
    y = Dense(512, activation='relu')(y_inputs)
    
    c = Concatenate()([f, y])
    d = Dense(256, activation='relu')(c)
    outputs = Dense(latent_dim, activation=None)(d)
    
    model = Model(inputs=[x_inputs, y_inputs], outputs=outputs)
    return model

# CNN model equipped with NSF feature
def CNN3():
    x_inputs = Input((400, 400, 3))
    z_inputs = Input((400, 400, 3))
    
    x = conv_block(x_inputs, 2)
    z = conv_block(z_inputs, 2)

    x = conv_block(x, 4)
    z = conv_block(z, 4)
    xz = Concatenate()([x, z])

    x = conv_block(xz, 8)
    z = conv_block(z, 8)
    xz = Concatenate()([x, z])

    x = conv_block(x, 16)
    z = conv_block(xz, 16)
    xz = Concatenate()([x, z])

    x = conv_block(xz, 32)
    z = conv_block(z, 32)
    xz = Concatenate()([x, z])

    x = conv_block(x, 64)
    z = conv_block(xz, 64)
    xz = Concatenate()([x, z])

    x = conv_block(xz, 128)
    z = conv_block(z, 128)
    xz = Concatenate()([x, z])

    x = conv_block(x, 256)
    z = conv_block(xz, 256)
    xz = Concatenate()([x, z])

    x = conv_block(xz, 512)
    z = conv_block(z, 512)

    f1 = Flatten()(x)
    f2 = Flatten()(z)
    c = Concatenate()([f1, f2])
    
    d = Dense(256, activation='relu')(c)
    outputs = Dense(latent_dim, activation=None)(d)
    
    model = Model(inputs=[x_inputs, z_inputs], outputs=outputs)
    return model

# CNN model equipped with both MT feature and NSF feature
def CNN4():
    x_inputs = Input((400, 400, 3))
    y_inputs = Input((1,))
    z_inputs = Input((400, 400, 3))
    
    x = conv_block(x_inputs, 2)
    z = conv_block(z_inputs, 2)

    x = conv_block(x, 4)
    z = conv_block(z, 4)
    xz = Concatenate()([x, z])

    x = conv_block(xz, 8)
    z = conv_block(z, 8)
    xz = Concatenate()([x, z])

    x = conv_block(x, 16)
    z = conv_block(xz, 16)
    xz = Concatenate()([x, z])

    x = conv_block(xz, 32)
    z = conv_block(z, 32)
    xz = Concatenate()([x, z])

    x = conv_block(x, 64)
    z = conv_block(xz, 64)
    xz = Concatenate()([x, z])

    x = conv_block(xz, 128)
    z = conv_block(z, 128)
    xz = Concatenate()([x, z])

    x = conv_block(x, 256)
    z = conv_block(xz, 256)
    xz = Concatenate()([x, z])

    x = conv_block(xz, 512)
    z = conv_block(z, 512)

    f1 = Flatten()(x)
    f2 = Flatten()(z)
    
    y = Dense(512, activation='relu')(y_inputs)
    
    c = Concatenate()([f1, f2, y])
    d = Dense(256, activation='relu')(c)
    outputs = Dense(latent_dim, activation=None)(d)
    
    model = Model(inputs=[x_inputs, y_inputs, z_inputs], outputs=outputs)
    return model

model1 = CNN1()
model2 = CNN2()
model3 = CNN3()
model4 = CNN4()
