import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import *
from tensorflow.keras.models import *
from tensorflow.keras.callbacks import *

tf.random.set_seed(42)
np.random.seed(42)

# Load data
vf = np.load('./augmentedDataSet/augmented_vf.npy')
img = np.load('./augmentedDataSet/augmented_img.npy')
img = np.expand_dims(img, axis=-1)
mt = np.load('./augmentedDataSet/augmented_mt.npy')
sif = np.load('./augmentedDataSet/augmented_SIF.npy')
target = np.load('./augmentedDataSet/augmented_target.npy')

# Split data
# img_train, img_test, mt_train, mt_test, sif_train, sif_test, y_train, y_test, vf_train, vf_test = train_test_split(
#     img, mt, sif, target, vf, test_size=0.2, random_state=42)

trainSize = 800

img_train, img_test = img[:trainSize], img[2000:]
mt_train, mt_test = mt[:trainSize], mt[2000:]
sif_train, sif_test = sif[:trainSize], sif[2000:]
y_train, y_test = target[:trainSize], target[2000:]
vf_train, vf_test = vf[:trainSize], vf[2000:]

#%%
def conv_block(inputs, filters):
    x = Conv2D(filters, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)
    return x

def dense_block(inputs, filters):
    x = Dense(filters)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)
    return x

def build_CNN():
    inputs1 = Input(shape=(256, 256, 1))
    inputs2 = Input(shape=(1,))
    # inputs3 = Input(shape=(1,))
    
    x = conv_block(inputs1, 4)
    x = conv_block(x, 8)
    x = conv_block(x, 16)
    x = conv_block(x, 32)
    x = conv_block(x, 64)
    x = conv_block(x, 128)
    x = conv_block(x, 256)
    x = conv_block(x, 512)
    
    x = Flatten()(x)
    x = Concatenate()([x, inputs2])
    x = dense_block(x, 256)
    
    outputs = Dense(2, activation='linear')(x)
    
    return Model(inputs=[inputs1, inputs2], outputs=outputs)

#%%
def print_elapsed_time(start_time, end_time):
    elapsed_time = end_time - start_time
    print(f'training completed in {elapsed_time:.2f} seconds')

batch_size = 64
epochs = 300
validation_split = 0.1

def scheduler(epoch, lr):
    if epoch > 0 and epoch % 100 == 0:
        return lr * 0.1
    return lr

model = build_CNN()

optimizer = optimizers.Adam(lr=1e-3)
model.compile(optimizer=optimizer, loss='mse')
lr_scheduler = LearningRateScheduler(scheduler)

print(f"training Model")

start_time = time.time()

history = model.fit(x=[img_train, mt_train],
                    y=y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=[[img_test, mt_test], y_test],
                    callbacks=[lr_scheduler])

end_time = time.time()

print_elapsed_time(start_time, end_time)

model.save('./model/MT_800.h5')

model = build_CNN()

optimizer = optimizers.Adam(lr=1e-3)
model.compile(optimizer=optimizer, loss='mse')
lr_scheduler = LearningRateScheduler(scheduler)

print(f"training Model")

start_time = time.time()

history = model.fit(x=[img_train, sif_train],
                    y=y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=[[img_test, sif_test], y_test],
                    callbacks=[lr_scheduler])

end_time = time.time()

print_elapsed_time(start_time, end_time)

model.save('./model/SIF_800.h5')

#%%
from sklearn.metrics import r2_score

y_pred = model.predict([img_test, mt_test, sif_test], verbose=0)

r2_score1 = r2_score(y_test[:, 0], y_pred[:, 0])
r2_score2 = r2_score(y_test[:, 1], y_pred[:, 1])

#%%
position = ['bottom', 'left', 'top', 'right']

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

for i in range(2):
    for pos in position:
        ax[i].spines[pos].set_visible(True)
        ax[i].spines[pos].set_linewidth(1)
        ax[i].set_aspect('equal')

ax[0].scatter(y_test[:, 0], y_pred[:, 0], c=y_test[:, 0], cmap='RdYlBu_r', edgecolor='k')
ax[0].plot([np.min(y_test[:, 0]), np.max(y_test[:, 0])],
           [np.min(y_test[:, 0]), np.max(y_test[:, 0])], 'k--')
ax[0].set_xlim([np.min(y_test[:, 0]), np.max(y_test[:, 0])])
ax[0].set_ylim([np.min(y_test[:, 0]), np.max(y_test[:, 0])])
ax[0].grid()
ax[0].set_xlabel('Target', font='arial', fontsize=15)
ax[0].set_ylabel('Prediction', font='arial', fontsize=15)
ax[0].set_title('Stiffness (MPa)', font='arial', fontsize=15)
ax[0].text(0.05, 0.95, f'R2: {r2_score1:.2f}',
           transform=ax[0].transAxes,
           font='arial', fontsize=15,
           verticalalignment='top',
           bbox=dict(boxstyle='round, pad=0.3', edgecolor='black', facecolor='white'))

ax[1].scatter(y_test[:, 1], y_pred[:, 1], c=y_test[:, 1], cmap='RdYlBu_r', edgecolor='k')
ax[1].plot([np.min(y_test[:, 1]), np.max(y_test[:, 1])],
           [np.min(y_test[:, 1]), np.max(y_test[:, 1])], 'k--')
ax[1].set_xlim([np.min(y_test[:, 1]), np.max(y_test[:, 1])])
ax[1].set_ylim([np.min(y_test[:, 1]), np.max(y_test[:, 1])])
ax[1].grid()
ax[1].set_xlabel('Target', font='arial', fontsize=15)
ax[1].set_ylabel('Prediction', font='arial', fontsize=15)
ax[1].set_title('Strength (MPa)', font='arial', fontsize=15)
ax[1].text(0.05, 0.95, f'R2: {r2_score2:.2f}',
           transform=ax[1].transAxes,
           font='arial', fontsize=15,
           verticalalignment='top',
           bbox=dict(boxstyle='round, pad=0.3', edgecolor='black', facecolor='white'))

#%%
def evaluate(y_target, y_pred):
    return np.abs(y_target - y_pred) / y_pred * 100

E = []
U = []

for i in range(len(y_test)):
    E.append(evaluate(y_test[i, 0], y_pred[i, 0]))
    U.append(evaluate(y_test[i, 1], y_pred[i, 1]))
