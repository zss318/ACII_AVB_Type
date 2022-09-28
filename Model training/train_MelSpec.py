# %% Packages
import pandas as pd
import numpy as np
import scipy.io as sio

import os

import tensorflow
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Dense, Concatenate, SpatialDropout2D
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, AveragePooling2D, Reshape, multiply, Permute, Concatenate, Conv2D, Add, Activation
from tensorflow.keras.layers import Input, Dropout, ZeroPadding2D, Flatten, SeparableConv2D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical


# %% Datagenerator
def f_datagen_ACII2022Type( path_in, df_class0, df_class1, df_class2, df_class3, df_class4, df_class5, df_class6, df_class7,  des_samps ):

    while True:

        batch_size = des_samps*8

        files2load_class0 = [path_in+"/"+file+".mat" for file in df_class0["filenames"].sample(des_samps)]
        files2load_class1 = [path_in+"/"+file+".mat" for file in df_class1["filenames"].sample(des_samps)]
        files2load_class2 = [path_in+"/"+file+".mat" for file in df_class2["filenames"].sample(des_samps)]
        files2load_class3 = [path_in+"/"+file+".mat" for file in df_class3["filenames"].sample(des_samps)]
        files2load_class4 = [path_in+"/"+file+".mat" for file in df_class4["filenames"].sample(des_samps)]
        files2load_class5 = [path_in+"/"+file+".mat" for file in df_class5["filenames"].sample(des_samps)]
        files2load_class6 = [path_in+"/"+file+".mat" for file in df_class6["filenames"].sample(des_samps)]
        files2load_class7 = [path_in+"/"+file+".mat" for file in df_class7["filenames"].sample(des_samps)]

        Feats_class0 = [sio.loadmat( file )["Feats"] for file in files2load_class0]
        Feats_class1 = [sio.loadmat( file )["Feats"] for file in files2load_class1]
        Feats_class2 = [sio.loadmat( file )["Feats"] for file in files2load_class2]
        Feats_class3 = [sio.loadmat( file )["Feats"] for file in files2load_class3]
        Feats_class4 = [sio.loadmat( file )["Feats"] for file in files2load_class4]
        Feats_class5 = [sio.loadmat( file )["Feats"] for file in files2load_class5]
        Feats_class6 = [sio.loadmat( file )["Feats"] for file in files2load_class6]
        Feats_class7 = [sio.loadmat( file )["Feats"] for file in files2load_class7]

        Feats = np.vstack( [Feats_class0, Feats_class1, Feats_class2, Feats_class3, Feats_class4, Feats_class5, Feats_class6, Feats_class7] )
        Feats = np.expand_dims(Feats, -1 )
        Labs = np.vstack( [0*np.ones(des_samps), 1*np.ones(des_samps), 2*np.ones(des_samps), 3*np.ones(des_samps), 4*np.ones(des_samps), 5*np.ones(des_samps), 6*np.ones(des_samps), 7*np.ones(des_samps) ] )
        Labs = to_categorical(Labs, 8)

        mixup_param = 0.2
        Mix_Coef = np.random.beta(mixup_param, mixup_param, batch_size//2 )
        X_Mix = Mix_Coef.reshape(batch_size//2, 1, 1, 1)
        y_Mix = Mix_Coef.reshape(batch_size//2, 1)

        X1 = Feats[ 0:batch_size//2, :, :, :]
        y1 = Labs[ 0:batch_size//2 ]
        X2 = Feats[ batch_size//2:, :, :, :]
        y2 = Labs[ batch_size//2: ]
        X = X1 * X_Mix + X2 * (1.0 - X_Mix)
        y = y1*y_Mix + y2*(1.0 - y_Mix)
        Feats = np.vstack( [Feats, X] )
        Labs = np.vstack( [Labs, y] )

        yield Feats, Labs

# %% Settings
path_in = "" # path to MelSpec

path_save_Results = "" # path to save trained models

model_ID = "ACII2022Type_A"
path_save_trained_model = path_save_Results + "/Trained Models - " + model_ID
os.mkdir( path_save_trained_model )

file_in_metadata_train = "" # path to metadata csv

df_train = pd.read_excel( file_in_metadata, index_col=None)
df_train["filename"] = df_train["File_ID"].str.replace("[","",regex=False).str.replace("]","",regex=False)
df_train_grouped = df_train.groupby("Voc_Type")
df_train_class0 = df_train_grouped.get_group('Laugh')
df_train_class1 = df_train_grouped.get_group('Grunt')
df_train_class2 = df_train_grouped.get_group('Cry')
df_train_class3 = df_train_grouped.get_group('Pant')
df_train_class4 = df_train_grouped.get_group('Gasp')
df_train_class5 = df_train_grouped.get_group('Other')
df_train_class6 = df_train_grouped.get_group('Groan')
df_train_class7 = df_train_grouped.get_group('Scream')


# %% Define model
def my_model_A( input_shape ):

    inputs = Input(shape=input_shape)


    bn = BatchNormalization(center=True, scale=True)(inputs)

    branch1 = Conv2D(16, kernel_size=(10,1), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4), use_bias=False )(bn)
    branch1 = BatchNormalization(center=False, scale=False)(branch1)
    branch1 = Activation('relu')(branch1)
    branch1 = SpatialDropout2D(0.2)(branch1)
    branch1 = MaxPooling2D(pool_size=(4, 2))(branch1)

    branch2 = Conv2D(16, kernel_size=(1,10), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4), use_bias=False )(bn)
    branch2 = BatchNormalization(center=False, scale=False)(branch2)
    branch2 = Activation('relu')(branch2)
    branch2 = SpatialDropout2D(0.2)(branch2)
    branch2 = MaxPooling2D(pool_size=(4, 2))(branch2)

    branch3 = Conv2D(16, kernel_size=(3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4), use_bias=False )(bn)
    branch3 = BatchNormalization(center=False, scale=False)(branch3)
    branch3 = Activation('relu')(branch3)
    branch3 = SpatialDropout2D(0.2)(branch3)
    branch3 = MaxPooling2D(pool_size=(4, 2))(branch3)

    one_path = Concatenate(axis=-1)([branch1, branch2, branch3])

    x = BatchNormalization(center=False, scale=False)(one_path)
    x = Conv2D( 32, kernel_size=(5,5), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4), use_bias=False)(x)
    x = BatchNormalization(center=False, scale=False)(x)
    x = Activation('relu')(x)
    x = SpatialDropout2D(0.3)(x)
    x = MaxPooling2D(pool_size=(2, 4), padding='same')(x)

    x = BatchNormalization(center=False, scale=False)(x)
    x = Conv2D( 32, kernel_size=(5,5), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4), use_bias=False)(x)
    x = BatchNormalization(center=False, scale=False)(x)
    x = Activation('relu')(x)
    x = SpatialDropout2D(0.3)(x)
    x = MaxPooling2D(pool_size=(2, 4), padding='same')(x)

    x = BatchNormalization(center=False, scale=False)(x)
    x = Conv2D( 64, kernel_size=(3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4), use_bias=False)(x)
    x = BatchNormalization(center=False, scale=False)(x)
    x = Activation('relu')(x)
    x = SpatialDropout2D(0.4)(x)
    x = MaxPooling2D(pool_size=(2, 4), padding='same')(x)

    OutputPath = BatchNormalization(center=False, scale=False)(x)
    OutputPath = Conv2D( 16, kernel_size=(3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4), use_bias=False)(OutputPath)
    OutputPath = GlobalMaxPooling2D()(OutputPath)
    OutputPath = BatchNormalization(center=True, scale=True)(OutputPath)

    OutputPath = Dense(8, activation="softmax")(OutputPath)
    model = Model(inputs=inputs, outputs=OutputPath)

    return model


# %% Train model

num_channels = 1
num_freq_bin = 128
num_time_bin = 150
num_classes = 8
max_lr = 0.2
num_epochs = 100
max_samples_Train = 7046
samps_per_class = 50
steps_per_epoch = int(np.floor(max_samples_Train/(samps_per_class))-1)

model_out = my_model_A( input_shape=[num_freq_bin, num_time_bin, num_channels] )


model_out.compile( loss='categorical_crossentropy',
                  optimizer = tensorflow.keras.optimizers.Adam(learning_rate=1e-2),
                  metrics=['accuracy'])

save_path = path_save_trained_model + "/model-{epoch:02d}-train-{accuracy:.4f}.hdf5"
checkpoint = tensorflow.keras.callbacks.ModelCheckpoint(save_path, monitor='accuracy', verbose=1, save_best_only=False, mode='max')
callbacks = [checkpoint]

history = model_out.fit(f_datagen_ACII2022Type( path_in, df_train_class0, df_train_class1, df_train_class2, df_train_class3, df_train_class4, df_train_class5, df_train_class6, df_train_class7, des_samps=samps_per_class),
                        epochs=num_epochs,
                        batch_size=samps_per_class*8,
                        steps_per_epoch=steps_per_epoch,
                        validation_data = None,
                        verbose=1,
                        initial_epoch=0,
                        callbacks=callbacks  )
