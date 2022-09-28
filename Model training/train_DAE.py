# %% Packages
import numpy as np
import pandas as pd
import scipy.io as sio

import tensorflow
from tensorflow.keras.layers import BatchNormalization, Activation, Dense, Dropout, Input, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical


# %% Settings

path_in = "" # feats
path_save = "" # trained models

fileIn_csv = "/type_info.csv"

df_metadata = pd.read_csv( fileIn_csv, index_col=None)
df_metadata["filename"] = df_metadata["File_ID"].str.replace("[","",regex=False).str.replace("]","",regex=False)

labels_string = ["Laugh", "Grunt", "Cry", "Pant", "Gasp", "Other", "Groan", "Scream","nan"]
labels_num = [0, 1, 2, 3, 4, 5, 6, 7, "x"]
label_mapping_dict = dict(zip(labels_string,labels_num))
df_metadata["num_labels"] = df_metadata["Voc_Type"].apply( lambda x: str(x) ).apply( lambda x: label_mapping_dict[x] )

dict_filename_to_lab = dict(zip(df_metadata["filename"],df_metadata["num_labels"]))
dict_filename_to_partition = dict(zip(df_metadata["filename"],df_metadata["Split"]))


# %% Dataframes for Train, Val, and Test

FileList = glob.glob( path_in + "/*mat" )
FileNames = [os.path.basename( file_in )[:-4] for file_in in FileList]

df_available_files = pd.DataFrame(  {"FileList":FileList,
                                     "filename":FileNames })

df_available_files["Split"] = df_available_files["filename"].apply( lambda x: dict_filename_to_partition[x] )

df_available_files["num_labels"] = df_available_files["filename"].apply( lambda x: dict_filename_to_lab[x] )

df_available_files_Train = df_available_files.groupby("Split").get_group("Train")
df_available_files_Val = df_available_files.groupby("Split").get_group("Val")
df_available_files_Test = df_available_files.groupby("Split").get_group("Test")

df_available_files_Train["num_labels"] = df_available_files_Train["num_labels"].apply( lambda x: str(x) )

df_Train_grouped = df_available_files_Train.groupby("num_labels")
df_Train_class0 = df_Train_grouped.get_group("0")
df_Train_class1 = df_Train_grouped.get_group("1")
df_Train_class2 = df_Train_grouped.get_group("2")
df_Train_class3 = df_Train_grouped.get_group("3")
df_Train_class4 = df_Train_grouped.get_group("4")
df_Train_class5 = df_Train_grouped.get_group("5")
df_Train_class6 = df_Train_grouped.get_group("6")
df_Train_class7 = df_Train_grouped.get_group("7")

# %% Classification model

def fcn_model( input_shape ):

    input_path = Input(shape=input_shape)

    x = BatchNormalization(center=True, scale=True)(input_path)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.2)(x)
    output_path = Dense(8, activation="softmax")(x)
    model = Model(inputs=input_path, outputs=output_path)

    return model


# %% Train model

num_epochs = 200
des_samps = 25
batch_size_use = (des_samps*8)
steps_per_epoch = num_epochs//batch_size_use
input_shape = (768,)
model_ID = "fcn_model"

path_save_trained_model = path_save + "/Trained Models - " + model_ID
os.mkdir( path_save_trained_model )

model = fcn_model( input_shape )


model.compile( loss="categorical_crossentropy",
                  optimizer = tensorflow.keras.optimizers.Adam(learning_rate=1e-4),
                  metrics=["accuracy"])

save_path = path_save_trained_model + "/model-{epoch:02d}-train-{accuracy:.4f}.hdf5"
checkpoint = tensorflow.keras.callbacks.ModelCheckpoint(save_path, monitor="accuracy", verbose=1, save_best_only=False, mode="max")
callbacks = [checkpoint]


for epoch in range(num_epochs):

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

    model.fit(Feats, Labs, batch_size=des_samps*8, epochs=epoch+1, initial_epoch=epoch, callbacks=callbacks)