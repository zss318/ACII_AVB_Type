
# %% Packages
import torchaudio
import torch

import numpy as np

import os, glob, tqdm
import scipy.io as sio

# %% Settings

path_in = "" # path to processed audio files

path_save = "" # path to save feats


# %% WAV2VEC2_ASR_BASE_960H

FeatName = "WAV2VEC2_ASR_BASE_960H"

pathSave_layer0 = path_save + "/" + FeatName + "_layer0"
os.mkdir( pathSave_layer0 )
pathSave_layer1 = path_save + "/" + FeatName + "_layer1"
os.mkdir( pathSave_layer1 )
pathSave_layer2 = path_save + "/" + FeatName + "_layer2"
os.mkdir( pathSave_layer2 )
pathSave_layer3 = path_save + "/" + FeatName + "_layer3"
os.mkdir( pathSave_layer3 )

####  LOAD MODEL
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model()


####  COMPUTE EMBEDDINGS AND STORE POOLED FEATURES

FileList = glob.glob( path_in + '/*wav' )

for aa in tqdm( range( len(FileList) ) ):

    file_in = FileList[aa]


    sig, fs = torchaudio.load(file_in)


    with torch.inference_mode():
      features, temp = model.extract_features(sig)



    layer_num = 0
    Feats_LLDs = np.squeeze( features[layer_num].detach().numpy() ).astype("float32")
    Feats_AvgPool = np.mean(Feats_LLDs, axis=0)
    fileSave_Feats_layer0 = pathSave_layer0 + "/" + os.path.basename( file_in )[:-4] + ".mat"
    sio.savemat( fileSave_Feats_layer0, {"Feats":Feats_AvgPool})


    layer_num = 1
    Feats_LLDs = np.squeeze( features[layer_num].detach().numpy() )
    Feats_AvgPool = np.mean(Feats_LLDs, axis=0)
    fileSave_Feats_layer1 = pathSave_layer1 + "/" + os.path.basename( file_in )[:-4] + ".mat"
    sio.savemat( fileSave_Feats_layer1, {"Feats":Feats_AvgPool})



    layer_num = 2
    Feats_LLDs = np.squeeze( features[layer_num].detach().numpy() )
    Feats_AvgPool = np.mean(Feats_LLDs, axis=0)
    fileSave_Feats_layer2 = pathSave_layer2 + "/" + os.path.basename( file_in )[:-4] + ".mat"
    sio.savemat( fileSave_Feats_layer2, {"Feats":Feats_AvgPool})



    layer_num = 3
    Feats_LLDs = np.squeeze( features[layer_num].detach().numpy() )
    Feats_AvgPool = np.mean(Feats_LLDs, axis=0)
    fileSave_Feats_layer3 = pathSave_layer3 + "/" + os.path.basename( file_in )[:-4] + ".mat"
    sio.savemat( fileSave_Feats_layer3, {"Feats":Feats_AvgPool})



