# %% Packages
import os, glob
import numpy as np
import scipy.io as sio
import librosa
import soundfile as sf
from tqdm import tqdm

# %% Settings

path_in = "" # path to processed audio files

path_save = "" # path to save MelSpec
os.mkdir( path_save )


num_freq_bin = 128
num_fft = 512
hop_length = int(num_fft / 2)


# %% Compute MelSpec for all files

FileList = glob.glob( path_in + '/*wav' )


for aa in tqdm( range( len(FileList) ) ):


    file_in = FileList[aa]
    file_save = path_save + "/" + os.path.basename( file_in )[:-4] + ".mat"


    sig, fs = sf.read( file_in )


    Feats = librosa.feature.melspectrogram(y=sig, fs=fs, n_fft=num_fft, hop_length=hop_length, n_mels=num_freq_bin, fmin=10, fmax=fs/2, htk=True, norm=None)


    Feats = np.log(Feats + 1e-8)
    Feats = (Feats - np.min(Feats)) / (np.max(Feats) - np.min(Feats))


    sio.savemat( file_save, {"Feats":Feats} )
