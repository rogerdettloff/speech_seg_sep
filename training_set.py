#!/usr/bin/env python
"""
This makes a data set used for training the neural network.  It includes clips
of speech (from CHiME3 wsj0 dataset) with a binary mask for the speech, and
other sounds (like siren, dog bark, cocktail party), with a mask of all zeros.

"""

from audio_utils import SoundClip
import numpy as np
import os
from pickle import dump, HIGHEST_PROTOCOL


def make_training_set(speech_files, other_files, duration, window_len):
    """

    Parameters
    ----------
    speech_files : list
        list of filenames of speech files to include.
    other_files : list
        list of filenames of other sound files to include
    duration : float
        duration (in seconds) to make each sound clip.
    window_len : int
        length (num of samples) of the stft window.

    Returns
    -------
    dict
        with {X: stft of clip, Y: mask corresponding to X}
    """
    X = []
    Y = []
    for filename in speech_files:
        full_clip = SoundClip.from_wav(filename, window_len)
        for start in np.arange(0, full_clip.duration - duration, 0.5):
            clip = full_clip.take_clip(start, duration)
            X.append(clip.Zxx)
            Y.append(clip.get_sp_mask(20.0, 0.001))

    for filename in other_files:
        full_clip = SoundClip.from_wav(filename, window_len)
        for start in np.arange(0, full_clip.duration - duration, 0.5):
            clip = full_clip.take_clip(start, duration)
            X.append(clip.Zxx)
            Y.append(np.zeros_like(np.abs(clip.Zxx)))

    # randomize the sound clips before saving...
    X = np.asarray(X)
    Y = np.asarray(Y)
    rand_idx = np.random.permutation(len(X))
    return {'X': X[rand_idx], 'Y': Y[rand_idx]}


if __name__ == "__main__":
    sp_path = "/shared/Projects/speech_signal_proc/CHiME3/data/audio/16kHz/isolated/"
    speech_ref = [os.path.join(sp_path, "et05_bth/F06_441C020K_BTH.CH1.wav"),
                  os.path.join(sp_path, "et05_bth/F06_441C020K_BTH.CH0.wav"),
                  os.path.join(sp_path, "et05_bth/F06_441C020K_BTH.CH2.wav"),
                  os.path.join(sp_path, "et05_bth/F06_441C020K_BTH.CH3.wav"),
                  os.path.join(sp_path, "et05_bth/F06_441C020K_BTH.CH4.wav"),
                  os.path.join(sp_path, "et05_bth/F06_441C020K_BTH.CH5.wav"),
                  os.path.join(sp_path, "et05_bth/F06_441C020K_BTH.CH6.wav"),
                  os.path.join(sp_path, "et05_bth/F06_442C020S_BTH.CH0.wav"),
                  os.path.join(sp_path, "et05_bth/F06_442C020S_BTH.CH1.wav"),
                  os.path.join(sp_path, "et05_bth/F06_442C020S_BTH.CH2.wav"),
                  os.path.join(sp_path, "et05_bth/F06_442C020S_BTH.CH3.wav"),
                  os.path.join(sp_path, "et05_bth/F06_442C020S_BTH.CH4.wav"),
                  os.path.join(sp_path, "et05_bth/F06_442C020S_BTH.CH5.wav"),
                  os.path.join(sp_path, "et05_bth/F06_442C020S_BTH.CH6.wav"),
                  ]

    interference_path = "/shared/Projects/speech_signal_proc/other_sounds"
    interferer = [os.path.join(interference_path, "siren_clip.wav"),
                  os.path.join(interference_path, "music_1_clip.wav"),
                  os.path.join(interference_path, "dogs_2_barking_clip.wav"),
                  os.path.join(interference_path,
                               "cocktail_party_english_clip.wav"),
                  ]

    train_set = make_training_set(speech_ref, interferer, 3.008, 1024)
    with open("/shared/Projects/speech_signal_proc/traing_set_09-21.pkl", 'wb') as f:
        dump(train_set, f, protocol=HIGHEST_PROTOCOL)
