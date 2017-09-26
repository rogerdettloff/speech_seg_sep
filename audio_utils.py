#!/usr/bin/env python
"""
This module contains tools that have been helpful for working with sound files,
including visualizing sound clips and masks, and making training and test data
sets.

"""

from copy import deepcopy
from matplotlib import pyplot as plt
import numpy as np
import os
from scipy.io import wavfile
from scipy import signal
import sounddevice as sd


class SoundClip(object):
    def __init__(self, fs, data, window_len=1024):
        """
        Construct a SoundClip object.

        Parameters
        ----------
        fs : float
            sampling frequency.
        data : np.ndarray
            dtype(int16) pcm signal.
        window_len : int
            length of the window for stft analysis.
        """
        self.fs = fs
        assert (isinstance(data, np.ndarray))
        assert (data.dtype == np.int16)  # we assume these are int for playback
        self.data = data
        self.window_len = window_len
        self.noverlap = self.window_len // 2
        self.window = signal.hann(self.window_len, sym=False)
        self.f, self.t, self.Zxx = signal.stft(
            self.data, self.fs,
            window=self.window,
            noverlap=self.noverlap,
            nperseg=self.window_len,
            nfft=self.window_len
        )

    @classmethod
    def from_wav(cls, filename, window_len=1024):
        """
         Construct a SoundClip from a .wav file.

        Parameters
        ----------
        filename : str
            name of .wav file containing the sound clip.
        window_len : int
            length of the window for stft analysis.

        Returns
        -------
        SoundClip

        """
        fs, data = wavfile.read(filename)
        return cls(fs, data, window_len)

    @classmethod
    def from_stft(cls, Zxx, fs, window_len=1024):
        """
         Construct a SoundClip from a STFT array.

        Parameters
        ----------
        Zxx : np.ndarray
            complex matrix containing the STFT of the sound clip.
        fs : float
            sampling frequency.
        window_len : int
            length of the window for stft analysis.

        Returns
        -------
        SoundClip

        """
        sc = cls(fs, np.zeros(window_len, dtype=np.int16), window_len)
        sc.Zxx = Zxx
        sc.istft()
        return sc

    @property
    def duration(self) -> float:
        return len(self.data) / self.fs  # seconds

    def play(self,  blocking=False):
        sd.play(self.data, self.fs,  blocking=blocking)

    def get_sp_mask(self, threshold=10.0, low_prob=0.0):
        """
        speech probability mask
        For now, this is a very simple threshold, which works okay if the clip
        is isolated speech.  This intended to be used to make a reference mask.

        Parameters
        ----------
        threshold : float
        low_prob : float

        Returns
        -------
        np.ndarray
            of shape same as Zxx.
        """
        m = np.where(np.abs(self.Zxx) > threshold)  # high probability of speech
        mask = np.ones(self.Zxx.shape,
                       dtype=np.float32) * low_prob  # low probability of speech
        mask[m] = 1.0  # m points to places with a high probability of being speech.
        return mask

    def plot_spectrogram(self):
        plt.figure()
        plt.pcolormesh(np.log(np.abs(self.Zxx)))
        plt.title('STFT Magnitude')
        plt.ylabel('Frequency index')
        plt.xlabel('Time index')
        # additional plot of slice through one time step.
        plt.figure()
        plt.plot(np.abs(self.Zxx[:, [27, 37, 47, 57, 67, 77]]))
        plt.show()

    def istft(self):
        """
        Reconstruct and update the time domain signal via inverse SFTF of the
        (presumably modified) Zxx.
        """
        _, x = signal.istft(
            self.Zxx, self.fs,
            window=self.window,
            noverlap=self.noverlap,
            nperseg=self.window_len,
            nfft=self.window_len
        )
        self.data = x.astype(np.int16)

    def apply_mask(self, mask):
        """
        This will make a copy of this instance, applay the given mask to the
        copy, masking out any unwanted time/frequencies.  Then update the
        masked_clip using istft, and return the new masked clip object.

        Parameters
        ----------
        mask : np.ndarray
            of shape like self.Zxx

        Returns
        -------
        SoundClip

        """
        masked_clip = deepcopy(self)
        masked_clip.Zxx = masked_clip.Zxx * mask
        masked_clip.istft()
        return masked_clip

    def take_clip(self, start_time=0.0, duration=3.0):
        """
        extract a shorter clip of this instance, return a new SoundClip
        containing the shorter signal.

        Parameters
        ----------
        start_time : float
            time (in seconds) to begin extracting the new clip from this clip.
        duration : float
            duration (in seconds) of the clip to take.

        Returns
        -------
        SoundClip

        """
        start_idx = int(start_time * self.fs)
        assert (start_idx <= len(self.data))
        end_idx = start_idx + int(duration * self.fs)
        assert (end_idx <= len(self.data))
        return SoundClip(self.fs, self.data[start_idx:end_idx], self.window_len)

    def mix_clip(self, other, start_time=0.0, scale=1.0):
        """
        mix self + (other * scale). Return a new SoundClip of the mixed signals.


        Parameters
        ----------
        other : SoundClip
            The other SoundClip to mix with self.
        start_time : float
            time (in seconds) to index into other, to start mixing at that point
        scale : float

        Returns
        -------
        SoundClip

        """
        assert (self.fs == other.fs)
        start_idx = int(start_time * self.fs)
        end_idx = start_idx + len(self.data)
        mixed_sig = self.data + other.data[start_idx:end_idx] * scale
        return SoundClip(self.fs, mixed_sig.astype(np.int16), self.window_len)


def test_sound_clip(sound_file, interf_file):
    """
    Test the SoundClip class...

    Parameters
    ----------
    sound_file : str
        name of .wav file containing the sound clip.
    interf_file : str
        name of .wav file containing interference clip.

    Returns
    -------

    """
    long_clip = SoundClip.from_wav(sound_file)
    #    long_clip.play()
    #    long_clip.plot_spectrogram()

    # take a shorter clip from the long_clip...
    clip = long_clip.take_clip(0.5, 3.008)

    # make, show, and test the mask
    ideal_mask = clip.get_sp_mask(threshold=30.0, low_prob=0.001)
    plt.figure()
    plt.pcolormesh(ideal_mask)
    plt.title('mask')
    plt.ylabel('Frequency index')
    plt.xlabel('Time index')

    masked_clip = clip.apply_mask(ideal_mask)
    masked_clip.play(blocking=True)
    masked_clip.plot_spectrogram()

    # try adding some interference and apply the mask
    interf_clip = SoundClip.from_wav(interf_file)
    noisy_clip = clip.mix_clip(interf_clip, start_time=0.0, scale=1.0)
    noisy_clip.play(blocking=True)
    noisy_clip.plot_spectrogram()
    noisy_masked_clip = noisy_clip.apply_mask(ideal_mask)
    noisy_masked_clip.play(blocking=True)
    noisy_masked_clip.plot_spectrogram()
    pass


if __name__ == "__main__":
    sp_path = "/shared/Projects/speech_signal_proc/CHiME3/data/audio/16kHz/isolated/"
    speech_ref = [os.path.join(sp_path, "et05_bth/F06_441C020K_BTH.CH1.wav"),
                  ]
    interference_path = "/shared/Projects/speech_signal_proc/other_sounds"
    interferer = [os.path.join(interference_path, "siren_clip.wav"),
                  ]
    test_sound_clip(speech_ref[0], interferer[0])
