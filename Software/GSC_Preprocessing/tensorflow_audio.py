from typing import Optional

import tensorflow as tf
from tensorflow import keras

import numpy as np

class Spectrogram(keras.Model):
    """
    """
    def __init__(self, 
                 sample_rate: int = 16000,
                 n_fft: int = 400,
                 win_length: Optional[int] = None,
                 hop_length: Optional[int] = None,
                 pad_end: bool = False,
                 power: float = 2.0) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else win_length//2
        self.pad_end = pad_end
        self.power = power
    
    def call(self, waveform: tf.Tensor) -> tf.Tensor:
        spectrogram = tf.abs(tf.signal.stft(
                signals = waveform,
                frame_length = self.win_length,
                frame_step = self.hop_length,
                fft_length = self.n_fft,
                pad_end = self.pad_end
            ))
        if self.power == 2:
            spectrogram = spectrogram*spectrogram
        return spectrogram

class MelSpectrogram(keras.Model):
    """
    """
    def __init__(self,
                 sample_rate: int = 16000,
                 n_fft: int = 400,
                 win_length: Optional[int] = None,
                 hop_length: Optional[int] = 160,
                 f_min: float = 0.0,
                 f_max: float = 3800,
                 pad_end: bool = False,
                 n_mels: int = 128,
                 power: float = 2.0,
                 power_to_db: bool = True) -> None:
          super().__init__()
          num_spectrogram_bins = n_fft//2+1
          self.spec = Spectrogram(sample_rate,
                                  n_fft,
                                  win_length,
                                  hop_length,
                                  pad_end,
                                  power)
          self.linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
              num_mel_bins = n_mels,
              num_spectrogram_bins = num_spectrogram_bins,
              sample_rate = sample_rate,
              lower_edge_hertz = f_min,
              upper_edge_hertz = f_max
            )
          self.power_to_db = power_to_db

    def call(self, waveform: tf.Tensor) -> tf.Tensor:
        spectrogram = self.spec(waveform)
        mel_spectrogram = tf.matmul(spectrogram, self.linear_to_mel_weight_matrix)
        if self.power_to_db:
            # Log mel spectrogram
            log_offset = 1e-6
            mel_spectrogram = tf.math.log(mel_spectrogram + log_offset)
        return mel_spectrogram

if __name__ == '__main__':
    f_mel = MelSpectrogram(sample_rate = 16000,
                           n_fft = 512,
                           win_length = 480,
                           hop_length = 160,
                           pad_end = True,
                           n_mels = 40,
                           power = 1,
                           power_to_db = True)