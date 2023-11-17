import numpy as np
import scipy.io
import scipy.fft
from tqdm import tqdm
from torchvision.transforms import transforms

from data.sequence_aug import *
import data.feature_indicators as feature_indicators


def time_signal_transforms(args):
    """ used data augmentation methods """
    if args.dataaug and args.train_mode != "finetune":
        data_transforms = transforms.Compose([
            Normalize(args.normalize_type),
            RandomZero(),
            RandomScale(),
            RandomStretch(),
            AddGaussian(),
            Retype()
        ])
    else:
        data_transforms = transforms.Compose([
            Normalize(args.normalize_type),
        ])
    return data_transforms


def signals_fft(signal, sampling_rate, fft_size):
    """ FFT """
    ysignal = signal[:, :fft_size]

    yfft = scipy.fft.fft(ysignal)
    xfft = scipy.fft.fftfreq(fft_size, 1 / sampling_rate)[:fft_size//2]

    Afft = 2.0 / fft_size * np.abs(yfft[:, :fft_size//2])

    Pfft = np.power(Afft, 2) / Afft.shape[1] * sampling_rate
    return xfft, yfft, Afft, Pfft


def process_signals(signal, sampling_rate, indicators):
    """ Obtain the time-frequency domain characteristics of the signal 
    
    Args:
        signal (Array): original time signal
        sampling_rate (int): sampling rate
        indicators (List): the name of time-frequency domain characteristics,
            The implemented characteristics can be found in file feature_indicators.py
    """
    if len(signal.shape) > 1:
        signal = signal.reshape(signal.shape[0], -1)
    elif len(signal.shape) == 1:
        signal = signal.reshape(1, -1)
    else:
        raise "signal.shape invalid"
    
    fft_size = 8192 if signal.shape[-1] > 8192 else 2048

    xfft, yfft, Afft, Pfft = signals_fft(signal, sampling_rate, fft_size)

    args = {
        "sig": signal,
        "Afft": Afft,
        "Pfft": Pfft
    }
    sig_findicators = []
    for indicator in tqdm(indicators):
        sig_findicators.append(getattr(feature_indicators, indicator)(args).reshape(-1,1))

    return np.concatenate(sig_findicators, axis=-1)


def prior_knowledge(sig):
    """ Obtain the 24 time-frequency features """
    indecators = [
        "mean", "std", "square_root_amplitude", "absolute_mean_value",
        "skewness", "kurtosis", "variance", "kurtosis_index",
        "peak_index", "waveform_index", "pulse_index", "skewness_index",
        "freq_mean_value", "freq_variance", "freq_skewness", "freq_steepness",
        "gravity_frequency", "freq_standard_deviation", "freq_root_mean_square",
        "average_freq", "regularity_degree", "variation_parameter",
        "eighth_order_moment", "sixteenth_order_moment",
    ]

    return process_signals(sig, 2048, indecators)
