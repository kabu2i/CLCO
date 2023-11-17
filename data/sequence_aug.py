import random
import torch
from torch import nn
import numpy as np
from scipy.signal import resample


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, seq):
        for t in self.transforms:
            seq = t(seq)
        return seq


class Reshape(object):
    def __call__(self, seq):
        return seq.transpose()


class Retype(object):
    def __call__(self, seq):
        return seq.astype(np.float32)


class AddGaussian(object):
    def __init__(self, sigma=0.01):
        self.sigma = sigma

    def __call__(self, seq):
        return seq + np.random.normal(loc=0, scale=self.sigma, size=seq.shape)


class RandomAddGaussian(object):
    def __init__(self, sigma=0.01):
        self.sigma = sigma

    def __call__(self, seq):
        if np.random.randint(2):
            return seq
        else:
            return seq + np.random.normal(loc=0, scale=self.sigma, size=seq.shape)


class Scale(object):
    def __init__(self, sigma=0.01):
        self.sigma = sigma

    def __call__(self, seq):
        scale_factor = np.random.normal(
            loc=1, scale=self.sigma, size=(seq.shape[0], 1))
        scale_matrix = np.matmul(scale_factor, np.ones((1, seq.shape[1])))
        return seq*scale_matrix


class RandomScale(object):
    def __init__(self, sigma=0.01):
        self.sigma = sigma

    def __call__(self, seq):
        if np.random.randint(2):
            return seq
        else:
            scale_factor = np.random.normal(
                loc=1, scale=self.sigma, size=(seq.shape[0], 1))
            scale_matrix = np.matmul(scale_factor, np.ones((1, seq.shape[1])))
            return seq*scale_matrix


class RandomStretch(object):
    def __init__(self, sigma=0.3):
        self.sigma = sigma

    def __call__(self, seq):
        if np.random.randint(2):
            return seq
        else:
            seq_aug = np.zeros(seq.shape)
            len = seq.shape[1]
            length = int(len * (1 + (random.random()-0.5)*self.sigma))
            for i in range(seq.shape[0]):
                y = resample(seq[i, :], length)
                if length < len:
                    if random.random() < 0.5:
                        seq_aug[i, :length] = y
                    else:
                        seq_aug[i, len-length:] = y
                else:
                    if random.random() < 0.5:
                        seq_aug[i, :] = y[:len]
                    else:
                        seq_aug[i, :] = y[length-len:]
            return seq_aug


class RandomCrop(object):
    def __init__(self, crop_len=20):
        self.crop_len = crop_len

    def __call__(self, seq):
        if np.random.randint(2):
            return seq
        else:
            max_index = seq.shape[1] - self.crop_len
            random_index = np.random.randint(max_index)
            seq[:, random_index:random_index+self.crop_len] = 0
            return seq


class Normalize(object):
    def __init__(self, type="mean-std"):  # "0-1","1-1","mean-std"
        self.type = type

    def __call__(self, seq):
        if self.type == "0-1":
            seq = (seq-seq.min())/(seq.max()-seq.min())
        elif self.type == "1-1":
            seq = 2*(seq-seq.min())/(seq.max()-seq.min()) + -1
        elif self.type == "mean-std":
            seq = (seq-seq.mean())/seq.std()
        else:
            raise NameError('This normalization is not included!')

        return seq


class SignalDropout(object):
    def __init__(self, p=0.1):
        self.dropout = nn.Dropout(p)

    def __call__(self, signal):
        signal = torch.from_numpy(signal)
        with torch.no_grad():
            signal = self.dropout(signal)

        signal = signal.numpy()
        signal = signal.astype(np.float32)

        return signal


class Normal(object):
    def __call__(self, seq):
        return seq


class RandomZero(object):
    def __init__(self, p=0.25):
        self.p = p

    def __call__(self, seq):
        arr = np.arange(seq.shape[-1])
        np.random.shuffle(arr)
        arr[arr < int(self.p * seq.shape[-1])] = 0
        arr[arr >= int(self.p * seq.shape[-1])] = 1
        return seq * arr.reshape(seq.shape)


class RandomIncrease(object):
    def __init__(self, p=0.25):
        self.p = p

    def __call__(self, seq):
        arr = np.concatenate((np.random.randint(4, 10, size=int(seq.shape[-1] * self.p)),
                              np.ones(int(seq.shape[-1] - seq.shape[-1] * self.p))))
        np.random.shuffle(arr)
        return seq * arr.reshape(seq.shape)


class RandomReduction(object):
    def __init__(self, p=0.25):
        self.p = p

    def __call__(self, seq):
        arr = np.concatenate((np.random.randint(4, 10, size=int(seq.shape[-1] * self.p)),
                              np.ones(int(seq.shape[-1] - seq.shape[-1] * self.p))))
        np.random.shuffle(arr)
        return seq / arr.reshape(seq.shape)


class TimeDisorder(object):
    def __init__(self, size=8):
        self.size = size

    def __call__(self, seq):
        seq_shape = seq.shape
        if seq.shape[-1] % self.size != 0:
            raise "Please make sure shuffled segments size can divide the size of signal."
        np.random.shuffle(seq.reshape((self.size, -1)))
        return seq.reshape(seq_shape)
