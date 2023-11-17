"""
Created on Sun Sep 26 10:12:08 2021

@author: weiyang
"""

import numpy as np
import scipy.io
import math
import os

from utils import *


class CWRUDataset:

    def __init__(self, args=None):
        '''Initialize class CWRUDataset

        Args:
            args (Dictionary): Program Arguments
        '''
        self.args = args
        self.length = args.length
        self.number = args.train_samples + args.test_samples + args.valid_samples
        self.signal_noise = False
        self.signal_std = False
        self.signal_nm = False
        self.sample_noise = False
        self.sample_std = False
        self.sample_nm = False
    
    def capture_cwru_DE_12k(self):
        '''Read the source file on Drive End, sampling rate 12k '''
        files = {}
        matfiles = []
        
        file_root = os.path.join(os.path.dirname(self.args.data), "CWRU_10_classes")
        for load in self.args.data_mode.split():
            filenames = os.listdir(os.path.join(file_root, load))
            for name in filenames:
                mat_path = os.path.join(file_root, load, name)
                mat_file = scipy.io.loadmat(mat_path)
                matfiles.append(mat_file)
                file_keys = mat_file.keys()
                for key in file_keys:
                    if 'DE' in key:
                        files[name] = mat_file[key].ravel()
        return files

    def capture_cwru_10_classes(self):
        '''Organized the source file on Drive End, sampling rate 12k
        into 10 classes of signals '''
        files = self.capture_cwru_DE_12k()
        signals = []
        files_sorted = {}
        key_order = ["normal", "IR007", "IR014", "IR021", "B007", "B014", "B021", "OR007", "OR014", "OR021"]
        for k in key_order:
            for i in files.keys():
                if k in i:
                    signals.append(files[i])
            files_sorted[k] = signals
            signals = []
        return files_sorted
    
    def capture_cwru_9_classes(self):
        '''Organized the source file on Drive End, sampling rate 12k
        into 9 classes of signals '''
        files = self.capture_cwru_DE_12k()
        signals = []
        files_sorted = {}
        key_order = ["IR007", "IR014", "IR021", "B007", "B014", "B021", "OR007", "OR014", "OR021"]
        for k in key_order:
            for i in files.keys():
                if k in i:
                    signals.append(files[i])
            files_sorted[k] = signals
            signals = []
        return files_sorted

    def capture_cwru_4_classes(self):
        '''Organized the source file on Drive End, sampling rate 12k
        into 4 classes of signals '''
        files = self.capture_cwru_DE_12k()
        files_sorted = {}
        key_order = ["normal", "IR007", "OR007", "B007"]
        signals = []
        for k in key_order:
            for i in files.keys():
                if k in i:
                    signals.append(files[i])
            files_sorted[k] = signals
            signals = []
        return files_sorted
    
    def capture_cwru_all(self):
        '''Read all the source file on Base, Drive End and Fan End, sampling 
        rate 12/48k '''
        files = {}

        file_root = os.path.join(os.path.dirname(self.args.data), "CWRU")
        file_dirs = ["Drive End 12k", "Drive End 48k", "Fan End 12k", "Normal"]
        for file_dir in file_dirs:
            for file in os.listdir(os.path.join(file_root, file_dir)):
                mat_file = scipy.io.loadmat(os.path.join(file_root, file_dir, file))
                for key in mat_file.keys():
                    if "time" in key:
                        files[key] = mat_file[key]
        return files
    
    def capture_cwru_normal(self):
        '''Read the source file on Normal condition (Drive End 12k) '''
        files = {}

        if os.path.exists(os.path.join(self.file_root, "Normal")):
            path = os.path.join(self.file_root, "Normal")
        else:
            assert "no normal dataset file"

        for file in os.listdir(path):
            mat_file = scipy.io.loadmat(os.path.join(path, file))
            for key in mat_file.keys():
                if "DE" in key:
                    files[key] = mat_file[key]
        return files

    def slice_enc(self):
        ''' slice labeled signal into data samples '''
        if 'CWRU_10_classes' in self.args.data:
            mat_data = self.capture_cwru_10_classes()
        elif 'CWRU_9_classes' in self.args.data:
            mat_data = self.capture_cwru_9_classes()
        elif 'CWRU_4_classes' in self.args.data:
            mat_data = self.capture_cwru_4_classes()
        else:
            raise 'no such cwru slice mode!!!'
        keys = mat_data.keys()

        label = 0
        train_dataset, test_dataset, valid_dataset = [], [], []
        train_labels, test_labels ,valid_labels = [], [], []
        for name in keys:
            class_train_dataset, class_test_dataset, class_valid_dataset = [], [], []
            for mat_single_data in mat_data[name]:
                valid_length = (self.args.valid_samples//len(mat_data[name])+1) * self.length
                valid_length = valid_length if valid_length < 25600 else 25600
                valid_data = mat_single_data[:valid_length]
                test_length = math.ceil(len(mat_single_data) * self.args.test_samples / self.number)
                test_length = test_length if test_length > 25600 else 25600
                test_data = mat_single_data[valid_length : valid_length + test_length]
                train_data = mat_single_data[test_length + valid_length :]
                class_train_dataset.append(self.slice_data(train_data, self.args.train_samples//len(mat_data[name])+1))
                class_test_dataset.append(self.slice_data(test_data, self.args.test_samples//len(mat_data[name])+1))
                class_valid_dataset.append(self.slice_data(valid_data, self.args.valid_samples//len(mat_data[name])+1))
            train_dataset.append(np.asarray(class_train_dataset).reshape((-1, 1, 2048))[:self.args.train_samples]),
            test_dataset.append(np.asarray(class_test_dataset).reshape((-1, 1, 2048))[:self.args.test_samples]),
            valid_dataset.append(np.asarray(class_valid_dataset).reshape((-1, 1, 2048))[:self.args.valid_samples]),
            train_labels += [label] * self.args.train_samples
            test_labels += [label] * self.args.test_samples
            valid_labels += [label] * self.args.valid_samples
            label += 1
            
        return np.asarray(train_dataset).reshape((-1, 1, 2048)),\
            np.asarray(test_dataset).reshape((-1, 1, 2048)),\
            np.asarray(valid_dataset).reshape((-1, 1, 2048)),\
            np.asarray(train_labels), np.asarray(test_labels),\
            np.asarray(valid_labels)
    
    def slice_enc_all(self):
        ''' slice all unlabeled signal into data samples '''
        mat_data = self.capture_cwru_all()

        label = 0
        train_dataset = []
        train_labels = []
        for name in mat_data.keys():
            train_data = mat_data[name]
            train_dataset.append(self.slice_data(train_data, self.args.train_samples))
            train_labels += [label] * self.args.train_samples
            label += 1
        
        return np.asarray(train_dataset).reshape((-1, 1, 2048)), np.asarray(train_labels)

    def slice_data(self, data, number):
        ''' slice single signal into data samples '''
        datasets = []
        self.step = int((data.shape[0] - self.length) / (self.number - 1))
        for x in range(number):
            datas = data[self.step * x:self.step * x + self.length]
            datasets.append(datas)
        return np.asarray(datasets)
