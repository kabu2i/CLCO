"""
Adapted from: https://github.com/ZhaoZhibin/DL-based-Intelligent-Diagnosis-Benchmark
"""
import os
import math

from tqdm import tqdm
from scipy.io import loadmat

from utils import *
from data.sequence_aug import *

#1 Undamaged (healthy) bearings(6X)
HBdata = ['K001',"K002",'K003','K004','K005','K006']
label1=[0,1,2,3,4,5]  #The undamaged (healthy) bearings data is labeled 1-9
#2 Artificially damaged bearings(12X)
ADBdata = ['KA01','KA03','KA05','KA06','KA07','KA08','KA09','KI01','KI03','KI05','KI07','KI08']
label2=[6,7,8,9,10,11,12,13,14,15,16,17]    #The artificially damaged bearings data is labeled 4-15
#3 Bearings with real damages caused by accelerated lifetime tests(14x)
# RDBdata = ['KA04','KA15','KA16','KA22','KA30','KB23','KB24','KB27','KI04','KI14','KI16','KI17','KI18','KI21']
# label3=[18,19,20,21,22,23,24,25,26,27,28,29,30,31]  #The artificially damaged bearings data is labeled 16-29

RDBdata = ['KA04','KA15','KA16','KA22','KA30','KB23','KB24','KB27','KI04','KI14','KI16','KI17','KI18','KI21']
label3=[i for i in range(14)]

#working condition
WC = ["N15_M07_F10","N09_M07_F10","N15_M01_F10","N15_M07_F04"]
state = WC[0] #WC[0] can be changed to different working states


class PUDataset:
    def __init__(self, args) -> None:
        '''Initialize class PUDataset

        Args:
            args (Dictionary): Program Arguments
        '''
        self.args = args
        self.file_root = args.data
        self.length = args.length
        self.number = (args.train_samples + args.test_samples + \
            args.valid_samples)
        self.signal_noise = False
        self.signal_std = False
        self.signal_nm = False
        self.mat_data = self.capture_files()

    def capture_files(self):
        '''generate Training Dataset and Testing Dataset

        Returns:
            files (Dictionary): mat files dict
        '''
        data = {}
        if self.args.data_mode == "artificial":
            MatData = [HBdata[0]] + ADBdata
        elif self.args.data_mode == "reallife":
            MatData = [HBdata[0]] + RDBdata
        elif self.args.data_mode == "reallife_subset":
            MatData = [HBdata[0]] + ['KA04', 'KA15', 'KI04', 'KB23', 'KB27']
        else:
            raise "no such data mode!!!"
        for k in tqdm(range(len(MatData))):
            signal = []
            for j in range(20):
                name = state+"_"+MatData[k]+f"_{j+1}"
                path=os.path.join(self.file_root,MatData[k],name+".mat")
                fl = loadmat(path)[name]
                signal.append(fl[0][0][2][0][6][2])
            data[MatData[k]] = signal
        return data
    
    def slice_enc(self):
        ''' slice labeled signal into data samples '''
        keys = self.mat_data.keys()

        label = 0
        train_dataset, test_dataset, valid_dataset = [], [], []
        train_labels, test_labels ,valid_labels = [], [], []
        for name in keys:
            class_train_dataset, class_test_dataset, class_valid_dataset = [], [], []
            for mat_single_data in self.mat_data[name]:
                valid_length = (self.args.valid_samples//20+1) * self.length
                valid_data = mat_single_data[:, :valid_length]
                test_length = math.ceil(len(mat_single_data) * (self.args.test_samples//20+1) / self.number)
                test_length = test_length if test_length > 25600 else 25600
                test_data = mat_single_data[:, valid_length : valid_length + test_length]
                train_data = mat_single_data[:, test_length + valid_length :]
                class_train_dataset.append(self.slice_data(train_data, self.args.train_samples//20+1))
                class_test_dataset.append(self.slice_data(test_data, self.args.test_samples//20+1))
                class_valid_dataset.append(self.slice_data(valid_data, self.args.valid_samples//20+1))
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

    def slice_data(self, data, number):
        ''' slice single signal into data samples '''
        datasets = []
        self.step = int((data.shape[1] - self.length) / (self.number - 1))
        for x in range(number):
            datas = data[:, self.step * x:self.step * x + self.length]
            datasets.append(datas)
        return datasets
