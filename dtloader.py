import os
from typing import Any, Optional, Tuple
import faiss
import numpy as np
from sklearn import preprocessing

from torch.utils.data import Dataset

from data.CWRU import CWRUDataset
from data.PU import PUDataset
from data.transforms import time_signal_transforms, prior_knowledge
from data.multi_view_data_injector import MultiViewDataInjector


class OneDimDataset(Dataset):
    """
    Dataset for 1D signals.
    """
    def __init__(self, mode, args):
        self.args = args
        if "CWRU" in args.data:
            data = CWRUDataset(args=self.args)
            train_x, train_y = data.slice_enc_all()
            _, test_x, valid_x, _, test_y, valid_y = data.slice_enc()
        elif "PU" in args.data:
            data = PUDataset(args=self.args)
            train_x, test_x, valid_x, train_y, test_y, valid_y = data.slice_enc()
        else:
            raise "Your path mask has data name, or no such data!!!"
        
        if args.train_mode in ['clco']:
            self.transform = MultiViewDataInjector([time_signal_transforms(args), time_signal_transforms(args)])
        else:
            self.transform = time_signal_transforms(args)
        
        if mode not in ['train', 'test', 'valid']:
            raise 'there is no such dataset!!!'
        signal, label = None, None
        if mode == 'train':
            signal, label = train_x, train_y
        if mode == 'test':
            signal, label = test_x, test_y
        if mode == 'valid':
            signal, label = valid_x, valid_y

        self.signal = signal.astype(np.float32)
        if args.train_mode in ['clco']:
            if args.cluster_mode == "prior_knowledge":
                statistical_features  = prior_knowledge(self.signal)
            elif args.cluster_mode == "pca":
                mat = faiss.PCAMatrix (2048, 24)
                mat.train(self.signal.reshape(self.signal.shape[0], -1))
                assert mat.is_trained
                statistical_features = mat.apply(self.signal.reshape(self.signal.shape[0], -1))
            elif args.cluster_mode == "ori":
                statistical_features  = self.signal.reshape(self.signal.shape[0], -1)
            else:
                raise "Cluster mode not supported!!!"
            statistical_features = preprocessing.scale(statistical_features, axis=0)
            d = statistical_features.shape[1]
            kmeans = faiss.Kmeans(d, args.ncentroids, niter=args.niter, verbose=True, gpu=True)
            kmeans.train(statistical_features)
            index = faiss.IndexFlatL2(d)
            index.add(statistical_features)
            _, I2 = index.search(kmeans.centroids, int(len(statistical_features) / args.ncentroids * args.select_data))
            self.label = np.zeros_like(label)
            for i, label_index in enumerate(I2):
                self.label[label_index] = i + 1
        else:
            self.label = label.astype(np.int64)
        self._len = len(self.signal)
        self.num_classes = set(self.label)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        lab: Optional[int]
        if self.label is not None:
            sig, lab = self.signal[idx], int(self.label[idx])
        else:
            sig, lab = self.signal[idx], None
        
        if self.transform is not None:
            sig = self.transform(sig)

        return sig, lab

    def __len__(self):
        return self._len
