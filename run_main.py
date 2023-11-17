from utils import grid_search

parameter = {
    'data': ["'PATH/TO/DATA'"], # '../dataset/CWRU_4_classes', '../dataset/CWRU_10_classes'
    'train-samples': [100],
    'test-samples': [1000],
    'data-mode': ["'0HP 1HP 2HP 3HP'"],
    'valid-samples': [1],
    'num-classes': [4],
    'backbone': ["resnet18_1d"],
    'train-mode': ["clco"],
    'loss': ["mpc"],
    'cluster-mode': ["prior_knowledge"],
    'ncentroids': [30],
    'select-data': [0.5],
    'max-epochs': [100],
    'weight-decay': [1e-5],
    'pretrain-lr-scheduler': ['\t'],
    'finetune-epochs': [200],
    'ftlr': [0.01],
    'active-log': ['\t'],
    'dataaug': ['\t'],
}

grid_search('main.py', parameter)
