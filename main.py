import os 
import torch
import argparse

from dtloader import OneDimDataset
from trainer import *
from utils import *

from torch.utils.tensorboard import SummaryWriter

model_names = sorted(name for name in models.__dict__
                    if name.islower() and not name.startswith("__"))

parser = argparse.ArgumentParser(description='PyTorch CLCO For Bearing Fault diagnosis.')
# dataset config
parser.add_argument('--data', default='PATH/TO/DATA', help='path to dataset. '
                    '(eg: "../dataset/CWRU_4_classes", "../dataset/CWRU_10_classes)", '
                    '"../dataset/PU Dataset)"')
parser.add_argument('--data-mode', default="0HP 1HP 2HP 3HP", help='data partition mode, which load'
                    ' or sense, contains "0HP 1HP 2HP 3HP" or "artificial", "reallife"')
parser.add_argument('--pretrainset', default='train', help='train dataset mode')
parser.add_argument('--finetune-trainset', default='valid', help='val dataset mode')
parser.add_argument('--finetune-testset', default='test', help='test dataset mode')
parser.add_argument('--finetuneset', default="reallife", help='finetune dataset partition mode')
parser.add_argument('--num-classes', default=4, type=int, help='num classes')
parser.add_argument('--length', default=2048, type=int, help='sample length.')
parser.add_argument('--train-samples', default=100, type=int, help='train samples.')
parser.add_argument('--test-samples', default=1000, type=int, help='test samples.')
parser.add_argument('--valid-samples', default=10, type=int, help='valid samples.')
parser.add_argument('--views', default=2, type=int, help='data augmentation views used in pre-training process.')
parser.add_argument('--dataaug', action='store_true', help='Activate data augmentation.')
parser.add_argument('--normalize-type', default='mean-std', type=str, choices=["0-1","1-1","mean-std"],
                    help='normalize type.')

# backbone and predictor configs
parser.add_argument('-a', '--backbone', metavar='ARCH', default='resnet18_1d',
                    choices=model_names, help='model architecture: ' +
                    ' | '.join(model_names) + ' (default: resnet18_1d)')
parser.add_argument('--mlp-hidden-size', default=128, type=int, metavar='N',
                    help='Mlp hidden size')
parser.add_argument('--projection-size', default=512, type=int, metavar='N',
                    help='Projection size')

# clco configs
parser.add_argument('--cluster-mode', default='prior_knowledge', type=str,
                    choices=['prior_knowledge', 'ori', 'pca'], help='Cluster mode')
parser.add_argument('--ncentroids', default=20, type=int,
                    help='Kmeans centroid number')
parser.add_argument('--niter', default=100, type=int,
                    help='Kmeans iter number')
parser.add_argument('--select-data', default=0.5, type=float,
                    help='Kmeans selected data ratio')
parser.add_argument('--loss', default='mpc', choices=['mpc', 'supcon', 'mcc'], 
                    type=str, help='multi label loss function')
parser.add_argument('--select-positive', default='cluster_postive', 
                    choices=['combine_inst_dis', 'only_inst_dis', 'cluster_postive'], 
                    type=str, help='add instance discrimination or only use instance '
                    'discrimination positive samples in physical hint method')
parser.add_argument('--random-pseudo-label', action='store_true', help='randomly generate'
                    ' pseudo-labels.')

# train options:
parser.add_argument('--train-mode', default='clco', type=str, choices=['clco', 
                    'finetune', 'evaluate'], help='train mode.')
parser.add_argument('--resume-mode', default='clco', type=str, choices=['clco'], 
                    help='resume mode.')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--max-epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=512, type=int, metavar='N',
                    help='mini-batch size (default: 512)')
parser.add_argument('--optimizer', default='sgd', type=str, help='choose optimizer.')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--pretrain-lr-scheduler', action='store_true', 
                    help='active pretrain lr scheduler.')
parser.add_argument('--base-lr', type=float, default=0.00001,
                    help='Base / Minimum Learing Rate to Begin Linear Warmup.')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--momentum', type=float, default=0.9, help='the momentum '
                    'for sgd')
parser.add_argument('--warm', default=10, type=int, help='lr scheduler warmup.')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('-m', default=0.996, type=float, help='EMA (default: 0.996)')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')
parser.add_argument('--temp', default='temp', type=str, help='Checkpoint path.')
parser.add_argument('--active-log', action='store_true', help='active log.')

# finetune setting
parser.add_argument('--resume', default=None, help='resume checkpoint("PATH/TO/FILE").')
parser.add_argument('--freeze', action='store_true', help='freeze fc parameter'
                    ' for random model initialization.')
parser.add_argument('--finetune-batch-size', default=64, type=int, metavar='N',
                    help='mini-batch size (default: 64)')
parser.add_argument('--finetune-lr-scheduler', action='store_true', 
                    help='active pretrain lr scheduler.')
parser.add_argument('--ftlr', default=0.01, type=float, help='finetune learning'
                    ' rate')
parser.add_argument('--fclr', default=0.01, type=float, help='finetune classifier'
                    'learning rate')
parser.add_argument('--ftwd', default=1e-5, type=float, 
                    help='finetune learning weight decay (default: 1e-5)')
parser.add_argument('--finetune-epochs', default=200, type=int, help='finetune'
                    ' epochs.')

args = parser.parse_args()

args.device = f'cuda:{args.gpu_index}' if not args.disable_cuda and torch.cuda.is_available() else 'cpu'

print(f"Training with: {args.device}")

if args.active_log:
     args.writer = SummaryWriter(time_file(args.temp))

     args.checkpoint_dir = os.path.join(args.writer.log_dir, 'checkpoint.pt')
     logging.basicConfig(filename=os.path.join(args.writer.log_dir, 'training.log'), level=logging.DEBUG)
     for k, v in args.__dict__.items():
          logging.info("{}: {}".format(k, v)) 

model_param = [
          'fc.weight', 'fc.bias',
     ]

def freeze_model(model, model_param, args):
     # freeze all layers but the last fc
     if args.freeze:
          for name, param in model.named_parameters():
               if name in model_param:
                    param.requires_grad = False

if args.train_mode in ['clco']:
     pretrain_dataset = OneDimDataset(args.pretrainset, args=args)
     contrastive_trainer(pretrain_dataset, args)
     model = getattr(models, args.backbone)(num_classes=args.num_classes).to(args.device)
     args.resume = args.checkpoint_dir[:-3] + "_best.pt"
     args.train_mode = 'finetune'
     model = load_clco(model, args)
     freeze_model(model, model_param, args)
     finetune_trainset = OneDimDataset(args.finetune_trainset, args=args)
     finetune_testset = OneDimDataset(args.finetune_testset, args=args)
     finetune(model, finetune_trainset, finetune_testset, args)
elif args.train_mode == 'finetune':
     finetune_trainset = OneDimDataset(args.finetune_trainset, args=args)
     finetune_testset = OneDimDataset(args.finetune_testset, args=args)
     model = getattr(models, args.backbone)(num_classes=args.num_classes).to(args.device)
     freeze_model(model, model_param, args)
     if args.resume:
          model = load_clco(model, args)
     finetune(model, finetune_trainset, finetune_testset, args=args)
elif args.train_mode == 'evaluate':
     finetune_testset = OneDimDataset(args.finetune_testset, args=args)
     val_loader = torch.utils.data.dataloader.DataLoader(finetune_testset,
                                                  batch_size=args.batch_size,
                                                  num_workers=0,
                                                  drop_last=False,
                                                  shuffle=False)
     model = getattr(models, args.backbone)(num_classes=args.num_classes).to(args.device)
     if args.resume:
          model = load_clco(model, args)
     evaluate(model, val_loader, 200, args)
else:
     raise NotImplementedError(f'There is no such self-supervised train strategy like {args.train_mode}!')

if args.active_log:
     args.writer.close()
