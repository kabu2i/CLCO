import os
import re
import time
import random
import torch


def time_file(temp):
    year_month = time.strftime('%Y-%m', time.localtime())
    day = time.strftime('%d', time.localtime())
    hour_min = time.strftime('%H-%M-%S', time.localtime())
    file_root = '../runs/{}/{}/{}'.format(year_month, day, hour_min)

    if not os.path.exists(file_root):
        os.makedirs(file_root)
    else:
        file_root += "_"
        os.makedirs(file_root)
    if not os.path.exists('../runs/temp'):
        os.makedirs('../runs/temp')
    with open(f'../runs/temp/{temp}.txt', 'w') as f:
        f.write(file_root)
    return file_root


def change_name(path):
    while os.path.exists(path):
        if os.path.isdir(path):
            path = path + "_"
        else:
            dir_ = re.findall("(.*)\.", path)
            file_type = re.findall("(\.[a-z].*)", path)
            path = os.path.join(dir_[0] + '_' + file_type[0])
    return path


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def grid_search(trainer, param, test=False, track=False):
    keys = [key for key in param.keys()]
    campare_keys, search_keys = [], []
    dim = 1
    len_param = [len(param[key]) for key in param.keys()]
    for i, l in enumerate(len_param):
        if l == 1:
            trainer += f' --{keys[i]} {param[keys[i]][0]}'
            campare_keys.append(keys[i])
        else:
            search_keys.append(keys[i])
            dim *= l
    campare_keys.extend(search_keys)
    trainer = 'python ' + trainer
    if len(search_keys) == 0:
        os.system(trainer)
    trainers, midtrainers = [], [trainer]
    for trainer in midtrainers:
        reset = trainer
        for key in search_keys:
            for i in param[key]:
                trainer += f' --{key} {i}'
                cond = re.findall('--(.*?) ', trainer)
                if cond == campare_keys:
                    trainers.append(trainer)
                    if test == False:
                        os.system(trainer)
                if len(trainers) == dim and track == True:
                    with open(change_name('../runs/grid_research.txt'), 'w') as f:
                        f.write(f'number of param set: {len(trainers)}\n')
                        f.write('parameters:\n')
                        for l in trainers:
                            f.write(f'{l}\n')
                    break
                midtrainers.append(trainer)
                trainer = reset
            if len(trainers) == dim:
                break
        if len(trainers) == dim:
            break


def random_search(trainer, param, limit):
    end = time.time()
    trainer = 'python ' + trainer
    reset = trainer
    trainers = []
    while (time.time() - end) < limit:
        for key in param.keys():
            if isinstance(param[key], list):
                trainer += f' --{key} {random.choice(param[key])}'
            if isinstance(param[key], tuple) and len(param[key]) == 2:
                trainer += f' --{key} {random.uniform(param[key][0], param[key][1])}'
        if trainer not in trainers:
            trainers.append(trainer)
            os.system(trainer)
        trainer = reset
    with open(change_name('../runs/random_research.txt'), 'w') as f:
        f.write(f'number of param set: {len(trainers)}\n')
        f.write('parameters:\n')
        for l in trainers:
            f.write(f'{l}\n')


def load_clco(model, args):
    """ Loads the pre-trained CLCO model parameters.

        Applies the loaded pre-trained params to the base encoder used in Linear Evaluation.

    Args:
        model (model): Randomly Initialised base encoder.

        args (dict): Program arguments/commandline arguments.

    Returns:
        model (model): Initialised base encoder with parameters from the CLCO encoder.
    """
    print("\nLoading the model: {}\n".format(args.resume))

    # Load the pretrained model
    checkpoint = torch.load(args.resume, map_location="cpu")

    # rename clco pre-trained keys
    state_dict = checkpoint['model']
    for k in list(state_dict.keys()):
        if k.startswith('encoder') and not k.startswith('encoder.fc'):
            state_dict[k[len("encoder."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]

    # Load the encoder parameters
    model.load_state_dict(state_dict, strict=False)

    return model
