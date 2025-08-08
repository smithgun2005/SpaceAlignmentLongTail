
import sys
import os
import time
import argparse
import torch
import numpy as np
import random
import datetime
from torch.backends import cudnn
from trainer import Trainer   
from model import ResNet_cifar, Resnet_LT
from imbalance_data import cifar10Imbanlance, cifar100Imbanlance, dataset_lt_data
from utils import util


def seed_everything(seed=42):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def remove_module_prefix(state_dict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

def print_model_param_nums(model):
    n_params = sum([p.numel() for p in model.parameters()])
    print(f'Number of params: {n_params/1e6:.2f}M')
    return n_params

def prepare_folders(args):
    if not os.path.exists(args.root_model):
        os.makedirs(args.root_model)
    if not os.path.exists(args.root_log):
        os.makedirs(args.root_log)

def get_model(args):
    if args.dataset in ["ImageNet-LT", "iNaturelist2018"]:
        print(f"=> creating model 'resnext50_32x4d'")
        net = Resnet_LT.resnext50_32x4d(num_classes=args.num_classes)
        return net
    else:
        print(f"=> creating model '{args.arch}'")
        if args.arch == 'resnet50':
            net = ResNet_cifar.resnet50(num_class=args.num_classes)
        elif args.arch == 'resnet18':
            net = ResNet_cifar.resnet18(num_class=args.num_classes)
        elif args.arch == 'resnet32':
            net = ResNet_cifar.resnet32(num_class=args.num_classes)
        elif args.arch == 'resnet34':
            net = ResNet_cifar.resnet34(num_class=args.num_classes)
        else:
            raise NotImplementedError
        return net

def get_dataset(args):
    transform_train, transform_val = util.get_transform(args.dataset)
    if args.dataset == 'cifar10':
        trainset = cifar10Imbanlance.Cifar10Imbanlance(
            transform=util.TwoCropTransform(transform_train),
            imbanlance_rate=args.imbanlance_rate, train=True, file_path=args.root)
        testset = cifar10Imbanlance.Cifar10Imbanlance(
            imbanlance_rate=args.imbanlance_rate, train=False, transform=transform_val, file_path=args.root)
        print("load cifar10")
        return trainset, testset
    if args.dataset == 'cifar100':
        trainset = cifar100Imbanlance.Cifar100Imbanlance(
            transform=util.TwoCropTransform(transform_train),
            imbanlance_rate=args.imbanlance_rate, train=True, file_path=os.path.join(args.root, 'cifar-100-python/'))
        testset = cifar100Imbanlance.Cifar100Imbanlance(
            imbanlance_rate=args.imbanlance_rate, train=False, transform=transform_val, file_path=os.path.join(args.root, 'cifar-100-python/'))
        print("load cifar100")
        return trainset, testset
    if args.dataset == 'ImageNet-LT':
        trainset = dataset_lt_data.LT_Dataset(args.root, args.dir_train_txt, util.TwoCropTransform(transform_train))
        testset = dataset_lt_data.LT_Dataset(args.root, args.dir_test_txt, transform_val)
        return trainset, testset
    if args.dataset == 'iNaturelist2018':
        trainset = dataset_lt_data.LT_Dataset(args.root, args.dir_train_txt, util.TwoCropTransform(transform_train))
        testset = dataset_lt_data.LT_Dataset(args.root, args.dir_test_txt, transform_val)
        return trainset, testset

def main():
    parser = argparse.ArgumentParser(description="Phase2 Null-space Trainer (All Layers)")
    parser.add_argument('--dataset', type=str, default='cifar100')
    parser.add_argument('--root', type=str, default='/data/', help="dataset setting")
    parser.add_argument('-a', '--arch', default='resnet32', choices=('resnet18', 'resnet34', 'resnet50', 'resnext50_32x4d'))
    parser.add_argument('--num_classes', default=100, type=int, help='number of classes')
    parser.add_argument('--imbanlance_rate', default=0.005, type=float)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('-b', '--batch_size', default=64, type=int)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=5e-3, type=float)
    parser.add_argument('--resample_weighting', default=0.0, type=float)
    parser.add_argument('--label_weighting', default=1.1, type=float)
    parser.add_argument('--contrast_weight', default=4, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('-p', '--print_freq', default=1000, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('-j', '--workers', default=4, type=int)
    parser.add_argument('--resume', default="", type=str)
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('--root_log', type=str, default='./logs')
    parser.add_argument('--root_model', type=str, default='./models')
    parser.add_argument('--store_name', type=str, default='result')
    args = parser.parse_args()

    seed_everything(args.seed)

    curr_time = datetime.datetime.now()
    args.store_name = '#'.join([
        "dataset: " + args.dataset, "arch: " + args.arch, "imbanlance_rate: " + str(args.imbanlance_rate),
        datetime.datetime.strftime(curr_time, '%Y-%m-%d %H:%M:%S')])
    prepare_folders(args)
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    # Load Model
    model = get_model(args)
    print_model_param_nums(model)
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    # Optionally resume
    if args.resume:
        checkpoint = torch.load(args.resume,weights_only=False)
        state_dict = checkpoint['state_dict']
        state_dict = remove_module_prefix(state_dict)
        model.load_state_dict(state_dict)
        args.start_epoch = args.start_epoch
        best_acc1 = checkpoint.get('best_acc1', 0)
        print(f"=> loaded checkpoint '{args.resume}' (epoch {args.start_epoch})")

    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

    train_dataset, val_dataset = get_dataset(args)
    num_classes = len(np.unique(train_dataset.targets))
    assert num_classes == args.num_classes

    cls_num_list = [0] * num_classes
    for label in train_dataset.targets:
        cls_num_list[label] += 1
    train_cls_num_list = np.array(cls_num_list)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True,worker_init_fn=seed_worker)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True,worker_init_fn=seed_worker)

    cls_weight = 1.0 / (np.array(cls_num_list) ** args.resample_weighting)
    cls_weight = cls_weight / np.sum(cls_weight) * len(cls_num_list)
    samples_weight = np.array([cls_weight[t] for t in train_dataset.targets])
    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()
    weighted_sampler = torch.utils.data.WeightedRandomSampler(
        samples_weight, len(samples_weight), replacement=True)
    weighted_train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, sampler=weighted_sampler,worker_init_fn=seed_worker)


    import logging
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.root_log, 'phase2_train_log'))
    fh.setFormatter(logging.Formatter(log_format))
    logger = logging.getLogger()
    logger.addHandler(fh)


    start_time = time.time()
    trainer = TrainerNullspaceAllLayers(
        args,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        weighted_train_loader=weighted_train_loader,
        per_class_num=train_cls_num_list,
        log=logger,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )
    trainer.train()
    end_time = time.time()
    print("It took {} to execute the program".format(
        time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))))

if __name__ == '__main__':
    main()
