import shutil
import torch
from torch.utils import data
import copy
import numpy as np
import os
from imbalance_data.cifar100Imbanlance import *
from imbalance_data.cifar10Imbanlance import *
from imbalance_data.dataset_lt_data import *
import utils.moco_loader as moco_loader
from utils.randaugment import rand_augment_transform

import torch

def build_P(K0: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """
    构造零空间投影矩阵 P = I - K0 (K0^T K0 + eps I)^{-1} K0^T
    K0: [D, n0]，每列是一个正确样本的特征向量
    返回 P: [D, D]
    """
    # K0^T K0 ∈ R^{n0×n0}
    KtK = K0.t().mm(K0)
    reg = eps * torch.eye(KtK.size(0), device=KtK.device)
    inv = torch.inverse(KtK + reg)      # ∈ R^{n0×n0}
    P = torch.eye(K0.size(0), device=K0.device) - K0.mm(inv).mm(K0.t())
    return P

def project_grad(param: torch.nn.Parameter, P: torch.Tensor) -> None:
    """
    将 param.grad 投影到 P 定义的子空间：
    param.grad ∈ R^{out×in}，P ∈ R^{in×in}，执行右乘
    """
    g = param.grad.data                    # [out, in]
    g_proj = g.mm(P)                       # [out, in]
    param.grad.data.copy_(g_proj)

def update_pa_lr(optimizer, fc_idx: int, ema_r: float,
                 g0: torch.Tensor, g1: torch.Tensor,
                 beta: float = 0.9, gamma_max: float = 10.0) -> float:
    """
    根据投影前后梯度范数比更新 EMA 并返回新的 ema_r
    同时在 optimizer.param_groups[fc_idx] 中写入 'gamma' 缩放因子
    """
    r = g1.norm().item() / (g0.norm().item() + 1e-12)
    ema_r = beta * ema_r + (1 - beta) * r
    scale = 1.0 / (ema_r + 1e-12)
    scale = max(1.0, min(scale, gamma_max))
    optimizer.param_groups[fc_idx]['gamma'] = scale
    return ema_r

class TwoCropTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class AverageMeter(object):

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.ceil(W * cut_rat).astype(int)
    cut_h = np.ceil(H * cut_rat).astype(int)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def get_transform(dataset):
    if dataset == "cifar10":
        mean = (0.49139968, 0.48215827, 0.44653124)
        std = (0.24703233, 0.24348505, 0.26158768)

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        return transform_train,transform_val

    if dataset == "cifar100":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        return transform_train, transform_val

    if dataset == "ImageNet-LT":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        rgb_mean = (0.485, 0.456, 0.406)
        ra_params = dict(translate_const=int(224 * 0.45),img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]), )
        augmentation_sim = [
            transforms.RandomResizedCrop(224),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)  # not strengthened
            ], p=1.0),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco_loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
        transform_val = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize])

        transform_train = transforms.Compose(augmentation_sim)

        return transform_train, transform_val

    if dataset == "iNaturelist2018":
        normalize = transforms.Normalize(mean=[0.466, 0.471, 0.380], std=[0.195, 0.194, 0.192])
        rgb_mean = (0.485, 0.456, 0.406)
        ra_params = dict(translate_const=int(224 * 0.45), img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]), )

        augmentation_sim = [
            transforms.RandomResizedCrop(224),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)  # not strengthened
            ], p=1.0),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco_loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
        transform_val = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize])

        transform_train = transforms.Compose(augmentation_sim)

        return transform_train, transform_val

def prepare_folders(args):
    folders_util = [args.root_log, args.root_model,
                    os.path.join(args.root_log, args.store_name),
                    os.path.join(args.root_model, args.store_name)]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.mkdir(folder)

def save_checkpoint(args, state, is_best, epoch):
    filename = '%s/%s/ckpt.pth.tar' % (args.root_model, args.store_name)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))
    if epoch % 20 == 0:
        filename = '%s/%s/%s_ckpt.pth.tar' % (args.root_model, args.store_name, str(epoch))
        torch.save(state, filename)

def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60.
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)

def GLMC_mixed(org1, org2, invs1, invs2, label_org, label_invs, label_org_w, label_invs_w, alpha=1):
    lam = np.random.beta(alpha, alpha)

    # mixup
    mixup_x = lam * org1 + (1 - lam) * invs1
    mixup_y = lam * label_org + (1 - lam) * label_invs
    mixup_y_w = lam * label_org_w + (1 - lam) * label_invs_w

    # cutmix
    bbx1, bby1, bbx2, bby2 = rand_bbox(org2.size(), lam)
    org2[:, :, bbx1:bbx2, bby1:bby2] = invs2[:, :, bbx1:bbx2, bby1:bby2]

    lam_cutmix = lam
    cutmix_y = lam_cutmix * label_org + (1 - lam_cutmix) * label_invs
    cutmix_y_w = lam_cutmix * label_org_w + (1 - lam_cutmix) * label_invs_w

    return mixup_x, org2, mixup_y, cutmix_y, mixup_y_w, cutmix_y_w

def print_model_param_nums(model=None):
    total = sum([param.nelement() for param in model.parameters()])
    print('  + Number of params: %.2fM' % (total / 1e6))
    return total