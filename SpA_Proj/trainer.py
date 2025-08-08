import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from sklearn.metrics import confusion_matrix
from utils import util
import matplotlib.pyplot as plt

class AverageMeter:
    def __init__(self, name, fmt=':f'):
        self.name, self.fmt = name, fmt
        self.reset()
    def reset(self):
        self.val = self.avg = self.sum = self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count else 0
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} (Avg:{avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def save_checkpoint(args, state, is_best, epoch):
    fn = (f"{args.root_model}/checkpoint_{epoch}.pth.tar")
    torch.save(state, fn)
    if is_best:
        torch.save(state, f"{args.root_model}/model_best.pth.tar")

class CenterAlignMonitor:
    def __init__(self, num_classes, feature_dim, device, ema_alpha=0.9):
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.device = device
        self.ema_alpha = ema_alpha
        self.class_centers = torch.zeros(num_classes, feature_dim).to(device)
        self.class_counts = torch.zeros(num_classes).to(device)
        self.align_history = []
        self.k0_count_history = []

    def update(self, features, targets):
        for c in range(self.num_classes):
            mask = (targets == c)
            if mask.sum() > 0:
                feat_c = features[mask].mean(0)
                self.class_centers[c] = self.ema_alpha * self.class_centers[c] + (1 - self.ema_alpha) * feat_c
                self.class_counts[c] += mask.sum().item()

    def get_aligned_classes(self, fc_weights, align_thresh=0.98, use_frobenius=False):
        centers = self.class_centers
        mu_g = centers.mean(dim=0, keepdim=True)
        centers_centered = centers - mu_g
        centers_normed = F.normalize(centers_centered, dim=1)
        fc_weights_normed = F.normalize(fc_weights, dim=1)
        if use_frobenius:
            centers_normed = centers_normed / centers_normed.norm(p='fro')
            fc_weights_normed = fc_weights_normed / fc_weights_normed.norm(p='fro')
        align_cos = (centers_normed * fc_weights_normed).sum(dim=1)
        aligned_classes = (align_cos > align_thresh).nonzero(as_tuple=True)[0]
        return aligned_classes, centers_normed, align_cos

    def log_and_plot(self, epoch=None):
        k0_arr = np.array(self.k0_count_history)
        print(f'[Monitor] Epoch {epoch} K0数量均值: {k0_arr.mean():.2f} (当前: {k0_arr[-1]})')

class Trainer(object):
    def __init__(
        self, args, model, train_loader, val_loader, weighted_train_loader, per_class_num, log,
        device='cuda', ema_alpha=0.9, align_thresh=0.7
    ):
        self.args = args
        self.device = device
        self.print_freq = args.print_freq
        self.epochs = args.epochs
        self.start_epoch = args.start_epoch
        self.num_classes = args.num_classes
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.weighted_train_loader = weighted_train_loader
        self.cls_num_list = per_class_num
        self.log = log
        self.model = model.to(device)
        self.align_thresh = align_thresh

        backbone = self.model.module if hasattr(self.model, "module") else self.model
        for param in backbone.parameters():
            param.requires_grad = True 

        params = [
            {"params": [p for n, p in backbone.named_parameters() if not n.startswith('fc') and p.requires_grad], "lr": args.lr},
            {"params": [p for n, p in backbone.named_parameters() if n.startswith('fc') and p.requires_grad], "lr": args.lr},
        ]
        self.optimizer = torch.optim.SGD(
            params,
            momentum=0.9, weight_decay=args.weight_decay
        )

        self.contrast_weight = args.contrast_weight
        self.label_weighting = args.label_weighting
        self.beta = args.beta
        self.update_weight()

        feature_dim = self.model.fc.weight.shape[1]
        self.center_monitor = CenterAlignMonitor(self.num_classes, feature_dim, self.device, ema_alpha=ema_alpha)

        self.cos_means = []
        self.acc_history = []

    def update_weight(self):
        per_cls_weights = 1.0 / (np.array(self.cls_num_list) ** self.label_weighting)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(self.cls_num_list)
        self.per_cls_weights = torch.FloatTensor(per_cls_weights).to(self.device)

    def paco_adjust_learning_rate(self, optimizer, epoch):
        warmup_epochs = 10
        lr = self.args.lr
        if epoch <= warmup_epochs:
            lr = self.args.lr / warmup_epochs * (epoch + 1)
        elif epoch <= 1000:
            lr *= 0.5 * (1. + np.cos(np.pi * (epoch - warmup_epochs + 1) / (self.epochs - warmup_epochs + 1)))
        else:
            lr = 0.0005
            if epoch <= warmup_epochs:
                lr = self.args.lr / warmup_epochs * (epoch - 160 + 1)
            lr *= 0.5 * (1. + np.cos(np.pi * (epoch - 160 - warmup_epochs + 1) / (self.epochs - warmup_epochs - 160 + 1)))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def train(self):
        best_acc1 = 0
        for epoch in range(self.start_epoch, self.epochs):
            self.paco_adjust_learning_rate(self.optimizer, epoch)
            print(f"Epoch {epoch + 1} learning rate: {self.optimizer.param_groups[0]['lr']}")

            batch_time = AverageMeter('Time', ':6.3f')
            data_time = AverageMeter('Data', ':6.3f')
            losses = AverageMeter('Loss', ':.4e')
            top1 = AverageMeter('Acc@1', ':6.2f')
            self.model.train()
            end = time.time()
            weighted_train_loader = iter(self.weighted_train_loader)
            for i, (inputs, targets) in enumerate(self.train_loader):
                input_org_1 = inputs[0].to(self.device)
                input_org_2 = inputs[1].to(self.device)
                target_org = targets.to(self.device)
                try:
                    input_invs, target_invs = next(weighted_train_loader)
                except:
                    weighted_train_loader = iter(self.weighted_train_loader)
                    input_invs, target_invs = next(weighted_train_loader)
                input_invs_1 = input_invs[0][:input_org_1.size(0)].to(self.device)
                input_invs_2 = input_invs[1][:input_org_2.size(0)].to(self.device)
                target_invs = target_invs.to(self.device)

                one_hot_org = torch.zeros(target_org.size(0), self.num_classes, device=self.device) \
                    .scatter_(1, target_org.view(-1, 1), 1)
                one_hot_org_w = self.per_cls_weights * one_hot_org

                one_hot_invs = torch.zeros(target_invs.size(0), self.num_classes, device=self.device) \
                    .scatter_(1, target_invs.view(-1, 1), 1)
                one_hot_invs = one_hot_invs[:one_hot_org.size(0)]
                one_hot_invs_w = self.per_cls_weights * one_hot_invs

                data_time.update(time.time() - end)

                lam = np.random.beta(self.beta, self.beta)
                mix_x, cut_x, mixup_y, mixcut_y, mixup_y_w, cutmix_y_w = util.GLMC_mixed(
                    org1=input_org_1, org2=input_org_2,
                    invs1=input_invs_1, invs2=input_invs_2,
                    label_org=one_hot_org, label_invs=one_hot_invs,
                    label_org_w=one_hot_org_w, label_invs_w=one_hot_invs_w
                )

              
                with torch.no_grad():
                    feats = self.model(input_org_1, get_feat=True)
                    self.center_monitor.update(feats, target_org)
                    fc_weights = self.model.fc.weight.data
                    aligned_classes, centers, align_cos = self.center_monitor.get_aligned_classes(
                        fc_weights, self.align_thresh)

                output_1, output_cb_1, z1, p1 = self.model(mix_x, train=True)
                output_2, output_cb_2, z2, p2 = self.model(cut_x, train=True)
                contrastive_loss = self.SimSiamLoss(p1, z2) + self.SimSiamLoss(p2, z1)

                loss_mix = -torch.mean(torch.sum(F.log_softmax(output_1, dim=1) * mixup_y, dim=1))
                loss_cut = -torch.mean(torch.sum(F.log_softmax(output_2, dim=1) * mixcut_y, dim=1))
                loss_mix_w = -torch.mean(torch.sum(F.log_softmax(output_cb_1, dim=1) * mixup_y_w, dim=1))
                loss_cut_w = -torch.mean(torch.sum(F.log_softmax(output_cb_2, dim=1) * cutmix_y_w, dim=1))

                alpha_epoch = 1 - (epoch / self.epochs) ** 2
                balance_loss = loss_mix + loss_cut
                rebalance_loss = loss_mix_w + loss_cut_w
                loss = alpha_epoch * balance_loss + (1 - alpha_epoch) * rebalance_loss + self.contrast_weight * contrastive_loss

                losses.update(loss.item(), inputs[0].size(0))
                self.optimizer.zero_grad()
                loss.backward()

                backbone = self.model.module if hasattr(self.model, "module") else self.model
                fc_weight = backbone.fc.weight

                if fc_weight.grad is not None:
                    centers = self.center_monitor.class_centers  
                    mu_g = centers.mean(dim=0, keepdim=True)
                    centers_centered = centers - mu_g
                    mu_normed = F.normalize(centers_centered, dim=1)  

                    grad = fc_weight.grad
                    grad_proj = torch.zeros_like(grad)
                    gamma = 0.6

               
                    for c in range(self.num_classes):
                        w = F.normalize(fc_weight.data[c], dim=0)
                        μ = mu_normed[c]

                        g = grad[c]
                        g_tan = g - torch.dot(g, w) * w
                        d = μ - torch.dot(μ, w) * w 
                        d_norm = torch.norm(d).clamp(min=1e-8)
                        d = d / d_norm
                        if torch.dot(g_tan, d) > 0:
                            g_tan = g_tan - torch.dot(g_tan, d) * d
                        grad_proj[c] = (1 - gamma) * g_tan + gamma * g

                    fc_weight.grad = grad_proj

                self.optimizer.step()
            acc1 = self.validate(epoch=epoch)
            cos_mean = align_cos.detach().cpu().numpy().mean()
            self.cos_means.append(float(cos_mean))
            self.acc_history.append(acc1)
            print(f'Epoch {epoch+1} | mean cos alignment: {cos_mean}')
            #np.save(f"/data/GLMC-main/acc&cos/cos_means_imb50_proj3.npy", np.array(self.cos_means))
            #np.save(f"/data/GLMC-main/acc&cos/accuracies_imb50_proj3.npy", np.array(self.acc_history))

            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            print('Best Prec@1: %.3f\n' % (best_acc1))
            save_checkpoint(self.args, {
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'best_acc1':  best_acc1,
            }, is_best, epoch + 1)

            if epoch == self.epochs - 1:
                torch.save({
                    'epoch': self.epochs,
                    'state_dict': self.model.state_dict(),
                    'best_acc1': best_acc1,
                }, f"/data/GLMC-main/result/model_ib=100_baseline")
            self.center_monitor.align_history.append(align_cos.detach().cpu().numpy())
            self.center_monitor.k0_count_history.append(len(aligned_classes))
            self.center_monitor.log_and_plot(epoch)


    def SimSiamLoss(self, p, z, version='simplified'):
        z = z.detach()
        if version == 'original':
            p = F.normalize(p, dim=1)
            z = F.normalize(z, dim=1)
            return -(p * z).sum(dim=1).mean()
        return -F.cosine_similarity(p, z, dim=-1).mean()

    def validate(self, epoch=None):
        batch_time = AverageMeter('Time', ':6.3f')
        top1 = AverageMeter('Acc@1', ':6.2f')
        eps = np.finfo(np.float64).eps
        self.model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            end = time.time()
            for i, (input, target) in enumerate(self.val_loader):
                input = input.to(self.device)
                target = target.to(self.device)
                output = self.model(input, train=False)
                acc1 = (output.argmax(1) == target).float().mean() * 100
                top1.update(acc1.item(), input.size(0))
                _, pred = torch.max(output, 1)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                batch_time.update(time.time() - end)
                end = time.time()
                if i % self.print_freq == 0:
                    print(f"Test: [{i}/{len(self.val_loader)}]\t"
                          f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                          f"Prec@1 {top1.val:.3f} ({top1.avg:.3f})")
            cf = confusion_matrix(all_targets, all_preds).astype(float)
            cls_cnt = cf.sum(axis=1)
            cls_hit = np.diag(cf)
            cls_acc = cls_hit / (cls_cnt + eps)
            print("EPOCH: {} val Results: Prec@1 {:.3f}".format(epoch + 1, top1.avg))
            many_shot = self.cls_num_list > 100
            medium_shot = (self.cls_num_list <= 100) & (self.cls_num_list > 20)
            few_shot = self.cls_num_list <= 20
            print("many avg, med avg, few avg",
                  float(sum(cls_acc[many_shot]) * 100 / (sum(many_shot) + eps)),
                  float(sum(cls_acc[medium_shot]) * 100 / (sum(medium_shot) + eps)),
                  float(sum(cls_acc[few_shot]) * 100 / (sum(few_shot) + eps))
                  )
        return top1.avg
