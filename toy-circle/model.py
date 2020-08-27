import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib
from utils import *
from modules import *


# ======================================================================================================================
def to_np(x):
    return x.detach().cpu().numpy()


def to_tensor(x, device="cuda"):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).to(device)
    else:
        x = x.to(device)
    return x


def chamfer_dist(p1, p2):
    diff = p1[:, :, None, :] - p2[:, None, :, :]
    d = (diff ** 2).sum(3)
    d1 = torch.min(d, 1)[0]
    d2 = torch.min(d, 2)[0]
    return d1.mean(1) + d2.mean(1)


def flat(x):
    n, m = x.shape[:2]
    return x.reshape(n * m, *x.shape[2:])


# ======================================================================================================================

class BaseModel(nn.Module):
    def __init__(self, opt):
        super(BaseModel, self).__init__()
        self.opt = opt
        self.device = opt.device

        self.train_log = opt.outf + '/train.log'
        self.model_path = opt.outf + '/model.pth'
        self.best_acc_tgt = 0

        mask_list = [1] * opt.num_source + [0] * opt.num_target
        self.domain_mask = torch.IntTensor(mask_list)
        # print('source domain', self.domain_mask == 1)
        # print('target domain', self.domain_mask == 0)

        self.tsne = TSNE(n_components=2)
        self.pca = PCA(n_components=2)

        self.set_num_domain(opt.num_domain)
        self.model_path = opt.outf + '/model.pth'

    #         self.wgan = opt.wgan if opt.wgan is not None else False

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def set_num_domain(self, num):
        self.num_domain = num
        # t is domain index: normalized to [0,1]
        self.t = np.linspace(0, 1, num).astype(np.float32)
        self.t_var = to_tensor(self.t, self.device)
        # z is domain class (0,1,2,...) will be used by some adaptation methods
        self.z = np.arange(num).astype(np.int64)
        self.z_var = to_tensor(self.z, self.device)

    def set_input(self, input):
        """
        :param
            input: x_seq, y_seq
            x_seq: Number of domain x Batch size x Data dim
            y_seq: Number of domain x Batch size x Label dim
        """
        self.x_seq, self.y_seq = input
        self.T, self.B = self.x_seq.shape[:2]
        self.t_seq = to_tensor(np.zeros((self.T, self.B, 1), dtype=np.float32), self.device) + self.t_var.reshape(self.T, 1, 1)
        self.z_seq = to_tensor(np.zeros((self.T, self.B), dtype=np.int64), self.device) + self.z_var.reshape(self.T, 1)

    def forward(self):
        self.e_seq = self.netE(self.x_seq, self.t_seq)  # encoder of the data
        self.f_seq = self.netF(self.e_seq)  # logit of the prediction
        self.g_seq = torch.argmax(self.f_seq.detach(), dim=2)  # class of the prediction

    def optimize_parameters(self):
        self.forward()  # forward prediction
        # update the discriminator D (optional)
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        # update the encoder E and predictor F
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    #         if self.wgan:
    #             clamp_range = 2.0
    #             for p in self.netF.parameters():
    #                 p.data.clamp_(-clamp_range, clamp_range)
    #             for p in self.netE.parameters():
    #                 p.data.clamp_(-clamp_range, clamp_range)
    #             for p in self.netD.parameters():
    #                 p.data.clamp_(-clamp_range, clamp_range)

    def learn(self, epoch, dataloader):
        self.epoch = epoch

        self.train()

        loss_curve = {
            loss: []
            for loss in self.loss_names
        }
        acc_curve = []

        for data in dataloader:
            x_seq, y_seq = [d[0][None, :, :] for d in data], [d[1][None, :] for d in data]
            x_seq = torch.cat(x_seq, 0).to(self.device)
            y_seq = torch.cat(y_seq, 0).to(self.device)

            self.set_input(input=(x_seq, y_seq))
            self.optimize_parameters()

            for loss in self.loss_names:
                loss_curve[loss].append(getattr(self, 'loss_' + loss).item())

            acc_curve.append(self.g_seq.eq(self.y_seq).to(torch.float).mean(-1, keepdim=True))

        loss_msg = '[Train][{}] Loss:'.format(epoch)
        for loss in self.loss_names:
            loss_msg += ' {} {:.3f}'.format(loss, np.mean(loss_curve[loss]))

        acc = to_np(torch.cat(acc_curve, 1).mean(-1))
        acc_msg = '[Train][{}] Accuracy: total average {:.1f}, in each domain {}'.format(epoch, acc.mean() * 100, np.around(acc * 100, decimals=1))

        if (epoch + 1) % 10 == 0:
            print(loss_msg)
            print(acc_msg)
        with open(self.train_log, 'a') as f:
            f.write(loss_msg + "\n" + acc_msg + "\n")
        for lr_scheduler in self.lr_schedulers:
            lr_scheduler.step()

    def save(self):
        torch.save(self.state_dict(), self.model_path)

    def load(self):
        print('===> Loading model from {}'.format(self.model_path))
        try:
            self.load_state_dict(torch.load(self.model_path))
            print('<=== Success!')
        except:
            print('<==== Failed!')

    def set_data_stats(self, dm, ds):
        self.data_m, self.data_s = dm, ds

    def visualize_F(self, phase=None):
        """
        visualize the predictor F
        """
        y_np = to_np(self.y_seq)
        g_np = to_np(self.g_seq)
        x_np = to_np(self.x_seq)

        if self.opt.normalize_domain:
            for i in range(len(x_np)):
                x_np[i] = x_np[i] * self.data_s[i] + self.data_m[i]

        fn = 'prediction.png'
        if phase is not None:
            fn = 'prediction_{}.png'.format(phase)

        for x, y, g in zip(x_np, y_np, g_np):
            for i in range(2):
                for j in range(2):
                    mark = ['+', '.'][i]
                    color = ['b', 'r'][j]
                    plt.plot(x[(y == i) & (g == j), 0], x[(y == i) & (g == j), 1], mark, color=color, markersize=10)
            plt.savefig(self.opt.outf + '/' + fn)
        plt.close()

    def test(self, epoch, dataloader):
        self.eval()

        acc_curve = []
        l_x = []
        l_y = []
        l_domain = []
        l_prob = []
        l_label = []

        for data in dataloader:
            x_seq, y_seq = [d[0][None, :, :] for d in data], [d[1][None, :] for d in data]
            x_seq = torch.cat(x_seq, 0).to(self.device)
            y_seq = torch.cat(y_seq, 0).to(self.device)

            self.set_input(input=(x_seq, y_seq))
            self.forward()
            acc_curve.append(self.g_seq.eq(self.y_seq).to(torch.float).mean(-1, keepdim=True))

            if self.opt.normalize_domain:
                x_np = to_np(x_seq)
                for i in range(len(x_np)):
                    x_np[i] = x_np[i] * self.data_s[i] + self.data_m[i]
                l_x.append(x_np)
            else:
                l_x.append(to_np(x_seq))

            l_y.append(to_np(y_seq))
            l_domain.append(to_np(self.z_seq))
            l_prob.append(to_np(self.f_seq))
            l_label.append(to_np(self.g_seq))

        x_all = np.concatenate(l_x, axis=1)
        y_all = np.concatenate(l_y, axis=1)
        domain_all = np.concatenate(l_domain, axis=1)
        prob_all = np.concatenate(l_prob, axis=1)
        label_all = np.concatenate(l_label, axis=1)

        d_all = dict()
        d_all['data'] = flat(x_all)
        d_all['gt'] = flat(y_all)
        d_all['domain'] = flat(domain_all)
        d_all['prob'] = flat(prob_all)
        d_all['label'] = flat(label_all)

        write_pickle(d_all, self.opt.outf + '/pred.pkl')

        acc = to_np(torch.cat(acc_curve, 1).mean(-1))
        acc_msg = '[Test][{}] Accuracy: total average {:.1f}, in each domain {}'.format(epoch, acc.mean() * 100, np.around(acc * 100, decimals=1))
        print(acc_msg)

    def init_weight(self, net=None):
        if net is None:
            net = self
        for m in net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
#                 nn.init.normal_(m.weight, mean=0, std=0.1)
#                 nn.init.xavier_normal_(m.weight, gain=10)
                nn.init.constant_(m.bias, val=0)


class SO(BaseModel):
    """
    Source Only Model
    """

    def __init__(self, opt):
        super(SO, self).__init__(opt)
        self.netE = FeatureNet(opt)
        self.netF = PredNet(opt)
        self.init_weight()
        non_D_parameters = list(self.netE.parameters()) + list(self.netF.parameters())
        self.optimizer_G = torch.optim.Adam(non_D_parameters, lr=opt.lr, betas=(opt.beta1, 0.999))
        self.lr_scheduler_G = lr_scheduler.ExponentialLR(optimizer=self.optimizer_G, gamma=0.5 ** (1 / 100))
        self.lr_schedulers = [self.lr_scheduler_G]
        self.loss_names = ['E_pred']

    def backward_G(self):
        # label supervise only from source domains
        self.y_seq_source = self.y_seq[self.domain_mask == 1]
        self.f_seq_source = self.f_seq[self.domain_mask == 1]
        self.loss_E_pred = F.nll_loss(flat(self.f_seq_source), flat(self.y_seq_source))
        self.loss_E = self.loss_E_pred
        self.loss_E.backward()

    def optimize_parameters(self):
        self.forward()  # forward prediction
        # update the encoder E and predictor F
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()


class ADDA(BaseModel):
    """
    Adversarial Discriminative Domain Adaptation (Tzeng et al., 2017)
    """

    def __init__(self, opt):
        super(ADDA, self).__init__(opt)
        self.netE = FeatureNet(opt)
        self.netF = PredNet(opt)
        self.netD = DiscNet(opt)
        non_D_parameters = list(self.netE.parameters()) + list(self.netF.parameters())
        self.optimizer_G = torch.optim.Adam(non_D_parameters, lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.lr_scheduler_G = lr_scheduler.ExponentialLR(optimizer=self.optimizer_G, gamma=0.5 ** (1 / 100))
        self.lr_scheduler_D = lr_scheduler.ExponentialLR(optimizer=self.optimizer_D, gamma=0.5 ** (1 / 100))
        self.lr_schedulers = [self.lr_scheduler_G, self.lr_scheduler_D]
        self.loss_names = ['D', 'E_gan', 'E_pred']
        self.lambda_gan = opt.lambda_gan

    def backward_G(self):
        # E_gan: encoder loss from discriminator
        self.d_seq = self.netD(self.e_seq)
        self.d_seq_target = self.d_seq[self.domain_mask == 0]
        self.loss_E_gan = - torch.log(self.d_seq_target + 1e-10).mean()
        # E_pred: encoder loss from prediction the label
        self.y_seq_source = self.y_seq[self.domain_mask == 1]
        self.f_seq_source = self.f_seq[self.domain_mask == 1]
        self.loss_E_pred = F.nll_loss(flat(self.f_seq_source), flat(self.y_seq_source))

        self.loss_E = self.loss_E_gan * self.lambda_gan + self.loss_E_pred
        self.loss_E.backward()

    def backward_D(self):
        self.d_seq = self.netD(self.e_seq.detach())
        self.d_seq_source = self.d_seq[self.domain_mask == 1]
        self.d_seq_target = self.d_seq[self.domain_mask == 0]
        # D: discriminator loss from classifying source v.s. target
        self.loss_D = - torch.log(self.d_seq_source + 1e-10).mean() \
                      - torch.log(1 - self.d_seq_target + 1e-10).mean()
        self.loss_D.backward()


class MDD(BaseModel):
    """
    Margin Disparity Discrepancy (Zhang et al., 2019)
    """

    def __init__(self, opt):
        super(MDD, self).__init__(opt)
        self.netE = FeatureNet(opt)
        self.netF = PredNet(opt)
        self.netD = PredNet(opt)  # it is an unusual "discriminator", it is a predictor but with largest discrepancy with netF

        non_D_parameters = list(self.netE.parameters()) + list(self.netF.parameters())
        self.optimizer_G = torch.optim.Adam(non_D_parameters, lr=opt.lr, betas=(opt.beta1, 0.999),
                                            weight_decay=opt.weight_decay)
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999),
                                            weight_decay=opt.weight_decay)

        self.lr_scheduler_G = lr_scheduler.ExponentialLR(optimizer=self.optimizer_G, gamma=0.5 ** (1 / 100))
        self.lr_scheduler_D = lr_scheduler.ExponentialLR(optimizer=self.optimizer_D, gamma=0.5 ** (1 / 100))
        self.lr_schedulers = [self.lr_scheduler_G, self.lr_scheduler_D]

        self.loss_names = ['E_pred', 'E_adv', 'ADV_src', 'ADV_tgt']

        self.lambda_src = opt.lambda_src
        self.lambda_tgt = opt.lambda_tgt

    def backward_G(self):
        self.loss_E_pred = F.nll_loss(flat(self.f_seq[self.domain_mask == 1]),
                                      flat(self.y_seq[self.domain_mask == 1]))

        self.f_adv, self.f_adv_softmax = self.netD(self.e_seq, return_softmax=True)
        self.loss_ADV_src = F.nll_loss(flat(self.f_adv[self.domain_mask == 1]),
                                       flat(self.g_seq[self.domain_mask == 1]))
        f_adv_tgt = torch.log(1 - self.f_adv_softmax[self.domain_mask == 0] + 1e-10)
        self.loss_ADV_tgt = F.nll_loss(flat(f_adv_tgt),
                                       flat(self.g_seq[self.domain_mask == 0]))
        self.loss_E_adv = -(self.loss_ADV_src * self.lambda_src + self.loss_ADV_tgt * self.lambda_tgt) / (self.lambda_src + self.lambda_tgt)
        self.loss_E = self.loss_E_pred + self.opt.lambda_gan * self.loss_E_adv
        self.loss_E.backward()

    def backward_D(self):
        self.f_adv, self.f_adv_softmax = self.netD(self.e_seq.detach(), return_softmax=True)
        # agreement with netF on source domain
        self.loss_ADV_src = F.nll_loss(flat(self.f_adv[self.domain_mask == 1]),
                                       flat(self.g_seq[self.domain_mask == 1]))
        f_adv_tgt = torch.log(1 - self.f_adv_softmax[self.domain_mask == 0] + 1e-10)
        # disagreement with netF on target domain
        self.loss_ADV_tgt = F.nll_loss(flat(f_adv_tgt),
                                       flat(self.g_seq[self.domain_mask == 0]))
        # minimize the agreement on source domain while maximize the disagreement on target domain
        self.loss_D = (self.loss_ADV_src * self.lambda_src + self.loss_ADV_tgt * self.lambda_tgt) / (self.lambda_src + self.lambda_tgt)
        self.loss_D.backward()


class DANN(BaseModel):
    def __init__(self, opt):
        super(DANN, self).__init__(opt)
        self.netE = FeatureNet(opt)
        self.netF = PredNet(opt)
        self.netD = ClassDiscNet(opt)
        non_D_parameters = list(self.netE.parameters()) + list(self.netF.parameters())
        self.optimizer_G = torch.optim.Adam(non_D_parameters, lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.lr_scheduler_G = lr_scheduler.ExponentialLR(optimizer=self.optimizer_G, gamma=0.5 ** (1 / 100))
        self.lr_scheduler_D = lr_scheduler.ExponentialLR(optimizer=self.optimizer_D, gamma=0.5 ** (1 / 100))
        self.lr_schedulers = [self.lr_scheduler_G, self.lr_scheduler_D]
        self.loss_names = ['D', 'E_gan', 'E_pred']
        self.lambda_gan = opt.lambda_gan

    def visualize_D(self):
        d_max = torch.argmax(self.d_seq, dim=2)
        d_np = to_np(d_max).flatten()
        t_np = to_np(self.z_seq).flatten()
        fig, ax = plt.subplots(1, 1, figsize=(12, 12), dpi=200)
        plt.plot(t_np, d_np, 'ro', alpha=0.5, markersize=10)
        plt.xlabel('true domain')
        plt.ylabel('disc pred domain')
        plt.xlim([-1, 16])
        plt.ylim([-1, 16])
        plt.savefig('./figures/disc.png')
        plt.close()

    def forward(self):
        self.e_seq = self.netE(self.x_seq, self.t_seq)
        self.f_seq = self.netF(self.e_seq)
        self.g_seq = torch.argmax(self.f_seq.detach(), dim=2)

        self.f_seq_softmax = torch.softmax(self.f_seq.detach(), dim=2)

    def backward_G(self):
        self.d_seq = self.netD(self.e_seq, self.f_seq_softmax)

        self.loss_E_gan = - F.nll_loss(flat(self.d_seq), flat(self.z_seq))

        self.y_seq_source = self.y_seq[self.domain_mask == 1]
        self.f_seq_source = self.f_seq[self.domain_mask == 1]

        self.loss_E_pred = F.nll_loss(flat(self.f_seq_source), flat(self.y_seq_source))

        self.loss_E = self.loss_E_gan * self.lambda_gan + self.loss_E_pred
        self.loss_E.backward()

    def backward_D(self):
        self.d_seq = self.netD(self.e_seq.detach(), self.f_seq_softmax)
        self.loss_D = F.nll_loss(flat(self.d_seq), flat(self.z_seq))
        self.loss_D.backward()


class CIDA(BaseModel):
    """
    Notice that we use L1 loss instead of MSE loss for the discriminator.
    MSE loss seems to lead worse performance than L1 for some unknown reasons.
    """

    def __init__(self, opt):
        super(CIDA, self).__init__(opt)

        self.netE = FeatureNet(opt)
        self.netF = PredNet(opt)
        self.netD = DiscNet(opt)
        non_D_parameters = list(self.netE.parameters()) + list(self.netF.parameters())
        self.optimizer_G = torch.optim.Adam(non_D_parameters, lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.lr_scheduler_G = lr_scheduler.ExponentialLR(optimizer=self.optimizer_G, gamma=0.5 ** (1 / opt.gamma))
        self.lr_scheduler_D = lr_scheduler.ExponentialLR(optimizer=self.optimizer_D, gamma=0.5 ** (1 / opt.gamma))
        self.lr_schedulers = [self.lr_scheduler_G, self.lr_scheduler_D]
        self.loss_names = ['D', 'E_gan', 'E_pred', 'E_pseudo']
        self.lambda_gan = opt.lambda_gan

        self.num_source = opt.num_source
        self.num_domain = opt.num_domain
        self.num_epoch = opt.num_epoch

        if opt.data == 'half-circle' or opt.data == 'quarter-circle':
            self.num_domain_step = 6
        elif opt.data == 'sine':
            self.num_domain_step = 2
        self.num_stage = int(np.ceil((self.num_domain - self.num_source) / self.num_domain_step))
        self.epoch_in_a_stage = opt.num_epoch // self.num_stage + bool(opt.num_epoch % self.num_stage > 0)
        self.pseudo_label = np.zeros((self.num_domain, 1000))
        
#         self.init_weight(self.netD)
#         self.init_weight(self.netF)
#         self.init_weight(self.netE)        

    def set_input(self, input):
        """
        :param
            input: x_seq, y_seq
            x_seq: Number of domain x Batch size x Data dim
            y_seq: Number of domain x Batch size x Label dim
        """
        self.x_seq, self.y_seq, self.idx_seq = input[0], input[1], input[2]
        self.T, self.B = self.x_seq.shape[:2]
        self.t_seq = to_tensor(np.zeros((self.T, self.B, 1), dtype=np.float32), self.device) + self.t_var.reshape(self.T, 1, 1)
        self.z_seq = to_tensor(np.zeros((self.T, self.B), dtype=np.int64), self.device) + self.z_var.reshape(self.T, 1)

        self.domain_weight = self.t_seq.clone()
        self.pseudo_weight = np.zeros((self.T))

        self.domain_weight[:, :, :] = 0
        self.pseudo_weight[:] = 0
        self.domain_weight[:self.num_source + self.num_domain_step * (1 + self.stage), :, :] = 1
        # self.pseudo_weight[self.num_source: self.num_source + self.num_domain_step * self.stage] = 1e-3

#         self.pseudo_weight[self.num_source: self.num_source + self.num_domain_step * self.stage] = 1
        self.pseudo_weight[self.num_source: self.num_source + self.num_domain_step * self.stage] = np.linspace(0.001, 0.001, self.num_domain_step * self.stage)

        self.p_seq = torch.zeros_like(self.y_seq)

        # print('pseudo weight', self.pseudo_weight, 'stage', self.stage)

        pseudo_acc = []
        for i in range(self.T):
            if self.pseudo_weight[i] == 0:
                continue
            idx = to_np(self.idx_seq[i])
            pseudo_label_i = self.pseudo_label[i][idx]
            for j in range(len(pseudo_label_i)):
                self.p_seq[i][j] = pseudo_label_i[j]
            pseudo_acc.append(to_np(self.p_seq[i].eq(self.y_seq[i])).mean())
        # print('pseudo_acc', pseudo_acc)

    def backward_G(self):
        self.d_seq = self.netD(self.e_seq)
        self.loss_E_gan = - F.l1_loss(flat(self.d_seq * self.domain_weight), flat(self.t_seq * self.domain_weight))
        #         self.loss_E_gan = - F.mse_loss(flat(self.d_seq[self.domain_mask >= 0]), flat(self.t_seq[self.domain_mask >= 0]))

        self.y_seq_source = self.y_seq[self.domain_mask == 1]
        self.f_seq_source = self.f_seq[self.domain_mask == 1]

        self.loss_E_pred = F.nll_loss(flat(self.f_seq_source), flat(self.y_seq_source))

        self.loss_E_pseudo = 0
        for i in range(self.T):
            self.loss_E_pseudo += F.nll_loss(self.f_seq[i], self.p_seq[i]) * self.pseudo_weight[i]
            # print('acc pseudo', i, self.p_seq[i].eq(self.y_seq[i]).to(float).mean())
        if self.pseudo_weight.sum() > 0:
            self.loss_E_pseudo /= self.pseudo_weight.sum()

        num_pseudo_domain = self.num_domain_step * self.stage
        alpha = self.num_source / (self.num_source + num_pseudo_domain * 0.5)
        self.loss_E = self.loss_E_gan * self.lambda_gan + self.loss_E_pred * alpha + self.loss_E_pseudo * (1 - alpha)
        self.loss_E.backward()

    def backward_D(self):
        self.d_seq = self.netD(self.e_seq.detach())
        self.loss_D = F.l1_loss(flat(self.d_seq * self.domain_weight), flat(self.t_seq * self.domain_weight))
        #         self.loss_D = F.mse_loss(flat(self.d_seq[self.domain_mask >= 0]), flat(self.t_seq[self.domain_mask >= 0]))

        self.loss_D.backward()

    def visualize_D(self):
        d_np = to_np(self.d_seq).flatten()
        t_np = to_np(self.t_seq).flatten()

        fig, ax = plt.subplots(1, 1, figsize=(12, 12), dpi=200)
        plt.plot(t_np, d_np, 'ro', alpha=0.5, markersize=10)
        plt.xlabel('true domain')
        plt.ylabel('disc pred domain')
        plt.xlim([-0.1, 1.1])
        plt.ylim([-0.1, 1.1])
        plt.savefig('./figures/disc.png')
        plt.close()

    def visualize_E(self):
        cmap = matplotlib.cm.get_cmap('jet')

        e_np = to_np(self.e_seq)
        y_np = to_np(self.y_seq)

        T, B, C = e_np.shape
        # _t = time.time()
        # tmp = self.tsne.fit_transform(flat(e_np)).reshape(T, B, 2)
        # print('tsne', time.time() - _t)
        # fig, ax = plt.subplots(1, 1, figsize=(12, 12), dpi=200)
        # for i, (e, y) in enumerate(zip(tmp, y_np)):
        #     plt.plot(e[y == 0, 0], e[y == 0, 1], '.', color=cmap(i / (T - 1))[:3], markersize=10)
        #     plt.plot(e[y == 1, 0], e[y == 1, 1], '+', color=cmap(i / (T - 1))[:3], markersize=10)
        # plt.savefig('./figures/encoding_tsne.png')
        # plt.close()

        tmp = self.pca.fit_transform(flat(e_np)).reshape(T, B, 2)
        fig, ax = plt.subplots(1, 1, figsize=(12, 12), dpi=200)
        for i, (e, y) in enumerate(zip(tmp, y_np)):
            plt.plot(e[y == 0, 0], e[y == 0, 1], '.', color=cmap(i / (T - 1))[:3], markersize=10)
            plt.plot(e[y == 1, 0], e[y == 1, 1], '+', color=cmap(i / (T - 1))[:3], markersize=10)
        plt.savefig('./figures/encoding_pca.png')
        plt.close()

    def learn(self, epoch, dataloader):
        self.epoch = epoch
        self.stage = epoch // self.epoch_in_a_stage

        self.train()

        loss_curve = {
            loss: []
            for loss in self.loss_names
        }
        acc_curve = []

        for data in dataloader:
            x_seq, y_seq, idx_seq = [d[0][None, :, :] for d in data], [d[1][None, :] for d in data], [d[2][None, :] for d in data]
            x_seq = torch.cat(x_seq, 0).to(self.device)
            y_seq = torch.cat(y_seq, 0).to(self.device)
            idx_seq = torch.cat(idx_seq, 0).to(self.device)

            self.set_input(input=(x_seq, y_seq, idx_seq))
            self.optimize_parameters()

            for loss in self.loss_names:
                loss_curve[loss].append(getattr(self, 'loss_' + loss).item())

            acc_curve.append(self.g_seq.eq(self.y_seq).to(torch.float).mean(-1, keepdim=True))

        loss_msg = '[Train][{}] Loss:'.format(epoch)
        for loss in self.loss_names:
            loss_msg += ' {} {:.3f}'.format(loss, np.mean(loss_curve[loss]))

        acc = to_np(torch.cat(acc_curve, 1).mean(-1))
        acc_msg = '[Train][{}] Accuracy: total average {:.1f}, in each domain {}'.format(epoch, acc.mean() * 100, np.around(acc * 100, decimals=1))

        if (epoch + 1) % 10 == 0:
            print(loss_msg)
            print(acc_msg)
        with open(self.train_log, 'a') as f:
            f.write(loss_msg + "\n" + acc_msg + "\n")
        for lr_scheduler in self.lr_schedulers:
            lr_scheduler.step()

    def test(self, epoch, dataloader):
        self.eval()
        self.epoch = epoch
        self.stage = epoch // self.epoch_in_a_stage

        acc_curve = []
        l_x = []
        l_y = []
        l_domain = []
        l_prob = []
        l_label = []

        is_anchor_epoch = (epoch + 1) % self.epoch_in_a_stage == 0
        if is_anchor_epoch:
            print(f'===> Anchor Epoch {epoch}', self.T)
            pseudo_acc = np.zeros((self.T))
            num_iter = 0

        for data in dataloader:
            x_seq, y_seq, idx_seq = [d[0][None, :, :] for d in data], [d[1][None, :] for d in data], [d[2][None, :] for d in data]
            x_seq = torch.cat(x_seq, 0).to(self.device)
            y_seq = torch.cat(y_seq, 0).to(self.device)
            idx_seq = torch.cat(idx_seq, 0).to(self.device)

            self.set_input(input=(x_seq, y_seq, idx_seq))
            self.forward()
            acc_curve.append(self.g_seq.eq(self.y_seq).to(torch.float).mean(-1, keepdim=True))

            if self.opt.normalize_domain:
                x_np = to_np(x_seq)
                for i in range(len(x_np)):
                    x_np[i] = x_np[i] * self.data_s[i] + self.data_m[i]
                l_x.append(x_np)
            else:
                l_x.append(to_np(x_seq))

            l_y.append(to_np(y_seq))
            l_domain.append(to_np(self.z_seq))
            l_prob.append(to_np(self.f_seq))
            l_label.append(to_np(self.g_seq))

            if is_anchor_epoch:
                for i in range(self.num_source + self.stage * self.num_domain_step, self.T):
                    if i >= self.num_source + (self.stage + 1) * self.num_domain_step:
                        break
                    pseudo_label_i = to_np(self.g_seq[i])
                    idx_i = to_np(self.idx_seq[i])
                    self.pseudo_label[i][idx_i] = pseudo_label_i

                    pseudo_acc[i] += np.mean(pseudo_label_i == to_np(self.y_seq[i]))

                num_iter += 1

        if is_anchor_epoch:
            print('==> pseudo acc')
            pseudo_acc = np.array(pseudo_acc) / num_iter
            print(pseudo_acc)
            if self.stage < self.num_stage - 2:
            #     self.init_weight(self.netD)
            #     self.init_weight(self.netF)
            #     self.init_weight(self.netE)
                non_D_parameters = list(self.netE.parameters()) + list(self.netF.parameters())
                self.optimizer_G = torch.optim.Adam(non_D_parameters, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
                self.lr_scheduler_G = lr_scheduler.ExponentialLR(optimizer=self.optimizer_G, gamma=0.5 ** (1 / self.opt.gamma))
                self.lr_scheduler_D = lr_scheduler.ExponentialLR(optimizer=self.optimizer_D, gamma=0.5 ** (1 / self.opt.gamma))
                self.lr_schedulers = [self.lr_scheduler_G, self.lr_scheduler_D]

        x_all = np.concatenate(l_x, axis=1)
        y_all = np.concatenate(l_y, axis=1)
        domain_all = np.concatenate(l_domain, axis=1)
        prob_all = np.concatenate(l_prob, axis=1)
        label_all = np.concatenate(l_label, axis=1)

        d_all = dict()
        d_all['data'] = flat(x_all)
        d_all['gt'] = flat(y_all)
        d_all['domain'] = flat(domain_all)
        d_all['prob'] = flat(prob_all)
        d_all['label'] = flat(label_all)

        write_pickle(d_all, self.opt.outf + '/pred.pkl')

        acc = to_np(torch.cat(acc_curve, 1).mean(-1))
        acc_msg = '[Test][{}] Accuracy: total average {:.1f}, in each domain {}'.format(epoch, acc.mean() * 100, np.around(acc * 100, decimals=1))
        print(acc_msg)


class PCIDA(BaseModel):
    """
    #TODO: Currently does not work!!!
    """

    def __init__(self, opt):
        super(PCIDA, self).__init__(opt)
        self.netE = FeatureNet(opt)
        self.netF = PredNet(opt)
        self.netD = ProbDiscNet(opt)

        non_D_parameters = list(self.netE.parameters()) + list(self.netF.parameters())
        self.optimizer_G = torch.optim.Adam(non_D_parameters, lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.lr_scheduler_G = lr_scheduler.ExponentialLR(optimizer=self.optimizer_G, gamma=0.5 ** (1 / 100))
        self.lr_scheduler_D = lr_scheduler.ExponentialLR(optimizer=self.optimizer_D, gamma=0.5 ** (1 / 100))
        self.lr_schedulers = [self.lr_scheduler_G, self.lr_scheduler_D]
        self.loss_names = ['D', 'E_gan', 'E_pred']
        self.lambda_gan = opt.lambda_gan

        self.prob_to_loss = lambda prob: torch.log(prob).mean()

    def forward(self):
        self.e_seq = self.netE(self.x_seq, self.t_seq)
        self.f_seq = self.netF(self.e_seq)
        self.g_seq = torch.argmax(self.f_seq.detach(), dim=2)

    def backward_G(self):
        self.d_mean_seq, self.d_std_seq, self.d_weight_seq = self.netD(self.e_seq)
        self.d_seq = self.d_mean_seq

        t_tmp = flat(self.t_seq)[:, :, None]
        self.d_prob_each = torch.exp(-torch.abs(t_tmp - flat(self.d_mean_seq)) / (flat(self.d_std_seq))) / flat(self.d_std_seq)
        self.d_prob = (self.d_prob_each * flat(self.d_weight_seq)).sum(-1)

        self.loss_E_gan = self.prob_to_loss(self.d_prob)

        self.y_seq_source = self.y_seq[self.domain_mask == 1]
        self.f_seq_source = self.f_seq[self.domain_mask == 1]

        self.loss_E_pred = F.nll_loss(flat(self.f_seq_source), flat(self.y_seq_source))

        self.loss_E = self.loss_E_gan * self.lambda_gan + self.loss_E_pred
        self.loss_E.backward()

    def backward_D(self):
        self.d_mean_seq, self.d_std_seq, self.d_weight_seq = self.netD(self.e_seq.detach())
        self.d_seq = self.d_mean_seq
        t_tmp = flat(self.t_seq)[:, :, None]
        self.d_prob_each = torch.exp(-torch.abs(t_tmp - flat(self.d_mean_seq)) / (flat(self.d_std_seq))) / flat(self.d_std_seq)
        self.d_prob = (self.d_prob_each * flat(self.d_weight_seq)).sum(-1)
        self.loss_D = - self.prob_to_loss(self.d_prob)
        self.loss_D.backward()

    def visualize_D(self):
        t_np = to_np(self.t_seq).flatten()
        color = ['r', 'g', 'b', 'y']

        fig, ax = plt.subplots(1, 1, figsize=(12, 12), dpi=200)
        for i in range(self.opt.nmix):
            d_np = to_np(self.d_seq[:, :, i]).flatten()
            plt.plot(t_np, d_np, '.', color=color[i], alpha=0.5, markersize=10)
            plt.xlabel('true domain')
            plt.ylabel('disc pred domain')
            plt.xlim([-0.1, 1.1])
            plt.ylim([-0.1, 1.1])
            plt.savefig('./figures/disc.png')
        plt.close()


class CUA(BaseModel):
    def __init__(self, opt):
        super(CUA, self).__init__(opt)

        self.opt = opt

        self.netE = FeatureNet(opt)
        self.netF = PredNet(opt)
        self.netD = DiscNet(opt)

        self.lambda_gan = opt.lambda_gan
        self.lambda_rpy = opt.lambda_rpy

    def prepare_trainer(self, init=True):
        print('======>preparing trainer', init)
        opt = self.opt
        if init:
            non_D_parameters = list(self.netE.parameters()) + list(self.netF.parameters())
            self.optimizer_G = torch.optim.Adam(non_D_parameters, lr=opt.lr, betas=(opt.beta1, 0.999),
                                                weight_decay=opt.weight_decay)
            self.lr_scheduler_G = lr_scheduler.ExponentialLR(optimizer=self.optimizer_G,
                                                             gamma=0.5 ** (1 / opt.lr_decay_period))
            self.lr_schedulers = [self.lr_scheduler_G]
            self.set_requires_grad(self.netF, True)
            self.set_requires_grad(self.netE, True)
            self.set_requires_grad(self.netD, False)
            self.loss_names = ['E_pred']

        else:
            # self.init_weight(self.netD)
            self.optimizer_G = torch.optim.Adam(self.netE.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999),
                                                weight_decay=opt.weight_decay)
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999),
                                                weight_decay=opt.weight_decay)
            self.lr_scheduler_G = lr_scheduler.ExponentialLR(optimizer=self.optimizer_G,
                                                             gamma=0.5 ** (1 / opt.lr_decay_period))
            self.lr_scheduler_D = lr_scheduler.ExponentialLR(optimizer=self.optimizer_D,
                                                             gamma=0.5 ** (1 / opt.lr_decay_period))
            self.lr_schedulers = [self.lr_scheduler_G, self.lr_scheduler_D]
            self.set_requires_grad(self.netF, False)
            self.set_requires_grad(self.netE, True)
            self.set_requires_grad(self.netD, True)
            self.loss_names = ['D', 'E_replay', 'E_gan']

    def set_input(self, input, init=True):
        if init:
            self.x_seq, self.y_seq = input
            self.T, self.B = self.x_seq.shape[:2]
            self.t_seq = to_tensor(np.zeros((self.T, self.B, 1), dtype=np.float32)) + self.t_var.reshape(self.T, 1, 1)
            self.z_seq = to_tensor(np.zeros((self.T, self.B), dtype=np.int64)) + self.z_var.reshape(self.T, 1)
        else:
            self.x_seq, self.y_seq, self.x_rpy, self.y_rpy = input

            self.T, self.B = self.x_seq.shape[:2]
            self.t_seq = to_tensor(np.zeros((self.T, self.B, 1), dtype=np.float32)) + self.t_var.reshape(self.T, 1, 1)

            self.t_src = self.t_seq[self.domain_mask == 1]
            self.t_tgt = self.t_seq[self.phase:self.phase + 1]
            self.y_src = self.y_seq[self.domain_mask == 1]
            self.y_tgt = self.y_seq[self.phase:self.phase + 1]

            T, B = self.x_rpy.shape[:2]
            self.t_rpy = to_tensor(np.zeros((T, B, 1), dtype=np.float32)) + self.t_var[:T].reshape(T, 1, 1)

    def forward(self, init=True):
        if init:
            self.e_seq = self.netE(self.x_seq, self.t_seq)
            self.f_seq = self.netF(self.e_seq)
            self.g_seq = torch.argmax(self.f_seq.detach(), dim=2)
        else:
            self.e_seq = self.netE(self.x_seq, self.t_seq)
            self.f_seq = self.netF(self.e_seq)
            self.g_seq = torch.argmax(self.f_seq.detach(), dim=2)

            self.e_rpy = self.netE(self.x_rpy, self.t_rpy)
            self.f_rpy = self.netF(self.e_rpy)
            self.g_rpy = torch.argmax(self.f_rpy.detach(), dim=2)

            self.e_src = self.e_seq[self.domain_mask == 1]
            self.f_src = self.f_seq[self.domain_mask == 1]
            self.g_src = self.g_seq[self.domain_mask == 1]

            self.e_tgt = self.e_seq[self.phase:self.phase + 1]
            self.f_tgt = self.f_seq[self.phase:self.phase + 1]
            self.g_tgt = self.g_seq[self.phase:self.phase + 1]

    def backward_D(self):
        self.d_src = self.netD(self.e_src.detach())
        self.d_tgt = self.netD(self.e_tgt.detach())
        self.loss_D = - torch.log(self.d_src + 1e-10).mean() \
                      - torch.log(1 - self.d_tgt + 1e-10).mean()
        self.loss_D.backward()
        # for visualization
        # self.d = torch.cat([self.d_src.detach(), self.d_tgt.detach()], 0)
        # self.t = torch.cat([self.t_src.detach(), self.t_tgt.detach()], 0)

    def backward_G(self, init=True):
        if init:
            self.y_source = self.y_seq[self.domain_mask == 1]
            self.f_source = self.f_seq[self.domain_mask == 1]
            self.loss_E_pred = F.nll_loss(flat(self.f_source), flat(self.y_source))
            self.loss_E_pred.backward()
        else:
            self.loss_E_replay = F.nll_loss(flat(self.f_rpy), flat(self.y_rpy))
            self.d_tgt = self.netD(self.e_tgt)
            self.loss_E_gan = - torch.log(self.d_tgt + 1e-10).mean()
            self.loss_E = self.loss_E_gan * self.lambda_gan + self.loss_E_replay * self.lambda_rpy
            self.loss_E.backward()

    def optimize_parameters(self, init=True):
        self.forward(init)
        if not init:
            self.set_requires_grad(self.netD, True)
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()

        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G(init)
        self.optimizer_G.step()

    def learn(self, epoch, dataloader, init=True):
        self.epoch = epoch
        self.train()
        loss_curve = {
            loss: []
            for loss in self.loss_names
        }
        acc_curve = []

        if init:
            for data in dataloader:
                x_seq, y_seq = [d[0][None, :, :] for d in data], [d[1][None, :] for d in data]
                x_seq = torch.cat(x_seq, 0).cuda()
                y_seq = torch.cat(y_seq, 0).cuda()

                self.set_input(input=(x_seq, y_seq))
                self.optimize_parameters(init)

                for loss in self.loss_names:
                    loss_curve[loss].append(getattr(self, 'loss_' + loss).item())

                acc_curve.append(self.g_seq.eq(self.y_seq).to(torch.float).mean(-1, keepdim=True))

            loss_msg = '[Train][{}] Loss:'.format(epoch)
            for loss in self.loss_names:
                loss_msg += ' {} {:.3f}'.format(loss, np.mean(loss_curve[loss]))

            acc = to_np(torch.cat(acc_curve, 1).mean(-1))
            acc_msg = '[Train][{}] Acc: {:.2f} {}'.format(epoch, acc.mean(), np.around(acc, decimals=2))
            print(loss_msg)
            print(acc_msg)
        else:
            dataloader, continual_dataloader = dataloader
            for data_1, data_2 in zip(dataloader, continual_dataloader):
                x_seq, y_seq = [d[0][None, :, :] for d in data_1], [d[1][None, :] for d in data_1]
                x_seq = torch.cat(x_seq, 0).cuda()
                y_seq = torch.cat(y_seq, 0).cuda()

                x_rpy, y_rpy = [d[0][None, :, :] for d in data_2], [d[1][None, :] for d in data_2]
                x_rpy = torch.cat(x_rpy, 0).cuda()
                y_rpy = torch.cat(y_rpy, 0).cuda()

                self.set_input([x_seq, y_seq, x_rpy, y_rpy], init)
                self.optimize_parameters(init)
                for loss in self.loss_names:
                    loss_curve[loss].append(getattr(self, 'loss_' + loss).detach().item())

                acc_curve.append(self.g_tgt.eq(self.y_tgt).to(torch.float).mean(-1, keepdim=True))

            loss_msg = '[Train][{}] Loss:'.format(epoch)
            for loss in self.loss_names:
                loss_msg += ' {} {:.3f}'.format(loss, np.mean(loss_curve[loss]))

            acc = to_np(torch.cat(acc_curve, 1).mean(-1))
            acc_msg = '[Train][{}] Acc: {:.2f} {}'.format(epoch, acc.mean(), np.around(acc, decimals=2))
            print(loss_msg)
            print(acc_msg)

        for lr_scheduler in self.lr_schedulers:
            lr_scheduler.step()

    def set_phase(self, phase):
        print('===> adapt phase', phase)
        self.phase = phase

    def gen_replay_dataset(self, dataloader):
        replay_data, replay_label = [], []
        for data in dataloader:
            x_seq, y_seq = [d[0][None, :, :] for d in data], [d[1][None, :] for d in data]
            x_seq = torch.cat(x_seq, 0).cuda()
            y_seq = torch.cat(y_seq, 0).cuda()
            self.set_input(input=(x_seq, y_seq))
            self.forward()
            replay_data.append(to_np(self.x_seq[self.phase]))
            replay_label.append(to_np(self.g_seq[self.phase]))
        replay_data = np.concatenate(replay_data, 0)
        replay_label = np.concatenate(replay_label, 0)
        print('Replay generated: ', replay_data.shape, replay_label.shape)
        return replay_data, replay_label


def get_model(model):
    model_pool = {
        'SO': SO,
        'ADDA': ADDA,
        'MDD': MDD,
        'DANN': DANN,
        'CDANN': DANN,  # DANN with a conditioned discriminator
        'CIDA': CIDA,
        'PCIDA': PCIDA,
        'CUA': CUA,
    }
    return model_pool[model]
