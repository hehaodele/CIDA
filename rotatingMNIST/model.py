import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import time
import os

from progressbar import ProgressBar
import torch.optim.lr_scheduler as lr_scheduler
from prettytable import PrettyTable

from PIL import Image
from matplotlib.cm import get_cmap
from tensorboardX import SummaryWriter as Logger


# ======================================================================================================================
def download():
    from torchvision.datasets import MNIST
    root = './data'
    processed_folder = os.path.join(root, 'MNIST', 'processed')
    if not os.path.isdir(processed_folder):
        dataset = MNIST(root=root, download=True)


def print_args(opt):
    for k, v in opt.items():
        print(f'{k}: {v}')


def set_default_args(opt):
    # dropout
    if opt in ['CIDA', 'PCIDA']:
        opt.dropout = 0.2
    else:
        opt.dropout = 0

    # model input/hidden/output size
    opt.nh = 512  # size of hidden neurons
    opt.nc = 10  # number of class
    opt.nz = 100  # size of features

    # training parameteres
    opt.num_epoch = 100
    opt.batch_size = 100
    opt.lr = 2e-4
    opt.weight_decay = 5e-4
    opt.beta1 = 0.9

    # loss parameters
    opt.lambda_gan = 2.0

    # experiment folder
    opt.exp = 'RotatingMNIST_' + opt.model
    opt.outf = './dump/' + opt.exp
    os.system('mkdir -p ' + opt.outf)
    print('Traing result will be saved in ', opt.outf)

    # dataset info
    opt.dim_domain = 1

    # parameters for CUA
    opt.continual_da = (opt.model == 'CUA')
    if opt.model == 'CUA':
        opt.num_da_step = 7
        opt.num_epoch_pre = 10
        opt.num_epoch_sub = 10
        opt.lr_decay_period = 50
        opt.lambda_rpy = 1.0


def to_np(x):
    return x.detach().cpu().numpy()


def to_tensor(x, device='cuda'):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).to(device)
    else:
        x = x.to(device)
    return x


class RotateMNIST(Dataset):
    def __init__(self, rotate_angle):
        root = './data'
        processed_folder = os.path.join(root, 'MNIST', 'processed')
        data_file = 'training.pt'
        self.data, self.targets = torch.load(os.path.join(processed_folder, data_file))
        self.rotate_angle = rotate_angle

    def __getitem__(self, index):
        from torchvision.transforms.functional import rotate
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        img = Image.fromarray(img.numpy(), mode='L')

        rot_min, rot_max = self.rotate_angle
        angle = np.random.rand() * (rot_max - rot_min) + rot_min

        img = rotate(img, angle)
        img = transforms.ToTensor()(img).to(torch.float)

        return img, \
               target, \
               np.array([angle / 360.0], dtype=np.float32), \
               np.array([angle / 360.0], dtype=np.float32)

    def __len__(self):
        return len(self.data)


# ======================================================================================================================
def init_weight_STN(stn):
    """ Initialize the weights/bias with (nearly) identity transformation
    reference: https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html
    """
    stn[-5].weight[:, -28 * 28:].data.zero_()
    stn[-5].bias.data.zero_()
    stn[-1].weight.data.zero_()
    stn[-1].bias.data.copy_(torch.tensor([1 - 1e-2, 1e-2, 1 - 1e-2], dtype=torch.float))


def convert_Avec_to_A(A_vec):
    """ Convert BxM tensor to BxNxN symmetric matrices """
    """ M = N*(N+1)/2"""
    if A_vec.dim() < 2:
        A_vec = A_vec.unsqueeze(dim=0)

    if A_vec.shape[1] == 10:
        A_dim = 4
    elif A_vec.shape[1] == 3:
        A_dim = 2
    else:
        raise ValueError("Arbitrary A_vec not yet implemented")

    idx = torch.triu_indices(A_dim, A_dim)
    A = A_vec.new_zeros((A_vec.shape[0], A_dim, A_dim))
    A[:, idx[0], idx[1]] = A_vec
    A[:, idx[1], idx[0]] = A_vec
    return A.squeeze()


class nnSqueeze(nn.Module):
    def __init__(self):
        super(nnSqueeze, self).__init__()

    def forward(self, x):
        return torch.squeeze(x)


class nnUnsqueeze(nn.Module):
    def __init__(self):
        super(nnUnsqueeze, self).__init__()

    def forward(self, x):
        return x[:, :, None, None]


class EncoderSTN(nn.Module):
    def __init__(self, opt):
        super(EncoderSTN, self).__init__()

        nh = 256

        self.fc_stn = nn.Sequential(
            nn.Linear(opt.dim_domain + 28 * 28, nh), nn.LeakyReLU(0.2), nn.Dropout(opt.dropout),
            nn.Linear(nh, nh), nn.BatchNorm1d(nh), nn.LeakyReLU(0.2), nn.Dropout(opt.dropout),
            nn.Linear(nh, 3),
        )

        nz = opt.nz

        self.conv = nn.Sequential(
            nn.Conv2d(1, nh, 3, 2, 1), nn.BatchNorm2d(nh), nn.ReLU(True), nn.Dropout(opt.dropout),  # 14 x 14
            nn.Conv2d(nh, nh, 3, 2, 1), nn.BatchNorm2d(nh), nn.ReLU(True), nn.Dropout(opt.dropout),  # 7 x 7
            nn.Conv2d(nh, nh, 3, 2, 1), nn.BatchNorm2d(nh), nn.ReLU(True), nn.Dropout(opt.dropout),  # 4 x 4
            nn.Conv2d(nh, nz, 4, 1, 0), nn.ReLU(True),  # 1 x 1
        )

        self.fc_pred = nn.Sequential(
            nn.Conv2d(nz, nh, 1, 1, 0), nn.BatchNorm2d(nh), nn.ReLU(True),
            nn.Conv2d(nh, nh, 1, 1, 0), nn.BatchNorm2d(nh), nn.ReLU(True),
            nnSqueeze(),
            nn.Linear(nh, 10),
        )

    def stn(self, x, u):
        # A_vec = self.fc_stn(u)
        A_vec = self.fc_stn(torch.cat([u, x.reshape(-1, 28 * 28)], 1))
        A = convert_Avec_to_A(A_vec)
        _, evs = torch.symeig(A, eigenvectors=True)
        tcos, tsin = evs[:, 0:1, 0:1], evs[:, 1:2, 0:1]

        self.theta_angle = torch.atan2(tsin[:, 0, 0], tcos[:, 0, 0])

        # clock-wise rotate theta
        theta_0 = torch.cat([tcos, tsin, tcos * 0], 2)
        theta_1 = torch.cat([-tsin, tcos, tcos * 0], 2)
        theta = torch.cat([theta_0, theta_1], 1)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

    def forward(self, x, u):
        """
        :param x: B x 1 x 28 x 28
        :param u: B x nu
        :return:
        """
        x = self.stn(x, u)
        z = self.conv(x)
        y = self.fc_pred(z)
        return F.log_softmax(y, dim=1), x, z


# ======================================================================================================================
class BaseModel(nn.Module):
    def __init__(self, opt):
        super(BaseModel, self).__init__()
        self.opt = opt
        self.tsne = TSNE(n_components=2)
        self.pca = PCA(n_components=2)

        self.train_log = opt.outf + '/train.log'
        self.model_path = opt.outf + '/model.pth'
        self.logger = Logger(opt.outf)
        self.best_acc_tgt = 0

    def init_weight(self, net=None):
        if net is None:
            net = self
        for m in net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, val=0)

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def save(self):
        torch.save(self.state_dict(), self.model_path)

    def load(self):
        try:
            print('load model from {}'.format(self.model_path))
            self.load_state_dict(torch.load(self.model_path))
            print('done!')
        except:
            print('failed!')

    def acc_reset_mnist(self):
        self.hit_domain, self.cnt_domain = np.zeros(8), np.zeros(8)
        self.acc_source, self.acc_target = 0, 0
        self.cnt_source, self.cnt_target = 0, 0
        self.hit_source, self.hit_target = 0, 0

    def acc_update_mnist(self):
        Y = to_np(self.y)
        G = to_np(self.g)
        T = to_np(self.domain)
        T = (T * 8).astype(np.int32)
        T[T >= 8] = 7
        hit = (Y == G).astype(np.float32)

        is_s = to_np(self.is_source)

        for i in range(8):
            self.hit_domain[i] += hit[T == i].sum()
            self.cnt_domain[i] += (T == i).sum()
        self.acc_domain = self.hit_domain / (self.cnt_domain + 1e-10)
        self.acc_source, self.acc_target = self.acc_domain[0], self.acc_domain[1:].mean()
        self.acc_domain = np.round(self.acc_domain, decimals=3)
        self.acc_source = np.round(self.acc_source, decimals=3)
        self.acc_target = np.round(self.acc_target, decimals=3)

        self.cnt_source += (is_s == 1).sum()
        self.cnt_target += (is_s == 0).sum()

        self.hit_source += (hit[is_s == 1]).sum()
        self.hit_target += (hit[is_s == 0]).sum()

    def set_input(self, input):
        self.x, self.y, self.u, self.domain = input
        self.domain = self.domain[:, 0]
        self.is_source = (self.domain < 1.0 / 8).to(torch.float)

    def print_log(self):
        print(self.loss_msg)
        print(self.acc_msg)
        with open(self.train_log, 'a') as f:
            f.write(self.loss_msg + '\n')
            f.write(self.acc_msg + '\n')

    def learn(self, epoch, dataloader):
        self.epoch = epoch
        self.train()

        loss_curve = {
            loss: []
            for loss in self.loss_names
        }
        self.acc_reset_mnist()

        bar = ProgressBar()

        for data in bar(dataloader):
            x, y, t, is_source = [to_tensor(_, self.opt.device) for _ in data]
            self.set_input(input=(x, y, t, is_source))
            self.optimize_parameters()

            for loss in self.loss_names:
                loss_curve[loss].append(getattr(self, 'loss_' + loss).detach().item())

            self.acc_update_mnist()

        self.loss_msg = '[Train][{}] Loss:'.format(epoch)
        for loss in self.loss_names:
            self.loss_msg += ' {} {:.3f}'.format(loss, np.mean(loss_curve[loss]))
        self.acc_msg = '[Train][{}] Acc: source {:.3f} ({}/{}) target {:.3f} ({}/{})'.format(
            epoch, self.acc_source, self.hit_source, self.cnt_source,
            self.acc_target, self.hit_target, self.cnt_target)
        self.print_log()

        for lr_scheduler in self.lr_schedulers:
            lr_scheduler.step()

        self.logger.add_scalar('AccSrc', self.acc_source, epoch)
        self.logger.add_scalar('AccTgt', self.acc_target, epoch)

    def test(self, epoch, dataloader, flag_save=True):
        self.eval()
        self.acc_reset()
        for data in dataloader:
            x, y, t, is_source = [to_tensor(_, self.opt.device) for _ in data]
            self.set_input(input=(x, y, t, is_source))
            self.forward()
            self.acc_update()

        self.best_acc_tgt = max(self.best_acc_tgt, self.acc_target)
        if self.best_acc_tgt == self.acc_target and flag_save:
            self.save()

        self.acc_msg = '[Test][{}] Acc: source {:.3f} target {:.3f} best {:.3f}'.format(
            epoch, self.acc_source, self.acc_target, self.best_acc_tgt)
        self.loss_msg = ''
        self.print_log()

    def eval_mnist(self, dataloader):
        self.eval()
        self.acc_reset_mnist()
        for data in dataloader:
            x, y, t, is_source = [to_tensor(_, self.opt.device) for _ in data]
            self.set_input(input=(x, y, t, is_source))
            self.forward()
            self.acc_update_mnist()
        self.loss_msg = ''
        self.acc_msg = f'Eval MNIST: {self.acc_domain} src: {self.acc_source} tgt:{self.acc_target}'
        self.print_log()
        return self.acc_target

    def gen_result_table(self, dataloader):

        res = PrettyTable()

        res.field_names = ["Accuracy"] + ["Source"] + [f"Target #{i}" for i in range(1, 8)]

        hit = np.zeros((10, 8))
        cnt = np.zeros((10, 8))

        for data in dataloader:
            x, y, t, is_source = [to_tensor(_, self.opt.device) for _ in data]
            self.set_input(input=(x, y, t, is_source))
            self.forward()

            Y = to_np(self.y)
            G = to_np(self.g)
            T = to_np(self.u)[:, 0]
            T = (T * 8).astype(np.int32)
            T[T >= 8] = 7

            for label, pred, domain in zip(Y, G, T):
                hit[label, domain] += int(label == pred)
                cnt[label, domain] += 1

        for c in range(10):
            res.add_row([f"Class {c}"] + list(np.round(100 * hit[c] / cnt[c], decimals=1)))

        res.add_row([f"Total"] + list(np.round(100 * hit.sum(0) / cnt.sum(0), decimals=1)))
        print(res)


class SO(BaseModel):
    def __init__(self, opt):
        super(SO, self).__init__(opt)
        self.netE = EncoderSTN(opt)
        init_weight_STN(self.netE.fc_stn)
        self.optimizer_G = torch.optim.Adam(self.netE.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
        self.lr_scheduler_G = lr_scheduler.ExponentialLR(optimizer=self.optimizer_G, gamma=0.5 ** (1 / 100))
        self.lr_schedulers = [self.lr_scheduler_G]
        self.loss_names = ['E_pred']

    def forward(self):
        self.f, self.x_align, self.e = self.netE(self.x, self.u)
        self.g = torch.argmax(self.f.detach(), dim=1)

    def backward_G(self):
        self.loss_E_pred = F.nll_loss(self.f[self.is_source == 1], self.y[self.is_source == 1])
        self.loss_E = self.loss_E_pred
        self.loss_E.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()


class DiscConv(nn.Module):
    def __init__(self, nin, nout):
        super(DiscConv, self).__init__()
        nh = 512
        self.net = nn.Sequential(
            nn.Conv2d(nin, nh, 1, 1, 0), nn.BatchNorm2d(nh), nn.LeakyReLU(),
            nn.Conv2d(nh, nh, 1, 1, 0), nn.BatchNorm2d(nh), nn.LeakyReLU(),
            nn.Conv2d(nh, nh, 1, 1, 0), nn.BatchNorm2d(nh), nn.LeakyReLU(),
            nnSqueeze(),
            nn.Linear(nh, nout),
        )

    def forward(self, x):
        return self.net(x)


class CIDA(BaseModel):
    def __init__(self, opt):
        super(CIDA, self).__init__(opt)

        self.opt = opt
        self.netE = EncoderSTN(opt)
        self.init_weight(self.netE)
        self.netD = DiscConv(nin=opt.nz, nout=opt.dim_domain)
        self.optimizer_G = torch.optim.Adam(self.netE.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
        self.lr_scheduler_G = lr_scheduler.ExponentialLR(optimizer=self.optimizer_G, gamma=0.5 ** (1 / 50))
        self.lr_scheduler_D = lr_scheduler.ExponentialLR(optimizer=self.optimizer_D, gamma=0.5 ** (1 / 50))
        self.lr_schedulers = [self.lr_scheduler_G, self.lr_scheduler_D]

        self.lambda_gan = opt.lambda_gan
        self.loss_names = ['D', 'E_gan', 'E_pred']

    def set_input(self, input):
        self.x, self.y, self.u, self.domain = input
        self.domain = self.domain[:, 0]
        self.is_source = (self.domain < 1.0 / 8).to(torch.float)

    def forward(self):
        self.f, self.x_align, self.e = self.netE(self.x, self.u)
        self.g = torch.argmax(self.f.detach(), dim=1)

    def backward_G(self):
        self.d = self.netD(self.e)

        E_gan_src = F.mse_loss(self.d[self.is_source == 1], self.u[self.is_source == 1])
        E_gan_tgt = F.mse_loss(self.d[self.is_source == 0], self.u[self.is_source == 0])
        self.loss_E_gan = - (E_gan_src + E_gan_tgt) / 2

        self.y_source = self.y[self.is_source == 1]
        self.f_source = self.f[self.is_source == 1]
        self.loss_E_pred = F.nll_loss(self.f_source, self.y_source)

        self.loss_E = self.loss_E_gan * self.lambda_gan + self.loss_E_pred
        self.loss_E.backward()

    def backward_D(self):
        self.d = self.netD(self.e.detach())
        D_src = F.mse_loss(self.d[self.is_source == 1], self.u[self.is_source == 1])
        D_tgt = F.mse_loss(self.d[self.is_source == 0], self.u[self.is_source == 0])
        self.loss_D = (D_src + D_tgt) / 2
        self.loss_D.backward()

    def optimize_parameters(self):
        self.forward()
        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()


def neg_guassian_likelihood(d, u):
    """return: -N(u; mu, var)"""
    B, dim = u.shape
    assert (d.shape[1] == dim * 2)
    mu, logvar = d[:, :dim], d[:, dim:]
    return 0.5 * (((u - mu) ** 2) / torch.exp(logvar) + logvar).mean()


class PCIDA(CIDA):
    def __init__(self, opt):
        super(PCIDA, self).__init__(opt)
        self.netD = DiscConv(nin=opt.nz, nout=opt.dim_domain * 2)  # mean, logvar
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
        self.lr_scheduler_D = lr_scheduler.ExponentialLR(optimizer=self.optimizer_D, gamma=0.5 ** (1 / 50))
        self.lr_schedulers = [self.lr_scheduler_G, self.lr_scheduler_D]

    def backward_G(self):
        self.d = self.netD(self.e)

        E_gan_src = neg_guassian_likelihood(self.d[self.is_source == 1], self.u[self.is_source == 1])
        E_gan_tgt = neg_guassian_likelihood(self.d[self.is_source == 0], self.u[self.is_source == 0])
        self.loss_E_gan = - (E_gan_src + E_gan_tgt) / 2

        self.y_source = self.y[self.is_source == 1]
        self.f_source = self.f[self.is_source == 1]
        self.loss_E_pred = F.nll_loss(self.f_source, self.y_source)

        self.loss_E = self.loss_E_gan * self.lambda_gan + self.loss_E_pred
        self.loss_E.backward()

    def backward_D(self):
        self.d = self.netD(self.e.detach())
        D_src = neg_guassian_likelihood(self.d[self.is_source == 1], self.u[self.is_source == 1])
        D_tgt = neg_guassian_likelihood(self.d[self.is_source == 0], self.u[self.is_source == 0])
        self.loss_D = (D_src + D_tgt) / 2
        self.loss_D.backward()


class ADDA(BaseModel):
    def __init__(self, opt):
        super(ADDA, self).__init__(opt)
        self.opt = opt

        self.netE = EncoderSTN(opt)
        self.netD = DiscConv(nin=opt.nz, nout=1)
        init_weight_STN(self.netE.fc_stn)

        self.optimizer_G = torch.optim.Adam(self.netE.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
        self.lr_scheduler_G = lr_scheduler.ExponentialLR(optimizer=self.optimizer_G, gamma=0.5 ** (1 / 100))
        self.lr_scheduler_D = lr_scheduler.ExponentialLR(optimizer=self.optimizer_D, gamma=0.5 ** (1 / 100))
        self.lr_schedulers = [self.lr_scheduler_G, self.lr_scheduler_D]
        self.loss_names = ['D', 'E_gan', 'E_pred']
        self.lambda_gan = opt.lambda_gan

    def forward(self):
        self.f, self.x_align, self.e = self.netE(self.x, self.u)
        self.g = torch.argmax(self.f.detach(), dim=1)

    def backward_G(self):
        self.d = torch.sigmoid(self.netD(self.e))
        self.d_t = self.d[self.is_source == 0]
        self.loss_E_gan = - torch.log(self.d_t + 1e-10).mean()  # make target domain looks like source
        self.loss_E_pred = F.nll_loss(self.f[self.is_source == 1], self.y[self.is_source == 1])
        self.loss_E = self.loss_E_gan * self.lambda_gan + self.loss_E_pred
        self.loss_E.backward()

    def backward_D(self):
        self.d = torch.sigmoid(self.netD(self.e.detach()))
        self.d_s = self.d[self.is_source == 1]
        self.d_t = self.d[self.is_source == 0]
        self.loss_D = - torch.log(self.d_s + 1e-10).mean() \
                      - torch.log(1 - self.d_t + 1e-10).mean()
        self.loss_D.backward()

    def optimize_parameters(self):
        self.forward()
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()


class DANN(BaseModel):
    def __init__(self, opt):
        super(DANN, self).__init__(opt)

        self.opt = opt

        self.netE = EncoderSTN(opt)
        self.netD = DiscConv(nin=opt.nz, nout=8)
        init_weight_STN(self.netE.fc_stn)

        self.optimizer_G = torch.optim.Adam(self.netE.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)

        self.lr_scheduler_G = lr_scheduler.ExponentialLR(optimizer=self.optimizer_G, gamma=0.5 ** (1 / 100))
        self.lr_scheduler_D = lr_scheduler.ExponentialLR(optimizer=self.optimizer_D, gamma=0.5 ** (1 / 100))
        self.lr_schedulers = [self.lr_scheduler_G, self.lr_scheduler_D]

        self.loss_names = ['D', 'E_gan', 'E_pred']
        self.lambda_gan = opt.lambda_gan

    def set_input(self, input):
        self.x, self.y, self.u, self.domain = input
        self.domain = self.domain[:, 0]
        self.is_source = (self.domain < 1.0 / 8).to(torch.float)
        self.t_class = (self.domain * 8).floor().to(torch.long)

    def forward(self):
        self.f, self.x_align, self.e = self.netE(self.x, self.u)
        self.g = torch.argmax(self.f.detach(), dim=1)

    def backward_G(self):
        self.d = torch.log_softmax(self.netD(self.e), dim=1)
        self.d_t = self.d[self.is_source == 0]
        self.loss_E_gan = - F.nll_loss(self.d, self.t_class)
        self.loss_E_pred = F.nll_loss(self.f[self.is_source == 1], self.y[self.is_source == 1])
        self.loss_E = self.loss_E_gan * self.lambda_gan + self.loss_E_pred
        self.loss_E.backward()

    def backward_D(self):
        self.d = torch.log_softmax(self.netD(self.e.detach()), dim=1)
        self.loss_D = F.nll_loss(self.d, self.t_class)
        self.loss_D.backward()

    def optimize_parameters(self):
        self.forward()
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()


class CUA(BaseModel):
    def __init__(self, opt):
        super(CUA, self).__init__(opt)
        self.opt = opt

        self.netE = EncoderSTN(opt)
        self.netD = DiscConv(nin=opt.nz, nout=1)
        init_weight_STN(self.netE.fc_stn)

        self.lambda_gan = opt.lambda_gan
        self.lambda_rpy = opt.lambda_rpy

    def prepare_trainer(self, init=True):
        print('======>preparing trainer', init)
        if init:
            non_D_parameters = list(self.netE.parameters())
            self.optimizer_G = torch.optim.Adam(non_D_parameters, lr=opt.lr, betas=(opt.beta1, 0.999),
                                                weight_decay=opt.weight_decay)
            self.lr_scheduler_G = lr_scheduler.ExponentialLR(optimizer=self.optimizer_G,
                                                             gamma=0.5 ** (1 / 50))
            self.lr_schedulers = [self.lr_scheduler_G]
            self.set_requires_grad(self.netE, True)
            self.set_requires_grad(self.netD, False)
            self.loss_names = ['E_pred']

        else:
            self.init_weight(self.netD)  # re-initialie the discriminator for the new adaptation task
            self.optimizer_G = torch.optim.Adam(self.netE.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999),
                                                weight_decay=opt.weight_decay)
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999),
                                                weight_decay=opt.weight_decay)
            self.lr_scheduler_G = lr_scheduler.ExponentialLR(optimizer=self.optimizer_G,
                                                             gamma=0.5 ** (1 / 50))
            self.lr_scheduler_D = lr_scheduler.ExponentialLR(optimizer=self.optimizer_D,
                                                             gamma=0.5 ** (1 / 50))
            self.lr_schedulers = [self.lr_scheduler_G, self.lr_scheduler_D]
            self.set_requires_grad(self.netE, True)
            self.set_requires_grad(self.netD, True)
            self.loss_names = ['D', 'E_replay', 'E_gan']

    def set_input(self, input, init=True):
        if init:  # pretraining phase
            self.x, self.y, self.u, self.domain = input
            self.domain = self.domain[:, 0]
            self.is_source = (self.domain < 1.0 / 8).to(torch.float)

        else:  # adaptation phase
            self.x_src, self.y_src, self.u_src, self.x_rpy, self.y_rpy, self.u_rpy, self.x_tgt, self.y_tgt, self.u_tgt = input
            self.y = torch.cat([self.y_src, self.y_tgt], 0)  # for computing metric (replay should be excluded)
            self.domain = torch.cat([self.u_src, self.u_tgt], 0)[:, 0]
            self.is_source = (self.domain < 1.0 / 8).to(torch.float)

    def forward(self, init=True):
        if init:
            self.f, self.x_align, self.e = self.netE(self.x, self.u)
            self.g = torch.argmax(self.f.detach(), dim=1)
        else:
            self.f_src, self.x2_src, self.e_src = self.netE(self.x_src, self.u_src)
            self.f_rpy, self.x2_rpy, self.e_rpy = self.netE(self.x_rpy, self.u_rpy)
            self.f_tgt, self.x2_tgt, self.e_tgt = self.netE(self.x_tgt, self.u_tgt)
            self.g_src = torch.argmax(self.f_src.detach(), dim=1)
            self.g_rpy = torch.argmax(self.f_rpy.detach(), dim=1)
            self.g_tgt = torch.argmax(self.f_tgt.detach(), dim=1)
            self.g = torch.cat([self.g_src, self.g_tgt], 0)  # no replay

    def backward_D(self):
        self.d_src = torch.sigmoid(self.netD(self.e_src.detach()))
        self.d_tgt = torch.sigmoid(self.netD(self.e_tgt.detach()))
        self.loss_D = - torch.log(self.d_src + 1e-10).mean() \
                      - torch.log(1 - self.d_tgt + 1e-10).mean()
        self.loss_D.backward()

    def backward_G(self, init=True):
        if init:
            self.y_source = self.y[self.is_source == 1]
            self.f_source = self.f[self.is_source == 1]
            self.loss_E_pred = F.nll_loss(self.f_source, self.y_source)
            self.loss_E_pred.backward()
        else:
            self.loss_E_replay = F.nll_loss(self.f_rpy, self.y_rpy)
            self.d_tgt = torch.sigmoid(self.netD(self.e_tgt))
            self.loss_E_gan = - torch.log(self.d_tgt + 1e-10).mean()
            self.loss_E = self.loss_E_gan * self.lambda_gan + self.loss_E_replay * self.lambda_rpy
            self.loss_E.backward()

    def optimize_parameters(self, init):
        self.forward(init)
        if not init:
            # optimize discriminator only in adaptation phase
            self.set_requires_grad(self.netD, True)
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()

        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G(init)
        self.optimizer_G.step()

    def learn(self, epoch, dataloader, init=True):
        self.train()
        loss_curve = {
            loss: []
            for loss in self.loss_names
        }
        self.acc_reset_mnist()
        bar = ProgressBar()
        for data in bar(dataloader):
            data_var = [to_tensor(_, self.opt.device) for _ in data]
            self.set_input(data_var, init)
            self.optimize_parameters(init)
            for loss in self.loss_names:
                loss_curve[loss].append(getattr(self, 'loss_' + loss).detach().item())
            self.acc_update_mnist()

        self.loss_msg = '[Train][{}] Loss:'.format(epoch)
        for loss in self.loss_names:
            self.loss_msg += ' {} {:.3f}'.format(loss, np.mean(loss_curve[loss]))
        self.acc_msg = '[Train][{}] Acc: source {:.3f} ({}/{}) target {:.3f} ({}/{})'.format(
            epoch, self.acc_source, self.hit_source, self.cnt_source,
            self.acc_target, self.hit_target, self.cnt_target)
        self.print_log()

        for lr_scheduler in self.lr_schedulers:
            lr_scheduler.step()

    def gen_data_tuple(self, dataloader):
        print('===> generate new replay dataset!')
        self.eval()
        all_x, all_g, all_t = [], [], []
        for data in dataloader:
            data_var = [to_tensor(_, self.opt.device) for _ in data]
            self.set_input(data_var, init=True)
            self.forward(init=True)
            all_x.append(to_np(self.x_tgt))
            all_g.append(to_np(self.g_tgt))
            all_t.append(to_np(self.u_tgt))
        x, g, t = np.concatenate(all_x, 0), np.concatenate(all_g, 0), np.concatenate(all_t, 0)
        print('generated: ', x.shape, g.shape, t.shape)
        return x, g, t


class SimpleDataset(Dataset):
    def __init__(self, x, y, u):
        self.x = x.astype(np.float32)
        self.y = y.astype(np.int64)
        self.u = u.astype(np.float32)

    def __getitem__(self, i):
        return self.x[i], self.y[i], self.u[i]

    def __len__(self):
        return len(self.x)

    def update(self, data):
        x, y, u = data
        self.x = np.concatenate([self.x, x], 0)
        self.y = np.concatenate([self.y, y], 0)
        self.u = np.concatenate([self.u, u], 0)
        return self


class ContinousRotateMNIST(Dataset):
    def __init__(self):
        self.ds_source = RotateMNIST(rotate_angle=(0, 45))
        self.ds_target = [RotateMNIST(rotate_angle=(45 * i, 45 * i + 45)) for i in range(1, 8)]

        tmp = {'x': [], 'y': [], 'u': []}
        for data in self.ds_source:
            tmp['x'] += [to_np(data[0])]
            tmp['y'] += [data[1]]
            tmp['u'] += [data[2]]
        tmp['x'] = np.array(tmp['x'])
        tmp['y'] = np.array(tmp['y'])
        tmp['u'] = np.array(tmp['u'])
        print(tmp['x'].shape, tmp['y'].shape, tmp['u'].shape)

        self.ds_replay = SimpleDataset(np.array(tmp['x']), np.array(tmp['y']), np.array(tmp['u']))
        self.phase = 0

    def __len__(self):
        return len(self.ds_source)

    def set_phase(self, p):
        self.phase = p

    @staticmethod
    def rand_sample(ds):
        n = len(ds)
        i = np.random.randint(n)
        return ds[i]

    def __getitem__(self, i):
        x_src, y_src, u_src, _ = self.ds_source[i]
        x_tgt, y_tgt, u_tgt, _ = self.ds_target[self.phase][i]
        x_rpy, y_rpy, u_rpy = self.rand_sample(self.ds_replay)
        return x_src, y_src, u_src, x_rpy, y_rpy, u_rpy, x_tgt, y_tgt, u_tgt


# ======================================================================================================================


if __name__ == '__main__':
    from easydict import EasyDict
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='SO', choices=['SO', 'ADDA', 'CIDA', 'DANN', 'PCIDA', 'CUA'])
    parser.add_argument('--lambda_gan', default=2.0, type=float)
    parser.add_argument('--num_epoch', default=100, type=int)
    parser.add_argument('--nz', default=100, type=int)
    parser.add_argument('--device', default='cuda', type=str)

    args = parser.parse_args()

    opt = EasyDict()
    # choose the method from ["CIDA", "PCIDA", "ADDA", "SO", "DANN", "CDANN", "MDD", "CUA"]
    opt.model = args.model
    # choose run on which device ["cuda", "cpu"]
    opt.device = args.device
    # set default args
    set_default_args(opt)
    # set args from command
    opt.lambda_gan = args.lambda_gan
    opt.num_epoch = args.num_epoch
    opt.nz = args.nz
    print_args(opt)

    download()
    dataset = RotateMNIST(rotate_angle=(0, 360))
    train_dataloader = DataLoader(
        dataset=dataset,
        shuffle=True,
        batch_size=opt.batch_size,
        num_workers=4,
    )
    test_dataloader = DataLoader(
        dataset=dataset,
        shuffle=True,  # for batch normalization
        batch_size=opt.batch_size,
        num_workers=4,
    )
    model_pool = {
        'SO': SO,
        'CIDA': CIDA,
        'PCIDA': PCIDA,
        'ADDA': ADDA,
        'DANN': DANN,
        'CUA': CUA,
    }
    model = model_pool[opt.model](opt)
    model = model.to(opt.device)

    best_acc_target = 0
    if not opt.continual_da:
        # Single Step Domain Adaptation
        for epoch in range(opt.num_epoch):
            model.learn(epoch, train_dataloader)
            if (epoch + 1) % 10 == 0:
                acc_target = model.eval_mnist(test_dataloader)
                if acc_target > best_acc_target:
                    print('Best acc target. saved.')
                    model.save()
    else:
        # continual DA training

        continual_dataset = ContinousRotateMNIST()

        print('===> pretrain the classifer')
        model.prepare_trainer(init=True)
        for epoch in range(opt.num_epoch_pre):
            model.learn(epoch, train_dataloader, init=True)
            if (epoch + 1) % 10 == 0:
                model.eval_mnist(test_dataloader)
        print('===> start continual DA')
        model.prepare_trainer(init=False)
        for phase in range(opt.num_da_step):
            continual_dataset.set_phase(phase)
            print(f'Phase {phase}/{opt.num_da_step}')
            print(f'#source {len(continual_dataset.ds_source)} #target {len(continual_dataset.ds_target[phase])} #replay {len(continual_dataset.ds_replay)}')
            continual_dataloader = DataLoader(
                dataset=continual_dataset,
                shuffle=True,
                batch_size=opt.batch_size,
                num_workers=4,
            )
            for epoch in range(opt.num_epoch_sub):
                model.learn(epoch, continual_dataloader, init=False)
                if (epoch + 1) % 10 == 0:
                    model.eval_mnist(test_dataloader)

            target_dataloader = DataLoader(
                dataset=continual_dataset.ds_target[phase],
                shuffle=True,
                batch_size=opt.batch_size,
                num_workers=4,
            )
            model.eval_mnist(test_dataloader)
            data_tuple = model.gen_data_tuple(target_dataloader)
            continual_dataset.ds_replay.update(data_tuple)  # add target data and prediction into replay buffer
