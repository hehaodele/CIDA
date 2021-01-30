import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class FeatureNet(nn.Module):
    def __init__(self, opt):
        super(FeatureNet, self).__init__()

        nx, nh = opt.nx, opt.nh

        self.fc1 = nn.Linear(nx, nh)
        self.fc2 = nn.Linear(nh * 2, nh * 2)
        self.fc3 = nn.Linear(nh * 2, nh * 2)
        self.fc4 = nn.Linear(nh * 2, nh * 2)
        self.fc_final = nn.Linear(nh * 2, nh)

        self.fc1_var = nn.Linear(1, nh)
        self.fc2_var = nn.Linear(nh, nh)

    def forward(self, x, t):
        re = x.dim() == 3
        if re:
            T, B, C = x.shape
            x = x.reshape(T * B, -1)
            t = t.reshape(T * B, -1)

        x = F.relu(self.fc1(x))
        t = F.relu(self.fc1_var(t))
        t = F.relu(self.fc2_var(t))

        # combine feature in the middle
        x = torch.cat((x, t), dim=1)

        # main
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc_final(x)

        if re:
            return x.reshape(T, B, -1)
        else:
            return x


class DiscNet(nn.Module):
    """
    Discriminator doing binary classification: source v.s. target
    """

    def __init__(self, opt):
        super(DiscNet, self).__init__()
        nh = opt.nh

        nin = nh
        self.fc3 = nn.Linear(nin, nh)
        self.bn3 = nn.BatchNorm1d(nh)

        self.fc4 = nn.Linear(nh, nh)
        self.bn4 = nn.BatchNorm1d(nh)

        self.fc5 = nn.Linear(nh, nh)
        self.bn5 = nn.BatchNorm1d(nh)

        self.fc6 = nn.Linear(nh, nh)
        self.bn6 = nn.BatchNorm1d(nh)

        self.fc7 = nn.Linear(nh, nh)
        self.bn7 = nn.BatchNorm1d(nh)

        if opt.no_bn:
            self.bn3 = Identity()
            self.bn4 = Identity()
            self.bn5 = Identity()
            self.bn6 = Identity()
            self.bn7 = Identity()

        self.fc_final = nn.Linear(nh, 1)
        if opt.model in ['ADDA', 'CUA']:
            print('===> Discrinimator Output Activation: sigmoid')
            self.output = lambda x: torch.sigmoid(x)
        else:
            print('===> Discrinimator Output Activation: identity')
            self.output = lambda x: x

    def forward(self, x):
        re = x.dim() == 3

        if re:
            T, B, C = x.shape
            x = x.reshape(T * B, -1)

        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        x = F.relu(self.bn5(self.fc5(x)))
        x = F.relu(self.bn6(self.fc6(x)))
        x = F.relu(self.bn7(self.fc7(x)))
        x = self.output(self.fc_final(x))

        if re:
            return x.reshape(T, B, -1)
        else:
            return x


class ClassDiscNet(nn.Module):
    """
    Discriminator doing multi-class classification on the domain
    """

    def __init__(self, opt):
        super(ClassDiscNet, self).__init__()
        nh = opt.nh
        nc = opt.nc
        nin = nh
        nout = opt.num_domain

        if opt.cond_disc:
            print('===> Conditioned Discriminator')
            nmid = nh * 2
            self.cond = nn.Sequential(
                nn.Linear(nc, nh),
                nn.ReLU(True),
                nn.Linear(nh, nh),
                nn.ReLU(True),
            )
        else:
            print('===> Unconditioned Discriminator')
            nmid = nh
            self.cond = None

        print(f'===> Discriminator will distinguish {nout} domains')

        self.fc3 = nn.Linear(nin, nh)
        self.bn3 = nn.BatchNorm1d(nh)

        self.fc4 = nn.Linear(nmid, nh)
        self.bn4 = nn.BatchNorm1d(nh)

        self.fc5 = nn.Linear(nh, nh)
        self.bn5 = nn.BatchNorm1d(nh)

        self.fc6 = nn.Linear(nh, nh)
        self.bn6 = nn.BatchNorm1d(nh)

        self.fc7 = nn.Linear(nh, nh)
        self.bn7 = nn.BatchNorm1d(nh)

        if opt.no_bn:
            self.bn3 = Identity()
            self.bn4 = Identity()
            self.bn5 = Identity()
            self.bn6 = Identity()
            self.bn7 = Identity()

        self.fc_final = nn.Linear(nh, nout)

    def forward(self, x, f_exp):
        re = x.dim() == 3

        if re:
            T, B, C = x.shape
            x = x.reshape(T * B, -1)
            f_exp = f_exp.reshape(T * B, -1)

        x = F.relu(self.bn3(self.fc3(x)))
        if self.cond is not None:
            f = self.cond(f_exp)
            x = torch.cat([x, f], dim=1)
        x = F.relu(self.bn4(self.fc4(x)))
        x = F.relu(self.bn5(self.fc5(x)))
        x = F.relu(self.bn6(self.fc6(x)))
        x = F.relu(self.bn7(self.fc7(x)))
        x = self.fc_final(x)
        x = torch.log_softmax(x, dim=1)
        if re:
            return x.reshape(T, B, -1)
        else:
            return x


class ProbDiscNet(nn.Module):
    def __init__(self, opt):
        super(ProbDiscNet, self).__init__()

        nmix = opt.nmix

        nh = opt.nh

        nin = nh
        nout = opt.dim_domain * nmix * 3

        self.fc3 = nn.Linear(nin, nh)
        self.bn3 = nn.BatchNorm1d(nh)

        self.fc4 = nn.Linear(nh, nh)
        self.bn4 = nn.BatchNorm1d(nh)

        self.fc5 = nn.Linear(nh, nh)
        self.bn5 = nn.BatchNorm1d(nh)

        self.fc6 = nn.Linear(nh, nh)
        self.bn6 = nn.BatchNorm1d(nh)

        if opt.no_bn:
            self.bn3 = Identity()
            self.bn4 = Identity()
            self.bn5 = Identity()
            self.bn6 = Identity()

        self.fc_final = nn.Linear(nh, nout)

        self.ndomain = opt.dim_domain
        self.nmix = nmix

    def forward(self, x):
        re = x.dim() == 3
        if re:
            T, B, C = x.shape
            x = x.reshape(T * B, -1)

        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        x = F.relu(self.bn5(self.fc5(x)))
        x = F.relu(self.bn6(self.fc6(x)))

        x = self.fc_final(x).reshape(-1, 3, self.ndomain, self.nmix)
        x_mean, x_std, x_weight = x[:, 0], x[:, 1], x[:, 2]
        # x_mean = torch.sigmoid(x_mean)
        x_std = torch.sigmoid(x_std) * 2 + 0.1
        x_weight = torch.softmax(x_weight, dim=1)

        if re:
            return x_mean.reshape(T, B, -1), x_std.reshape(T, B, -1), x_weight.reshape(T, B, -1)
        else:
            return x_mean, x_std, x_weight


class PredNet(nn.Module):
    def __init__(self, opt):
        super(PredNet, self).__init__()

        nh, nc = opt.nh, opt.nc
        nin = nh
        self.fc3 = nn.Linear(nin, nh)
        self.bn3 = nn.BatchNorm1d(nh)
        self.fc4 = nn.Linear(nh, nh)
        self.bn4 = nn.BatchNorm1d(nh)
        self.fc_final = nn.Linear(nh, nc)
        if opt.no_bn:
            self.bn3 = Identity()
            self.bn4 = Identity()

    def forward(self, x, return_softmax=False):
        re = x.dim() == 3
        if re:
            T, B, C = x.shape
            x = x.reshape(T * B, -1)

        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        x = self.fc_final(x)
        x_softmax = F.softmax(x, dim=1)
        x = F.log_softmax(x, dim=1)

        if re:
            x = x.reshape(T, B, -1)
            x_softmax = x_softmax.reshape(T, B, -1)

        if return_softmax:
            return x, x_softmax
        else:
            return x

# ======================================================================================================================
