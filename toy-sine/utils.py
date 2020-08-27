import torch.nn.functional as F
import _pickle as pickle
import torch

def plain_log(filename, text):
    fp = open(filename,'a')
    fp.write(text)
    fp.close()

def read_pickle(name):
    with open(name,'rb') as f:
        data = pickle.load(f)
    return data

def write_pickle(data, name):
    with open(name,'wb') as f:
        pickle.dump(data, f)


def masked_cross_entropy(pred, label, mask, weight=None):
    """ get masked cross entropy loss, for those training data padded with zeros at the end """
    length, classnum = pred.size()

    temp = pred * mask.unsqueeze(1)
    loss = F.nll_loss(temp, label, reduction="sum")
    loss = loss / (mask.sum(0) + 1e-10)

    return loss

def gaussian_loss(pred_m, pred_s, label, mean=0, norm=15):
    """gaussian loss taking mean and (log) variance as input"""
    length, dim = pred_m.size()
    term1 = torch.sum((pred_m - label) ** 2 / (torch.exp(pred_s))) / length / dim

    term2 = 0.5 * torch.sum(pred_s) / length / dim

    delta = norm // 2 + 1 - torch.abs(label - mean) * norm * 1.0
    delta = delta.clone().detach()
    delta.data.clamp_(1.0, 10.0)
    term3 = 0.01 * torch.sum((1 / torch.exp(pred_s) - delta) ** 2) / length / dim #0.05

    return term1 + term2 + term3

def inverse_weighted_mse_loss(pred, label, mean, norm=15):
    """inverse weighted"""
    length, dim = pred.size()
    delta = norm // 2 + 1 - torch.abs(label - mean) * norm * 1.0
    delta = delta.clone().detach()
    delta.data.clamp_(1.0, 10.0)
    term1 = torch.sum((pred - label) ** 2 * delta) / length / dim
    return term1

def inverse_quadratic_mse_loss(pred, label, mean, norm=15):
    """inverse weighted"""
    length, dim = pred.size()
    delta = 1 / (torch.abs(label - mean)) ** 2
    delta = delta.clone().detach()
    delta.data.clamp_(1.0, 5.0)
    term1 = torch.sum((pred - label) ** 2 * delta) / length / dim
    return term1

def inverse_weighted_l1_loss(pred, label, mean, norm=15):
    """inverse weighted"""
    length, dim = pred.size()
    delta = norm // 2 + 1 - torch.abs(label - mean) * norm * 1.0
    delta = delta.clone().detach()
    delta.data.clamp_(1.0, 10.0)
    term1 = torch.sum(torch.abs(pred - label) * delta) / length / dim
    return term1

def ln_loss(pred, label, mean, norm=15, n=3.0):
    """inverse weighted"""
    length, dim = pred.size()
    term1 = torch.sum(torch.abs(pred - label) ** n) / length / dim
    return term1

def mixture_loss(pred1, pred2, label, lambda_m=0.1):
    """mixture loss taking two prediction as input"""
    term1 = F.mse_loss(pred1, label)
    term2 = F.mse_loss(pred2, label)
    length, dim = pred1.size()
    term3 = - lambda_m * torch.sum(F.sigmoid(pred1 - pred2) ** 2) / length / dim
    return term1 + term2 + term3