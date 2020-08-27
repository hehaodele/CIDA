import numpy as np
from torch.utils.data import DataLoader, Dataset


class ToyDataset(Dataset):
    def __init__(self, pkl, domain_id, opt=None):
        idx = pkl['domain'] == domain_id
        self.data = pkl['data'][idx].astype(np.float32)
        self.label = pkl['label'][idx].astype(np.int64)

        if opt.normalize_domain:
            print('===> Normalize in every domain')
            self.data_m, self.data_s = self.data.mean(0, keepdims=True), self.data.std(0, keepdims=True)
            self.data = (self.data - self.data_m) / self.data_s

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx], idx

    def __len__(self):
        return len(self.data)


class ReplayDataset(Dataset):
    def __init__(self, data, label, opt):
        self.data, self.label = data.copy(), label.copy()
        self.use_resample = opt.use_resample

    def __getitem__(self, i):
        if self.use_resample:
            idx = np.random.randint(len(self.data))
        else:
            idx = i
        return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.data)


class SeqToyDataset(Dataset):
    def __init__(self, datasets, size=1000):
        self.datasets = datasets
        self.size = size
        print('SeqDataset Size {} Sub Size {}'.format(
            size, [len(ds) for ds in datasets]
        ))

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        return [ds[i] for ds in self.datasets]
