import torch
from torch.utils.data import Dataset
import random
import numpy as np


device = 'cuda:0'
embedding = {chr(65 + i): torch.randn(128, device=device) for i in range(26)}


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj1 = torch.nn.Linear(128, 64, bias=True, device=device)
        self.relu = torch.nn.ReLU()
        self.proj2 = torch.nn.Linear(64, 1, bias=True, device=device)

    def forward(self, x):
        x = self.proj1(x)
        x = self.relu(x)
        x = self.proj2(x)
        return x


class Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def forward(self, rnd_chr, logits):
        indices = np.argsort(rnd_chr).tolist()
        indices = torch.tensor(indices, dtype=torch.int64, device=device)[:, None]

        logits = torch.gather(logits, dim=0, index=indices)
        logits = logits[1:] - logits[:-1]

        labels = torch.ones_like(logits, dtype=torch.float32)

        probs = logits.detach().sigmoid()
        diffs = torch.count_nonzero((probs > 0.5).type(labels.dtype) != labels)

        return self.criterion(logits, labels), diffs.item()


def sample():
    rnd_chr = [chr(65 + random.randint(a=0,b=25)) for _ in range(26)]
    rnd_chr = list(set(rnd_chr))

    rnd_emb = [embedding[ch] for ch in rnd_chr]
    rnd_emb = torch.stack(rnd_emb, dim=0)

    return rnd_chr, rnd_emb


class RandomData(Dataset):
    def __getitem__(self, index):
        return sample()
    
    def __len__(self):
        return 100
    

model = Model()
ds = RandomData()
loss_fcn = Loss()
optim = torch.optim.Adam(model.parameters(), lr=1e-3)

model.train()

for idx, (rnd_chr, rnd_emb) in enumerate(ds):
    logits = model(rnd_emb)
    loss, diffs = loss_fcn(rnd_chr, logits)
    loss.backward()
    optim.step()
    optim.zero_grad()

    print(f"{idx:<5d}: {diffs}")