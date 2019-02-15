import torch
from .invertible import Seq, AdditiveCoupling, AffineCoupling


class MLP(torch.nn.Module):
    def __init__(self, n_in, n_hid, n_out):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_in, n_hid),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hid, n_hid),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hid, n_out)
        )
        self.n_input = n_in
        self.n_output = n_out

    def forward(self, x):
        return self.net(x)



def real_nvp(n_flows, data_dim, n_hidden, additive=False):
    assert data_dim % 2 == 0
    n_half = data_dim // 2
    if additive:
        return Seq(*[AdditiveCoupling(MLP(n_half, n_hidden, n_half)) for _ in range(n_flows)])
    else:
        return Seq(*[AffineCoupling(MLP(n_half, n_hidden, data_dim)) for _ in range(n_flows)])

