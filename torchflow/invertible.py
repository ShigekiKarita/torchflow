import torch


def half(x):
    assert x.shape[-1] % 2 == 0, "last dim should be even but {}".format(x.shape)
    return x.split(x.shape[-1] // 2, dim=-1)


class Invertible(torch.nn.Module):
    def forward(self, x):
        """Returns f(x), log |det df/dx| """
        pass

    def inverse(self, z):
        """Returns f^-1(x), log |det df^-1/dx| """
        pass


class Coupling(Invertible):
    def split(self, x):
        if hasattr(self.net, "n_input"):
            a = self.net.n_input
            b = x.shape[-1] - a
            return x.split([a, b], dim=-1)
        else:
            return half(x)

    def isplit(self, x):
        if hasattr(self.net, "n_input"):
            b = self.net.n_input
            a = x.shape[-1] - b
            return x.split([a, b], dim=-1)
        else:
            return half(x)


class AdditiveCoupling(Coupling):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x_a, x_b = self.split(x)
        x_b = x_b + self.net(x_a)
        return torch.cat((x_b, x_a), dim=-1), torch.tensor(0.0)

    def inverse(self, y):
        y_a, y_b = self.isplit(y)
        y_a = y_a - self.net(y_b)
        return torch.cat((y_b, y_a), dim=-1), torch.tensor(0.0)


class AffineCoupling(Coupling):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x_a, x_b = self.split(x)
        s, t = half(self.net(x_a))
        x_b = torch.exp(s)*x_b + t
        return torch.cat((x_b, x_a), dim=-1), s.sum()

    def inverse(self, z):
        z_a, z_b = self.isplit(z)
        s, t = half(self.net(z_b))
        z_a = (z_a - t) / torch.exp(s)
        return torch.cat([z_b, z_a], dim=-1), -s.sum()


class Seq(Invertible):
    def __init__(self, *seq):
        super().__init__()
        self.net = torch.nn.ModuleList(seq)
        for net in self.net:
            assert isinstance(net, Invertible)

    def forward(self, x):
        log_det_jacobian = 0
        for net in self.net:
            x, ldj = net(x)
            log_det_jacobian += ldj
        return x, log_det_jacobian

    def inverse(self, z):
        log_det_jacobian = 0
        for net in reversed(self.net):
            z, ldj = net.inverse(z)
            log_det_jacobian += ldj
        return z, log_det_jacobian


class Parallel(Invertible):
    def __init__(self, *modules):
        super().__init__()
        self.net = torch.nn.ModuleList(modules)
        for m in self.net:
            assert isinstance(m, Invertible)
        self.sizes = None

    def forward(self, x):
        log_det_jacobian = 0
        zs = []
        self.sizes = []
        for net in self.net:
            z, ldj = net(x)
            zs.append(z)
            self.sizes.append(z.shape[-1])
            log_det_jacobian += ldj
        return torch.cat(zs, dim=-1), log_det_jacobian

    def inverse(self, zs):
        if self.sizes is None:
            self.sizes = [zs.shape[-1] // len(self.modules) for _ in self.modules]
        # TODO: average?
        z = zs.split(self.sizes, dim=-1)[0]
        return self.net[0].inverse(z)


def population(x):
    """assume x [batch, shape0, shape1, ..., feat]"""
    n = 1
    for i in x.shape[0:-1]:
        n *= i
    return n


class ActNorm(Invertible):
    def __init__(self, n, data_init=True, eps=1e-6):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(n))
        self.bias = torch.nn.Parameter(torch.zeros(n))
        self.eps = eps
        self.data_init = data_init

    def forward(self, x):
        assert x.shape[-1] == self.scale.shape[0]
        xf = x.view(-1, x.shape[-1])
        if self.data_init:
            self.scale[:] = 1 / (xf.std(dim=0) + self.eps)
            self.bias[:] = -xf.mean(dim=0)
            self.data_init = False
        return self.scale * x + self.bias, self.logdet(x)

    def inverse(self, y):
        return (y - self.bias) / self.scale, -self.logdet(y)

    def logdet(self, x):
        return population(x) * self.scale.log().sum()


class Linear(Invertible):
    def __init__(self, n):
        super().__init__()
        # invertible initialization
        q, r = torch.randn(n, n).qr()
        self.weight = torch.nn.Parameter(q)

    def forward(self, x):
        return x.matmul(self.weight), self.logdet(x)

    def inverse(self, y):
        return y.matmul(self.weight.inverse()), -self.logdet(y)

    def logdet(self, x):
        return population(x) * self.weight.det().abs().log()

