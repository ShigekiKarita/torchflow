import torch


def jacobian(f, x, detach=True):
    x = x.detach()
    x.requires_grad = True
    z, ldj = f(x)
    assert x.shape[:-1] == z.shape[:-1], "f(x) should has the same shape of x except for the last dim"
    n_in = x.shape[-1]
    n_out = z.shape[-1]
    zf = z.view(-1, n_out)
    n_batch = zf.shape[0]
    ret = torch.empty(n_batch, n_out, n_in, device=x.device)
    for i in range(n_out):
        zi = zf[:, i]
        dzi_dx = torch.autograd.grad(zi, x, torch.ones_like(zi),
                                     allow_unused=True, retain_graph=True)[0]
        dzi_dx = dzi_dx.view(n_batch, n_in)
        if detach:
            dzi_dx = dzi_dx.detach()
        ret[:, i, :] = dzi_dx
    return ret.view(*x.shape[:-1], n_out, n_in)


def sum_log_det(js):
    return sum(j.det().abs().log() for j in js.view(-1, js.shape[-2], js.shape[-1]))


def assert_invertible_and_logdet(f, x):
    x.requires_grad = True
    z, ldj = f(x)
    ix, ildj = f.inverse(z)
    torch.testing.assert_allclose(ix, x)
    torch.testing.assert_allclose(sum_log_det(jacobian(f, x)), ldj)
    torch.testing.assert_allclose(sum_log_det(jacobian(f.inverse, z)), ildj)
