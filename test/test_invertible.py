import torch
import torchflow.model as M
import torchflow.invertible as I
import torchflow.testing as T


def test_additive_coupling():
    net = M.MLP(1, 2, 1)
    x = torch.randn(3, 2)
    f = I.AdditiveCoupling(net)
    T.assert_invertible_and_logdet(f, x)


def test_affine_coupling():
    net = M.MLP(1, 2, 2)
    x = torch.randn(3, 4, 5, 2)
    f = I.AffineCoupling(net)
    T.assert_invertible_and_logdet(f, x)


def test_additive_coupling_split_larger():
    net = M.MLP(1, 2, 2)
    x = torch.randn(3, 4, 5, 3)
    f = I.AdditiveCoupling(net)
    T.assert_invertible_and_logdet(f, x)

def test_additive_coupling_split_smaller():
    net = M.MLP(2, 2, 1)
    x = torch.randn(3, 4, 5, 3)
    f = I.AdditiveCoupling(net)
    T.assert_invertible_and_logdet(f, x)



def test_seq():
    x = torch.randn(3, 4, 5, 2)
    f1 = I.AdditiveCoupling(M.MLP(1, 2, 1))
    f2 = I.AffineCoupling(M.MLP(1, 2, 2))
    f = I.Seq(f1, f2)
    z, ldj = f(x)
    ix, ildj = f.inverse(z)
    torch.testing.assert_allclose(z, f2(f1(x)[0])[0])
    torch.testing.assert_allclose(x, f1.inverse(f2.inverse(z)[0])[0])
    T.assert_invertible_and_logdet(f, x)


def test_parallel():
    x = torch.randn(3, 4, 5, 2)
    f1 = I.AdditiveCoupling(M.MLP(1, 2, 1))
    f2 = I.AffineCoupling(M.MLP(1, 2, 2))
    f = I.Parallel(f1, f2)
    z, ldj = f(x)
    ix, ildj = f.inverse(z)
    torch.testing.assert_allclose(ix, x)
    # torch.testing.assert_allclose(z, f2(f1(x)[0])[0])
    # torch.testing.assert_allclose(x, f1.inverse(f2.inverse(z)[0])[0])
    # T.assert_invertible_and_logdet(f, x)



def test_act_norm():
    x = torch.randn(3, 4, 5, 2)
    f = I.ActNorm(2)
    T.assert_invertible_and_logdet(f, x)


def test_linear():
    x = torch.randn(3, 4, 5, 2)
    f = I.Linear(2)
    T.assert_invertible_and_logdet(f, x)
