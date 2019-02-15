import torch
import torchflow.model as M
import torchflow.testing as T


def test_real_nvp():
    x = torch.randn(3, 4, 5, 2)
    f = M.real_nvp(2, 2, 5, additive=True)
    T.assert_invertible_and_logdet(f, x)

    f = M.real_nvp(2, 2, 5, additive=False)
    T.assert_invertible_and_logdet(f, x)
