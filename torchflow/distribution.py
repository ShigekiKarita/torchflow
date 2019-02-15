import torch


def cuda_(d):
    assert isinstance(d, torch.distributions.Distribution)
    for k in dir(d):
        attr = getattr(d, k)
        if isinstance(attr, torch.Tensor):
            try:
                setattr(d, k, attr.cuda())
            except AttributeError:
                pass
