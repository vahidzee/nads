import torch
import typing as th


def numerical_jacobian(fn: torch.nn.Module, x: torch.Tensor, scale: float = 100,
                       batch_size: th.Optional[int] = None) -> torch.Tensor:
    """
    Calculate the jacobian of the function w.r.t. the given input tensor x using numerical approximation. If
    `batch_size` is specified, inputs will be split into batches of batch_size.
    Original code can be found in https://github.com/LTS4/neural-anisotropy-directions/blob/master/utils.py#L82
    """
    shape = list(x.shape)
    n_dims = x.numel()
    batch_size = n_dims if batch_size is None else batch_size
    v = torch.eye(n_dims, device=x.device).view([n_dims] + shape)
    jac = torch.zeros(n_dims, device=x.device)
    residual = 1 if n_dims % batch_size > 0 else 0
    with torch.no_grad():
        for n in range(n_dims // batch_size + residual):
            batch_plus = x[None, :] + scale * v[n * batch_size: (n + 1) * batch_size]
            batch_minus = x[None, :] - scale * v[n * batch_size: (n + 1) * batch_size]

            jac[n * batch_size: (n + 1) * batch_size] = ((fn(batch_plus) - fn(batch_minus)) / (2 * scale)).detach()[:,
                                                        0]
    return jac.view(shape)


def backprop_jacobian(fn: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Calculate the jacobian of the function w.r.t. the given input tensor x using regular backprop.
    """
    x.requires_grad_()
    batch = x[None]
    return torch.autograd.grad(fn(batch), batch, allow_unused=True)[0][0]
