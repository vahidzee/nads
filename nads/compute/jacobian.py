import torch


def numerical_jacobian(fn: torch.nn.Module, x: torch.Tensor, scale: float = 100) -> torch.Tensor:
    """
    Calculate the jacobian of the function w.r.t. the given input tensor x using numerical approximation.
    Original code can be found in https://github.com/LTS4/neural-anisotropy-directions/blob/master/utils.py#L82
    """
    shape = list(x.shape)
    n_dims = x.numel()
    v = torch.eye(n_dims, device=x.device).view([n_dims] + shape)
    jac = torch.zeros(n_dims, device=x.device)
    residual = 1 if n_dims % n_dims > 0 else 0
    with torch.no_grad():
        for n in range(n_dims // n_dims + residual):
            batch_plus = x[None, :] + scale * v[n * n_dims: (n + 1) * n_dims]
            batch_minus = x[None, :] - scale * v[n * n_dims: (n + 1) * n_dims]

            jac[n * n_dims: (n + 1) * n_dims] = ((fn(batch_plus) - fn(batch_minus)) / (2 * scale)).detach()[:, 0]
    return jac.view(shape)


def backprop_jacobian(fn: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Calculate the jacobian of the function w.r.t. the given input tensor x using regular backprop.
    """
    x.requires_grad_()
    batch = x[None]
    return torch.autograd.grad(fn(batch), batch, allow_unused=True)[0][0]
