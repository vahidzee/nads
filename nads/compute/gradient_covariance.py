import typing as th
import torch
from .jacobian import numerical_jacobian
from nads.nads import NADs


class GradientCovariance:
    """
    Gradient Covariance computation by MCMC sampling.
    Original code can be found in https://github.com/LTS4/neural-anisotropy-directions/blob/master/nad_computation.py#L9
    """
    def __init__(
            self,
            eval_point,
            model_cls: th.Type[type],
            model_params: th.Optional[dict] = None,
            device: str = 'cpu',
            force_eval=True,
            jacobian_fn: th.Callable[[th.Callable, torch.Tensor], torch.Tensor] = numerical_jacobian,
    ):
        """
        Gradient Covariance computation of NADs. `eval_points` is the initial input point for evaluating the expected
        w.r.t model parameters. `model_cls` should be a class variable for the desired model and `model_params` is an
        optional dictionary consisting the desired parameters to initialize the model with.`device` is where you want
        the computations to reside. if `force_eval` is set True
        """
        assert eval_point is not None, 'Evaluation point cannot be None'

        self.eval_point = eval_point.to(device)
        self.model_cls = model_cls
        self.model_params = model_params if model_params is not None else dict()
        self.device = device
        self.force_eval = force_eval
        self.jacobian_fn = jacobian_fn

    @property
    def sample_model(self):
        model = self.model_cls(**self.model_params).to(self.device)
        if self.force_eval:
            model = model.eval()
        return model

    @property
    def sample_grad(self):
        return self.jacobian_fn(self.sample_model, self.eval_point)

    def nads(self, num_samples=1000, grad_active=False):
        sample_gradient = torch.stack([self.sample_grad.view(-1) for i in range(num_samples)])
        with torch.set_grad_enabled(grad_active):
            u, s, vh = torch.linalg.svd(sample_gradient, full_matrices=False)
        return NADs(
            nads=vh.view(-1, *self.eval_point.shape)[:self.eval_point.numel()].detach(),
            eigen_values=s[:self.eval_point.numel()].detach(),
            name=self.model_cls.__name__,
            model_params=self.model_params,
        )
