# Neural Anisotropy Directions

This package is a rearrangement of some bits and pieces adapted from
the [original repository](https://github.com/LTS4/neural-anisotropy-directions)
by [@gortizji](https://github.com/gortizji) et al., plus some additional object-oriented implementation to make the code
as self-contained and easy to follow as possible.  
You can find the paper "Neural Anisotropy Directions" paper [here](https://arxiv.org/abs/2006.09717).

## Installation

This package relies on `torch>=1.9`, but everything other than the Fourier based operations should work just fine for
lower versions of python.

To install the package, enter the following command in your command-line interface:

```shell
pip install nads
```

## Usage

Assuming that you have a model class like `Model` and some initialization parameters
like `arg1=value1, arg2=value2, ...`
to compute the NADs for this architecture using the _Gradient Covariance_ method described in the paper, you can do as
follows:

```python
...
from nads.compute import GradientCovariance

compute = GradientCovariance(
    eval_point=torch.rand(...),  # some arbitrary input point to feed to the network
    model_cls=Model,  # your Model class
    model_params=dict(arg1=value1, arg2=value2, ...),  # initialization parameters for Model architecture
    device='cpu',  # which hardware do you want the computations to take place on
    force_eval=True,  # whether to force the model to eval state by doing model.eval() after each model initialization
)
nads = compute.nads(
    num_samples=2048, # number of MCMC samples to make for nads calculation
)
...
```

The resulting object has a bunch of useful properties such as saving (`.save(path)`), visualization of eigenvalues'
 spectrum (`.visualize_spectrum()`) and nads themselves (`.visualize_nads()`). You can slice it just like any tensor, 
and it will give you the sliced and accordingly. By calling the `.to(device)` method, you can move its tensors to your  
hardware of choice. You can also use the `.load(path)` to load up a previously saved NADs object.
For more information regarding each method, consult their docstrings.

The `data` module also contains a bunch of helpful data utils, such as the `DirectionalLinearDataset` class, which creates a
linearly separable dataset just as described in the paper and the `create_rfft2_direction` function that can be used to
create the desired canonical direction in the rfft2 vector space. 

## Todo
- [ ] Add arbitrary dataset poisoning functionality
- [ ] Add qualitative metrics for nads like using KLD or similar methods for measuring how uniformly distributed the  
eigenvalue spectrum is
- [ ] Add NADs computation for a grid of model parameters functionality
- [ ] Add grid-search functionality to reach the most uniformly distributed model architecture described by a set of
model parameters and a model class



