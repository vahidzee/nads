from dataclasses import dataclass
from .visualization import visualize_spectrum, visualize_nads
from .utils import make_subdirectories
import typing as th
import torch


@dataclass
class NADs:
    """
    Utility dataclass which manages NADs saving/loading, visualization and potentially other operations
    """
    nads: torch.Tensor
    eigen_values: torch.Tensor
    name: th.Optional[str] = None
    model_params: th.Optional[dict] = None

    def to(self, device):
        assert self.nads is not None and self.eigen_values is not None, \
            'neither `nads` nor `eigen_velues` should be None'
        self.nads = self.nads.to(device)
        self.eigen_values = self.eigen_values.to(device)

    def save(self, path):
        """
        Save nads & eigen values along with the name of the model and its initialization parameters to the provided
        path. Subdirectories are automatically created.
        """
        make_subdirectories(path)
        self.to('cpu')
        torch.save(
            dict(nads=self.nads, eigen_values=self.eigen_values, name=self.name, model_params=self.model_params), path)

    def __getitem__(self, item):
        return self.nads.__getitem__(item)

    @classmethod
    def load(cls, path):
        """
        Load a NADs object from the provided path.
        """
        return NADs(**torch.load(path))

    def visualize_spectrum(self, **kwargs):
        """
        Visualize the NAD eigen_values spectrum. Check `visualization.visualize_spectrum` to see how keyword arguments
        affect the results.
        """
        kwargs['title'] = kwargs.get('title', f'{self.name} NADs eigen spectrum' if self.name else None)
        kwargs['x_label'] = kwargs.get('x_label', 'NAD Index')
        kwargs['y_label'] = kwargs.get('y_label', '$\sigma$')
        visualize_spectrum(self.eigen_values.tolist(), **kwargs)

    def visualize_nads(self, **kwargs):
        """
        Visualize the NAD tesnor images. Check `visualization.visualize_nads` to see how keyword arguments affect the
        results.
        """
        kwargs['title'] = kwargs.get('title', f'{self.name} NADs' if self.name else None)
        kwargs['indices'] = kwargs.get('indices', (0, 10))
        visualize_nads(self.nads, **kwargs)
