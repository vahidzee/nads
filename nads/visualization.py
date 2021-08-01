from matplotlib import pyplot as plt
import typing as th
from .utils import make_subdirectories
import torch


def visualize_image(
        image: torch.Tensor,
        transform: th.Optional[th.Callable[[torch.Tensor], torch.Tensor]] = None,
        title: th.Optional[str] = None,
        no_axis: bool = True,
        cmap: str = 'BrBG',
        subplot: th.Optional[th.Tuple[int, int, int]] = None,
        path: th.Optional[str] = None
):
    """
    Visualize the given image (in the respective subplot if one was provided) with the provided settings. `transform`
    can be a function which transforms the image before visualizing it (e.g. fft2, etc.)
    """
    img = transform(image) if transform is not None else image
    if len(img.shape) == 3 and img.shape[0] == 3:
        img = img.permute(1, 2, 0)
    if len(img.shape) == 3 and img.shape[0] == 1:
        img = img[0]
    if subplot is not None:
        plt.subplot(subplot[0], subplot[1], subplot[2])
    value_max = max(img.max().abs().item(), img.min().abs().item())
    plt.imshow(img, cmap=cmap, vmin=-value_max, vmax=value_max)
    if title:
        plt.title(title)
    if no_axis:
        plt.axis('off')
    if path:
        make_subdirectories(path)
        plt.savefig(path)


def visualize_nads(
        nads: torch.Tensor,
        indices: th.Optional[th.Union[int, th.Tuple[int, int]]] = None,
        transform: th.Optional[th.Callable[[torch.Tensor], torch.Tensor]] = None,
        cmap: str = 'BrBG',
        columns_count: int = 5,
        title: th.Optional[str] = None,
        no_axis: bool = True,
        path: th.Optional[str] = None,
):
    """
    Visualize the given nad tensors (filter by indices if any where provided) with the provided settings. `transform`
    can be a function which transforms the image before visualizing it (e.g. fft2, etc.)
    Most of this code is based on the original
        https://github.com/LTS4/neural-anisotropy-directions/blob/master/nad_computation.py#L9
    """
    if indices is not None and isinstance(indices, int):
        visualize_image(
            nads[indices], transform=transform, cmap=cmap, no_axis=no_axis,
            title=f'NAD idx={indices}{" - " + title if title is not None else ""}'
        )
    else:
        indices = indices if indices is not None else (0, len(nads) + 1)
        count = indices[1] - indices[0]
        rows_count = count // columns_count

        plt.figure(figsize=(columns_count * 4, rows_count * 4))

        if title is not None:
            plt.suptitle(title)

        for i in range(*indices):
            visualize_image(
                nads[i].reshape(32, 32), transform=transform, cmap=cmap, no_axis=no_axis,
                title=f'NAD idx={i}', subplot=(rows_count, columns_count, i + 1))

        if path:
            make_subdirectories(path)
            plt.savefig(path)
        else:
            plt.show()


def visualize_spectrum(
        spectrum,
        x_label: th.Optional[str] = None,
        y_label: th.Optional[str] = None,
        title: th.Optional[str] = None,
        path: th.Optional[str] = None,
        plot_type: str = 'semilogy'
) -> None:
    """
    Visualize given spectrum with the given `plot_type`. The resulting graph will either be shown or saved to `path` in
    case one was provided.
    """
    getattr(plt, plot_type)(spectrum)
    plt.grid()
    if x_label:
        plt.xlabel(x_label)
    if y_label:
        plt.ylabel(y_label)
    plt.tight_layout()

    if title:
        plt.title(title)
    if path:
        make_subdirectories(path)
        plt.savefig(path)
    else:
        plt.show()
