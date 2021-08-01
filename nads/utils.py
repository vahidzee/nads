import os


def make_subdirectories(path: str, exist_ok: bool = True) -> None:
    """
    Create the subdirectories of the provided path and raise error if exists and `exist_ok` is false.
    """
    base_dir = '/'.join(path.split('/')[:-1])
    if base_dir:
        os.makedirs(base_dir, exist_ok=exist_ok)
