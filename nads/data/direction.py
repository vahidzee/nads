import torch


def create_rfft2_direction(shape: tuple, direction: tuple, direction_value=1j):
    """
    Create canonical rfft2 base vector in the spatial space
    This code is the functional implementation of the original main code in:
        https://github.com/LTS4/neural-anisotropy-directions/blob/master/directional_bias.py
    """
    v = torch.zeros(shape)  # Create empty vector
    v_fft = torch.fft.rfft2(v)
    v_fft[direction] = direction_value  # Select coordinate in fourier space
    v = torch.fft.irfft2(v_fft, s=shape[-2:])
    return v / v.norm()
