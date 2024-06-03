import numpy as np

def create_gaussian_kernel(kernel_size: int, sigma: float = 1.0) -> np.ndarray:
    """
    Creates a 2D Gaussian kernel for image filtering.

    Args:
        kernel_size (int): The size of the kernel (must be an odd integer).
        sigma (float, optional): The standard deviation of the Gaussian distribution. Defaults to 1.0.

    Returns:
        np.ndarray: The Gaussian kernel.
    """
    ax = np.linspace(-(kernel_size - 1) / 2., (kernel_size - 1) / 2., kernel_size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)

def create_mean_kernel(kernel_size: int) -> np.ndarray:
    """
    Creates a 2D mean (average) kernel for image filtering.

    Args:
        kernel_size (int): The size of the kernel (must be an odd integer).

    Returns:
        np.ndarray: The mean kernel.
    """
    return np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)

def create_sharpen_kernel(kernel_size: int) -> np.ndarray:
    """
    Creates a sharpening kernel for image filtering.

    Args:
        kernel_size (int): The size of the kernel (must be an odd integer).

    Returns:
        np.ndarray: The sharpening kernel.
    """
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    kernel[center, center] = 2
    kernel -= np.ones((kernel_size, kernel_size)) / kernel_size ** 2
    return kernel

def create_unsharp_mask_kernel(kernel_size: int, amount: float = 1.0) -> np.ndarray:
    """
    Creates an unsharp mask kernel for image sharpening.

    Args:
        kernel_size (int): The size of the kernel (must be an odd integer).
        amount (float, optional): The sharpening amount (0.0 to 2.0). Defaults to 1.0.

    Returns:
        np.ndarray: The unsharp mask kernel.
    """
    blurred = create_gaussian_kernel(kernel_size)
    sharpened = create_sharpen_kernel(kernel_size)
    return (1 - amount) * blurred + amount * sharpened