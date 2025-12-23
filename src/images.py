"""
images.py
Target image generation functions
"""

import numpy as np
from PIL import Image


def create_gradient_image(size: int = 50, direction: str = 'horizontal') -> np.ndarray:
    """
    Create gradient target image

    Args:
        size: Image size (width and height)
        direction: 'horizontal', 'vertical', or 'radial'

    Returns:
        Grayscale image as numpy array
    """
    img = np.zeros((size, size), dtype=np.uint8)

    if direction == 'horizontal':
        for i in range(size):
            img[:, i] = int((i / (size - 1)) * 255)

    elif direction == 'vertical':
        for i in range(size):
            img[i, :] = int((i / (size - 1)) * 255)

    elif direction == 'radial':
        center = size / 2
        max_dist = np.sqrt(2) * center
        for i in range(size):
            for j in range(size):
                dist = np.sqrt((i - center)**2 + (j - center)**2)
                img[i, j] = int((dist / max_dist) * 255)

    elif direction == 'diagonal':
        for i in range(size):
            for j in range(size):
                value = ((i + j) / (2 * (size - 1))) * 255
                img[i, j] = int(value)

    else:
        raise ValueError(f"Unknown gradient direction: {direction}")

    return img


def create_circle_image(size: int = 50, radius_ratio: float = 0.3) -> np.ndarray:
    """
    Create circle target image

    Args:
        size: Image size (width and height)
        radius_ratio: Circle radius as fraction of image size

    Returns:
        Grayscale image as numpy array
    """
    img = np.zeros((size, size), dtype=np.uint8)
    center = size / 2
    radius = size * radius_ratio

    for i in range(size):
        for j in range(size):
            dist = np.sqrt((i - center)**2 + (j - center)**2)
            if dist < radius:
                img[i, j] = 255

    return img


def create_ring_image(size: int = 50,
                      inner_ratio: float = 0.2,
                      outer_ratio: float = 0.4) -> np.ndarray:
    """
    Create ring (annulus) target image

    Args:
        size: Image size (width and height)
        inner_ratio: Inner radius as fraction of image size
        outer_ratio: Outer radius as fraction of image size

    Returns:
        Grayscale image as numpy array
    """
    img = np.zeros((size, size), dtype=np.uint8)
    center = size / 2
    inner_radius = size * inner_ratio
    outer_radius = size * outer_ratio

    for i in range(size):
        for j in range(size):
            dist = np.sqrt((i - center)**2 + (j - center)**2)
            if inner_radius < dist < outer_radius:
                img[i, j] = 255

    return img


def create_checkerboard_image(size: int = 50, squares: int = 8) -> np.ndarray:
    """
    Create checkerboard target image

    Args:
        size: Image size (width and height)
        squares: Number of squares along one dimension

    Returns:
        Grayscale image as numpy array
    """
    img = np.zeros((size, size), dtype=np.uint8)
    square_size = size // squares

    for i in range(size):
        for j in range(size):
            row_idx = i // square_size
            col_idx = j // square_size
            if (row_idx + col_idx) % 2 == 0:
                img[i, j] = 255

    return img


def create_stripes_image(size: int = 50,
                         stripes: int = 8,
                         direction: str = 'horizontal') -> np.ndarray:
    """
    Create striped pattern target image

    Args:
        size: Image size (width and height)
        stripes: Number of stripes
        direction: 'horizontal' or 'vertical'

    Returns:
        Grayscale image as numpy array
    """
    img = np.zeros((size, size), dtype=np.uint8)
    stripe_size = size // stripes

    if direction == 'horizontal':
        for i in range(size):
            stripe_idx = i // stripe_size
            if stripe_idx % 2 == 0:
                img[i, :] = 255

    elif direction == 'vertical':
        for j in range(size):
            stripe_idx = j // stripe_size
            if stripe_idx % 2 == 0:
                img[:, j] = 255

    else:
        raise ValueError(f"Unknown stripe direction: {direction}")

    return img


def create_cross_image(size: int = 50, thickness: int = 10) -> np.ndarray:
    """
    Create cross pattern target image

    Args:
        size: Image size (width and height)
        thickness: Thickness of cross lines

    Returns:
        Grayscale image as numpy array
    """
    img = np.zeros((size, size), dtype=np.uint8)
    center = size // 2
    half_thick = thickness // 2

    # Horizontal line
    img[center - half_thick:center + half_thick, :] = 255

    # Vertical line
    img[:, center - half_thick:center + half_thick] = 255

    return img


def create_rectangle_image(size: int = 50,
                           width_ratio: float = 0.6,
                           height_ratio: float = 0.4) -> np.ndarray:
    """
    Create centered rectangle target image

    Args:
        size: Image size (width and height)
        width_ratio: Rectangle width as fraction of image size
        height_ratio: Rectangle height as fraction of image size

    Returns:
        Grayscale image as numpy array
    """
    img = np.zeros((size, size), dtype=np.uint8)

    rect_width = int(size * width_ratio)
    rect_height = int(size * height_ratio)

    start_x = (size - rect_width) // 2
    start_y = (size - rect_height) // 2

    img[start_y:start_y + rect_height, start_x:start_x + rect_width] = 255

    return img


def create_triangle_image(size: int = 50) -> np.ndarray:
    """
    Create triangle target image

    Args:
        size: Image size (width and height)

    Returns:
        Grayscale image as numpy array
    """
    img = np.zeros((size, size), dtype=np.uint8)

    for i in range(size):
        for j in range(size):
            # Simple isosceles triangle pointing up
            if i >= size // 4 and j >= (size // 2 - i + size // 4) and j <= (size // 2 + i - size // 4):
                img[i, j] = 255

    return img


def create_bull_eye_image(size: int = 50, rings: int = 3) -> np.ndarray:
    """
    Create bull's eye (concentric circles) target image

    Args:
        size: Image size (width and height)
        rings: Number of rings

    Returns:
        Grayscale image as numpy array
    """
    img = np.zeros((size, size), dtype=np.uint8)
    center = size / 2
    max_radius = size / 2

    for i in range(size):
        for j in range(size):
            dist = np.sqrt((i - center)**2 + (j - center)**2)
            ring_idx = int((dist / max_radius) * rings)
            if ring_idx % 2 == 0:
                img[i, j] = 255

    return img


def load_image(filepath: str, size: int = 50, grayscale: bool = True) -> np.ndarray:
    """
    Load and preprocess an image file.
    Forces grayscale because the GA only supports 2D images.
    """

    img = Image.open(filepath)

    # Resize image
    img = img.resize((size, size), Image.Resampling.LANCZOS)

    # ALWAYS convert to grayscale (1 channel)
    img = img.convert('L')

    # Convert to numpy array (shape: size x size)
    img_array = np.array(img).astype(np.uint8)

    return img_array


def save_image(image: np.ndarray, filepath: str):
    """
    Save image array to file

    Args:
        image: Image as numpy array
        filepath: Output file path
    """
    img = Image.fromarray(image.astype(np.uint8))
    img.save(filepath)
    print(f"Image saved to {filepath}")


def create_custom_pattern(size: int = 50, pattern_func=None) -> np.ndarray:
    """
    Create custom pattern using a user-defined function

    Args:
        size: Image size (width and height)
        pattern_func: Function that takes (i, j, size) and returns pixel value

    Returns:
        Grayscale image as numpy array
    """
    if pattern_func is None:
        # Default: diagonal gradient
        pattern_func = lambda i, j, s: int(((i + j) / (2 * (s - 1))) * 255)

    img = np.zeros((size, size), dtype=np.uint8)

    for i in range(size):
        for j in range(size):
            img[i, j] = pattern_func(i, j, size)

    return img