import cv2
import numpy as np
import matplotlib.pyplot as plt

def read_image(image_path: str) -> np.ndarray:
    """
    Reads an image from the specified file path.

    Args:
        image_path (str): The path to the image file.

    Returns:
        np.ndarray: The image data as a NumPy array.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image

def pad_image(image: np.ndarray, padding_size: tuple[int, int], padding_mode: str = 'constant', constant_values: tuple[int, int] = (0, 0)) -> np.ndarray:
    """
    Pads an image with the specified padding size and mode.

    Args:
        image (np.ndarray): The image to be padded.
        padding_size (tuple[int, int]): The padding size for the top/bottom and left/right sides of the image.
        padding_mode (str, optional): The padding mode. Defaults to 'constant'.
            - 'constant': Pads with a constant value.
            - 'edge': Pads with the edge values of the image.
            - 'reflect': Pads with the reflection of the image.
            - 'symmetric': Pads with the reflection of the image mirrored along the edge.
        constant_values (tuple[int, int], optional): The values to use for constant padding for the top/bottom and left/right sides of the image. Defaults to (0, 0).

    Returns:
        np.ndarray: The padded image.

    Raises:
        ValueError: If an invalid padding mode is provided.
    """

    if padding_mode not in ['constant', 'edge', 'reflect', 'symmetric']:
        raise ValueError("Invalid padding mode. Choose from 'constant', 'edge', 'reflect', or 'symmetric'.")

    # If padding_mode is 'constant', use the provided constant_values
    if padding_mode == 'constant':
        padded_image = np.pad(image, pad_width=((padding_size[0], padding_size[0]), (padding_size[1], padding_size[1]), (0, 0)), mode=padding_mode, constant_values=constant_values)
    else:
        padded_image = np.pad(image, pad_width=((padding_size[0], padding_size[0]), (padding_size[1], padding_size[1]), (0, 0)), mode=padding_mode)

    return padded_image


def display_image_list(images, rows=None, cols=None, figsize=None, titles=None):
  """Displays a list of NumPy arrays representing images in a grid layout.

  Args:
      images (list): List of NumPy arrays containing image data.
      rows (int, optional): Number of rows in the grid layout. If None,
          automatically calculated based on the number of images. Defaults to None.
      cols (int, optional): Number of columns in the grid layout. If None,
          automatically calculated based on the number of images. Defaults to None.
      figsize (tuple, optional): Size of the figure in inches. Defaults to None,
          which uses Matplotlib's default figure size.
      titles (list, optional): List of titles for each image. If None,
          no titles are displayed. Defaults to None.
  """

  num_images = len(images)

  # Determine rows and columns if not specified
  if rows is None and cols is None:
    rows, cols = calculate_grid_layout(num_images)
  elif rows is None:
    rows = 1
    cols = num_images // rows
  elif cols is None:
    cols = 1
    rows = num_images // cols

  # Create the figure
  if figsize is None:
    figsize = (cols * 3, rows * 3)  # Adjust figure size based on number of images
  fig, axes = plt.subplots(rows, cols, figsize=figsize)

  # Iterate through images and display them on subplots
  for i in range(num_images):
    img = images[i]
    ax = axes.flat[i]  # Get the current subplot axis

    # Display the image
    ax.imshow(img)

    # Turn off axes visibility (optional)
    ax.axis('off')

    # Add title if provided
    if titles and len(titles) > i:
      ax.set_title(titles[i])

  # Adjust layout to prevent overlapping elements
  plt.tight_layout()

  # Display the plot
  plt.show()

def calculate_grid_layout(num_images):
  """Calculates an appropriate grid layout for displaying a given number of images.

  Args:
      num_images (int): Number of images to display.

  Returns:
      tuple: A tuple containing the number of rows and columns in the grid layout.
  """

  # Start with a square grid
  rows = cols = int(np.sqrt(num_images))

  # Adjust rows and columns while keeping the aspect ratio close to 1
  while rows * cols < num_images:
    if rows <= cols:
      rows += 1
    else:
      cols += 1

  return rows, cols

if __name__ == "__main__":
    # Load an image
    image_path = "demo/cat.jpg"
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Pad the image
    padding_size = (50, 50)
    padded_image = pad_image(image, padding_size, padding_mode='reflect', constant_values=(1, 1))
    
    # Display the original and padded images
    display_image_list([image, padded_image], titles=["Original Image", "Padded Image"])