import torch
from PIL import Image
import numpy as np
import matplotlib
import models # Import the centralized models module
from utils.visualization import colorize # Use absolute import from backend perspective

# --- Depth Processing Functions ---

def get_grayscale_depth(image_path, output_path):
    """
    Get the grayscale depth of an image, invert it, and save it to a file.

    Args:
        image_path (str): Path to the input image.
        output_path (str): Path to save the inverted grayscale depth image.

    Raises:
        RuntimeError: If the ZoeDepth model failed to load.
        FileNotFoundError: If the input image path does not exist.
        Exception: For other image processing or saving errors.
    """
    # Get the model from the central models module
    zoe_depth_model = models.get_zoe_depth_model()
    if zoe_depth_model is None:
        raise RuntimeError("ZoeDepth model is not loaded. Cannot generate depth.")

    try:
        # Load the image
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Input image not found at {image_path}")
        raise
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        raise

    try:
        # Infer depth using ZoeDepth model (ensure model is on the correct device)
        # Model was already moved to DEVICE during loading
        # Use the centrally loaded model
        depth_tensor = zoe_depth_model.infer_pil(image, output_type="tensor") # Output is on DEVICE

        # Colorize the depth map in grayscale
        # colorize function handles tensor conversion and processing
        grayscale_depth = colorize(depth_tensor, cmap="gray") # Returns numpy array

        # Invert the grayscale depth
        inverted_depth = 255 - grayscale_depth  # Invert grayscale values

        # Save the inverted grayscale depth image
        # Ensure we only save RGB channels if grayscale_depth has 4 channels (RGBA)
        Image.fromarray(inverted_depth[..., :3]).save(output_path)
        print(f"Inverted grayscale depth image saved at {output_path}")

    except Exception as e:
        print(f"Error during depth inference or saving for {image_path}: {e}")
        raise
