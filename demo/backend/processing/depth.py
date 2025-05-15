import torch
from PIL import Image
import numpy as np
import matplotlib
import models # Import the centralized models module
from utils.visualization import colorize # Use absolute import from backend perspective

# --- Depth Processing Functions ---

def get_grayscale_depth(image: Image.Image) -> np.ndarray:
    """
    Get the grayscale depth of a PIL image, invert it, and return as NumPy array.

    Args:
        image (PIL.Image.Image): Input PIL image (RGB).

    Returns:
        np.ndarray: Inverted grayscale depth image as a NumPy array.

    Raises:
        RuntimeError: If the ZoeDepth model failed to load or during inference.
        TypeError: If the input is not a PIL Image.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("Input must be a PIL Image.")

    # Get the model from the central models module
    zoe_depth_model = models.get_zoe_depth_model()
    if zoe_depth_model is None:
        raise RuntimeError("ZoeDepth model is not loaded. Cannot generate depth.")

    try:
        # Ensure image is RGB
        rgb_image = image.convert("RGB")

        # Infer depth using ZoeDepth model
        # Model was already moved to DEVICE during loading
        depth_tensor = zoe_depth_model.infer_pil(rgb_image, output_type="tensor") # Output is on DEVICE

        # Colorize the depth map in grayscale
        # colorize function handles tensor conversion and processing
        grayscale_depth_np = colorize(depth_tensor, cmap="gray") # Returns numpy array

        # Invert the grayscale depth
        inverted_depth_np = 255 - grayscale_depth_np  # Invert grayscale values

        # Return only RGB channels if grayscale_depth has 4 channels (RGBA)
        # The colorize function might return RGBA, we usually want 3 channels or 1 for depth.
        # Since it's grayscale, it should be (H, W) or (H, W, 1) or (H, W, 3) or (H, W, 4)
        # For saving as a typical grayscale image, ensure it's 2D or 3D with 1 channel.
        # If it's (H,W,3) or (H,W,4) and grayscale, all color channels are the same.
        if inverted_depth_np.ndim == 3 and inverted_depth_np.shape[2] >= 3:
            return inverted_depth_np[..., 0].astype(np.uint8) # Take one channel for grayscale
        elif inverted_depth_np.ndim == 2:
            return inverted_depth_np.astype(np.uint8)
        else: # Should not happen with cmap='gray' if colorize works as expected
             raise RuntimeError(f"Unexpected depth map shape: {inverted_depth_np.shape}")


    except Exception as e:
        print(f"Error during depth inference for the provided image: {e}")
        raise RuntimeError(f"Error during depth inference: {e}")
