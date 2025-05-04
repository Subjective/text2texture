import base64
import io
import logging
import numpy as np
import cv2
from PIL import Image

# Get logger (assuming root logger is configured elsewhere, e.g., in app.py)
logger = logging.getLogger(__name__)

def generate_checkerboard_heightmap(mask_region: np.ndarray, checker_size: int = 8, height: float = 1.0) -> np.ndarray:
    """
    Generates a heightmap with a checkerboard pattern within the mask region.

    Args:
        mask_region: A 2D boolean numpy array where True indicates the region for the checkerboard.
        checker_size: The size of each square in the checkerboard pattern.
        height: The value to assign to the 'high' squares of the checkerboard.

    Returns:
        A 2D numpy array of the same shape as mask_region, with the checkerboard pattern
        applied within the mask, and zeros elsewhere.
    """
    if mask_region.ndim != 2 or mask_region.dtype != bool:
        raise ValueError("mask_region must be a 2D boolean numpy array.")

    h, w = mask_region.shape
    checkerboard = np.zeros((h, w), dtype=np.float32)

    # Create the checkerboard pattern based on coordinates
    # # (i // checker_size) % 2 == (j // checker_size) % 2 determines the square color
    # for i in range(h):
    #     for j in range(w):
    #         if mask_region[i, j]: # Only apply within the mask
    #             if (i // checker_size) % 2 == (j // checker_size) % 2:
    #                 checkerboard[i, j] = height # Or 0, depending on starting corner preference
    #             else:
    #                 checkerboard[i, j] = 0 # Or height

    # Vectorized approach (potentially faster for large images)
    y_idx, x_idx = np.indices(mask_region.shape)
    checkerboard_pattern = ((y_idx // checker_size) % 2 == (x_idx // checker_size) % 2).astype(np.float32) * height
    checkerboard[mask_region] = checkerboard_pattern[mask_region]

    return checkerboard

def generate_vertical_stripe_heightmap(mask_region: np.ndarray, stripe_size: int = 8, height: float = 1.0) -> np.ndarray:
    """
    Generates a heightmap with vertical stripes within the mask region.

    Args:
        mask_region: A 2D boolean numpy array where True indicates the region for the stripes.
        stripe_size: The width of each stripe.
        height: The value to assign to the 'high' stripes.

    Returns:
        A 2D numpy array of the same shape as mask_region, with vertical stripes
        applied within the mask, and zeros elsewhere.
    """
    if mask_region.ndim != 2 or mask_region.dtype != bool:
        raise ValueError("mask_region must be a 2D boolean numpy array.")

    h, w = mask_region.shape
    stripes = np.zeros((h, w), dtype=np.float32)

    # Vectorized approach for vertical stripes
    _, x_idx = np.indices(mask_region.shape)
    stripe_pattern = ((x_idx // stripe_size) % 2 == 0).astype(np.float32) * height
    stripes[mask_region] = stripe_pattern[mask_region]

    return stripes

def generate_horizontal_stripe_heightmap(mask_region: np.ndarray, stripe_size: int = 8, height: float = 1.0) -> np.ndarray:
    """
    Generates a heightmap with horizontal stripes within the mask region.

    Args:
        mask_region: A 2D boolean numpy array where True indicates the region for the stripes.
        stripe_size: The height of each stripe.
        height: The value to assign to the 'high' stripes.

    Returns:
        A 2D numpy array of the same shape as mask_region, with horizontal stripes
        applied within the mask, and zeros elsewhere.
    """
    if mask_region.ndim != 2 or mask_region.dtype != bool:
        raise ValueError("mask_region must be a 2D boolean numpy array.")

    h, w = mask_region.shape
    stripes = np.zeros((h, w), dtype=np.float32)

    # Vectorized approach for horizontal stripes
    y_idx, _ = np.indices(mask_region.shape)
    stripe_pattern = ((y_idx // stripe_size) % 2 == 0).astype(np.float32) * height
    stripes[mask_region] = stripe_pattern[mask_region]

    return stripes

def generate_heightmap_by_texture_type(mask_region: np.ndarray, texture_type: str = "checkerboard", feature_size: int = 8, height: float = 1.0) -> np.ndarray:
    """
    Generates a heightmap with the specified texture pattern within the mask region.

    Args:
        mask_region: A 2D boolean numpy array where True indicates the region for the texture.
        texture_type: The type of texture to generate ("checkerboard", "vertical_stripes", 
                     "horizontal_stripes", or "auto").
        feature_size: The size of the feature (checker size or stripe width).
        height: The value to assign to the 'high' areas of the texture.

    Returns:
        A 2D numpy array of the same shape as mask_region, with the specified texture pattern
        applied within the mask, and zeros elsewhere.
    """
    if texture_type == "checkerboard":
        return generate_checkerboard_heightmap(mask_region, feature_size, height)
    elif texture_type == "vertical_stripes":
        return generate_vertical_stripe_heightmap(mask_region, feature_size, height)
    elif texture_type == "horizontal_stripes":
        return generate_horizontal_stripe_heightmap(mask_region, feature_size, height)
    elif texture_type == "auto":
        # TODO: For now, just use checkerboard for "auto"
        return generate_checkerboard_heightmap(mask_region, feature_size, height)
    else:
        # Default to checkerboard if an invalid type is specified
        logger.warning(f"Unknown texture type '{texture_type}'. Using checkerboard instead.")
        return generate_checkerboard_heightmap(mask_region, feature_size, height)

def decode_base64_image(base64_string: str) -> np.ndarray | None:
    """Decodes a Base64 string into an NumPy array (RGB format)."""
    try:
        # Remove data URI prefix if present
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]
        img_bytes = base64.b64decode(base64_string)
        pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return np.array(pil_image)
    except Exception as e:
        logger.error(f"Error decoding Base64 image: {e}", exc_info=True)
        return None

def encode_mask_to_base64_png(mask: np.ndarray) -> str | None:
    """Encodes a boolean mask numpy array to a Base64 PNG string with transparency."""
    if mask.ndim != 2:
        logger.error(f"Mask must be 2D, but got shape {mask.shape}")
        return None
    try:
        mask_bool = mask.astype(bool)
        h, w = mask_bool.shape
        # Create a 4-channel BGRA image (OpenCV uses BGRA)
        bgra_mask = np.zeros((h, w, 4), dtype=np.uint8) # Transparent background
        # Set foreground pixels to a visible color (e.g., blue) and opaque
        bgra_mask[mask_bool] = [255, 0, 0, 255] # Blue: B=255, G=0, R=0, Alpha=255

        # Encode the BGRA mask to PNG format in memory
        success, buffer = cv2.imencode('.png', bgra_mask)
        if not success:
            logger.error("Failed to encode mask to PNG format.")
            return None

        # Encode the PNG buffer to Base64 string
        png_base64 = base64.b64encode(buffer).decode('utf-8')
        return png_base64
    except Exception as e:
        logger.error(f"Error encoding mask to Base64 PNG: {e}", exc_info=True)
        return None
