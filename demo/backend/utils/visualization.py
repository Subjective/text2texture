# Adapted from ZoeDepth
import numpy as np
import matplotlib
import torch # Keep torch import for type hinting if used in original signature

def colorize(value, vmin=None, vmax=None, cmap='gray_r', invalid_val=-99, invalid_mask=None, background_color=(128, 128, 128, 255), gamma_corrected=False, value_transform=None):
    """Converts a 2D array (depth map, heatmap, etc.) to a color image using matplotlib colormaps.

    Args:
        value (torch.Tensor | np.ndarray): Input 2D data. Shape: (H, W) or (1, H, W) or (1, 1, H, W).
                                           All singular dimensions are squeezed. If torch.Tensor, it's detached and moved to CPU.
        vmin (float, optional): Value mapped to the start color of cmap. If None, uses 2nd percentile of valid data. Defaults to None.
        vmax (float, optional): Value mapped to the end color of cmap. If None, uses 85th percentile of valid data. Defaults to None.
        cmap (str, optional): Matplotlib colormap name. Defaults to 'gray_r'.
        invalid_val (int, optional): Value indicating invalid pixels. Defaults to -99.
        invalid_mask (np.ndarray, optional): Boolean mask for invalid regions (overrides invalid_val if provided). Defaults to None.
        background_color (tuple[int], optional): RGBA color for invalid pixels. Defaults to (128, 128, 128, 255).
        gamma_corrected (bool, optional): Apply gamma correction (2.2) to the output color image. Defaults to False.
        value_transform (Callable, optional): Function to apply to valid pixel values *before* colormapping. Defaults to None.

    Returns:
        np.ndarray: Colored image as a NumPy array (H, W, 4) with dtype uint8.
    """
    try:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()

        if not isinstance(value, np.ndarray):
             raise TypeError(f"Input 'value' must be a NumPy array or PyTorch Tensor, got {type(value)}")

        value = value.squeeze()
        if value.ndim == 0: # Handle scalar input
             value = value[np.newaxis, np.newaxis]
        elif value.ndim != 2:
             raise ValueError(f"Input value must be 2D after squeezing, but got shape {value.shape}")


        if invalid_mask is None:
            # Ensure invalid_val comparison works correctly with floating point NaNs if they are used
            if np.isnan(invalid_val):
                 invalid_mask = np.isnan(value)
            else:
                 invalid_mask = value == invalid_val
        elif invalid_mask.shape != value.shape:
             raise ValueError(f"Provided invalid_mask shape {invalid_mask.shape} does not match value shape {value.shape}")

        mask = np.logical_not(invalid_mask) # Mask of valid pixels

        # Handle case where mask is all False (all pixels invalid)
        if not np.any(mask):
             # print("Warning: All values are invalid.") # Reduce noise
             # Create an array filled with the background color
             colored_image = np.full((*value.shape, 4), background_color, dtype=np.uint8)
             return colored_image


        # Normalize valid values
        valid_values = value[mask]
        vmin = np.percentile(valid_values, 2) if vmin is None else vmin
        vmax = np.percentile(valid_values, 85) if vmax is None else vmax

        # Ensure vmin and vmax are valid numbers
        if np.isnan(vmin): vmin = np.min(valid_values) if valid_values.size > 0 else 0
        if np.isnan(vmax): vmax = np.max(valid_values) if valid_values.size > 0 else 1
        if np.isinf(vmin) or np.isinf(vmax):
             print(f"Warning: Encountered inf values in vmin/vmax calculation. Clamping might occur.")
             # Attempt to use nanmin/nanmax if percentiles failed badly
             vmin = np.nanmin(valid_values) if np.isinf(vmin) else vmin
             vmax = np.nanmax(valid_values) if np.isinf(vmax) else vmax


        if vmin > vmax:
             # print(f"Warning: vmin ({vmin}) > vmax ({vmax}). Swapping them.") # Reduce noise
             vmin, vmax = vmax, vmin
        if vmin == vmax:
             # Avoid division by zero, map all valid values to 0.5 (mid-colormap)
             normalized_values = np.full_like(value, 0.5, dtype=np.float64)
        else:
             # Normalize to [0, 1], ensuring float division
             normalized_values = (value.astype(np.float64) - vmin) / (vmax - vmin)

        # Apply value transform if provided, only on valid pixels
        if value_transform:
            transformed_valid = value_transform(normalized_values[mask])
            # Ensure transformed values are clipped to [0, 1] for colormapper
            normalized_values[mask] = np.clip(transformed_valid, 0, 1)


        # Get colormap; handle potential errors
        try:
             cmapper = matplotlib.colormaps[cmap]
        except KeyError:
             print(f"Warning: Colormap '{cmap}' not found. Using 'viridis'.")
             cmapper = matplotlib.colormaps['viridis']

        # Apply colormap (expects values in [0, 1])
        # We need to handle the invalid values (masked) separately.
        # Create a temporary array with valid values normalized and invalid values set to NaN
        # so the colormapper ignores them initially.
        value_for_cmap = normalized_values.copy()
        value_for_cmap[invalid_mask] = np.nan # Let cmapper handle NaN initially

        colored_image = cmapper(value_for_cmap, bytes=True)  # Returns (H, W, 4) RGBA uint8

        # Explicitly set background color for invalid pixels
        # Cmapper might assign its 'bad' color to NaNs, we overwrite it.
        colored_image[invalid_mask] = background_color

        if gamma_corrected:
            # gamma correction
            # Operate on a float copy to avoid issues with uint8 calculations
            img_float = colored_image.astype(np.float32) / 255.0
            # Apply gamma correction only to RGB channels, leave alpha untouched
            img_float[..., :3] = np.power(img_float[..., :3], 2.2)
            colored_image = (img_float * 255.0).clip(0, 255).astype(np.uint8)

        return colored_image

    except Exception as e:
        print(f"Error during colorization: {e}")
        # Return a placeholder background image on error
        shape = value.shape if isinstance(value, np.ndarray) else (1,1) # Basic shape if input was bad
        return np.full((*shape, 4), background_color, dtype=np.uint8)
