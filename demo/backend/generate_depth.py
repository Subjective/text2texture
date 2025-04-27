import torch
from PIL import Image
import argparse
import numpy as np
import matplotlib

# Load ZoeDepth model using Torch Hub
repo = "isl-org/ZoeDepth"
model_zoe_n = torch.hub.load(repo, "ZoeD_N", pretrained=True)

# Set the device
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
model_zoe_n = model_zoe_n.to(DEVICE)

def get_grayscale_depth(image_path, output_path):
    """
    Get the grayscale depth of an image, invert it, and save it to a file.

    Args:
        image_path (str): Path to the input image.
        output_path (str): Path to save the inverted grayscale depth image.
    """
    # Load the image
    image = Image.open(image_path).convert("RGB")

    # Infer depth using ZoeDepth model
    depth_tensor = model_zoe_n.infer_pil(image, output_type="tensor")

    # Colorize the depth map in grayscale
    grayscale_depth = colorize(depth_tensor, cmap="gray")

    # Invert the grayscale depth
    inverted_depth = 255 - grayscale_depth  # Invert grayscale values

    # Save the inverted grayscale depth image
    Image.fromarray(inverted_depth[:, :, :3]).save(output_path)
    print(f"Inverted grayscale depth image saved at {output_path}")

def colorize(value, vmin=None, vmax=None, cmap='gray_r', invalid_val=-99, invalid_mask=None, background_color=(128, 128, 128, 255), gamma_corrected=False, value_transform=None):
    """Converts a depth map to a color image.

    Args:
        value (torch.Tensor, numpy.ndarry): Input depth map. Shape: (H, W) or (1, H, W) or (1, 1, H, W). All singular dimensions are squeezed
        vmin (float, optional): vmin-valued entries are mapped to start color of cmap. If None, value.min() is used. Defaults to None.
        vmax (float, optional):  vmax-valued entries are mapped to end color of cmap. If None, value.max() is used. Defaults to None.
        cmap (str, optional): matplotlib colormap to use. Defaults to 'magma_r'.
        invalid_val (int, optional): Specifies value of invalid pixels that should be colored as 'background_color'. Defaults to -99.
        invalid_mask (numpy.ndarray, optional): Boolean mask for invalid regions. Defaults to None.
        background_color (tuple[int], optional): 4-tuple RGB color to give to invalid pixels. Defaults to (128, 128, 128, 255).
        gamma_corrected (bool, optional): Apply gamma correction to colored image. Defaults to False.
        value_transform (Callable, optional): Apply transform function to valid pixels before coloring. Defaults to None.

    Returns:
        numpy.ndarray, dtype - uint8: Colored depth map. Shape: (H, W, 4)
    """
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()

    value = value.squeeze()
    if invalid_mask is None:
        invalid_mask = value == invalid_val
    mask = np.logical_not(invalid_mask)

    # normalize
    vmin = np.percentile(value[mask],2) if vmin is None else vmin
    vmax = np.percentile(value[mask],85) if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.

    # squeeze last dim if it exists
    # grey out the invalid values

    value[invalid_mask] = np.nan
    cmapper = matplotlib.cm.get_cmap(cmap)
    if value_transform:
        value = value_transform(value)
        # value = value / value.max()
    value = cmapper(value, bytes=True)  # (nxmx4)

    # img = value[:, :, :]
    img = value[...]
    img[invalid_mask] = background_color

    #     return img.transpose((2, 0, 1))
    if gamma_corrected:
        # gamma correction
        img = img / 255
        img = np.power(img, 2.2)
        img = img * 255
        img = img.astype(np.uint8)
    return img

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate inverted grayscale depth map from an image.")
    parser.add_argument("input_path", type=str, help="Path to the input image.")
    parser.add_argument("output_path", type=str, help="Path to save the inverted grayscale depth image.")
    args = parser.parse_args()

    # Run the function with provided arguments
    get_grayscale_depth(args.input_path, args.output_path)
