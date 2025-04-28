import numpy as np
from stl import mesh
from PIL import Image
import os
from utils.mesh_io import write_ply, write_stl # Use absolute import from backend perspective

def modify_top_z(base_top, pixel_value, depth, mode):
    """
    Compute the top Z coordinate for a given pixel value and mode.

    :param base_top: The nominal top (base_height + block_thickness)
    :param pixel_value: Grayscale value [0, 1]
    :param depth: Maximum extra height (used positively for protrude, negatively for carve)
    :param mode: Either 'protrude' or 'carve'
    :return: Adjusted Z coordinate.
    """
    if mode == "protrude":
        return base_top + (pixel_value * depth)
    elif mode == "carve":
        return base_top - (pixel_value * depth)
    else:
        raise ValueError("Invalid mode: choose 'protrude' or 'carve'.")

def generate_block_from_heightmap(
    heightmap_path,
    output_path,
    block_width=100.0,
    block_length=100.0,
    block_thickness=10.0,
    depth=5.0,
    base_height=0.0,
    mode="protrude",
    invert=False,
    color_reference=None
):
    """
    Generate a rectangular block (of size block_width x block_length x block_thickness)
    and modify its top surface using a heightmap. Corrected face winding.

    Args:
        heightmap_path (str): Path to the grayscale heightmap image.
        output_path (str): Path to save the output model (.stl or .ply).
        block_width (float): Width of the block (X dimension).
        block_length (float): Length of the block (Y dimension).
        block_thickness (float): Base thickness of the block (Z dimension).
        depth (float): Maximum depth/height of the heightmap effect.
        base_height (float): Z-coordinate of the bottom face.
        mode (str): 'protrude' (heightmap adds height) or 'carve' (heightmap subtracts height).
        invert (bool): Invert the heightmap values before applying.
        color_reference (str, optional): Path to an RGB image to use for vertex colors.
                                         If provided, output is forced to PLY format.

    Returns:
        None. Saves the generated model to the output_path.

    Raises:
        FileNotFoundError: If the heightmap_path or color_reference (if provided) is not found.
        ValueError: If heightmap dimensions are too small, or mode is invalid.
        Exception: For other image loading or file writing errors.
    """
    # 1) Load heightmap as grayscale and normalize to [0, 1]
    try:
        img = Image.open(heightmap_path).convert('L')
    except FileNotFoundError:
        print(f"Error: Heightmap file not found at {heightmap_path}")
        raise
    except Exception as e:
        print(f"Error opening heightmap {heightmap_path}: {e}")
        raise

    width_px, height_px = img.size
    if width_px < 2 or height_px < 2:
        raise ValueError("Heightmap must be at least 2x2 pixels.")
    pixels = np.array(img, dtype=np.float32) / 255.0
    if invert:
        pixels = 1.0 - pixels

    # Optionally load the reference image for vertex colors.
    use_color = False
    ref_pixels = None # Initialize ref_pixels
    if color_reference:
        try:
            ref_img = Image.open(color_reference).convert('RGB')
            ref_pixels = np.array(ref_img, dtype=np.uint8)
            if ref_pixels.shape[0] != height_px or ref_pixels.shape[1] != width_px:
                raise ValueError("Reference image dimensions do not match the heightmap.")
            use_color = True
            # Automatically set output format to PLY if color is used
            output_ext = os.path.splitext(output_path)[1].lower()
            if output_ext != ".ply":
                 print("Warning: Color reference provided, changing output format to PLY.")
                 output_path = os.path.splitext(output_path)[0] + ".ply"

        except FileNotFoundError:
             print(f"Warning: Color reference file not found at {color_reference}. Generating without color.")
             use_color = False
        except ValueError as e: # Catch specific dimension mismatch error
             print(f"Warning: {e}. Generating without color.")
             use_color = False
        except Exception as e:
             print(f"Warning: Error loading color reference '{color_reference}': {e}. Generating without color.")
             use_color = False


    # 2) Create vertices for the top (modified by heightmap) and bottom (flat)
    num_vertices_top = width_px * height_px
    num_vertices_bottom = width_px * height_px
    total_vertices = num_vertices_top + num_vertices_bottom

    vertices = np.zeros((total_vertices, 3), dtype=np.float32)
    vertex_colors = np.zeros((total_vertices, 3), dtype=np.uint8) if use_color else None

    def vertex_index(x, y, is_top=True):
        """Return index of the vertex at (x,y) in the top or bottom layer."""
        base = 0 if is_top else num_vertices_top
        # Clamp indices to avoid out-of-bounds (already handled by loop ranges)
        # x_clamped = min(x, width_px - 1)
        # y_clamped = min(y, height_px - 1)
        return base + (y * width_px + x) # Use direct y, x as loops are correct

    base_top = base_height + block_thickness
    # Fill top vertices
    for y in range(height_px):
        for x in range(width_px):
            X_real = (x / (width_px - 1)) * block_width if width_px > 1 else block_width / 2
            Y_real = ((height_px - 1 - y) / (height_px - 1)) * block_length if height_px > 1 else block_length / 2 # Y=0 pixel is max Y coord
            z_top = modify_top_z(base_top, pixels[y, x], depth, mode)
            idx = vertex_index(x, y, is_top=True)
            vertices[idx] = [X_real, Y_real, z_top]
            if use_color and ref_pixels is not None: # Check ref_pixels exists
                vertex_colors[idx] = ref_pixels[y, x]

    # Fill bottom vertices
    default_bottom_color = np.array([128, 128, 128], dtype=np.uint8)
    for y in range(height_px):
        for x in range(width_px):
            X_real = (x / (width_px - 1)) * block_width if width_px > 1 else block_width / 2
            Y_real = ((height_px - 1 - y) / (height_px - 1)) * block_length if height_px > 1 else block_length / 2
            z_bottom = base_height
            idx = vertex_index(x, y, is_top=False)
            vertices[idx] = [X_real, Y_real, z_bottom]
            if use_color:
                vertex_colors[idx] = default_bottom_color

    # 3) Create faces (triangles) for top, bottom, and side walls.
    faces = []
    def add_square(v1, v2, v3, v4):
        """Adds two triangles (v1,v2,v3) and (v3,v4,v1) to the faces list."""
        faces.append([v1, v2, v3]) # Triangle 1
        faces.append([v3, v4, v1]) # Triangle 2

    # --- Top and Bottom Faces ---
    if width_px > 1 and height_px > 1: # Only create faces if grid exists
        for y in range(height_px - 1):
            for x in range(width_px - 1):
                v1_top = vertex_index(x,     y,     True)
                v2_top = vertex_index(x + 1, y,     True)
                v3_top = vertex_index(x + 1, y + 1, True)
                v4_top = vertex_index(x,     y + 1, True)
                v1_bot = vertex_index(x,     y,     False)
                v2_bot = vertex_index(x + 1, y,     False)
                v3_bot = vertex_index(x + 1, y + 1, False)
                v4_bot = vertex_index(x,     y + 1, False)
                add_square(v1_top, v4_top, v3_top, v2_top) # Top face
                add_square(v1_bot, v2_bot, v3_bot, v4_bot) # Bottom face

        # --- Side Walls ---
        # Side walls along X (Left: x=0, Right: x=width_px-1)
        for y in range(height_px - 1):
            # Left side (x=0).
            v1_t = vertex_index(0, y,     True)
            v1_b = vertex_index(0, y,     False)
            v4_t = vertex_index(0, y + 1, True)
            v4_b = vertex_index(0, y + 1, False)
            add_square(v1_t, v1_b, v4_b, v4_t) # Correct order for Left

            # Right side (x = width_px - 1).
            v2_t = vertex_index(width_px - 1, y,     True)
            v2_b = vertex_index(width_px - 1, y,     False)
            v3_t = vertex_index(width_px - 1, y + 1, True)
            v3_b = vertex_index(width_px - 1, y + 1, False)
            add_square(v2_t, v3_t, v3_b, v2_b) # Correct order for Right

        # Side walls along Y (Near: y=height_px-1 -> Y_real=0, Far: y=0 -> Y_real=max)
        for x in range(width_px - 1):
            # Near side (y = height_px - 1).
            v4_t = vertex_index(x,     height_px - 1, True)
            v4_b = vertex_index(x,     height_px - 1, False)
            v3_t = vertex_index(x + 1, height_px - 1, True)
            v3_b = vertex_index(x + 1, height_px - 1, False)
            add_square(v4_t, v4_b, v3_b, v3_t) # Correct order for Near

            # Far side (y = 0).
            v1_t = vertex_index(x,     0, True)
            v1_b = vertex_index(x,     0, False)
            v2_t = vertex_index(x + 1, 0, True)
            v2_b = vertex_index(x + 1, 0, False)
            add_square(v2_t, v2_b, v1_b, v1_t) # Correct order for Far

    faces = np.array(faces, dtype=np.int32)

    # 4) Export the model.
    output_ext = os.path.splitext(output_path)[1].lower()

    # Ensure the output directory exists before trying to save
    output_dir = os.path.dirname(output_path)
    if output_dir: # Only create if not saving in current dir
        os.makedirs(output_dir, exist_ok=True)

    try:
        if use_color:
            if output_ext != ".ply":
                print(f"Warning: Output format is '{output_ext}', but color requires PLY. Saving as PLY.")
                output_path = os.path.splitext(output_path)[0] + ".ply"
            print("Color reference provided – exporting as a PLY file with vertex colors.")
            write_ply(output_path, vertices, faces, vertex_colors)
        elif output_ext == ".ply":
             print("No color reference, but output is PLY. Assigning default colors.")
             # Create default colors if PLY output is requested without a color map
             vertex_colors = np.full((total_vertices, 3), 128, dtype=np.uint8) # Gray
             # Optionally color top face differently
             if width_px > 0 and height_px > 0: # Check dimensions before indexing
                 for y in range(height_px):
                     for x in range(width_px):
                         idx = vertex_index(x, y, is_top=True)
                         vertex_colors[idx] = [200, 200, 200] # Light gray for top
             write_ply(output_path, vertices, faces, vertex_colors)
        else: # Default to STL if no color and not PLY
            if output_ext != ".stl":
                 print(f"Warning: Output format is '{output_ext}'. Saving as STL.")
                 output_path = os.path.splitext(output_path)[0] + ".stl"
            print("No color reference – exporting as an STL file.")
            # Call the utility function to write the STL file
            write_stl(output_path, vertices, faces)

        print(f"Saved model to: {output_path}")

    except Exception as e:
        print(f"Error saving model to {output_path}: {e}")
        raise # Re-raise the exception after logging
