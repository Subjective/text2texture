import argparse
import numpy as np
from stl import mesh
from PIL import Image
import os # Added for file extension check

def write_ply(filename, vertices, faces, vertex_colors):
    """
    Write an ASCII PLY file with per-vertex colors.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("element vertex {}\n".format(len(vertices)))
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("element face {}\n".format(len(faces)))
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for i, v in enumerate(vertices):
            # Handle potential None colors if logic error occurs
            if vertex_colors is not None and i < len(vertex_colors):
                r, g, b = vertex_colors[i]
                f.write("{:.6f} {:.6f} {:.6f} {} {} {}\n".format(v[0], v[1], v[2],
                                                               int(r), int(g), int(b)))
            else: # Fallback if color is missing for some reason
                 f.write("{:.6f} {:.6f} {:.6f} 128 128 128\n".format(v[0], v[1], v[2]))

        for face in faces:
            # Each face is a triangle (3 vertices)
            f.write("3 {} {} {}\n".format(face[0], face[1], face[2]))

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

    [...] # Rest of the docstring
    """
    # 1) Load heightmap as grayscale and normalize to [0, 1]
    img = Image.open(heightmap_path).convert('L')
    width_px, height_px = img.size
    if width_px < 2 or height_px < 2:
        raise ValueError("Heightmap must be at least 2x2 pixels.")
    pixels = np.array(img, dtype=np.float32) / 255.0
    if invert:
        pixels = 1.0 - pixels

    # Optionally load the reference image for vertex colors.
    use_color = False
    if color_reference:
        try:
            ref_img = Image.open(color_reference).convert('RGB')
            ref_pixels = np.array(ref_img, dtype=np.uint8)
            if ref_pixels.shape[0] != height_px or ref_pixels.shape[1] != width_px:
                raise ValueError("Reference image dimensions do not match the heightmap.")
            use_color = True
            # Automatically set output format to PLY if color is used
            if not output_path.lower().endswith(".ply"):
                 print("Warning: Color reference provided, changing output format to PLY.")
                 output_path = os.path.splitext(output_path)[0] + ".ply"

        except FileNotFoundError:
             print(f"Warning: Color reference file not found at {color_reference}. Generating without color.")
             use_color = False
        except Exception as e:
             print(f"Warning: Error loading color reference: {e}. Generating without color.")
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
            X_real = (x / (width_px - 1)) * block_width
            Y_real = ((height_px - 1 - y) / (height_px - 1)) * block_length # Y=0 pixel is max Y coord
            z_top = modify_top_z(base_top, pixels[y, x], depth, mode)
            idx = vertex_index(x, y, is_top=True)
            vertices[idx] = [X_real, Y_real, z_top]
            if use_color:
                vertex_colors[idx] = ref_pixels[y, x]

    # Fill bottom vertices
    default_bottom_color = np.array([128, 128, 128], dtype=np.uint8)
    for y in range(height_px):
        for x in range(width_px):
            X_real = (x / (width_px - 1)) * block_width
            Y_real = ((height_px - 1 - y) / (height_px - 1)) * block_length
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
    # [ This part remains the same - correctly generating top/bottom faces ]
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
        # Left side (x=0). Correct.
        v1_t = vertex_index(0, y,     True)
        v1_b = vertex_index(0, y,     False)
        v4_t = vertex_index(0, y + 1, True)
        v4_b = vertex_index(0, y + 1, False)
        add_square(v1_t, v1_b, v4_b, v4_t) # Correct order for Left

        # Right side (x = width_px - 1). Needs correction.
        v2_t = vertex_index(width_px - 1, y,     True)
        v2_b = vertex_index(width_px - 1, y,     False)
        v3_t = vertex_index(width_px - 1, y + 1, True)
        v3_b = vertex_index(width_px - 1, y + 1, False)
        # Flip the winding empirically by swapping 2nd and 4th args
        add_square(v2_t, v3_t, v3_b, v2_b) # *** EMPIRICAL FIX ORDER ***

    # Side walls along Y (Near: y=height_px-1 -> Y_real=0, Far: y=0 -> Y_real=max)
    for x in range(width_px - 1):
        # Near side (y = height_px - 1). Correct.
        v4_t = vertex_index(x,     height_px - 1, True)
        v4_b = vertex_index(x,     height_px - 1, False)
        v3_t = vertex_index(x + 1, height_px - 1, True)
        v3_b = vertex_index(x + 1, height_px - 1, False)
        add_square(v4_t, v4_b, v3_b, v3_t) # Correct order for Near

        # Far side (y = 0). Correct.
        v1_t = vertex_index(x,     0, True)
        v1_b = vertex_index(x,     0, False)
        v2_t = vertex_index(x + 1, 0, True)
        v2_b = vertex_index(x + 1, 0, False)
        add_square(v2_t, v2_b, v1_b, v1_t) # Correct order for Far

    faces = np.array(faces, dtype=np.int32)

    # 4) Export the model.
    output_ext = os.path.splitext(output_path)[1].lower()

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
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        output_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
        for i, f in enumerate(faces):
            for j in range(3):
                output_mesh.vectors[i][j] = vertices[f[j]]
        output_mesh.save(output_path)

    print(f"Saved model to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a block with a modified top face from a heightmap." # Simplified description
    )
    parser.add_argument("heightmap_path", help="Path to the grayscale heightmap image.")
    parser.add_argument("output_path", help="Output file path (.stl or .ply).")
    parser.add_argument("--block_width", type=float, default=100.0, help="X dimension (default: 100).")
    parser.add_argument("--block_length", type=float, default=100.0, help="Y dimension (default: 100).")
    parser.add_argument("--block_thickness", type=float, default=10.0, help="Base thickness (default: 10).")
    parser.add_argument("--depth", type=float, default=5.0, help="Max protrusion/carve depth (default: 5).")
    parser.add_argument("--base_height", type=float, default=0.0, help="Z offset for bottom (default: 0).")
    parser.add_argument("--mode", choices=["protrude", "carve"], default="protrude",
                        help="Mode: 'protrude' or 'carve' (default: protrude).")
    parser.add_argument("--invert", action="store_true", help="Invert heightmap.")
    parser.add_argument("--color_reference", help="Path to color image (forces PLY output).")
    args = parser.parse_args()

    try:
        generate_block_from_heightmap(
            heightmap_path=args.heightmap_path,
            output_path=args.output_path,
            block_width=args.block_width,
            block_length=args.block_length,
            block_thickness=args.block_thickness,
            depth=args.depth,
            base_height=args.base_height,
            mode=args.mode,
            invert=args.invert,
            color_reference=args.color_reference
        )
    except FileNotFoundError as e:
        print(f"Error: Input file not found - {e}")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
