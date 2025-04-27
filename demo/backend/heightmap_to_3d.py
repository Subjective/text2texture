import argparse
import numpy as np
from stl import mesh
from PIL import Image

def write_ply(filename, vertices, faces, vertex_colors):
    """
    Write an ASCII PLY file with per-vertex colors.
    """
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
            r, g, b = vertex_colors[i]
            f.write("{:.6f} {:.6f} {:.6f} {} {} {}\n".format(v[0], v[1], v[2],
                                                               int(r), int(g), int(b)))
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
    and modify its top surface using a heightmap.

    - The bottom of the block is at Z = base_height.
    - The nominal top (before modification) is at Z = base_height + block_thickness.
    - If mode is 'protrude', white pixels (value=1) will raise the top by up to depth.
    - If mode is 'carve', white pixels (value=1) will lower the top by up to depth.

    Optionally, if a color_reference image is provided (and matches the heightmap dimensions),
    its RGB values are applied to the top face vertices. In that case the model is exported as
    a PLY file; otherwise, an STL file is generated.

    :param heightmap_path: Path to the grayscale heightmap image.
    :param output_path: Output file path (STL if no color, PLY if colored).
    :param block_width: X dimension of the block.
    :param block_length: Y dimension of the block.
    :param block_thickness: Base thickness (height) of the block.
    :param depth: Maximum extra height for protruding or carving.
    :param base_height: Z offset for the bottom of the block.
    :param mode: 'protrude' (default) to raise the top, or 'carve' to cut into the block.
    :param invert: If True, invert the heightmap (swap black and white).
    :param color_reference: Optional path to a reference color image (must match heightmap dimensions).
    """
    # 1) Load heightmap as grayscale and normalize to [0, 1]
    img = Image.open(heightmap_path).convert('L')
    width_px, height_px = img.size
    pixels = np.array(img, dtype=np.float32) / 255.0
    if invert:
        pixels = 1.0 - pixels

    # Optionally load the reference image for vertex colors.
    use_color = False
    if color_reference:
        ref_img = Image.open(color_reference).convert('RGB')
        ref_pixels = np.array(ref_img, dtype=np.uint8)
        if ref_pixels.shape[0] != height_px or ref_pixels.shape[1] != width_px:
            raise ValueError("Reference image dimensions do not match the heightmap.")
        use_color = True

    # 2) Create vertices for the top (modified by heightmap) and bottom (flat)
    num_vertices_top = width_px * height_px
    num_vertices_bottom = width_px * height_px
    total_vertices = num_vertices_top + num_vertices_bottom

    vertices = np.zeros((total_vertices, 3), dtype=np.float32)
    vertex_colors = np.zeros((total_vertices, 3), dtype=np.uint8) if use_color else None

    def vertex_index(x, y, is_top=True):
        """Return index of the vertex at (x,y) in the top or bottom layer."""
        base = 0 if is_top else num_vertices_top
        return base + (y * width_px + x)

    base_top = base_height + block_thickness
    # Fill top vertices (modified by heightmap and optionally colored)
    for y in range(height_px):
        for x in range(width_px):
            X_real = (x / (width_px - 1)) * block_width
            Y_real = (y / (height_px - 1)) * block_length
            # Adjust top Z coordinate based on the selected mode.
            z_top = modify_top_z(base_top, pixels[y, x], depth, mode)
            idx = vertex_index(x, y, is_top=True)
            vertices[idx] = [X_real, Y_real, z_top]
            if use_color:
                vertex_colors[idx] = ref_pixels[y, x]

    # Fill bottom vertices (flat at base_height; assign a default color if using color)
    default_bottom_color = np.array([200, 200, 200], dtype=np.uint8)
    for y in range(height_px):
        for x in range(width_px):
            X_real = (x / (width_px - 1)) * block_width
            Y_real = (y / (height_px - 1)) * block_length
            z_bottom = base_height
            idx = vertex_index(x, y, is_top=False)
            vertices[idx] = [X_real, Y_real, z_bottom]
            if use_color:
                vertex_colors[idx] = default_bottom_color

    # 3) Create faces (triangles) for top, bottom, and side walls.
    faces = []
    def add_square(v1, v2, v3, v4):
        faces.append([v1, v2, v3])
        faces.append([v3, v4, v1])

    for y in range(height_px - 1):
        for x in range(width_px - 1):
            # Top face
            v1_top = vertex_index(x,     y,     True)
            v2_top = vertex_index(x + 1, y,     True)
            v3_top = vertex_index(x + 1, y + 1, True)
            v4_top = vertex_index(x,     y + 1, True)
            add_square(v1_top, v2_top, v3_top, v4_top)

            # Bottom face (reverse winding for outward normals)
            v1_bot = vertex_index(x,     y,     False)
            v2_bot = vertex_index(x + 1, y,     False)
            v3_bot = vertex_index(x + 1, y + 1, False)
            v4_bot = vertex_index(x,     y + 1, False)
            add_square(v1_bot, v4_bot, v3_bot, v2_bot)

    # Side walls (connect top edge to bottom edge)
    for y in range(height_px - 1):
        for side_x in [0, width_px - 1]:
            v_top1 = vertex_index(side_x, y,     True)
            v_top2 = vertex_index(side_x, y + 1, True)
            v_bot1 = vertex_index(side_x, y,     False)
            v_bot2 = vertex_index(side_x, y + 1, False)
            add_square(v_top1, v_top2, v_bot2, v_bot1)
    for x in range(width_px - 1):
        for side_y in [0, height_px - 1]:
            v_top1 = vertex_index(x,     side_y, True)
            v_top2 = vertex_index(x + 1, side_y, True)
            v_bot1 = vertex_index(x,     side_y, False)
            v_bot2 = vertex_index(x + 1, side_y, False)
            add_square(v_top1, v_top2, v_bot2, v_bot1)

    faces = np.array(faces, dtype=np.int32)

    # 4) Export the model.
    if use_color:
        print("Color reference provided – exporting as a PLY file with vertex colors.")
        write_ply(output_path, vertices, faces, vertex_colors)
    else:
        output_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
        for i, f in enumerate(faces):
            for j in range(3):
                output_mesh.vectors[i][j] = vertices[f[j]]
        output_mesh.save(output_path)
    print(f"Saved model to: {output_path}")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Generate a block with a modified top face from a heightmap, either by protruding or carving."
    )
    parser.add_argument("heightmap_path", help="Path to the grayscale heightmap image.")
    parser.add_argument("output_path", help="Output file path (STL if no color, PLY if colored).")
    parser.add_argument("--block_width", type=float, default=100.0, help="X dimension of the block (default: 100).")
    parser.add_argument("--block_length", type=float, default=100.0, help="Y dimension of the block (default: 100).")
    parser.add_argument("--block_thickness", type=float, default=10.0, help="Base thickness of the block (default: 10).")
    parser.add_argument("--depth", type=float, default=5.0, help="Maximum extra height (default: 5).")
    parser.add_argument("--base_height", type=float, default=0.0, help="Z offset for the bottom of the block (default: 0).")
    parser.add_argument("--mode", choices=["protrude", "carve"], default="protrude",
                        help="Mode for top modification: 'protrude' to raise or 'carve' to cut into the block (default: protrude).")
    parser.add_argument("--invert", action="store_true", help="Invert the heightmap (swap black and white).")
    parser.add_argument("--color_reference", help="Path to a reference color image (must match heightmap dimensions).")
    args = parser.parse_args()

    # Run the function with provided arguments
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
