import os
from stl import mesh # Import for STL mesh operations
import numpy as np # Although not directly used, often needed alongside mesh data

def write_ply(filename, vertices, faces, vertex_colors=None):
    """
    Write an ASCII PLY file with optional per-vertex colors.

    Args:
        filename (str): Path to the output PLY file.
        vertices (np.ndarray): Array of vertex coordinates, shape (N, 3).
        faces (np.ndarray): Array of face indices, shape (M, 3).
        vertex_colors (np.ndarray, optional): Array of vertex colors (RGB), shape (N, 3), dtype uint8.
                                              If None, default gray colors will be written if format requires.
                                              Defaults to None.
    """
    num_vertices = len(vertices)
    num_faces = len(faces)
    has_colors = vertex_colors is not None

    # Ensure the output directory exists
    output_dir = os.path.dirname(filename)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(filename, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_vertices}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if has_colors:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write(f"element face {num_faces}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

        for i, v in enumerate(vertices):
            if has_colors:
                # Ensure vertex_colors array is valid and long enough
                if i < len(vertex_colors):
                    r, g, b = vertex_colors[i]
                    f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {int(r)} {int(g)} {int(b)}\n")
                else:
                    # Fallback if color array is somehow shorter than vertices
                    f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} 128 128 128\n")
            else:
                # Write only vertex coordinates if no colors provided
                f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        for face in faces:
            # Each face is a triangle (3 vertices)
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")

    print(f"Saved PLY file to: {filename}")

def write_stl(filename, vertices, faces):
    """
    Write a binary STL file.

    Args:
        filename (str): Path to the output STL file.
        vertices (np.ndarray): Array of vertex coordinates, shape (N, 3).
        faces (np.ndarray): Array of face indices, shape (M, 3).
    """
    num_faces = len(faces)
    if num_faces == 0:
        print("Warning: No faces provided to write_stl. Saving empty STL file.")
        # Create an empty mesh and save it
        output_mesh = mesh.Mesh(np.zeros(0, dtype=mesh.Mesh.dtype))
    else:
        # Create the mesh object
        output_mesh = mesh.Mesh(np.zeros(num_faces, dtype=mesh.Mesh.dtype))
        for i, f in enumerate(faces):
            # For each face, assign the 3 vertices from the vertices array
            for j in range(3):
                output_mesh.vectors[i][j] = vertices[f[j]]

    # Ensure the output directory exists
    output_dir = os.path.dirname(filename)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save the mesh to file
    try:
        output_mesh.save(filename)
        print(f"Saved STL file to: {filename}")
    except Exception as e:
        print(f"Error saving STL file to {filename}: {e}")
        raise # Re-raise the exception
