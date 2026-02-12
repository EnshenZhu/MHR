"""
This script demonstrates how to use the Momentum Human Rig (MHR) model and
visualize the resulting mesh using Open3D.  It is based on the original
``demo.py`` that ships with the MHR repository but adds an Open3D viewer so you
can interact with the mesh directly from Python instead of only exporting a
PLY file.

The original ``demo.py`` uses the ``trimesh`` library to create a mesh from
the vertices returned by the MHR model and then writes it to disk.  Open3D
provides a convenient ``draw_geometries`` function which takes a list of
geometry objects (point clouds, triangle meshes or images) and renders them
together【732350169453825†L68-L75】.  To construct an Open3D ``TriangleMesh`,` you
pass the vertex positions and face indices as ``Vector3dVector`` and
``Vector3iVector`` respectively【535057006977722†L980-L990】.  Calling
``compute_vertex_normals()`` on the mesh computes per–vertex normals for
correct shading before rendering【535057006977722†L1044-L1057】.

Usage:

    python demo_open3d.py

Requirements:
    pip install open3d trimesh

"""

import torch
import open3d as o3d
from mhr.mhr import MHR
import trimesh


def _prepare_input_data(batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate random identity, pose and facial expression coefficients.

    Args:
        batch_size: Number of independent samples to generate.

    Returns:
        A tuple ``(identity_coeffs, model_parameters, face_expr_coeffs)``.
    """
    identity_coeffs = 0.8 * torch.randn(batch_size, 45).cpu()
    model_parameters = 0.2 * (torch.rand(batch_size, 204) - 0.5).cpu()
    face_expr_coeffs = 0.05 * torch.randn(batch_size, 72).cpu()
    return identity_coeffs, model_parameters, face_expr_coeffs


def visualize_with_open3d(vertices: torch.Tensor, faces: torch.Tensor) -> None:
    """Create an Open3D triangle mesh from vertices and faces and display it.

    Open3D's ``TriangleMesh`` can be constructed by passing the vertices and
    triangle indices as Open3D utility vectors【535057006977722†L980-L990】.  Before
    rendering we compute vertex normals so that lighting works correctly【535057006977722†L1044-L1057】.  The
    ``draw_geometries`` function then opens an interactive viewer window where
    the mesh can be rotated, scaled and translated using the mouse and common
    keyboard shortcuts【732350169453825†L68-L75】.

    Args:
        vertices: A NumPy-like array of shape ``(V, 3)`` representing vertex
            positions in metres.
        faces: A NumPy-like array of shape ``(F, 3)`` with integer indices
            defining the triangular faces of the mesh.
    """
    # Convert PyTorch tensors to NumPy arrays on the CPU.
    verts_np = vertices.cpu().numpy()
    faces_np = faces.astype(int)

    # Build an Open3D triangle mesh from vertices and faces.
    mesh_o3d = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(verts_np),
        triangles=o3d.utility.Vector3iVector(faces_np),
    )
    # Compute vertex normals for better shading.
    mesh_o3d.compute_vertex_normals()

    # Optionally colour the mesh uniformly (grey) to make shading clearer.
    mesh_o3d.paint_uniform_color([0.7, 0.7, 0.7])

    # Create a coordinate frame to show axes for reference.
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)

    # Draw the mesh and axes together.  This call blocks until the user closes
    # the viewer window.
    o3d.visualization.draw_geometries([mesh_o3d, axes])


def run() -> None:
    """Generate a test mesh from the MHR model and visualize it with Open3D."""
    # Load MHR model with Level of Detail 1 on CPU
    mhr_model = MHR.from_files(device=torch.device("cpu"), lod=1)

    batch_size = 1
    identity_coeffs, model_parameters, face_expr_coeffs = _prepare_input_data(batch_size)

    with torch.no_grad():
        verts, _ = mhr_model(identity_coeffs, model_parameters, face_expr_coeffs)

    # verts has shape (batch_size, N_vertices, 3); take the first sample
    verts0 = verts[0]
    # faces is provided by the loaded character inside the model
    faces = mhr_model.character.mesh.faces

    # Save as PLY using trimesh for comparison
    mesh = trimesh.Trimesh(vertices=verts0.numpy(), faces=faces, process=False)
    output_mesh_path = "./test_open3d.ply"
    mesh.export(output_mesh_path)
    print(f"Saved example MHR mesh to {output_mesh_path}")

    # Visualize the mesh with Open3D
    visualize_with_open3d(verts0, faces)


def compare_with_torchscript_model() -> None:
    """Compare the Python model to the TorchScript model as in the original demo."""
    print("Comparing MHR model with TorchScripted model.")
    scripted_model = torch.jit.load("./assets/mhr_model.pt")
    mhr_model = MHR.from_files(device=torch.device("cpu"), lod=1)

    batch_size = 16
    identity_coeffs, model_parameters, face_expr_coeffs = _prepare_input_data(batch_size)

    with torch.no_grad():
        verts, _ = mhr_model(identity_coeffs, model_parameters, face_expr_coeffs)
        verts_ts, _ = scripted_model(identity_coeffs, model_parameters, face_expr_coeffs)
        print(f"Average per‑vertex offsets {torch.abs(verts - verts_ts).mean()} cm.")
        print(f"Max per‑vertex offsets {torch.abs(verts - verts_ts).max()} cm.")


if __name__ == "__main__":
    run()
    compare_with_torchscript_model()