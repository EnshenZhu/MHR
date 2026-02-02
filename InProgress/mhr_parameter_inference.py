"""
Utility script for inferring MHR parameters from a human mesh.

This example demonstrates how to recover identity, pose (model) and facial
expression parameters of the Momentum Human Rig (MHR) from a mesh that is
compatible with the SMPL/SMPL‑X topology.  The MHR repository provides
`tools/mhr_smpl_conversion` which can convert SMPL or SMPL‑X parameters
or vertices into MHR parameters via a two‑stage process: first the mesh
is retargeted to the MHR template using barycentric interpolation, and
then a PyTorch optimisation fits the MHR model to the target vertices to
recover the latent parameters.

The code below illustrates how to load a SMPL‑X mesh from disk, set up
the necessary body models and call the conversion routine.  It is assumed
that you have downloaded the MHR assets (see the main MHR `README.md`
for instructions) and that you have a local copy of the SMPL‑X model
files.  The mesh must share the same vertex ordering as SMPL or SMPL‑X;
arbitrary meshes need to be fitted to SMPL first before conversion.

Note: running this script on GPU requires a recent PyTorch build and a
CUDA‑capable device.  Remove `.cuda()` calls or set `device="cpu"` to
run on CPU only.  In a headless environment, make sure to install the
`trimesh` and `smplx` packages.
"""

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import trimesh
import smplx

from mhr.mhr import MHR
from tools.mhr_smpl_conversion.conversion import Conversion


def load_mesh_vertices(mesh_path: Path) -> np.ndarray:
    """Load vertices from a mesh file into a numpy array.

    Parameters
    ----------
    mesh_path: Path
        Path to an OBJ or PLY file containing a SMPL/SMPL‑X mesh.

    Returns
    -------
    np.ndarray
        Array of shape (1, V, 3) containing the mesh vertices.  The
        conversion API supports batches of frames, so a single mesh is
        wrapped in a batch dimension.
    """
    mesh = trimesh.load(mesh_path, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Expected a single mesh in {mesh_path}, got {type(mesh)}")
    # Ensure vertices are float32 and in a batch of size 1
    verts = mesh.vertices.astype(np.float32)[None, ...]
    return verts


def infer_mhr_parameters(
    smpl_vertices: np.ndarray,
    smpl_model_path: Path,
    mhr_lod: int = 1,
    smpl_model_type: str = "smplx",
    gender: str = "neutral",
    device: str = "cpu",
) -> Dict[str, np.ndarray]:
    """Estimate MHR identity, pose and expression parameters from SMPL/SMPL‑X vertices.

    Parameters
    ----------
    smpl_vertices: np.ndarray
        A batch of SMPL/SMPL‑X vertices with shape (N, V, 3).  The vertex
        ordering must match the official SMPL/SMPL‑X model.  If you only
        have a single mesh, provide shape (1, V, 3).
    smpl_model_path: Path
        Path to the directory containing SMPL/SMPL‑X model files (e.g.
        `models/smplx` downloaded from the SMPL-X website).
    mhr_lod: int, optional
        Level of detail to load for the MHR model (default: 1).  Higher
        values provide more vertices but consume more memory.
    smpl_model_type: str, optional
        Which SMPL model to use: "smpl" for 6890‑vertex SMPL or
        "smplx" for 10475‑vertex SMPL‑X (default: "smplx").
    gender: str, optional
        Gender of the SMPL model ("male", "female" or "neutral").  Only
        relevant when using gender‑specific SMPL models.
    device: str, optional
        Device on which to perform optimisation ("cpu" or "cuda").  On
        systems without a GPU, set this to "cpu".

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary with keys ``identity_coeffs`` (shape [N, 45]),
        ``lbs_model_params`` (shape [N, 204]) and ``face_expr_coeffs``
        (shape [N, 72]) containing the inferred MHR parameters for each
        frame in the input batch.
    """
    # Convert input vertices to torch tensor on the selected device
    verts_tensor = torch.tensor(smpl_vertices, dtype=torch.float32, device=device)

    # Load the MHR model from downloaded assets.  Use from_files to locate
    # assets in the current working directory; ensure `assets.zip` has been
    # downloaded and extracted as described in the MHR README.  The `lod`
    # argument chooses the level of detail.
    mhr_model = MHR.from_files(device=torch.device(device), lod=mhr_lod)

    # Create the SMPL or SMPL‑X model.  smplx.create will load the
    # appropriate JSON/NPZ files from the given model path.  Setting
    # `use_face_contour=True` is important for SMPL‑X conversion.
    if smpl_model_type.lower() == "smplx":
        smpl_model = smplx.create(
            model_path=str(smpl_model_path),
            model_type="smplx",
            gender=gender,
            use_face_contour=True,
            num_betas=10,
            num_expression_coeffs=10,
        ).to(device)
    else:
        smpl_model = smplx.create(
            model_path=str(smpl_model_path),
            model_type="smpl",
            gender=gender,
            use_face_contour=False,
        ).to(device)

    # Instantiate the conversion class.  The conversion tool uses
    # barycentric interpolation and optimisation to find the best MHR
    # parameters corresponding to the input SMPL mesh.
    converter = Conversion(
        mhr_model=mhr_model,
        smpl_model=smpl_model,
        smpl_model_type=smpl_model_type.lower(),
        device=torch.device(device),
    )

    # Perform conversion.  If you have SMPL parameter dictionaries
    # (pose, betas, translation, etc.) you can pass them via
    # ``smpl_parameters=dict(...)``.  Here we pass vertices directly.
    conversion_result = converter.convert_smpl2mhr(
        smpl_vertices=verts_tensor, single_identity=True, is_tracking=False
    )

    # The result stores vertices, meshes and parameters.  The latent
    # coefficients live in the ``result_parameters`` attribute under keys
    # 'lbs_model_params', 'identity_coeffs' and 'face_expr_coeffs'.
    params = conversion_result.result_parameters
    # Move tensors back to CPU and convert to numpy
    return {
        "identity_coeffs": params["identity_coeffs"].cpu().numpy(),
        "lbs_model_params": params["lbs_model_params"].cpu().numpy(),
        "face_expr_coeffs": params["face_expr_coeffs"].cpu().numpy(),
    }


def main() -> None:
    """Command‑line interface for the parameter inference utility."""
    parser = argparse.ArgumentParser(
        description="Infer MHR parameters from a SMPL/SMPL‑X mesh"
    )
    parser.add_argument(
        "--mesh",
        type=Path,
        required=True,
        help="Path to an OBJ/PLY file with SMPL/SMPL‑X vertices",
    )
    parser.add_argument(
        "--smpl-model-path",
        type=Path,
        required=True,
        help="Directory containing SMPL/SMPL‑X model files (downloaded separately)",
    )
    parser.add_argument(
        "--lod",
        type=int,
        default=1,
        help="Level of detail for the MHR model (0–6)",
    )
    parser.add_argument(
        "--smpl-type",
        type=str,
        default="smplx",
        choices=["smpl", "smplx"],
        help="Type of body model corresponding to the input mesh",
    )
    parser.add_argument(
        "--gender",
        type=str,
        default="neutral",
        choices=["male", "female", "neutral"],
        help="Gender of the SMPL/SMPL‑X model",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for running the optimiser",
    )
    args = parser.parse_args()

    smpl_vertices = load_mesh_vertices(args.mesh)
    params = infer_mhr_parameters(
        smpl_vertices,
        smpl_model_path=args.smpl_model_path,
        mhr_lod=args.lod,
        smpl_model_type=args.smpl_type,
        gender=args.gender,
        device=args.device,
    )

    print("Inferred MHR parameters:")
    for name, arr in params.items():
        print(f"  {name}: shape={arr.shape}")

    # Save the parameters to disk as NumPy arrays
    out_dir = Path("mhr_parameters")
    out_dir.mkdir(exist_ok=True)
    np.save(out_dir / "identity_coeffs.npy", params["identity_coeffs"])
    np.save(out_dir / "lbs_model_params.npy", params["lbs_model_params"])
    np.save(out_dir / "face_expr_coeffs.npy", params["face_expr_coeffs"])
    print(f"Saved parameter arrays to {out_dir}/")


if __name__ == "__main__":
    main()