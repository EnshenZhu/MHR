import torch
from mhr.mhr import MHR

def mhr_tester():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}.")

    # Load MHR model on the selected device
    mhr_model = MHR.from_files(device=device, lod=1)

    # Define parameters
    batch_size = 2
    identity_coeffs = (0.8 * torch.randn(batch_size, 45)).to(device)              # Identity
    model_parameters = (0.2 * (torch.rand(batch_size, 204) - 0.5)).to(device)     # Pose
    face_expr_coeffs = (0.3 * torch.randn(batch_size, 72)).to(device)             # Facial expression

    # Optional safety check (helps debug future device issues quickly)
    assert next(mhr_model.parameters()).device == identity_coeffs.device, (
        f"Device mismatch: model on {next(mhr_model.parameters()).device}, "
        f"inputs on {identity_coeffs.device}"
    )

    # Generate mesh vertices and skeleton information
    with torch.no_grad():
        vertices, skeleton_state = mhr_model(identity_coeffs, model_parameters, face_expr_coeffs)

    return vertices, skeleton_state

if __name__ == "__main__":
    print("start")
    this_vertices, this_skeleton_state=mhr_tester()
    print(this_vertices)
    print(this_skeleton_state)
    print("end")