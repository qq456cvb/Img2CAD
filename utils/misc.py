import os
import numpy as np
import torch
import trimesh
import nvdiffrast.torch as dr
import cv2

def render_nvdiffrast_rgb(input_path, dist=2.2, device='cuda', ctx=None):
    """
    Render an RGB image of a mesh using nvdiffrast.
    
    Parameters:
      input_path (str): Path to the mesh file.
      dist (float): Distance used to translate the object along -Z.
      device (str): Torch device to use.
      
    Returns:
      rgb_img (np.ndarray): Rendered 8-bit RGB image (H x W x 3).
    """
    # ---------------------------
    # 1. Load mesh and set up transforms
    # ---------------------------
    mesh = trimesh.load(input_path, force='mesh')
    
    # Define helper functions for rotation matrices:
    def rotx(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[1, 0, 0, 0],
                         [0, c, -s, 0],
                         [0, s,  c, 0],
                         [0, 0, 0, 1]], dtype=np.float32)
    
    def roty(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[ c, 0, s, 0],
                         [ 0, 1, 0, 0],
                         [-s, 0, c, 0],
                         [0, 0, 0, 1]], dtype=np.float32)
    
    # Build the mesh pose matrix (rotate then translate along -Z).
    y_angle = -20 / 180 * np.pi
    x_angle = 30 / 180 * np.pi
    # Rotation: combine rotx then roty; note order matters.
    mesh_pose = rotx(x_angle) @ roty(y_angle)
    
    # Center the mesh by subtracting the center of its bounds.
    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2
    trans_center = np.eye(4, dtype=np.float32)
    trans_center[:3, 3] = -center
    
    # Create a translation along -Z (you can think of this as including the camera offset into the model)
    trans_dist = np.eye(4, dtype=np.float32)
    trans_dist[:3, 3] = np.array([0, 0, -dist], dtype=np.float32)
    
    # The final model transformation: center, then rotate, then shift.
    model_matrix = trans_dist @ mesh_pose @ trans_center 

    # ---------------------------
    # 2. Prepare mesh attributes
    # ---------------------------
    # Get vertices and faces from the mesh
    vertices = torch.tensor(mesh.vertices, device=device, dtype=torch.float32)  # (V, 3)
    faces = torch.tensor(mesh.faces, device=device, dtype=torch.int32)           # (F, 3)
    
    # Make sure the mesh has normals. (trimesh usually computes these.)
    if (not hasattr(mesh.visual, 'vertex_colors')) or (mesh.visual.vertex_colors is None):
        # If no vertex colors, assign a uniform gray.
        vertex_colors = torch.ones_like(vertices) * 0.5
    else:
        # Normalize colors from 0–255 to 0–1 and consider only RGB channels.
        vertex_colors = torch.tensor(mesh.visual.vertex_colors[:, :3] / 255.0,
                                     device=device, dtype=torch.float32)
    
    # Convert vertices to homogeneous coordinates so that we can apply the 4x4 transformation.
    ones = torch.ones((vertices.shape[0], 1), device=device)
    vertices_h = torch.cat([vertices, ones], dim=1)  # (V, 4)
    model_matrix_torch = torch.tensor(model_matrix, device=device, dtype=torch.float32)
    
    # Transform vertices from object space to world space.
    vertices_world = (model_matrix_torch @ vertices_h.T).T  # (V, 4)
    vertices_world = vertices_world[:, :3]  # drop the homogeneous coordinate

    # For vertex normals, apply just the rotation part.
    if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None and len(mesh.vertex_normals):
        normals = torch.tensor(mesh.vertex_normals, device=device, dtype=torch.float32)
    else:
        # Fall back to a default normal if needed.
        normals = torch.zeros_like(vertices)
        normals[:, 2] = 1.0
    rotation_matrix = model_matrix_torch[:3, :3]
    normals_trans = (rotation_matrix @ normals.T).T
    normals_trans = normals_trans / (torch.norm(normals_trans, dim=1, keepdim=True) + 1e-8)
    
    # ---------------------------
    # 3. Set up camera projection
    # ---------------------------
    # For consistency with your original intrinsics:
    #   intrinsics = [[400, 0, 200],
    #                 [0, 400, 200],
    #                 [0, 0,   1]]
    # and image resolution 400x400.
    H, W = 400, 400
    intrinsics = np.array([[400, 0, 200],
                           [0, 400, 200],
                           [0,   0,   1]], dtype=np.float32)
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    
    # We need a perspective projection matrix that maps from camera space to clip space.
    # One common convention (similar to OpenGL) is to use:
    #   [ 2fx/W      0         1 - 2cx/W        0 ]
    #   [   0     2fy/H       2cy/H - 1         0 ]
    #   [   0        0       -(far+near)/(far-near)  -2far*near/(far-near) ]
    #   [   0        0            -1              0 ]
    near, far = 0.1, 1000.0
    proj = np.array([
        [2 * fx / W,      0,             1 - 2 * cx / W,            0],
        [0,         2 * fy / H,         2 * cy / H - 1,               0],
        [0,                0,       -(far + near) / (far - near),   -2 * far * near / (far - near)],
        [0,                0,                    -1,                   0]
    ], dtype=np.float32)
    projection = torch.tensor(proj, device=device, dtype=torch.float32)

    # Here we assume the camera is at the origin looking down -Z (i.e. view matrix is identity).
    view = torch.eye(4, device=device, dtype=torch.float32)
    
    # Form the full model-view-projection matrix.
    # (Since we already baked the model transformation, our mvp = projection * view * model_matrix.)
    mvp = projection @ view @ model_matrix_torch

    # Transform original vertices_h (object-space) to clip space.
    clip_coords = (mvp @ vertices_h.T).T  # (V, 4)
    
    # ---------------------------
    # 4. Rasterize with nvdiffrast
    # ---------------------------
    # Create a nvdiffrast CUDA context.
    if ctx is None:
        ctx = dr.RasterizeCudaContext()
    
    # Rasterize: here we pass the clip-space vertices along with face connectivity.
    # nvdiffrast expects the vertices in homogeneous clip space (before perspective division).
    rast, _ = dr.rasterize(ctx, clip_coords[None].contiguous(), faces.contiguous(), (H, W))
    
    # ---------------------------
    # 5. Interpolate attributes and perform shading
    # ---------------------------
    # Interpolate vertex normals and colors to the pixel grid.
    interp_normals = dr.interpolate(normals_trans.contiguous(), rast, faces)[0][0]
    interp_colors  = dr.interpolate(vertex_colors.contiguous(), rast, faces)[0][0]
    
    # Compute simple Lambertian shading with a directional light.
    # In your original code the directional light has color ones and intensity 10.
    # Here we assume the light is coming from the direction (0, 0, 1) in world space.
    light_dir = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=torch.float32)
    light_dir = light_dir / torch.norm(light_dir)
    # Compute dot product at each pixel; unsqueeze to match (H,W,1).
    dot = torch.clamp(torch.sum(interp_normals * light_dir.view(1, 1, 3), dim=2, keepdim=True), min=0.0)
    shading = dot * 2.0 + 0.4  # combine directional and ambient
    
    # Modulate vertex colors with shading.
    pixel_colors = interp_colors * shading
    
    # Rasterization returns -1 for pixels where no triangle was drawn.
    # Create a mask (of shape H x W) so that pixels not covered by any triangle are set to black.
    mask = (rast[0, ..., 0] != -1).unsqueeze(-1).float()
    pixel_colors = pixel_colors * mask
    
    # Clamp the color values to [0, 1] and convert to a uint8 image.
    rgb = pixel_colors.clamp(0.0, 1.0).cpu().numpy()
    rgb_img = (rgb * 255).astype(np.uint8)[::-1]  # (H, W, 3)
    
    return rgb_img