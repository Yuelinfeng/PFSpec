from pathlib import Path
import json
import numpy as np
import open3d as o3d
from PIL import Image

scene_name = Path(
    "draft_a/vfront_pilot_scene.txt"
).read_text().strip()
scene_id = Path(scene_name).stem

root = Path("/root/autodl-tmp/3D-FRONT-pilot") / scene_id
out = root / "depth_pilot"
out.mkdir(parents=True, exist_ok=True)

W, H = 640, 480
FOV_X_DEG = 70.0
fx = W / (2 * np.tan(np.deg2rad(FOV_X_DEG / 2)))
fy = fx
cx, cy = W / 2, H / 2

def unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / max(np.linalg.norm(v), 1e-12)

mesh_path = root / "pilot_full_zup.obj"
print("Loading:", mesh_path, flush=True)

mesh = o3d.io.read_triangle_mesh(str(mesh_path))
mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

scene = o3d.t.geometry.RaycastingScene()
scene.add_triangles(mesh_t)

bounds = mesh.get_axis_aligned_bounding_box()
center = bounds.get_center()

print("Mesh loaded", flush=True)
print("Center:", center, flush=True)

# Camera positions inside the room, in Z-up coordinates.
camera_views = [
    (
        "living_0",
        center + [-0.8, 1.8, 0.0],
        center + [0.9, -1.3, 0.0],
    ),
    (
        "living_1",
        center + [1.5, 0.5, 0.0],
        center + [0.0, -2.0, 0.0],
    ),
    (
        "living_2",
        center + [-1.0, -1.0, 0.0],
        center + [1.5, -2.0, 0.0],
    ),
]

u, v = np.meshgrid(
    np.arange(W, dtype=np.float32) + 0.5,
    np.arange(H, dtype=np.float32) + 0.5,
)

frames = []

for name, eye, target in camera_views:
    print(f"\n===== {name} =====", flush=True)

    forward = unit(target - eye)
    right = unit(np.cross(forward, [0, 0, 1]))
    up = unit(np.cross(right, forward))

    dirs = (
        forward[None, None, :]
        + ((u - cx) / fx)[..., None]
          * right[None, None, :]
        + ((cy - v) / fy)[..., None]
          * up[None, None, :]
    )

    origins = np.broadcast_to(
        eye.astype(np.float32),
        (H, W, 3)
    )

    rays = np.concatenate(
        [origins, dirs.astype(np.float32)],
        axis=-1
    ).astype(np.float32)

    ans = scene.cast_rays(o3d.core.Tensor(rays))
    t_hit = ans["t_hit"].numpy()

    hit = np.isfinite(t_hit) & (t_hit > 0)

    # Directions have unit forward-axis projection,
    # so t_hit is camera-forward Z depth, in metres.
    depth = np.where(hit, t_hit, 0).astype(np.float32)

    depth_path = out / f"{name}_depth.npy"
    np.save(depth_path, depth)

    # Grayscale visualization; not the metric depth storage.
    near, far = 0.2, 12.0
    normalized = np.clip(
        (depth - near) / (far - near),
        0, 1
    )

    preview = np.zeros((H, W), dtype=np.uint8)
    preview[hit] = (
        (1 - normalized[hit]) * 230 + 20
    ).astype(np.uint8)

    preview_path = out / f"{name}_depth.png"
    Image.fromarray(preview).save(preview_path)

    # OpenGL camera-to-world convention:
    # X right, Y up, Z backward.
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 0] = right
    c2w[:3, 1] = up
    c2w[:3, 2] = -forward
    c2w[:3, 3] = eye

    frames.append({
        "name": name,
        "depth_file": depth_path.name,
        "preview_file": preview_path.name,
        "camera_to_world": c2w.tolist(),
    })

    print("Camera:", eye, flush=True)
    print("Target:", target, flush=True)
    print("Hit ratio:", float(hit.mean()), flush=True)
    if hit.any():
        print("Depth range:",
              float(depth[hit].min()),
              float(depth[hit].max()),
              flush=True)
    print("Saved:", depth_path, flush=True)
    print("Saved:", preview_path, flush=True)

metadata = {
    "width": W,
    "height": H,
    "fx": float(fx),
    "fy": float(fy),
    "cx": float(cx),
    "cy": float(cy),
    "coordinate_system": "Z-up world; OpenGL camera-to-world",
    "depth_type": "camera-forward Z depth, metres",
    "frames": frames,
}

with (out / "cameras.json").open("w") as f:
    json.dump(metadata, f, indent=2)

print("\nDEPTH PILOT SUCCESSFUL", flush=True)
