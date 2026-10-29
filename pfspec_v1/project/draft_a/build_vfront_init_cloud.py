import json
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image
from scenezoo import get_dataset

SCENE_ID = (
    Path("draft_a/vfront_pilot_scene.txt")
    .read_text()
    .strip()
    .removesuffix(".json")
)

ROOT = Path("/root/autodl-tmp/3D-FRONT-pilot") / SCENE_ID
OUT = ROOT / "gaussian_init"
VOXEL_SIZE = 0.025

with (ROOT / "depth_pilot/cameras.json").open() as f:
    cfg = json.load(f)

W, H = int(cfg["width"]), int(cfg["height"])
fx, fy = float(cfg["fx"]), float(cfg["fy"])
cx, cy = float(cfg["cx"]), float(cfg["cy"])

assert cfg["depth_type"] == "camera-forward Z depth, metres"

print("Loading scene mesh...", flush=True)
dataset = get_dataset("3dfront", ROOT)
mesh = dataset.get_mesh(SCENE_ID, mesh_type="full")

ray_scene = o3d.t.geometry.RaycastingScene()
ray_scene.add_triangles(
    o3d.t.geometry.TriangleMesh.from_legacy(mesh)
)

points_all = []
colors_all = []
rng = np.random.default_rng(2026)

for frame in cfg["frames"]:
    name = frame["name"]
    print(f"\n===== {name} =====", flush=True)

    depth = np.load(
        ROOT / "depth_pilot" / frame["depth_file"],
        allow_pickle=False,
    )

    rgb = np.asarray(
        Image.open(
            ROOT / "rgb_pilot" / f"{name}_rgb.png"
        ).convert("RGB")
    )

    assert depth.shape == (H, W)
    assert rgb.shape == (H, W, 3)

    c2w = np.asarray(
        frame["camera_to_world"], dtype=np.float64
    )
    R = c2w[:3, :3]
    eye = c2w[:3, 3]

    assert np.allclose(R.T @ R, np.eye(3), atol=1e-5)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-5)

    valid = np.isfinite(depth) & (depth > 0)
    ys, xs = np.where(valid)

    z = depth[ys, xs].astype(np.float64)
    u = xs.astype(np.float64) + 0.5
    v = ys.astype(np.float64) + 0.5

    # OpenGL camera coordinates: forward is negative Z.
    x_cam = (u - cx) / fx * z
    y_cam = (cy - v) / fy * z
    z_cam = -z

    cam_points = np.column_stack(
        (x_cam, y_cam, z_cam)
    )

    world = cam_points @ R.T + eye

    # Independently recast a subset of rays against
    # the original scene mesh to verify numeric depth.
    k = min(2048, len(z))
    chosen = rng.choice(len(z), size=k, replace=False)

    directions = (
        world[chosen] - eye[None, :]
    ) / z[chosen, None]

    rays = np.concatenate(
        [
            np.broadcast_to(eye, (k, 3)),
            directions
        ],
        axis=1,
    ).astype(np.float32)

    recast = ray_scene.cast_rays(
        o3d.core.Tensor(rays)
    )["t_hit"].numpy()

    err = np.abs(
        recast.astype(np.float64) - z[chosen]
    )

    finite_ratio = np.isfinite(err).mean()
    p99_error = (
        float(np.percentile(err[np.isfinite(err)], 99))
        if np.isfinite(err).any()
        else float("inf")
    )

    print("Valid pixels:", len(z))
    print("Depth re-raycast finite:", round(float(finite_ratio), 6))
    print("Depth error P99 (m):", round(p99_error, 6))
    print("World min:", world.min(axis=0).round(3).tolist())
    print("World max:", world.max(axis=0).round(3).tolist())

    if finite_ratio < 0.99 or p99_error > 0.01:
        raise RuntimeError(
            f"{name}: depth geometry mismatch; "
            "stop before voxelization"
        )

    points_all.append(world.astype(np.float32))
    colors_all.append(rgb[ys, xs])

xyz = np.concatenate(points_all, axis=0)
rgb = np.concatenate(colors_all, axis=0)

print("\n===== FUSION =====", flush=True)
print("Raw points:", len(xyz))

voxel_keys = np.floor(
    xyz.astype(np.float64) / VOXEL_SIZE
).astype(np.int32)

_, inverse = np.unique(
    voxel_keys,
    axis=0,
    return_inverse=True,
)

n_voxels = int(inverse.max()) + 1
counts = np.bincount(
    inverse, minlength=n_voxels
).astype(np.float64)

voxel_xyz = np.column_stack([
    np.bincount(
        inverse, weights=xyz[:, i], minlength=n_voxels
    ) / counts
    for i in range(3)
]).astype(np.float32)

voxel_rgb = np.column_stack([
    np.bincount(
        inverse, weights=rgb[:, i], minlength=n_voxels
    ) / counts
    for i in range(3)
])

voxel_rgb = np.clip(
    np.rint(voxel_rgb), 0, 255
).astype(np.uint8)

print("Voxel size (m):", VOXEL_SIZE)
print("Unique voxels:", n_voxels)
print("Compression ratio:", round(len(xyz) / n_voxels, 3))

OUT.mkdir(parents=True, exist_ok=True)

np.savez_compressed(
    OUT / "voxel_init_025m.npz",
    xyz=voxel_xyz,
    rgb=voxel_rgb,
    voxel_size=np.float32(VOXEL_SIZE),
)

cloud = o3d.geometry.PointCloud()
cloud.points = o3d.utility.Vector3dVector(
    voxel_xyz.astype(np.float64)
)
cloud.colors = o3d.utility.Vector3dVector(
    voxel_rgb.astype(np.float64) / 255.0
)

ply_path = OUT / "voxel_init_025m.ply"
assert o3d.io.write_point_cloud(
    str(ply_path), cloud, write_ascii=False
)

print("Saved NPZ:", OUT / "voxel_init_025m.npz")
print("Saved PLY:", ply_path)
print("\nVOXEL INITIALIZATION COMPLETE")
