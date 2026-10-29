import json
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image
from scenezoo import get_dataset

scene_id = Path(
    "draft_a/vfront_pilot_scene.txt"
).read_text().strip().removesuffix(".json")

root = (
    Path("/root/autodl-tmp/3D-FRONT-pilot")
    / scene_id
)

depth_dir = root / "depth_pilot"
rgb_dir = root / "rgb_pilot"
rgb_dir.mkdir(parents=True, exist_ok=True)

with (depth_dir / "cameras.json").open() as f:
    cameras = json.load(f)

W = cameras["width"]
H = cameras["height"]
fx = cameras["fx"]
fy = cameras["fy"]
cx = cameras["cx"]
cy = cameras["cy"]

print("Loading SceneZoo geometry...", flush=True)

dataset = get_dataset("3dfront", root)
mesh = dataset.get_mesh(scene_id, mesh_type="full")

faces = np.asarray(mesh.triangles)
uvs = np.asarray(mesh.triangle_uvs)
material_ids = np.asarray(mesh.triangle_material_ids)
textures = mesh.textures

print("Triangles:", len(faces))
print("UV entries:", len(uvs))
print("Material IDs:", len(material_ids))
print("Textures:", len(textures))

assert len(faces) > 0
assert len(uvs) == len(faces) * 3
assert len(material_ids) == len(faces)
assert len(textures) > 0

uvs = uvs.reshape(-1, 3, 2)

print("Building CPU raycasting scene...", flush=True)

mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

ray_scene = o3d.t.geometry.RaycastingScene()
ray_scene.add_triangles(mesh_t)

uu, vv = np.meshgrid(
    np.arange(W, dtype=np.float32) + 0.5,
    np.arange(H, dtype=np.float32) + 0.5,
)

for frame in cameras["frames"]:
    name = frame["name"]

    print("\n===== RENDERING", name, "=====", flush=True)

    c2w = np.asarray(
        frame["camera_to_world"],
        dtype=np.float64
    )

    right = c2w[:3, 0]
    up = c2w[:3, 1]
    forward = -c2w[:3, 2]
    eye = c2w[:3, 3]

    directions = (
        forward[None, None, :]
        + ((uu - cx) / fx)[..., None] * right
        + ((cy - vv) / fy)[..., None] * up
    )

    origins = np.broadcast_to(eye, (H, W, 3))

    rays = np.concatenate(
        [origins, directions],
        axis=-1
    ).astype(np.float32)

    result = ray_scene.cast_rays(
        o3d.core.Tensor(rays)
    )

    distance = result["t_hit"].numpy().reshape(-1)
    triangle_ids = result[
        "primitive_ids"
    ].numpy().reshape(-1)

    bary = result[
        "primitive_uvs"
    ].numpy().reshape(-1, 2)

    hit = np.isfinite(distance) & (distance > 0)

    rgb = np.zeros(
        (W * H, 3),
        dtype=np.uint8
    )

    # Magenta identifies hit pixels with missing materials.
    rgb[hit] = [255, 0, 255]

    hit_pixels = np.flatnonzero(hit)

    hit_triangles = triangle_ids[hit_pixels]
    hit_materials = material_ids[hit_triangles]

    textured_count = 0

    for material_id in np.unique(hit_materials):
        if material_id < 0 or material_id >= len(textures):
            print("Invalid material:", material_id)
            continue

        selection = hit_materials == material_id
        pixel_ids = hit_pixels[selection]
        tri_ids = hit_triangles[selection]
        weights = bary[pixel_ids]

        w1 = weights[:, 0]
        w2 = weights[:, 1]
        w0 = 1.0 - w1 - w2

        tri_uv = uvs[tri_ids]

        tex_uv = (
            w0[:, None] * tri_uv[:, 0]
            + w1[:, None] * tri_uv[:, 1]
            + w2[:, None] * tri_uv[:, 2]
        )

        texture = np.asarray(
            textures[int(material_id)]
        )

        if texture.ndim == 2:
            texture = np.repeat(
                texture[..., None], 3, axis=2
            )

        if texture.ndim != 3 or texture.shape[2] < 3:
            print("Invalid texture:", material_id)
            continue

        texture = texture[..., :3]

        if np.issubdtype(texture.dtype, np.floating):
            texture = (
                np.clip(texture, 0, 1) * 255
            ).astype(np.uint8)

        th, tw = texture.shape[:2]

        # OBJ UV uses bottom-left origin;
        # image arrays use top-left origin.
        tx = np.floor(
            (tex_uv[:, 0] % 1.0) * tw
        ).astype(np.int64) % tw

        ty = np.floor(
            (1.0 - tex_uv[:, 1] % 1.0) * th
        ).astype(np.int64) % th

        rgb[pixel_ids] = texture[ty, tx]
        textured_count += len(pixel_ids)

    rgb_image = rgb.reshape(H, W, 3)

    output = rgb_dir / f"{name}_rgb.png"
    Image.fromarray(rgb_image).save(output)

    mask = (hit.reshape(H, W) * 255).astype(np.uint8)
    Image.fromarray(mask).save(
        rgb_dir / f"{name}_hit_mask.png"
    )

    hit_ratio = hit.mean()
    coverage = textured_count / max(hit.sum(), 1)

    print("Hit ratio:", round(float(hit_ratio), 4))
    print("Material coverage:", round(float(coverage), 4))
    print("Saved RGB:", output, flush=True)

print("\nCPU RGB PILOT COMPLETED", flush=True)
