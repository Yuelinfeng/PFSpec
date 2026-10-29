from pathlib import Path
import numpy as np
import open3d as o3d
from scenezoo import get_dataset

scene_name = Path(
    "draft_a/vfront_pilot_scene.txt"
).read_text().strip()

scene_id = Path(scene_name).stem

root = (
    Path("/root/autodl-tmp/3D-FRONT-pilot")
    / scene_id
)

dataset = get_dataset("3dfront", root)

print("Scene:", scene_id, flush=True)
print("Rooms:", dataset.get_room_ids(scene_id), flush=True)

for kind in ["layout", "furniture", "full"]:
    print(f"\n===== {kind.upper()} =====", flush=True)

    mesh = dataset.get_mesh(scene_id, mesh_type=kind)

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    assert len(vertices) > 0
    assert len(triangles) > 0
    assert np.isfinite(vertices).all()

    print("Vertices:", len(vertices), flush=True)
    print("Triangles:", len(triangles), flush=True)
    print("Textures:", len(mesh.textures), flush=True)

    bounds = mesh.get_axis_aligned_bounding_box()

    print("Bounds min:", bounds.get_min_bound(), flush=True)
    print("Bounds max:", bounds.get_max_bound(), flush=True)
    print("Extent:", bounds.get_extent(), flush=True)

    output = root / f"pilot_{kind}_zup.obj"

    success = o3d.io.write_triangle_mesh(
        str(output),
        mesh,
        write_triangle_uvs=True,
    )

    if not success:
        raise RuntimeError(f"Export failed: {output}")

    print("Saved:", output, flush=True)

print("\nPilot geometry export completed.", flush=True)
