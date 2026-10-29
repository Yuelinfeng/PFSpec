from pathlib import Path
import numpy as np
import open3d as o3d

scene_name = Path("draft_a/vfront_pilot_scene.txt").read_text().strip()
scene_id = Path(scene_name).stem

pilot_root = Path("/root/autodl-tmp/3D-FRONT-pilot") / scene_id
out_dir = pilot_root / "preview"
out_dir.mkdir(parents=True, exist_ok=True)

mesh_files = {
    "full": pilot_root / "pilot_full_zup.obj",
    "layout": pilot_root / "pilot_layout_zup.obj",
    "furniture": pilot_root / "pilot_furniture_zup.obj",
}

width, height = 1280, 960

def render_views(mesh_path: Path, tag: str):
    if not mesh_path.is_file():
        raise FileNotFoundError(mesh_path)

    print(f"\n===== LOADING {tag.upper()} =====", flush=True)
    print("Mesh path:", mesh_path, flush=True)

    mesh = o3d.io.read_triangle_mesh(str(mesh_path), enable_post_processing=True)

    if mesh.is_empty():
        raise RuntimeError(f"Empty mesh: {mesh_path}")

    mesh.compute_vertex_normals()

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    print("Vertices:", len(vertices), flush=True)
    print("Triangles:", len(triangles), flush=True)

    aabb = mesh.get_axis_aligned_bounding_box()
    center = aabb.get_center()
    extent_xyz = aabb.get_extent()
    radius = max(np.linalg.norm(extent_xyz), 1.0)

    print("Bounds min:", aabb.get_min_bound(), flush=True)
    print("Bounds max:", aabb.get_max_bound(), flush=True)
    print("Extent:", extent_xyz, flush=True)
    print("Center:", center, flush=True)
    print("Radius:", radius, flush=True)

    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])

    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultLit"

    renderer.scene.add_geometry(tag, mesh, material)

    # Try to enable sun lighting if available
    try:
        renderer.scene.scene.enable_sun_light(True)
        renderer.scene.scene.set_sun_light(
            np.array([0.35, 0.25, -1.0], dtype=np.float32),
            np.array([1.0, 1.0, 1.0], dtype=np.float32),
            75000.0,
        )
    except Exception:
        pass

    views = {
        "iso_front": (
            center + np.array([ 1.30, -1.35, 0.95]) * radius,
            np.array([0.0, 0.0, 1.0]),
        ),
        "iso_back": (
            center + np.array([-1.20,  1.20, 0.85]) * radius,
            np.array([0.0, 0.0, 1.0]),
        ),
        "top": (
            center + np.array([0.0, 0.0, 2.3]) * radius,
            np.array([0.0, 1.0, 0.0]),
        ),
    }

    for view_name, (eye, up) in views.items():
        print(f"Rendering {tag}:{view_name}", flush=True)
        renderer.setup_camera(60.0, center, eye, up)
        image = renderer.render_to_image()

        out_path = out_dir / f"{tag}_{view_name}.png"
        ok = o3d.io.write_image(str(out_path), image)
        if not ok:
            raise RuntimeError(f"Failed to save image: {out_path}")
        print("Saved:", out_path, flush=True)

    del renderer


print("Scene ID:", scene_id, flush=True)
print("Pilot root:", pilot_root, flush=True)
print("Preview output:", out_dir, flush=True)

for tag, path in mesh_files.items():
    render_views(path, tag)

print("\nPreview rendering completed successfully.", flush=True)
