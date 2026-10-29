from pathlib import Path
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw

scene_name = Path(
    "draft_a/vfront_pilot_scene.txt"
).read_text().strip()

scene_id = Path(scene_name).stem

root = (
    Path("/root/autodl-tmp/3D-FRONT-pilot")
    / scene_id
)

out_dir = root / "preview_cpu"
out_dir.mkdir(parents=True, exist_ok=True)

W, H = 1280, 960

def unit(x):
    x = np.asarray(x, dtype=np.float64)
    return x / max(np.linalg.norm(x), 1e-12)

def load_mesh(kind):
    path = root / f"pilot_{kind}_zup.obj"

    print("Loading:", path, flush=True)

    mesh = o3d.io.read_triangle_mesh(str(path))

    vertices = np.asarray(
        mesh.vertices, dtype=np.float64
    )
    faces = np.asarray(
        mesh.triangles, dtype=np.int32
    )

    if not len(vertices) or not len(faces):
        raise RuntimeError(f"Empty geometry: {path}")

    print(
        kind,
        "vertices:", len(vertices),
        "triangles:", len(faces),
        flush=True
    )

    return vertices, faces

data = {
    "layout": load_mesh("layout"),
    "furniture": load_mesh("furniture"),
}

all_vertices = np.concatenate(
    [data[k][0] for k in data],
    axis=0
)

bounds_min = all_vertices.min(axis=0)
bounds_max = all_vertices.max(axis=0)
center = (bounds_min + bounds_max) / 2

print("Bounds min:", bounds_min, flush=True)
print("Bounds max:", bounds_max, flush=True)
print("Center:", center, flush=True)

def render(name, direction, visible):
    print("\nRendering:", name, flush=True)

    direction = unit(direction)
    forward = -direction

    if abs(direction[2]) > 0.95:
        up_hint = np.array([0, 1, 0])
    else:
        up_hint = np.array([0, 0, 1])

    right = unit(np.cross(forward, up_hint))
    up = unit(np.cross(right, forward))

    # Orthographic projection of the entire scene.
    rel = all_vertices - center

    all_x = rel @ right
    all_y = rel @ up

    xmin, xmax = all_x.min(), all_x.max()
    ymin, ymax = all_y.min(), all_y.max()

    scale = 0.86 * min(
        W / max(xmax - xmin, 1e-6),
        H / max(ymax - ymin, 1e-6)
    )

    midx = (xmin + xmax) / 2
    midy = (ymin + ymax) / 2

    triangles_2d = []
    depths = []
    colors = []
    alphas = []

    light = unit([0.4, -0.5, 0.8])

    for kind in visible:
        vertices, faces = data[kind]
        rel = vertices - center

        px = (rel @ right - midx) * scale + W / 2
        py = H / 2 - (rel @ up - midy) * scale

        screen = np.stack([px, py], axis=1)
        corners = screen[faces]

        p0 = vertices[faces[:, 0]]
        p1 = vertices[faces[:, 1]]
        p2 = vertices[faces[:, 2]]

        normals = np.cross(p1 - p0, p2 - p0)
        lengths = np.linalg.norm(
            normals, axis=1
        ).clip(min=1e-12)

        normals = normals / lengths[:, None]

        brightness = (
            0.48 +
            0.48 * np.abs(normals @ light)
        )

        if kind == "layout":
            base = np.array([161, 178, 192])
            alpha = 85 if len(visible) > 1 else 255
        else:
            base = np.array([205, 149, 98])
            alpha = 255

        rgb = np.clip(
            brightness[:, None] * base[None, :],
            0, 255
        ).astype(np.uint8)

        # Larger depth means farther from the camera.
        depth = (
            (p0 + p1 + p2) / 3 - center
        ) @ forward

        triangles_2d.append(corners)
        depths.append(depth)
        colors.append(rgb)
        alphas.append(
            np.full(len(faces), alpha, dtype=np.uint8)
        )

    triangles_2d = np.concatenate(triangles_2d)
    depths = np.concatenate(depths)
    colors = np.concatenate(colors)
    alphas = np.concatenate(alphas)

    image = Image.new(
        "RGB", (W, H), (248, 249, 251)
    )

    draw = ImageDraw.Draw(image, "RGBA")

    # Draw far triangles first (CPU painter's algorithm).
    order = np.argsort(-depths)

    for i in order:
        tri = triangles_2d[i]

        area2 = abs(
            (tri[1, 0] - tri[0, 0]) *
            (tri[2, 1] - tri[0, 1]) -
            (tri[1, 1] - tri[0, 1]) *
            (tri[2, 0] - tri[0, 0])
        )

        if area2 < 0.3:
            continue

        color = tuple(
            int(x) for x in colors[i]
        ) + (int(alphas[i]),)

        points = [
            (int(p[0]), int(p[1]))
            for p in tri
        ]

        draw.polygon(points, fill=color)

    draw.text(
        (20, 20),
        f"{name} | Geometry preview | Z-up",
        fill=(30, 40, 50, 255)
    )

    output = out_dir / f"{name}.png"
    image.save(output)

    print("Saved:", output, flush=True)

render(
    "full_iso_front",
    [1.2, -1.4, 0.9],
    ["layout", "furniture"]
)

render(
    "full_iso_back",
    [-1.2, 1.4, 0.9],
    ["layout", "furniture"]
)

render(
    "full_top",
    [0, 0, 1],
    ["layout", "furniture"]
)

render(
    "furniture_iso_front",
    [1.2, -1.4, 0.9],
    ["furniture"]
)

render(
    "layout_iso_front",
    [1.2, -1.4, 0.9],
    ["layout"]
)

print("\nCPU PREVIEW SUCCESSFUL", flush=True)
