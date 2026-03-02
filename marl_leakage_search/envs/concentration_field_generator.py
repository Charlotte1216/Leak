"""
generate_concentration_field.py
Generate and save random concentration fields as .npz files.
"""
import math
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    # When imported as a package
    from .concentration_field import ConcentrationField
except ImportError:
    # When executed as a script
    from concentration_field import ConcentrationField


def _random_sources(
    num_sources: int,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    q_range: Tuple[float, float],
) -> List[Dict[str, float]]:
    sources = []
    for _ in range(num_sources):
        sources.append(
            {
                "x": random.uniform(*x_range),
                "y": random.uniform(*y_range),
                "Q": random.uniform(*q_range),
            }
        )
    return sources


def _random_obstacles(
    num_obstacles: int,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    r_range: Tuple[float, float],
) -> List[Dict[str, float]]:
    obstacles = []
    for _ in range(num_obstacles):
        obstacles.append(
            {
                "x": random.uniform(*x_range),
                "y": random.uniform(*y_range),
                "radius": random.uniform(*r_range),
            }
        )
    return obstacles


def generate_concentration_field(
    grid_size: Tuple[int, int] = (100, 100),
    x_range: Tuple[float, float] = (0.0, 100.0),
    y_range: Tuple[float, float] = (0.0, 100.0),
    source_count_range: Tuple[int, int] = (1, 8),
    obstacle_count_range: Tuple[int, int] = (0, 8),
    wind_speed_range: Tuple[float, float] = (0.5, 3.0),
    wind_dir_range: Tuple[float, float] = (0.0, 2.0 * math.pi),
    source_q_range: Tuple[float, float] = (5.0, 15.0),
    obstacle_radius_range: Tuple[float, float] = (4.0, 16.0),
    plume_params: Dict[str, float] = None,
    vortex_params: Dict[str, float] = None,
    noise_std: float = 0.0,
    t: float = 0.0,
) -> Dict[str, np.ndarray]:
    """
    Generate one random concentration field.
    Returns a dict with concentration grid and metadata.
    """
    nx, ny = grid_size
    xs = np.linspace(x_range[0], x_range[1], nx)
    ys = np.linspace(y_range[0], y_range[1], ny)
    X, Y = np.meshgrid(xs, ys, indexing="xy")

    num_sources = random.randint(source_count_range[0], source_count_range[1])
    num_obstacles = random.randint(obstacle_count_range[0], obstacle_count_range[1])
    wind_speed = random.uniform(*wind_speed_range)
    wind_dir = random.uniform(*wind_dir_range)

    sources = _random_sources(num_sources, x_range, y_range, source_q_range)
    obstacles = _random_obstacles(num_obstacles, x_range, y_range, obstacle_radius_range)

    plume_params = plume_params or {"L": 100.0, "sigma_y": 5.0}
    vortex_params = vortex_params or {}

    field = ConcentrationField(
        sources=sources,
        obstacles=obstacles,
        wind_speed=wind_speed,
        wind_dir=wind_dir,
        plume_params=plume_params,
        vortex_params=vortex_params,
        noise_std=noise_std,
    )

    C = field.concentration(X, Y, t=t)

    sources_array = np.array([[s["x"], s["y"], s["Q"]] for s in sources], dtype=float)
    obstacles_array = np.array(
        [[o["x"], o["y"], o["radius"]] for o in obstacles], dtype=float
    )

    return {
        "concentration": C,
        "x": xs,
        "y": ys,
        "sources": sources_array,
        "obstacles": obstacles_array,
        "wind_speed": np.array([wind_speed], dtype=float),
        "wind_dir": np.array([wind_dir], dtype=float),
        "plume_params": np.array([plume_params], dtype=object),
        "vortex_params": np.array([vortex_params], dtype=object),
        "noise_std": np.array([noise_std], dtype=float),
        "t": np.array([t], dtype=float),
    }


def save_concentration_field(
    data: Dict[str, np.ndarray],
    output_dir: str,
    filename_prefix: str = "concentration_field",
) -> str:
    """
    Save a concentration field dict to an .npz file.
    Returns the saved filepath.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    num_sources = data["sources"].shape[0]
    num_obstacles = data["obstacles"].shape[0]
    wind_speed = float(data["wind_speed"][0])

    file_name = (
        f"{filename_prefix}_{num_sources}sources_"
        f"{num_obstacles}obstacles_{wind_speed:.1f}windspeed.npz"
    )

    file_path = output_path / file_name
    if file_path.exists():
        stem = file_path.stem
        suffix = file_path.suffix
        counter = 1
        while True:
            candidate = output_path / f"{stem}_{counter}{suffix}"
            if not candidate.exists():
                file_path = candidate
                break
            counter += 1
    np.savez_compressed(file_path, **data)
    return str(file_path)


def generate_multiple(
    num_fields: int,
    output_dir: str,
    **kwargs,
) -> List[str]:
    """
    Generate and save multiple random concentration fields.
    Returns list of saved file paths.
    """
    output_path = Path(output_dir)
    if output_path.exists():
        for item in output_path.iterdir():
            if item.is_file() or item.is_symlink():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
    else:
        output_path.mkdir(parents=True, exist_ok=True)

    saved = []
    for _ in range(num_fields):
        data = generate_concentration_field(**kwargs)
        saved.append(save_concentration_field(data, output_dir))
    return saved


if __name__ == "__main__":
    # Example usage
    output_dir = os.path.join(os.path.dirname(__file__), "generated_fields")
    generate_multiple(num_fields=100, output_dir=output_dir)
