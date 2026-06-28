"""Export EchoAvatar skeleton reference BVHs for Blender inspection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.utils import rotation_conversions as rc
from utils.anim import bvh, quat


def write_bvh(
    path: Path,
    *,
    rotations_deg: np.ndarray,
    positions: np.ndarray,
    names: np.ndarray,
    parents: np.ndarray,
    offsets: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bvh.save(
        str(path),
        {
            "rotations": rotations_deg.astype(np.float32),
            "positions": positions.astype(np.float32),
            "offsets": offsets.astype(np.float32),
            "parents": parents.astype(np.int32),
            "names": [str(name) for name in names],
            "order": "zyx",
            "frametime": 1.0 / 60.0,
        },
    )


def export_rest_pose(output_dir: Path) -> Path:
    meta = np.load(ROOT_DIR / "stats" / "zm_meta_info.npz")
    names = meta["names"]
    parents = meta["parents"]
    offsets = meta["offsets"]
    frame_count = 1
    joint_count = len(names)

    path = output_dir / "echoavatar_rest.bvh"
    write_bvh(
        path,
        rotations_deg=np.zeros((frame_count, joint_count, 3), dtype=np.float32),
        positions=np.zeros((frame_count, joint_count, 3), dtype=np.float32),
        names=names,
        parents=parents,
        offsets=offsets,
    )
    return path


def export_mean_source_pose(output_dir: Path) -> Path:
    meta = np.load(ROOT_DIR / "stats" / "zm_meta_info.npz")
    names = meta["names"]
    parents = meta["parents"]
    offsets = meta["offsets"]
    mean, _std = np.load(ROOT_DIR / "stats" / "body_mean_std_30fps.npy")

    rot6d = torch.from_numpy(mean[:-3].astype(np.float32)).reshape(1, 88, 6)
    rot_mats = rc.rotation_6d_to_matrix(rot6d)
    rotations_wxyz = rc.matrix_to_quaternion(rot_mats).numpy()
    rotations_deg = np.degrees(quat.to_euler(rotations_wxyz, order="zyx"))

    positions = np.zeros((1, len(names), 3), dtype=np.float32)
    positions[0, 0] = mean[-3:].astype(np.float32)

    path = output_dir / "echoavatar_mean_source_pose.bvh"
    write_bvh(
        path,
        rotations_deg=rotations_deg,
        positions=positions,
        names=names,
        parents=parents,
        offsets=offsets,
    )
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export EchoAvatar reference BVHs for Blender inspection."
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT_DIR / "exports" / "echoavatar_reference"),
        help="Directory where BVH files will be written.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    paths = [
        export_rest_pose(output_dir),
        export_mean_source_pose(output_dir),
    ]

    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
