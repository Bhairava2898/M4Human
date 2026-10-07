"""Raw synchronized multimodal loader for M4Human.

Expected raw layout
-------------------
<root>/
  p1/
    actions/
      1/
        mmwave/*.mat
        rgb/*.png
        depth/*.png
    parameters/1.json
    ins_ext/...

Each sample returns synchronized radar point cloud, RGB, depth point cloud,
and MoCap 3D keypoints. Geometric outputs are expressed in the radar frame.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .data_loader_Load_data import (
    gender_info,
    load_json_data,
    load_mat_file,
    load_transformation_matrices,
)
from .dataset_config_clean import info_init, split_indices
from .lmdb_utils import extract_frame_index


NON_VALID_INDICATOR = {
    (1, 12, 58),
    (1, 13, 491),
    (1, 31, 324),
    (1, 36, 284),
    (1, 37, 43),
    (5, 43, 551),
    (5, 43, 552),
    (5, 47, 136),
    (5, 47, 140),
}


@dataclass(frozen=True)
class RawFrameRecord:
    radar_path: str
    rgb_path: str
    depth_path: str
    json_path: str
    json_index: int
    indicator: Tuple[int, int, int]
    frame_key: str
    gender: int


def _numeric_subject_dirs(
    root_dir: str,
    select_subjects: Optional[Sequence[int]],
) -> List[str]:
    selected = None if select_subjects is None else {int(x) for x in select_subjects}
    out = []

    for name in os.listdir(root_dir):
        path = os.path.join(root_dir, name)
        match = re.fullmatch(r"p(\d+)", name.lower())

        if not os.path.isdir(path) or match is None:
            continue

        subject_id = int(match.group(1))
        if selected is None or subject_id in selected:
            out.append(name)

    return sorted(out, key=lambda x: int(re.findall(r"\d+", x)[0]))


def build_raw_index(
    root_dir: str,
    select_subjects: Optional[Sequence[int]] = None,
    select_actions: Optional[Sequence[int]] = None,
    drop_known_invalid: bool = True,
) -> List[RawFrameRecord]:
    """Build the raw synchronized RGB/depth/radar index.

    M4Human's historical RGB-D loader enumerated the union of modality frame
    keys and used that union position as the JSON/annotation index. We retain
    that behavior here so the raw modalities stay aligned with the annotations.
    """
    root_dir = os.path.abspath(os.path.expanduser(root_dir))
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"M4Human raw root does not exist: {root_dir}")

    action_set = None if select_actions is None else {int(x) for x in select_actions}
    records: List[RawFrameRecord] = []

    for subject_dir in _numeric_subject_dirs(root_dir, select_subjects):
        subject_id = int(re.findall(r"\d+", subject_dir)[0])
        subject_root = os.path.join(root_dir, subject_dir)
        actions_root = os.path.join(subject_root, "actions")
        parameters_root = os.path.join(subject_root, "parameters")

        if not os.path.isdir(actions_root) or not os.path.isdir(parameters_root):
            continue

        for action_id in range(1, 51):
            if action_set is not None and action_id not in action_set:
                continue

            json_path = os.path.join(parameters_root, f"{action_id}.json")
            radar_dir = os.path.join(actions_root, str(action_id), "mmwave")
            rgb_dir = os.path.join(actions_root, str(action_id), "rgb")
            depth_dir = os.path.join(actions_root, str(action_id), "depth")

            if not os.path.isfile(json_path):
                continue
            if not all(os.path.isdir(p) for p in (radar_dir, rgb_dir, depth_dir)):
                continue

            radar_files = {
                extract_frame_index(f): os.path.join(radar_dir, f)
                for f in os.listdir(radar_dir)
                if f.lower().endswith(".mat")
            }
            rgb_files = {
                extract_frame_index(f): os.path.join(rgb_dir, f)
                for f in os.listdir(rgb_dir)
                if f.lower().endswith(".png")
            }
            depth_files = {
                extract_frame_index(f): os.path.join(depth_dir, f)
                for f in os.listdir(depth_dir)
                if f.lower().endswith(".png")
            }

            common = set(radar_files) & set(rgb_files) & set(depth_files)
            union = set(radar_files) | set(rgb_files) | set(depth_files)

            def frame_sort_key(key: str):
                try:
                    return (0, int(key))
                except ValueError:
                    return (1, key)

            for json_index, frame_key in enumerate(sorted(union, key=frame_sort_key)):
                if frame_key not in common:
                    continue

                indicator = (subject_id, action_id, json_index)
                if drop_known_invalid and indicator in NON_VALID_INDICATOR:
                    continue

                gender_key = f"P{subject_id}"
                if gender_key not in gender_info:
                    raise KeyError(f"Missing gender metadata for {gender_key}")

                records.append(
                    RawFrameRecord(
                        radar_path=radar_files[frame_key],
                        rgb_path=rgb_files[frame_key],
                        depth_path=depth_files[frame_key],
                        json_path=json_path,
                        json_index=json_index,
                        indicator=indicator,
                        frame_key=frame_key,
                        gender=int(gender_info[gender_key]),
                    )
                )

    records.sort(key=lambda rec: rec.indicator)
    return records


def _translation_to_meters(
    translation: np.ndarray,
    scale: Union[str, float],
) -> np.ndarray:
    translation = np.asarray(translation, dtype=np.float32).reshape(3)

    if isinstance(scale, (int, float)):
        return translation / float(scale)

    if scale != "auto":
        raise ValueError("translation scale must be 'auto' or a numeric divisor")

    return (
        translation / 1000.0
        if float(np.linalg.norm(translation)) > 10.0
        else translation
    )


def _points_to_meters(
    points: np.ndarray,
    scale: Union[str, float],
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)

    if isinstance(scale, (int, float)):
        return points / float(scale)

    if scale != "auto":
        raise ValueError("point scale must be 'auto' or a numeric divisor")

    finite = np.abs(points[np.isfinite(points)])
    if finite.size == 0:
        return points

    return points / 1000.0 if float(np.median(finite)) > 20.0 else points


def _fixed_size_points(
    points: np.ndarray,
    target: Optional[int],
) -> Tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float32)
    n_points = len(points)

    if target is None:
        return points, np.ones(n_points, dtype=bool)

    if target <= 0:
        raise ValueError("target point count must be positive or None")

    if n_points >= target:
        indices = np.linspace(0, n_points - 1, target, dtype=np.int64)
        return points[indices], np.ones(target, dtype=bool)

    out = np.zeros((target, points.shape[1]), dtype=np.float32)
    mask = np.zeros(target, dtype=bool)

    if n_points:
        out[:n_points] = points
        mask[:n_points] = True

    return out, mask


def _depth_to_camera_points(
    depth_image: np.ndarray,
    intrinsic: np.ndarray,
    depth_scale: Union[str, float],
    stride: int,
    min_depth_m: float,
    max_depth_m: float,
) -> np.ndarray:
    depth = np.asarray(depth_image)

    if depth.ndim == 3:
        depth = depth[..., 0]

    if depth.ndim != 2:
        raise ValueError(f"Expected HxW depth image, got shape {depth.shape}")

    depth = depth.astype(np.float32)

    if isinstance(depth_scale, (int, float)):
        depth_m = depth / float(depth_scale)
    elif depth_scale == "auto":
        valid_raw = depth[np.isfinite(depth) & (depth > 0)]
        if valid_raw.size and float(np.median(valid_raw)) > 20.0:
            depth_m = depth / 1000.0
        else:
            depth_m = depth
    else:
        raise ValueError("depth_scale must be 'auto' or a numeric divisor")

    stride = max(1, int(stride))
    v, u = np.mgrid[
        0 : depth.shape[0] : stride,
        0 : depth.shape[1] : stride,
    ]
    z = depth_m[::stride, ::stride]

    valid = (
        np.isfinite(z)
        & (z >= min_depth_m)
        & (z <= max_depth_m)
    )

    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32)

    u = u[valid].astype(np.float32)
    v = v[valid].astype(np.float32)
    z = z[valid].astype(np.float32)

    K = np.asarray(intrinsic, dtype=np.float32).reshape(3, 3)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    if abs(float(fx)) < 1e-8 or abs(float(fy)) < 1e-8:
        raise ValueError("Invalid camera intrinsic: fx/fy cannot be zero")

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    return np.stack((x, y, z), axis=1).astype(np.float32)


def _camera_to_radar(
    camera_points_m: np.ndarray,
    radar_to_cam_R: np.ndarray,
    radar_to_cam_t_m: np.ndarray,
) -> np.ndarray:
    """Transform camera-frame points into the radar frame.

    The calibration convention used by M4Human is
        p_cam = R_radar_to_cam @ p_radar + t_radar_to_cam.
    """
    R = np.asarray(radar_to_cam_R, dtype=np.float32).reshape(3, 3)
    t = np.asarray(radar_to_cam_t_m, dtype=np.float32).reshape(1, 3)

    return ((camera_points_m - t) @ R).astype(np.float32)


def _vicon_to_radar(
    vicon_points_m: np.ndarray,
    vicon_to_cam_R: np.ndarray,
    vicon_to_cam_t_m: np.ndarray,
    radar_to_cam_R: np.ndarray,
    radar_to_cam_t_m: np.ndarray,
) -> np.ndarray:
    R_vicon_to_cam = np.asarray(
        vicon_to_cam_R,
        dtype=np.float32,
    ).reshape(3, 3)
    t_vicon_to_cam = np.asarray(
        vicon_to_cam_t_m,
        dtype=np.float32,
    ).reshape(1, 3)

    camera_points = (
        np.asarray(vicon_points_m, dtype=np.float32) @ R_vicon_to_cam.T
        + t_vicon_to_cam
    )

    return _camera_to_radar(
        camera_points,
        radar_to_cam_R,
        radar_to_cam_t_m,
    )


class M4HumanRawMultimodalDataset(Dataset):
    """Synchronized raw M4Human loader.

    Main outputs
    ------------
    radar_pc:
        [Nr, 4], xyz in metres in the radar frame plus intensity.
    rgb:
        [3, H, W], float32 in [0, 1].
    depth_pc:
        [Nd, 3], metres in the radar frame.
    keypoints_3d:
        [J, 3], MoCap keypoints in metres in the radar frame.
    radar_mask:
        [Nr], valid-point mask after padding.
    depth_mask:
        [Nd], valid-point mask after padding.
    indicator:
        [3] = [subject_id, action_id, synchronized_frame_index].

    Notes
    -----
    The historical M4Human RGB-D loader uses synchronized depth PNGs alongside
    RGB PNGs. This implementation assumes those depth images are registered to
    the RGB camera and uses cam_ins.txt to unproject them.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "all",
        scale_id: str = "p1",
        split_id: str = "s1",
        select_subjects: Optional[Sequence[int]] = None,
        select_actions: Optional[Sequence[int]] = None,
        radar_num_points: Optional[int] = 1000,
        depth_num_points: Optional[int] = 8192,
        depth_stride: int = 2,
        min_depth_m: float = 0.1,
        max_depth_m: float = 8.0,
        depth_scale: Union[str, float] = "auto",
        joint_scale: Union[str, float] = "auto",
        vicon_translation_scale: Union[str, float] = "auto",
        radar_translation_scale: Union[str, float] = "auto",
        include_depth_image: bool = False,
        include_smplx_params: bool = True,
        include_calibration: bool = False,
        drop_known_invalid: bool = True,
    ):
        if split not in {"all", "train", "val", "test"}:
            raise ValueError("split must be one of: all, train, val, test")

        self.root_dir = os.path.abspath(os.path.expanduser(root_dir))
        self.radar_num_points = radar_num_points
        self.depth_num_points = depth_num_points
        self.depth_stride = depth_stride
        self.min_depth_m = min_depth_m
        self.max_depth_m = max_depth_m
        self.depth_scale = depth_scale
        self.joint_scale = joint_scale
        self.vicon_translation_scale = vicon_translation_scale
        self.radar_translation_scale = radar_translation_scale
        self.include_depth_image = include_depth_image
        self.include_smplx_params = include_smplx_params
        self.include_calibration = include_calibration

        records = build_raw_index(
            self.root_dir,
            select_subjects=select_subjects,
            select_actions=select_actions,
            drop_known_invalid=drop_known_invalid,
        )

        if split != "all":
            indicators = [list(rec.indicator) for rec in records]
            scale_cfg = info_init(scale_id=scale_id)
            split_map = split_indices(
                indicators,
                scale_cfg=scale_cfg,
                scheme=split_id,
            )
            keep = {tuple(x) for x in split_map[split]}
            records = [rec for rec in records if rec.indicator in keep]

        self.records = records
        self._json_cache: Dict[str, dict] = {}

    def __len__(self) -> int:
        return len(self.records)

    def _json(self, path: str) -> dict:
        data = self._json_cache.get(path)

        if data is None:
            data = load_json_data(path)
            if data is None:
                raise RuntimeError(f"Failed to load annotation JSON: {path}")
            self._json_cache[path] = data

        return data

    def __getitem__(self, idx: int) -> Dict[str, object]:
        rec = self.records[idx]

        radar_data = load_mat_file(rec.radar_path)
        if radar_data is None:
            raise RuntimeError(f"Failed to load radar MAT: {rec.radar_path}")

        rgb_bgr = cv2.imread(rec.rgb_path, cv2.IMREAD_COLOR)
        if rgb_bgr is None:
            raise RuntimeError(f"Failed to load RGB image: {rec.rgb_path}")
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

        depth = cv2.imread(rec.depth_path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise RuntimeError(f"Failed to load depth image: {rec.depth_path}")

        annotations = self._json(rec.json_path)
        json_index = rec.json_index

        if json_index >= len(annotations["joints"]):
            raise IndexError(
                f"JSON index {json_index} exceeds joints length "
                f"{len(annotations['joints'])} for {rec.json_path}. "
                f"Indicator={rec.indicator}, frame_key={rec.frame_key}"
            )

        ins_ext_dir = os.path.normpath(
            os.path.join(
                os.path.dirname(rec.json_path),
                "..",
                "ins_ext",
            )
        )

        calibration = load_transformation_matrices(
            ins_ext_dir,
            list(rec.indicator),
        )

        if calibration is None or any(x is None for x in calibration):
            raise RuntimeError(
                f"Failed to load calibration for indicator {rec.indicator}"
            )

        (
            cam_intrinsic,
            vicon_to_cam_R,
            vicon_to_cam_t,
            radar_to_cam_R,
            radar_to_cam_t,
        ) = calibration

        vicon_to_cam_t_m = _translation_to_meters(
            vicon_to_cam_t,
            self.vicon_translation_scale,
        )
        radar_to_cam_t_m = _translation_to_meters(
            radar_to_cam_t,
            self.radar_translation_scale,
        )

        radar_pc = np.asarray(
            radar_data["RawPoints"],
            dtype=np.float32,
        )

        if radar_pc.ndim != 2 or radar_pc.shape[1] < 4:
            raise ValueError(
                f"Expected radar RawPoints [N, >=4], got "
                f"{radar_pc.shape} at {rec.radar_path}"
            )

        radar_pc = radar_pc[:, :4]
        radar_pc = radar_pc[np.isfinite(radar_pc).all(axis=1)]
        radar_pc, radar_mask = _fixed_size_points(
            radar_pc,
            self.radar_num_points,
        )

        depth_camera = _depth_to_camera_points(
            depth,
            cam_intrinsic,
            depth_scale=self.depth_scale,
            stride=self.depth_stride,
            min_depth_m=self.min_depth_m,
            max_depth_m=self.max_depth_m,
        )

        depth_radar = _camera_to_radar(
            depth_camera,
            radar_to_cam_R,
            radar_to_cam_t_m,
        )

        depth_radar, depth_mask = _fixed_size_points(
            depth_radar,
            self.depth_num_points,
        )

        joints_vicon = _points_to_meters(
            np.asarray(
                annotations["joints"][json_index],
                dtype=np.float32,
            ),
            self.joint_scale,
        )

        keypoints_radar = _vicon_to_radar(
            joints_vicon,
            vicon_to_cam_R,
            vicon_to_cam_t_m,
            radar_to_cam_R,
            radar_to_cam_t_m,
        )

        sample: Dict[str, object] = {
            "radar_pc": torch.from_numpy(radar_pc).float(),
            "radar_mask": torch.from_numpy(radar_mask),
            "rgb": (
                torch.from_numpy(np.ascontiguousarray(rgb))
                .permute(2, 0, 1)
                .float()
                / 255.0
            ),
            "depth_pc": torch.from_numpy(depth_radar).float(),
            "depth_mask": torch.from_numpy(depth_mask),
            "keypoints_3d": torch.from_numpy(keypoints_radar).float(),
            "indicator": torch.tensor(rec.indicator, dtype=torch.long),
            "frame_key": rec.frame_key,
            "gender": torch.tensor(rec.gender, dtype=torch.long),
        }

        if self.include_depth_image:
            depth_np = np.asarray(depth)

            if depth_np.ndim == 2:
                depth_np = depth_np[None]
            else:
                depth_np = depth_np.transpose(2, 0, 1)

            sample["depth"] = torch.from_numpy(
                np.ascontiguousarray(depth_np)
            ).float()

        if self.include_smplx_params:
            sample["smplx"] = {
                "betas": torch.as_tensor(
                    np.asarray(annotations["betas"][json_index])[:10],
                    dtype=torch.float32,
                ),
                "pose_body": torch.as_tensor(
                    np.asarray(annotations["pose_body"][json_index]),
                    dtype=torch.float32,
                ),
                "trans": torch.as_tensor(
                    np.asarray(annotations["trans"][json_index]),
                    dtype=torch.float32,
                ),
                "root_orient": torch.as_tensor(
                    np.asarray(annotations["root_orient"][json_index]),
                    dtype=torch.float32,
                ),
                "gender": torch.tensor(rec.gender, dtype=torch.long),
            }

        if self.include_calibration:
            sample["calibration"] = {
                "cam_intrinsic": torch.as_tensor(
                    np.asarray(cam_intrinsic),
                    dtype=torch.float32,
                ),
                "vicon_to_cam_R": torch.as_tensor(
                    np.asarray(vicon_to_cam_R),
                    dtype=torch.float32,
                ),
                "vicon_to_cam_t_m": torch.as_tensor(
                    vicon_to_cam_t_m,
                    dtype=torch.float32,
                ),
                "radar_to_cam_R": torch.as_tensor(
                    np.asarray(radar_to_cam_R),
                    dtype=torch.float32,
                ),
                "radar_to_cam_t_m": torch.as_tensor(
                    radar_to_cam_t_m,
                    dtype=torch.float32,
                ),
            }

        return sample
