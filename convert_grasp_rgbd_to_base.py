#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
from pathlib import Path
import numpy as np


def load_json(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到文件: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[SAVE] {path}")


def as_T(mat, name):
    arr = np.asarray(mat, dtype=np.float64)
    if arr.shape != (4, 4):
        raise ValueError(f"{name} 不是 4x4 矩阵，shape={arr.shape}")
    return arr


def transform_point(T, p):
    p = np.asarray(p, dtype=np.float64).reshape(3)
    ph = np.ones(4, dtype=np.float64)
    ph[:3] = p
    out = T @ ph
    return out[:3]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--grasp-json",
        default="output_pipeline4_kettle/final_topdown_graspnet/contact_grasp_top20_rgbd_with_width.json",
    )
    parser.add_argument(
        "--calib-json",
        default="robot_calib/T_camera_to_base.json",
    )
    parser.add_argument(
        "--out-json",
        default="output_pipeline4_kettle/final_topdown_graspnet/contact_grasp_top20_base_with_width.json",
    )

    args = parser.parse_args()

    grasp_data = load_json(args.grasp_json)
    calib_data = load_json(args.calib_json)

    if "T_camera_to_base" not in calib_data:
        raise KeyError("calib-json 里找不到 T_camera_to_base")

    T_camera_to_base = as_T(calib_data["T_camera_to_base"], "T_camera_to_base")

    out = dict(grasp_data)
    out["source_grasp_json"] = str(Path(args.grasp_json).resolve())
    out["source_calib_json"] = str(Path(args.calib_json).resolve())
    out["coordinate_frame"] = "robot base frame"
    out["definition"] = "T_grasp_base = T_camera_to_base @ T_grasp_rgbd"
    out["T_camera_to_base"] = T_camera_to_base.tolist()

    new_grasps = []

    for g in grasp_data.get("grasps", []):
        T_grasp_rgbd = as_T(g["T_grasp_rgbd"], "T_grasp_rgbd")
        T_grasp_base = T_camera_to_base @ T_grasp_rgbd

        g2 = dict(g)
        g2["T_grasp_base"] = T_grasp_base.tolist()
        g2["position_base"] = T_grasp_base[:3, 3].tolist()
        g2["rotation_base"] = T_grasp_base[:3, :3].tolist()

        if "contact_point_rgbd" in g:
            g2["contact_point_base"] = transform_point(
                T_camera_to_base,
                g["contact_point_rgbd"]
            ).tolist()

        new_grasps.append(g2)

    out["grasps"] = new_grasps

    if new_grasps:
        out["best_grasp_base"] = new_grasps[0]

    save_json(out, args.out_json)

    print("\n" + "=" * 80)
    print("转换完成")
    print("=" * 80)

    if new_grasps:
        best = new_grasps[0]
        print("Top-1 score:", best.get("score"))
        print("position_rgbd:", best.get("position_rgbd"))
        print("position_base:", best.get("position_base"))
        print("gripper_width_m:", best.get("gripper_width_m"))
        print("has T_grasp_base:", "T_grasp_base" in best)


if __name__ == "__main__":
    main()
