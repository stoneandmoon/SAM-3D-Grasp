#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pipeline4_with_contact_graspnet.py

作用：
  把原来的 pipeline4 和 Contact-GraspNet 抓取配置生成融合成一个总控脚本。

整体流程：
  1. 在当前 grasp_env 中运行 pipeline4 / pipeline4_kettle / pipeline4_interactive。
  2. 检查 topdown_graspnet_input.npy 是否存在。
  3. 通过 subprocess 切换到 contact_graspnet_env，运行 Contact-GraspNet inference。
  4. 读取 predictions_topdown_graspnet_input.npz。
  5. 读取 rgbd_to_topdown_graspnet.json。
  6. 将 Contact-GraspNet 预测的 grasp 从 topdown 坐标系转换回 RGB-D camera frame。
  7. 导出最终抓取配置：
       output_pipeline4_kettle/final_topdown_graspnet/contact_grasp_top20_rgbd_with_width.json

重要：
  - 不要把 grasp_env 和 contact_graspnet_env 合并。
  - 本脚本本身在 grasp_env 中运行。
  - Contact-GraspNet 部分通过 bash + conda activate contact_graspnet_env 单独运行。
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent


# =========================================================
# 基础工具
# =========================================================

def run_cmd(cmd, cwd=None, env=None, check=True):
    print("\n" + "=" * 100)
    print("[RUN]")
    if isinstance(cmd, list):
        print(" ".join(str(x) for x in cmd))
    else:
        print(cmd)
    print("=" * 100)

    result = subprocess.run(
        cmd,
        cwd=str(cwd or REPO_ROOT),
        shell=isinstance(cmd, str),
        env=env,
    )

    if check and result.returncode != 0:
        raise RuntimeError(f"命令失败，returncode={result.returncode}")

    return result.returncode


def run_bash(script, cwd=None, check=True):
    print("\n" + "=" * 100)
    print("[BASH]")
    print(script)
    print("=" * 100)

    result = subprocess.run(
        ["bash", "-lc", script],
        cwd=str(cwd or REPO_ROOT),
    )

    if check and result.returncode != 0:
        raise RuntimeError(f"Bash 命令失败，returncode={result.returncode}")

    return result.returncode


def ensure_exists(path, name="file"):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{name} 不存在: {path}")
    return path


def to_homogeneous(points):
    points = np.asarray(points, dtype=np.float64)
    ones = np.ones((points.shape[0], 1), dtype=np.float64)
    return np.concatenate([points, ones], axis=1)


def transform_points(T, points):
    points = np.asarray(points, dtype=np.float64)
    ph = to_homogeneous(points)
    out = (T @ ph.T).T[:, :3]
    return out


def is_invalid_grasp(T):
    T = np.asarray(T)
    return np.allclose(T, -1.0) or np.isnan(T).any()


# =========================================================
# Step A：运行原 pipeline4
# =========================================================

def find_pipeline_script(user_script=None):
    candidates = []

    if user_script:
        candidates.append(user_script)

    candidates.extend([
        "pipeline4_kettle.py",
        "pipeline4.py",
        "pipeline4_interactive.py",
    ])

    for c in candidates:
        p = REPO_ROOT / c
        if p.exists():
            return p

    raise FileNotFoundError(
        "找不到 pipeline 脚本。请确认存在 pipeline4_kettle.py / pipeline4.py / pipeline4_interactive.py"
    )


def run_pipeline4(pipeline_script, skip_pipeline=False):
    if skip_pipeline:
        print("[Skip] 跳过 pipeline4，直接使用已有输出。")
        return

    pipeline_script = find_pipeline_script(pipeline_script)

    print(f"[Pipeline4] 使用脚本: {pipeline_script}")

    env = os.environ.copy()
    env.setdefault("TORCH_HOME", "/root/.cache/torch")
    env["PYOPENGL_PLATFORM"] = "egl"
    env["HYDRA_FULL_ERROR"] = "1"

    run_cmd(
        [sys.executable, str(pipeline_script)],
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )


# =========================================================
# Step B：运行 Contact-GraspNet inference
# =========================================================

def run_contact_graspnet_inference(
    topdown_input_npy,
    conda_sh,
    contact_env,
    ckpt_dir,
    contact_graspnet_dir,
    pred_npz_name="predictions_topdown_graspnet_input.npz",
):
    topdown_input_npy = Path(topdown_input_npy).resolve()
    contact_graspnet_dir = Path(contact_graspnet_dir).resolve()

    ensure_exists(topdown_input_npy, "Contact-GraspNet 输入 npy")
    ensure_exists(contact_graspnet_dir / "contact_graspnet" / "inference.py", "Contact-GraspNet inference.py")

    pred_npz = contact_graspnet_dir / "results" / pred_npz_name

    bash_script = f"""
set -e

source "{conda_sh}"
conda activate "{contact_env}"

cd "{contact_graspnet_dir}"

echo "[ENV] python=$(which python)"
python -V

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYOPENGL_PLATFORM=egl
export QT_QPA_PLATFORM=offscreen

mkdir -p results
rm -f "{pred_npz}"

xvfb-run -a python contact_graspnet/inference.py \\
  --ckpt_dir "{ckpt_dir}" \\
  --np_path "{topdown_input_npy}"

echo "[DONE] Contact-GraspNet inference finished."
"""

    run_bash(bash_script, cwd=contact_graspnet_dir, check=True)

    ensure_exists(pred_npz, "Contact-GraspNet prediction npz")

    return pred_npz


# =========================================================
# Step C：读取 topdown -> RGB-D 变换
# =========================================================

def _as_matrix(v):
    arr = np.asarray(v, dtype=np.float64)
    if arr.shape == (4, 4):
        return arr
    return None


def load_T_topdown_to_rgbd(topdown_json_path):
    """
    读取 topdown/rgbd 变换 JSON。

    优先支持：
      T_topdown_to_rgbd
      T_graspnet_to_rgbd
      T_cgn_to_rgbd
      topdown_to_rgbd

    如果只有反向：
      T_rgbd_to_topdown
      T_rgbd_to_graspnet
      rgbd_to_topdown

    就自动求逆。
    """
    topdown_json_path = Path(topdown_json_path)
    ensure_exists(topdown_json_path, "topdown 变换 JSON")

    with open(topdown_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    direct_keys = [
        "T_topdown_to_rgbd",
        "T_graspnet_to_rgbd",
        "T_cgn_to_rgbd",
        "T_topdown_to_camera",
        "topdown_to_rgbd",
        "graspnet_to_rgbd",
    ]

    inverse_keys = [
        "T_rgbd_to_topdown",
        "T_rgbd_to_graspnet",
        "T_camera_to_topdown",
        "rgbd_to_topdown",
        "rgbd_to_graspnet",
    ]

    for k in direct_keys:
        if k in data:
            T = _as_matrix(data[k])
            if T is not None:
                print(f"[Transform] 使用 direct key: {k}")
                return T, data

            if isinstance(data[k], dict):
                for subk in ["matrix", "T", "value"]:
                    if subk in data[k]:
                        T = _as_matrix(data[k][subk])
                        if T is not None:
                            print(f"[Transform] 使用 direct key: {k}.{subk}")
                            return T, data

    for k in inverse_keys:
        if k in data:
            T = _as_matrix(data[k])
            if T is not None:
                print(f"[Transform] 使用 inverse key: {k}，自动求逆")
                return np.linalg.inv(T), data

            if isinstance(data[k], dict):
                for subk in ["matrix", "T", "value"]:
                    if subk in data[k]:
                        T = _as_matrix(data[k][subk])
                        if T is not None:
                            print(f"[Transform] 使用 inverse key: {k}.{subk}，自动求逆")
                            return np.linalg.inv(T), data

    # 兜底：递归搜索所有 4x4 矩阵
    possible = []

    def walk(obj, prefix=""):
        if isinstance(obj, dict):
            for kk, vv in obj.items():
                walk(vv, f"{prefix}.{kk}" if prefix else kk)
        else:
            T = _as_matrix(obj)
            if T is not None:
                possible.append((prefix, T))

    walk(data)

    if len(possible) == 1:
        key, T = possible[0]
        print(f"[Transform] 警告：只找到一个 4x4 矩阵，默认它是 T_topdown_to_rgbd: {key}")
        return T, data

    keys = list(data.keys())
    raise RuntimeError(
        "无法从 topdown JSON 中确定 T_topdown_to_rgbd。\n"
        f"JSON 路径: {topdown_json_path}\n"
        f"顶层 keys: {keys}\n"
        "请检查 rgbd_to_topdown_graspnet.json 里到底保存的矩阵字段名。"
    )


# =========================================================
# Step D：将 Contact-GraspNet 输出转换为 RGB-D 抓取 JSON
# =========================================================

def convert_predictions_to_rgbd_json(
    pred_npz,
    topdown_json,
    out_dir,
    top_k=20,
):
    pred_npz = Path(pred_npz)
    topdown_json = Path(topdown_json)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ensure_exists(pred_npz, "Contact-GraspNet prediction npz")
    ensure_exists(topdown_json, "topdown transform json")

    data = np.load(pred_npz, allow_pickle=True)

    required = ["pred_grasps_cam", "scores", "contact_pts"]
    for k in required:
        if k not in data.files:
            raise RuntimeError(f"{pred_npz} 缺少字段: {k}")

    pred_grasps = data["pred_grasps_cam"].item()
    scores = data["scores"].item()
    contact_pts = data["contact_pts"].item()

    gripper_openings = None
    if "gripper_openings" in data.files:
        gripper_openings = data["gripper_openings"].item()
        print("[Width] 已读取 gripper_openings")
    else:
        print("[Width] 警告：prediction npz 中没有 gripper_openings")

    T_topdown_to_rgbd, topdown_meta = load_T_topdown_to_rgbd(topdown_json)

    pred_grasps_rgbd = {}
    contact_pts_rgbd = {}
    all_items = []

    for obj_id, grasps in pred_grasps.items():
        grasps = np.asarray(grasps)
        obj_scores = np.asarray(scores[obj_id])
        obj_contacts = np.asarray(contact_pts[obj_id])

        rgbd_grasps = []
        rgbd_contacts = []

        for i, T_topdown in enumerate(grasps):
            if is_invalid_grasp(T_topdown):
                rgbd_grasps.append(T_topdown)
                rgbd_contacts.append(obj_contacts[i])
                continue

            T_rgbd = T_topdown_to_rgbd @ np.asarray(T_topdown, dtype=np.float64)
            rgbd_grasps.append(T_rgbd)

            c_rgbd = transform_points(T_topdown_to_rgbd, obj_contacts[i].reshape(1, 3))[0]
            rgbd_contacts.append(c_rgbd)

            item = {
                "obj_id": str(obj_id),
                "index": int(i),
                "score": float(obj_scores[i]),
                "T_grasp_rgbd": T_rgbd.tolist(),
                "position_rgbd": T_rgbd[:3, 3].tolist(),
                "rotation_rgbd": T_rgbd[:3, :3].tolist(),
                "contact_point_rgbd": c_rgbd.tolist(),
            }

            if gripper_openings is not None and obj_id in gripper_openings:
                opening = float(np.asarray(gripper_openings[obj_id][i]).reshape(-1)[0])
                item["gripper_opening_m"] = opening
                item["gripper_width_m"] = opening

            all_items.append(item)

        pred_grasps_rgbd[obj_id] = np.asarray(rgbd_grasps, dtype=np.float64)
        contact_pts_rgbd[obj_id] = np.asarray(rgbd_contacts, dtype=np.float64)

    all_items.sort(key=lambda x: x["score"], reverse=True)
    top_items = all_items[: int(top_k)]

    out_npz = out_dir / "predictions_topdown_graspnet_rgbd.npz"
    save_kwargs = {
        "pred_grasps_cam": pred_grasps_rgbd,
        "scores": scores,
        "contact_pts": contact_pts_rgbd,
    }

    if gripper_openings is not None:
        save_kwargs["gripper_openings"] = gripper_openings

    np.savez(out_npz, **save_kwargs)

    out_json = out_dir / "contact_grasp_top20_rgbd_with_width.json"

    missing_width = 0
    for item in top_items:
        if "gripper_width_m" not in item:
            item["gripper_opening_m"] = None
            item["gripper_width_m"] = None
            missing_width += 1

    json_data = {
        "source_prediction_npz": str(pred_npz.resolve()),
        "topdown_json": str(topdown_json.resolve()),
        "coordinate_frame": "RGB-D camera frame",
        "definition": "T_grasp_rgbd = T_topdown_to_rgbd @ T_grasp_topdown",
        "num_valid_grasps": len(all_items),
        "top_k": int(top_k),
        "gripper_width_source": str(pred_npz.resolve()) if gripper_openings is not None else None,
        "gripper_width_field": "gripper_openings" if gripper_openings is not None else None,
        "gripper_width_unit": "meter",
        "num_grasps_missing_width": missing_width,
        "grasps": top_items,
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    print("\n" + "█" * 80)
    print("✅ RGB-D 抓取配置已生成")
    print(f"NPZ : {out_npz}")
    print(f"JSON: {out_json}")
    print(f"Top-K: {top_k}")
    print(f"有效抓取数: {len(all_items)}")
    print(f"缺失开合度: {missing_width}")
    if top_items:
        print(f"Top-1 score: {top_items[0]['score']:.4f}")
        print(f"Top-1 position_rgbd: {top_items[0]['position_rgbd']}")
        print(f"Top-1 gripper_width_m: {top_items[0].get('gripper_width_m')}")
    print("█" * 80)

    return out_json


# =========================================================
# 主程序
# =========================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pipeline-script",
        default=None,
        help="原始 pipeline 脚本，例如 pipeline4_kettle.py / pipeline4.py / pipeline4_interactive.py。默认自动寻找。",
    )

    parser.add_argument(
        "--skip-pipeline",
        action="store_true",
        help="跳过 SAM3D/RGB-D pipeline，直接使用已有 topdown_graspnet_input.npy。",
    )

    parser.add_argument(
        "--out-root",
        default="output_pipeline4_kettle",
        help="pipeline 输出目录，默认 output_pipeline4_kettle。",
    )

    parser.add_argument(
        "--conda-sh",
        default="/home/vipuser/miniconda3/etc/profile.d/conda.sh",
        help="conda.sh 路径。",
    )

    parser.add_argument(
        "--contact-env",
        default="contact_graspnet_env",
        help="Contact-GraspNet conda 环境名。",
    )

    parser.add_argument(
        "--contact-dir",
        default="contact_graspnet",
        help="Contact-GraspNet 目录，默认 contact_graspnet。",
    )

    parser.add_argument(
        "--ckpt-dir",
        default="checkpoints/scene_test_2048_bs3_hor_sigma_001",
        help="Contact-GraspNet checkpoint 目录，相对于 contact_graspnet 目录。",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="导出 Top-K 抓取候选。",
    )

    args = parser.parse_args()

    out_root = REPO_ROOT / args.out_root

    topdown_dir = out_root / "final_topdown_graspnet"
    topdown_input_npy = topdown_dir / "topdown_graspnet_input.npy"
    topdown_json = topdown_dir / "rgbd_to_topdown_graspnet.json"

    contact_dir = REPO_ROOT / args.contact_dir

    # 1. 跑原 pipeline4
    run_pipeline4(
        pipeline_script=args.pipeline_script,
        skip_pipeline=args.skip_pipeline,
    )

    # 2. 检查 topdown 输入
    if not topdown_input_npy.exists():
        raise FileNotFoundError(
            f"没有找到 Contact-GraspNet 输入文件: {topdown_input_npy}\n"
            "说明你的原 pipeline 还没有自动生成 topdown_graspnet_input.npy。\n"
            "请先用你原来的 prepare_topdown_graspnet_input.py 生成它，"
            "或者把 prepare_topdown_graspnet_input.py 的 --help 发我，我可以继续把这一步也接进来。"
        )

    if not topdown_json.exists():
        raise FileNotFoundError(
            f"没有找到 topdown 坐标变换 JSON: {topdown_json}\n"
            "这个文件用于把 Contact-GraspNet 的 topdown 坐标系抓取结果转回 RGB-D camera frame。"
        )

    # 3. 跑 Contact-GraspNet
    pred_npz = run_contact_graspnet_inference(
        topdown_input_npy=topdown_input_npy,
        conda_sh=args.conda_sh,
        contact_env=args.contact_env,
        ckpt_dir=args.ckpt_dir,
        contact_graspnet_dir=contact_dir,
    )

    # 4. 转换为 RGB-D 抓取 JSON，含 gripper_width_m
    final_grasp_json = convert_predictions_to_rgbd_json(
        pred_npz=pred_npz,
        topdown_json=topdown_json,
        out_dir=topdown_dir,
        top_k=args.top_k,
    )

    print("\n" + "█" * 80)
    print("🎉 端到端融合 pipeline 完成")
    print(f"物体点云: {out_root / 'final' / 'sam3d_pure_in_rgbd.ply'}")
    print(f"物体位姿: {out_root / 'final' / 'sam3d_pose_in_rgbd.json'}")
    print(f"最终抓取配置: {final_grasp_json}")
    print("█" * 80)


if __name__ == "__main__":
    main()