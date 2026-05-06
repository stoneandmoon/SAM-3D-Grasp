#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
dimos_sam3d_grasp_skill.py

作用：
  把 SAM3D-Grasp + Contact-GraspNet 端到端流程封装成 DimOS Skill。

它调用你已经跑通的：
  pipeline4_with_contact_graspnet.py

最终输出：
  1. object_pcd_rgbd
  2. object_pose_rgbd_json
  3. grasp_config_json
  4. best_grasp

注意：
  - DimOS skill 本身不直接 import SAM3D / TensorFlow。
  - 它通过 subprocess 调用 pipeline4_with_contact_graspnet.py。
  - 这样可以安全跨两个 conda 环境：
      grasp_env              跑 SAM3D / RGB-D pipeline
      contact_graspnet_env   跑 Contact-GraspNet
"""

import os
import json
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List


# =========================================================
# DimOS imports
# =========================================================
# 不同 DimOS 版本 API 可能略有差异：
# 官方新文档推荐 @skill；
# 老例子里常见 @rpc。
# 这里做兼容：优先使用 @skill，失败就退回 @rpc。
from dimos.core.module import Module
from dimos.core.core import rpc

try:
    from dimos.agents.annotation import skill
except Exception:
    skill = rpc


REPO_ROOT = Path("/root/SAM-3D-Grasp")


# =========================================================
# 工具函数
# =========================================================

def _run_bash(script: str, cwd: Path = REPO_ROOT, timeout: Optional[int] = None) -> str:
    """
    运行 bash 脚本，并返回 stdout + stderr。
    """
    result = subprocess.run(
        ["bash", "-lc", script],
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )

    output = result.stdout or ""

    if result.returncode != 0:
        raise RuntimeError(
            f"命令执行失败，returncode={result.returncode}\n\n"
            f"执行脚本：\n{script}\n\n"
            f"输出：\n{output}"
        )

    return output


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON 不存在: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _summarize_grasp_config(grasp_json: Path, top_n: int = 3) -> Dict[str, Any]:
    """
    读取最终抓取配置 JSON，并返回 DimOS/Agent 更容易使用的摘要。
    """
    data = _load_json(grasp_json)
    grasps = data.get("grasps", [])

    best = grasps[0] if grasps else None

    return {
        "grasp_config_json": str(grasp_json),
        "coordinate_frame": data.get("coordinate_frame", "RGB-D camera frame"),
        "top_k": data.get("top_k"),
        "num_valid_grasps": data.get("num_valid_grasps"),
        "num_grasps_missing_width": data.get("num_grasps_missing_width"),
        "best_grasp": best,
        "top_grasps": grasps[:top_n],
    }


def _build_result(out_root: Path, top_n: int = 3) -> Dict[str, Any]:
    """
    整理最终返回给 DimOS 的结果。
    """
    object_pcd = out_root / "final" / "sam3d_pure_in_rgbd.ply"
    object_pose = out_root / "final" / "sam3d_pose_in_rgbd.json"
    grasp_json = out_root / "final_topdown_graspnet" / "contact_grasp_top20_rgbd_with_width.json"

    if not object_pcd.exists():
        raise FileNotFoundError(f"物体点云不存在: {object_pcd}")

    if not object_pose.exists():
        raise FileNotFoundError(f"物体位姿 JSON 不存在: {object_pose}")

    if not grasp_json.exists():
        raise FileNotFoundError(f"抓取配置 JSON 不存在: {grasp_json}")

    grasp_summary = _summarize_grasp_config(grasp_json, top_n=top_n)

    return {
        "status": "success",
        "object_pcd_rgbd": str(object_pcd),
        "object_pose_rgbd_json": str(object_pose),
        "grasp_config_json": str(grasp_json),
        "grasp_summary": grasp_summary,
        "best_grasp": grasp_summary.get("best_grasp"),
    }


# =========================================================
# DimOS Skill Module
# =========================================================

class SAM3DContactGraspSkill(Module):
    """
    DimOS Skill:
      语言 + RGB-D → SAM3D 目标重建 / RGB-D 配准 → Contact-GraspNet 抓取配置。

    当前版本提供两个 skill：

    1. generate_grasp_from_existing_scene
       使用已有 output_pipeline4_kettle 的中间结果，快速重新生成抓取配置。
       对应你刚刚跑通的 --skip-pipeline。

    2. generate_grasp_end_to_end
       从 pipeline4 开始完整跑，再生成抓取配置。
       适合最终演示，但耗时更长。
    """

    @skill
    def generate_grasp_from_existing_scene(
        self,
        out_root: str = "output_pipeline4_kettle",
        top_k: int = 20,
    ) -> Dict[str, Any]:
        """
        使用已有 pipeline 输出，生成最终抓取配置。

        适用场景：
          - 已经有 output_pipeline4_kettle/final_topdown_graspnet/topdown_graspnet_input.npy
          - 只想重新跑 Contact-GraspNet 和 JSON 整理
          - 调试 DimOS skill 是否能调用成功

        返回：
          object_pcd_rgbd
          object_pose_rgbd_json
          grasp_config_json
          best_grasp
        """

        out_root_path = REPO_ROOT / out_root

        cmd = f"""
set -e
source /home/vipuser/miniconda3/etc/profile.d/conda.sh
conda activate grasp_env

cd "{REPO_ROOT}"

export TORCH_HOME=/root/.cache/torch
export PYOPENGL_PLATFORM=egl
export QT_QPA_PLATFORM=offscreen

python pipeline4_with_contact_graspnet.py \\
  --skip-pipeline \\
  --out-root "{out_root}" \\
  --top-k {int(top_k)}
"""

        output = _run_bash(cmd, cwd=REPO_ROOT)

        result = _build_result(out_root_path, top_n=min(3, int(top_k)))
        result["mode"] = "existing_scene"
        result["log_tail"] = output[-4000:]

        return result

    @skill
    def generate_grasp_end_to_end(
        self,
        pipeline_script: str = "pipeline4.py",
        out_root: str = "output_pipeline4_kettle",
        top_k: int = 20,
    ) -> Dict[str, Any]:
        """
        完整运行 SAM3D-Grasp + Contact-GraspNet 端到端 pipeline。

        适用场景：
          - 最终演示
          - 重新从 RGB-D/语音开始跑
          - 需要同时更新物体点云、物体位姿和抓取配置

        注意：
          当前 pipeline_script 里如果还是写死 TEST_IMAGE / TEST_DEPTH / TEST_AUDIO，
          那它仍然会用脚本内部配置。
          后续要接真实相机时，需要把 pipeline_script 改成可接收当前 RGB-D 的版本。
        """

        out_root_path = REPO_ROOT / out_root

        cmd = f"""
set -e
source /home/vipuser/miniconda3/etc/profile.d/conda.sh
conda activate grasp_env

cd "{REPO_ROOT}"

export TORCH_HOME=/root/.cache/torch
export PYOPENGL_PLATFORM=egl
export QT_QPA_PLATFORM=offscreen

python pipeline4_with_contact_graspnet.py \\
  --pipeline-script "{pipeline_script}" \\
  --out-root "{out_root}" \\
  --top-k {int(top_k)}
"""

        output = _run_bash(cmd, cwd=REPO_ROOT)

        result = _build_result(out_root_path, top_n=min(3, int(top_k)))
        result["mode"] = "end_to_end"
        result["pipeline_script"] = pipeline_script
        result["log_tail"] = output[-4000:]

        return result

    @skill
    def get_best_grasp(
        self,
        grasp_config_json: str = "output_pipeline4_kettle/final_topdown_graspnet/contact_grasp_top20_rgbd_with_width.json",
    ) -> Dict[str, Any]:
        """
        读取已有抓取配置 JSON，返回 Top-1 抓取。

        返回的 best_grasp 包含：
          T_grasp_rgbd
          position_rgbd
          rotation_rgbd
          contact_point_rgbd
          score
          gripper_width_m
        """

        grasp_path = Path(grasp_config_json)

        if not grasp_path.is_absolute():
            grasp_path = REPO_ROOT / grasp_path

        summary = _summarize_grasp_config(grasp_path, top_n=3)

        return {
            "status": "success",
            "grasp_config_json": str(grasp_path),
            "coordinate_frame": summary["coordinate_frame"],
            "best_grasp": summary["best_grasp"],
            "top_grasps": summary["top_grasps"],
        }