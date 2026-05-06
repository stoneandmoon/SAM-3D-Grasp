#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
sam3d_grasp_core.py

作用：
  把 pipeline4_interactive.py 封装成两个标准接口：

  1. grasp_from_files(...)
     离线文件接口：
       输入已有 RGB 图、Depth 图、语音文件或文字指令。
       用于现在调试和复现实验。

  2. grasp_current_scene(...)
     当前场景接口：
       由 scene_provider.capture_rgbd() 获取当前 RGB-D；
       由 voice_provider.record_once() 获取当前语音；
       然后调用同一个 pipeline 生成抓取位姿。

注意：
  这个文件不重写 SAM3 / SAM3D / 点云配准算法。
  它只是把你已经跑通的 pipeline4_interactive.py 变成可被程序调用的函数接口。
"""

import os
import json
import inspect
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Tuple


# =========================================================
# 环境设置
# =========================================================

REPO_ROOT = Path(__file__).resolve().parent

os.environ.setdefault("TORCH_HOME", "/root/.cache/torch")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("HYDRA_FULL_ERROR", "1")

# 确保相对路径都按 /root/SAM-3D-Grasp 解析
os.chdir(str(REPO_ROOT))


# =========================================================
# 导入你已经跑通的 pipeline
# =========================================================

from pipeline4_interactive import (
    Pipeline4TableAxisVisibleShell,
    auto_get_intrinsics,
)


# =========================================================
# 路径工具
# =========================================================

def _resolve_path(path_like: str, must_exist: bool = False) -> Path:
    """
    兼容以下路径写法：
      data/...
      ./data/...
      SAM-3D-Grasp/data/...
      /root/SAM-3D-Grasp/data/...
    """
    if path_like is None:
        raise ValueError("路径不能为空")

    raw = str(path_like).strip().strip('"').strip("'")

    if not raw:
        raise ValueError("路径不能为空")

    p = Path(raw)

    if p.is_absolute():
        q = p
    else:
        raw_norm = raw.replace("\\", "/")
        repo_prefix = REPO_ROOT.name + "/"

        if raw_norm.startswith(repo_prefix):
            q = REPO_ROOT / raw_norm[len(repo_prefix):]
        else:
            q = REPO_ROOT / p

    if must_exist and not q.exists():
        raise FileNotFoundError(f"找不到文件: {q}")

    return q


def _make_out_root(prefix: str = "run") -> Path:
    """
    自动创建本次运行输出目录。
    """
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = REPO_ROOT / "output_pipeline4_skill" / f"{prefix}_{run_id}"
    out_root.mkdir(parents=True, exist_ok=True)
    return out_root


def _make_pipeline():
    """
    创建 pipeline 实例。

    兼容两种类定义：
      1. Pipeline4TableAxisVisibleShell(ds_key)
      2. Pipeline4TableAxisVisibleShell(ds_key, force_target_en=None)
    """
    deepseek_key = os.environ.get("DEEPSEEK_KEY", "<YOUR_DEEPSEEK_KEY>")

    try:
        sig = inspect.signature(Pipeline4TableAxisVisibleShell)
        params = sig.parameters

        if "force_target_en" in params:
            return Pipeline4TableAxisVisibleShell(
                ds_key=deepseek_key,
                force_target_en=None,
            )

        return Pipeline4TableAxisVisibleShell(
            ds_key=deepseek_key,
        )

    except Exception:
        # 兜底写法
        try:
            return Pipeline4TableAxisVisibleShell(
                ds_key=deepseek_key,
            )
        except TypeError:
            return Pipeline4TableAxisVisibleShell(
                ds_key=deepseek_key,
                force_target_en=None,
            )


# =========================================================
# 输出结果整理
# =========================================================

def _collect_result(out_root: Path, camera_intrinsics: Dict[str, Any]) -> Dict[str, Any]:
    """
    pipeline 跑完后，把关键输出路径整理成标准 JSON。
    """
    out_root = Path(out_root)

    result = {
        "status": "success",
        "out_root": str(out_root),

        "mask_png": str(out_root / "00_mask" / "target_mask.png"),
        "mask_npy": str(out_root / "00_mask" / "target_mask.npy"),
        "mask_vis": str(out_root / "00_mask" / "target_2d_result.jpg"),

        "single_sam3d_ply": str(out_root / "01_sam3d_single" / "sam3d_single_clean.ply"),
        "single_pose_json": str(out_root / "01_sam3d_single" / "sam3d_pose.json"),

        "table_plane_json": str(out_root / "02_real_table" / "table_plane.json"),
        "raw_partial_ply": str(out_root / "03_real_partial_raw" / "duck_partial_raw_camera.ply"),

        "joint_sam3d_ply": str(out_root / "05_joint_sam3d" / "sam3d_object_table_joint_clean.ply"),

        "single_table_normal_json": str(out_root / "06_axis_transfer" / "table_normal_in_single_duck_frame.json"),

        "init_transform_json": str(out_root / "07_table_axis_init_align" / "table_axis_constrained_transform.json"),
        "visible_transform_json": str(out_root / "08_visible_shell_align" / "visible_shell_transform.json"),

        "final_pcd": str(out_root / "final" / "sam3d_pure_in_rgbd.ply"),
        "final_pose_json": str(out_root / "final" / "sam3d_pose_in_rgbd.json"),

        "camera_intrinsics": camera_intrinsics,
    }

    final_pose_json = Path(result["final_pose_json"])

    if final_pose_json.exists():
        with open(final_pose_json, "r", encoding="utf-8") as f:
            result["pose"] = json.load(f)

    required_keys = [
        "mask_png",
        "mask_vis",
        "final_pcd",
        "final_pose_json",
    ]

    missing = []
    for key in required_keys:
        if not Path(result[key]).exists():
            missing.append(result[key])

    if missing:
        result["status"] = "partial_success"
        result["missing_files"] = missing

    return result


# =========================================================
# 内部统一执行函数
# =========================================================

def _run_pipeline_once(
    rgb_path: str,
    depth_path: str,
    audio_path: Optional[str] = None,
    text_command: Optional[str] = None,
    out_root: Optional[str] = None,
) -> Dict[str, Any]:
    """
    两个接口最终都会调用这个函数。

    输入：
      rgb_path
      depth_path
      audio_path 或 text_command
      out_root

    输出：
      标准结果 dict
    """

    rgb_path = _resolve_path(rgb_path, must_exist=True)
    depth_path = _resolve_path(depth_path, must_exist=True)

    if audio_path:
        audio_path = _resolve_path(audio_path, must_exist=True)
    else:
        audio_path = None

    if not audio_path and not text_command:
        raise ValueError("audio_path 和 text_command 至少需要提供一个")

    if out_root is None:
        out_root = _make_out_root(prefix="run")
    else:
        out_root = _resolve_path(out_root, must_exist=False)
        out_root.mkdir(parents=True, exist_ok=True)

    # 自动读取相机内参
    fx, fy, cx, cy, width, height = auto_get_intrinsics(
        image_file=str(rgb_path),
        depth_file=str(depth_path),
        default_fx=607.0,
        default_fy=607.0,
    )

    camera_intrinsics = {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "width": width,
        "height": height,
    }

    print("\n" + "=" * 100)
    print("[SAM3D-Grasp Skill] 输入")
    print("=" * 100)
    print(f"RGB   : {rgb_path}")
    print(f"Depth : {depth_path}")
    print(f"Audio : {audio_path}")
    print(f"Text  : {text_command}")
    print(f"K     : {camera_intrinsics}")
    print(f"Out   : {out_root}")
    print("=" * 100)

    pipeline = _make_pipeline()

    run_sig = inspect.signature(pipeline.run)

    run_kwargs = {
        "audio_file": str(audio_path) if audio_path else None,
        "image_file": str(rgb_path),
        "depth_file": str(depth_path),

        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "width": width,
        "height": height,

        "out_root": str(out_root),

        "reuse_single_sam3d_ply": None,
        "reuse_joint_ply": None,

        "fixed_scale": -1.0,
        "height_percentile": 98.0,

        "yaw_step_deg": 5.0,
        "fine_yaw_step_deg": 1.0,

        "visible_scale_min": -1.0,
        "visible_scale_max": -1.0,
        "visible_scale_steps": 5,
        "visible_yaw_range_deg": 25.0,
        "visible_depth_margin": 0.006,
        "camera_z_mode": "min",
    }

    if "text_command" in run_sig.parameters:
        run_kwargs["text_command"] = text_command
    else:
        if text_command:
            raise RuntimeError(
                "你的 pipeline4_interactive.py 的 run() 还不支持 text_command。"
                "请使用已经支持文字模式的版本。"
            )

    pipeline.run(**run_kwargs)

    return _collect_result(out_root, camera_intrinsics)


# =========================================================
# 接口 1：离线文件接口
# =========================================================

def grasp_from_files(
    rgb_path: str,
    depth_path: str,
    audio_path: Optional[str] = None,
    text_command: Optional[str] = None,
    out_root: Optional[str] = None,
) -> Dict[str, Any]:
    """
    离线文件接口。

    用途：
      用已经保存好的 RGB 图、Depth 图、语音文件或文字指令跑 pipeline。

    例子：
      result = grasp_from_files(
          rgb_path="data/test/000002/rgb/000000.png",
          depth_path="data/test/000002/depth/000000.png",
          audio_path="data/test_audio3.m4a"
      )

    注意：
      data/test/... 只是测试数据。
      最终机器人在线运行时不会写死这些路径。
    """
    return _run_pipeline_once(
        rgb_path=rgb_path,
        depth_path=depth_path,
        audio_path=audio_path,
        text_command=text_command,
        out_root=out_root,
    )


# =========================================================
# Provider 抽象：给在线当前场景接口用
# =========================================================

class CurrentSceneProvider:
    """
    当前场景提供器。

    真正接机器人时，需要实现 capture_rgbd()。

    capture_rgbd() 返回：
      rgb_path, depth_path
    """

    def capture_rgbd(self) -> Tuple[str, str]:
        raise NotImplementedError


class VoiceProvider:
    """
    语音提供器。

    真正接机器人时，需要实现 record_once()。

    record_once() 返回：
      audio_path
    """

    def record_once(self) -> str:
        raise NotImplementedError


# =========================================================
# Mock Provider：现在先用已有文件模拟当前相机和麦克风
# =========================================================

class MockSceneProvider(CurrentSceneProvider):
    """
    用已有 RGB-D 文件模拟当前相机场景。

    以后接 RealSense / Orbbec 时，只需要替换这个 Provider。
    """

    def __init__(self, rgb_path: str, depth_path: str):
        self.rgb_path = str(_resolve_path(rgb_path, must_exist=True))
        self.depth_path = str(_resolve_path(depth_path, must_exist=True))

    def capture_rgbd(self) -> Tuple[str, str]:
        print("[MockSceneProvider] 使用已有 RGB-D 模拟当前相机场景")
        print(f"[MockSceneProvider] RGB  : {self.rgb_path}")
        print(f"[MockSceneProvider] Depth: {self.depth_path}")
        return self.rgb_path, self.depth_path


class MockVoiceProvider(VoiceProvider):
    """
    用已有音频文件模拟当前录音。

    以后接真实麦克风时，只需要替换这个 Provider。
    """

    def __init__(self, audio_path: str):
        self.audio_path = str(_resolve_path(audio_path, must_exist=True))

    def record_once(self) -> str:
        print("[MockVoiceProvider] 使用已有音频模拟当前语音")
        print(f"[MockVoiceProvider] Audio: {self.audio_path}")
        return self.audio_path


# =========================================================
# 接口 2：在线当前场景接口
# =========================================================

def grasp_current_scene(
    scene_provider: CurrentSceneProvider,
    voice_provider: Optional[VoiceProvider] = None,
    text_command: Optional[str] = None,
    out_root: Optional[str] = None,
) -> Dict[str, Any]:
    """
    在线当前场景接口。

    流程：
      1. scene_provider.capture_rgbd()
         从当前深度相机采集 RGB 图和 Depth 图。

      2. 如果 text_command 不为空：
           使用文字指令。
         否则：
           voice_provider.record_once()
           录制当前语音文件。

      3. 调用同一个 SAM3D-Grasp pipeline。

    未来真实调用：
      result = grasp_current_scene(
          scene_provider=realsense_provider,
          voice_provider=mic_provider,
      )

    或者 Agent 已经把语音转文字：
      result = grasp_current_scene(
          scene_provider=realsense_provider,
          text_command="抓取白色水壶",
      )
    """

    if scene_provider is None:
        raise ValueError("scene_provider 不能为空")

    rgb_path, depth_path = scene_provider.capture_rgbd()

    audio_path = None

    if text_command:
        print(f"[grasp_current_scene] 使用文字指令: {text_command}")
    else:
        if voice_provider is None:
            raise ValueError("没有 text_command 时，必须提供 voice_provider")
        audio_path = voice_provider.record_once()

    if out_root is None:
        out_root = _make_out_root(prefix="current_scene")

    return _run_pipeline_once(
        rgb_path=rgb_path,
        depth_path=depth_path,
        audio_path=audio_path,
        text_command=text_command,
        out_root=str(out_root),
    )
