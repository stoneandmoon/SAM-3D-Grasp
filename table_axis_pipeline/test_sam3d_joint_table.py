#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import shutil
import argparse
import cv2
import open3d as o3d

os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["PYOPENGL_PLATFORM"] = "egl"

# 建议固定 torch hub 目录，避免跑到 /home/vipuser/.cache 之类的位置
os.environ.setdefault("TORCH_HOME", "/root/.cache/torch")

sys.path.insert(0, os.path.abspath("."))


def patch_torch_hub_for_local_dinov2():
    """
    解决 SAM3D 加载 DINOv2 时访问 GitHub 失败的问题。

    原本 SAM3D 内部会调用：
      torch.hub.load("facebookresearch/dinov2", ...)

    这个函数会访问 GitHub。
    这里把它重定向为本地加载：
      torch.hub.load("/root/.cache/torch/hub/facebookresearch_dinov2_main", ..., source="local")
    """
    import torch

    local_candidates = [
        os.environ.get("DINOV2_LOCAL_REPO", ""),
        "/root/.cache/torch/hub/facebookresearch_dinov2_main",
        os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov2_main"),
        "/home/vipuser/.cache/torch/hub/facebookresearch_dinov2_main",
    ]

    local_repo = None
    for p in local_candidates:
        if p and os.path.isdir(p) and os.path.exists(os.path.join(p, "hubconf.py")):
            local_repo = p
            break

    print("[DINOv2 Patch] torch hub dir:", torch.hub.get_dir())

    if local_repo is None:
        print("\n[DINOv2 Patch][WARN] 没有找到本地 DINOv2 repo。")
        print("请先运行：")
        print("  mkdir -p /root/.cache/torch/hub")
        print("  git clone --depth 1 https://github.com/facebookresearch/dinov2.git \\")
        print("    /root/.cache/torch/hub/facebookresearch_dinov2_main")
        print()
        print("如果你已经有本地 dinov2 repo，可以这样指定：")
        print("  export DINOV2_LOCAL_REPO=/path/to/dinov2")
        print()
        return

    print("[DINOv2 Patch] 使用本地 DINOv2 repo:", local_repo)

    original_load = torch.hub.load

    def patched_load(repo_or_dir, model, *args, **kwargs):
        if isinstance(repo_or_dir, str) and repo_or_dir == "facebookresearch/dinov2":
            print(f"[DINOv2 Patch] redirect torch.hub.load({repo_or_dir}, {model}) -> local")
            kwargs["source"] = "local"

            # local source 下 trust_repo 不需要，某些 torch 版本可能不认
            kwargs.pop("trust_repo", None)

            return original_load(local_repo, model, *args, **kwargs)

        return original_load(repo_or_dir, model, *args, **kwargs)

    torch.hub.load = patched_load


# 必须在导入 SAM3DEngine 前/模型实例化前 patch
patch_torch_hub_for_local_dinov2()

from modules.sam3d_engine import SAM3DEngine


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def copy_if_exists(src, dst):
    if os.path.exists(src):
        d = os.path.dirname(dst)
        if d:
            ensure_dir(d)
        shutil.copy2(src, dst)
        print(f"[Copy] {src} -> {dst}")
        return True
    return False


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        default="data/test/000002/rgb/000000.png",
        help="RGB image path"
    )

    parser.add_argument(
        "--mask",
        default="output_duck_table_joint_mask/duck_plus_local_table_mask.png",
        help="联合 mask 路径：鸭子 + 局部桌面"
    )

    parser.add_argument(
        "--out-dir",
        default="output_3d_duck_table_joint",
        help="保存 joint SAM3D 输出的目录"
    )

    parser.add_argument(
        "--clean-ply",
        default="sam3d_duck_table_joint_clean.ply",
        help="额外保存到根目录的 clean ply 文件名"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("🚀 启动 SAM-3D 联合模型提取模块：鸭子 + 局部桌面")
    print("=" * 80)

    image_file = args.image
    mask_file = args.mask
    out_dir = args.out_dir
    clean_ply_path = args.clean_ply

    print(f"[Input] image: {image_file}")
    print(f"[Input] mask : {mask_file}")
    print(f"[Output] out-dir: {out_dir}")
    print(f"[Output] clean ply: {clean_ply_path}")

    if not os.path.exists(image_file):
        print(f"❌ 找不到 RGB 图像: {image_file}")
        sys.exit(1)

    if not os.path.exists(mask_file):
        print(f"❌ 找不到联合 mask: {mask_file}")
        print("请确认这个文件存在，例如：")
        print("  ./output_duck_table_joint_mask/duck_plus_local_table_mask.png")
        sys.exit(1)

    mask_img = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        print(f"❌ 无法读取掩码文件: {mask_file}")
        sys.exit(1)

    mask_bool = mask_img > 127
    mask_pixels = int(mask_bool.sum())

    print(f"[Mask] foreground pixels = {mask_pixels}")

    if mask_pixels <= 0:
        print("❌ mask 是空的，不能送入 SAM3D")
        sys.exit(1)

    print("\n[Step 1] 正在用联合 mask 生成 3D 模型...")
    print("说明：这一步是同时生成 鸭子 + 局部桌面，不是分别生成。")

    sam3d = SAM3DEngine()
    success = sam3d.generate_3d(image_file, mask_bool)

    if not success:
        print("\n❌ 3D 重建失败。")
        print("如果日志仍然显示 GitHub 502，请确认本地 DINOv2 repo 是否存在：")
        print("  ls -lh /root/.cache/torch/hub/facebookresearch_dinov2_main")
        sys.exit(1)

    default_output_dir = "./output_3d"
    raw_obj_path = os.path.join(default_output_dir, "reconstructed_mesh.obj")
    raw_ply_path = os.path.join(default_output_dir, "reconstructed_mesh.ply")
    pose_json_path = os.path.join(default_output_dir, "sam3d_pose.json")

    print("\n[Step 2] 检查 SAM3D 输出文件...")

    if not os.path.exists(raw_obj_path) and not os.path.exists(raw_ply_path):
        print("❌ 没有找到 SAM3D 输出 mesh：")
        print(f"  {raw_obj_path}")
        print(f"  {raw_ply_path}")
        sys.exit(1)

    ensure_dir(out_dir)

    copy_if_exists(image_file, os.path.join(out_dir, "input_rgb.png"))
    copy_if_exists(mask_file, os.path.join(out_dir, "input_joint_mask.png"))

    copy_if_exists(raw_obj_path, os.path.join(out_dir, "reconstructed_mesh.obj"))
    copy_if_exists(raw_ply_path, os.path.join(out_dir, "reconstructed_mesh_raw.ply"))
    copy_if_exists(pose_json_path, os.path.join(out_dir, "sam3d_pose.json"))

    print("\n[Step 3] 正在清洗并导出单一 PLY...")

    if os.path.exists(raw_obj_path):
        mesh = o3d.io.read_triangle_mesh(raw_obj_path)

        if mesh is None or len(mesh.vertices) == 0:
            print(f"❌ Open3D 读取 OBJ 失败或为空: {raw_obj_path}")
            sys.exit(1)

        mesh.compute_vertex_normals()

        out_mesh_ply = os.path.join(out_dir, "reconstructed_mesh.ply")

        ok = o3d.io.write_triangle_mesh(out_mesh_ply, mesh)
        if not ok:
            print(f"❌ 写出 PLY 失败: {out_mesh_ply}")
            sys.exit(1)

        ok = o3d.io.write_triangle_mesh(clean_ply_path, mesh)
        if not ok:
            print(f"❌ 写出根目录 clean PLY 失败: {clean_ply_path}")
            sys.exit(1)

        print(f"[Save] joint mesh ply: {out_mesh_ply}")
        print(f"[Save] root clean ply : {clean_ply_path}")

    else:
        pcd = o3d.io.read_point_cloud(raw_ply_path)

        if pcd is not None and len(pcd.points) > 0:
            out_mesh_ply = os.path.join(out_dir, "reconstructed_mesh.ply")
            o3d.io.write_point_cloud(out_mesh_ply, pcd)
            o3d.io.write_point_cloud(clean_ply_path, pcd)
            print(f"[Save] joint pcd ply : {out_mesh_ply}")
            print(f"[Save] root clean ply: {clean_ply_path}")
        else:
            print(f"❌ 无法读取 PLY: {raw_ply_path}")
            sys.exit(1)

    print("\n" + "█" * 60)
    print("✅ 联合生成完成！")
    print("这次输出是：鸭子 + 局部桌面，在同一个 SAM3D 坐标系里。")
    print()
    print("重点文件：")
    print(f"  1. 联合生成 mesh/点云: {os.path.abspath(os.path.join(out_dir, 'reconstructed_mesh.ply'))}")
    print(f"  2. 根目录 clean ply  : {os.path.abspath(clean_ply_path)}")
    print(f"  3. 输入 joint mask   : {os.path.abspath(os.path.join(out_dir, 'input_joint_mask.png'))}")
    print("█" * 60)


if __name__ == "__main__":
    main()