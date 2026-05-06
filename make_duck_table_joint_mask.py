#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_duck_table_joint_mask.py

作用：
  1. 读取 RGB / depth / pure_duck_mask.png / table_plane.json
  2. 根据真实桌面平面，从 depth 图里提取鸭子附近的局部桌面 patch
  3. 生成联合 mask：
       duck_plus_local_table_mask.png
  4. 这个联合 mask 用来一次性喂给 SAM3D，让 SAM3D 同时生成：
       鸭子 + 局部桌面支撑面

为什么不是分别生成？
  因为分别生成鸭子和桌子时，两次 SAM3D 输出不一定在同一个坐标系。
  要用桌面反推 SAM3D 侧竖直方向，必须同一次生成“鸭子 + 桌面”。

典型运行：

python make_duck_table_joint_mask.py \
  --rgb ./data/test/000002/rgb/000000.png \
  --depth ./data/test/000002/depth/000000.png \
  --duck-mask ./pure_duck_mask.png \
  --table-plane ./output_table_axis_probe_xy/table_plane.json \
  --fx 607.0 \
  --fy 607.0 \
  --cx 320.0 \
  --cy 240.0 \
  --out-dir ./output_duck_table_joint_mask

输出：
  output_duck_table_joint_mask/local_table_patch_mask.png
  output_duck_table_joint_mask/duck_plus_local_table_mask.png
  output_duck_table_joint_mask/debug_overlay.png

可选：生成后自动调用 SAM3D。
例如你的 SAM3D 入口支持 --mask 参数时：

python make_duck_table_joint_mask.py \
  --rgb ./data/test/000002/rgb/000000.png \
  --depth ./data/test/000002/depth/000000.png \
  --duck-mask ./pure_duck_mask.png \
  --table-plane ./output_table_axis_probe_xy/table_plane.json \
  --fx 607.0 --fy 607.0 --cx 320.0 --cy 240.0 \
  --out-dir ./output_duck_table_joint_mask \
  --sam3d-cmd "python pipeline3.py --rgb {rgb} --mask {mask} --out-dir ./output_3d_duck_table_joint"
"""

import os
import json
import argparse
import subprocess
import numpy as np
import cv2


# ============================================================
# 基础工具
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_rgb(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到 RGB: {path}")
    rgb = cv2.imread(path, cv2.IMREAD_COLOR)
    if rgb is None:
        raise RuntimeError(f"读取 RGB 失败: {path}")
    return rgb


def load_depth(path, depth_scale=1000.0):
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到 depth: {path}")

    depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise RuntimeError(f"读取 depth 失败: {path}")

    depth = depth.astype(np.float32)

    # 常见情况：16-bit depth 单位是 mm，需要转成 m
    if np.nanmax(depth) > 20:
        depth = depth / float(depth_scale)

    depth[~np.isfinite(depth)] = 0.0
    return depth


def load_mask(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到 mask: {path}")

    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"读取 mask 失败: {path}")

    return (mask > 127).astype(np.uint8)


def bbox_from_mask(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise RuntimeError("duck mask 是空的，请检查 pure_duck_mask.png")
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def expand_bbox(x0, y0, x1, y1, W, H, pad):
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(W - 1, x1 + pad)
    y1 = min(H - 1, y1 + pad)
    return x0, y0, x1, y1


# ============================================================
# 鲁棒读取 table_plane.json
# ============================================================

def load_table_plane(json_path):
    """
    超鲁棒读取 table_plane.json。

    目标读取：
      normal = [nx, ny, nz]
      d      = float

    平面形式：
      normal · x + d = 0

    兼容格式示例：
      {"normal": [...], "d": ...}
      {"plane_normal": [...], "plane_d": ...}
      {"normal_camera": [...], "d_camera": ...}
      {"plane_model": [a,b,c,d]}
      {"plane": [a,b,c,d]}
      {"coefficients": [a,b,c,d]}
      嵌套 dict 里的以上字段
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"找不到 table_plane.json: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("\n[DEBUG] table_plane.json 顶层 keys:")
    if isinstance(data, dict):
        print("  ", list(data.keys()))
    else:
        print("  ", type(data), data)

    def is_num(x):
        return isinstance(x, (int, float, np.integer, np.floating))

    def as_float_list(x):
        if isinstance(x, np.ndarray):
            x = x.tolist()
        if isinstance(x, (list, tuple)) and all(is_num(v) for v in x):
            return [float(v) for v in x]
        return None

    def recursive_items(obj, prefix=""):
        items = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                p = f"{prefix}.{k}" if prefix else str(k)
                items.append((p, v))
                items.extend(recursive_items(v, p))
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                p = f"{prefix}[{i}]"
                items.append((p, v))
                items.extend(recursive_items(v, p))
        return items

    def normalize_plane(normal, d, source_desc):
        normal = np.asarray(normal, dtype=np.float64)
        d = float(d)

        norm = float(np.linalg.norm(normal))
        if norm < 1e-12:
            raise RuntimeError(f"平面 normal 长度接近 0，来源: {source_desc}")

        normal = normal / norm
        d = d / norm

        print(f"[Plane] 使用来源: {source_desc}")
        print(f"[Plane] normal = {normal.tolist()}")
        print(f"[Plane] d      = {d:.8f}")
        print("[Plane] 平面形式: normal · x + d = 0")

        return normal, d

    # ------------------------------------------------------------
    # 1. 优先查找同一个 dict 中的 normal + d
    # ------------------------------------------------------------
    normal_keys = [
        "normal",
        "plane_normal",
        "normal_camera",
        "table_normal",
        "table_plane_normal",
        "n",
    ]
    d_keys = [
        "d",
        "plane_d",
        "d_camera",
        "table_d",
        "table_plane_d",
        "offset",
    ]

    for path, value in recursive_items(data):
        if not isinstance(value, dict):
            continue

        for nk in normal_keys:
            if nk not in value:
                continue

            n_arr = as_float_list(value[nk])
            if n_arr is None or len(n_arr) != 3:
                continue

            for dk in d_keys:
                if dk in value and is_num(value[dk]):
                    return normalize_plane(
                        n_arr,
                        value[dk],
                        f"{path}: {nk} + {dk}"
                    )

    # ------------------------------------------------------------
    # 2. 查找长度为 4 的 plane 参数
    # ------------------------------------------------------------
    candidates4 = []
    for path, value in recursive_items(data):
        arr = as_float_list(value)
        if arr is not None and len(arr) == 4:
            lower = path.lower()
            score = 0
            if "plane" in lower:
                score += 10
            if "model" in lower:
                score += 5
            if "coeff" in lower or "coef" in lower:
                score += 5
            if "table" in lower:
                score += 4
            if "camera" in lower:
                score += 2
            candidates4.append((score, path, arr))

    if len(candidates4) > 0:
        candidates4 = sorted(candidates4, key=lambda x: -x[0])
        score, path, arr = candidates4[0]
        return normalize_plane(
            arr[:3],
            arr[3],
            f"{path} = {arr}"
        )

    # ------------------------------------------------------------
    # 3. 全局查找 normal 候选和 d 候选
    # ------------------------------------------------------------
    all_items = recursive_items(data)

    normal_candidates = []
    d_candidates = []

    for path, value in all_items:
        lower = path.lower()

        arr = as_float_list(value)
        if arr is not None and len(arr) == 3:
            score = 0
            if "normal" in lower:
                score += 10
            if "plane" in lower:
                score += 5
            if "table" in lower:
                score += 4
            if "camera" in lower:
                score += 2
            normal_candidates.append((score, path, arr))

        if is_num(value):
            score = 0
            if lower.endswith(".d") or lower == "d":
                score += 10
            if "plane_d" in lower:
                score += 10
            if "offset" in lower:
                score += 6
            if "table" in lower:
                score += 4
            if "camera" in lower:
                score += 2
            d_candidates.append((score, path, float(value)))

    normal_candidates = sorted(normal_candidates, key=lambda x: -x[0])
    d_candidates = sorted(d_candidates, key=lambda x: -x[0])

    if len(normal_candidates) > 0 and len(d_candidates) > 0:
        n_score, n_path, n_arr = normal_candidates[0]
        d_score, d_path, d_val = d_candidates[0]

        return normalize_plane(
            n_arr,
            d_val,
            f"normal from {n_path}, d from {d_path}"
        )

    print("\n[ERROR] 无法自动识别 table_plane.json 格式。文件内容如下：")
    print(json.dumps(data, indent=2, ensure_ascii=False))

    raise RuntimeError(
        "table_plane.json 格式不识别。请把上面打印出来的 json 内容发我。"
    )


# ============================================================
# 生成局部桌面 patch mask
# ============================================================

def project_depth_to_camera_xyz(depth, fx, fy, cx, cy):
    H, W = depth.shape[:2]
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

    z = depth.astype(np.float32)
    x = (xx.astype(np.float32) - float(cx)) * z / float(fx)
    y = (yy.astype(np.float32) - float(cy)) * z / float(fy)

    return x, y, z


def clean_binary_mask(mask_u8, min_area=40):
    """
    形态学清理 + 连通域过滤。
    """
    mask_u8 = (mask_u8 > 0).astype(np.uint8) * 255

    k3 = np.ones((3, 3), np.uint8)
    k5 = np.ones((5, 5), np.uint8)

    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, k3)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, k5)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

    cleaned = np.zeros_like(mask_u8)
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= min_area:
            cleaned[labels == i] = 255

    return (cleaned > 127).astype(np.uint8)


def compute_local_table_patch_mask_once(
    depth,
    duck_mask,
    plane_normal,
    plane_d,
    fx,
    fy,
    cx,
    cy,
    bbox_pad,
    plane_thresh,
    prefer_bottom=True,
    min_component_area=40,
):
    """
    单次参数下生成局部桌面 patch。
    """
    H, W = depth.shape[:2]

    valid = depth > 1e-6

    x, y, z = project_depth_to_camera_xyz(depth, fx, fy, cx, cy)

    # 点到桌面平面的有符号距离
    dist = (
        plane_normal[0] * x +
        plane_normal[1] * y +
        plane_normal[2] * z +
        plane_d
    )

    table_candidate = (np.abs(dist) < float(plane_thresh)) & valid

    # 限定在鸭子 bbox 周围
    x0, y0, x1, y1 = bbox_from_mask(duck_mask)
    ex0, ey0, ex1, ey1 = expand_bbox(x0, y0, x1, y1, W, H, int(bbox_pad))

    local_region = np.zeros((H, W), dtype=bool)
    local_region[ey0:ey1 + 1, ex0:ex1 + 1] = True

    table_mask = table_candidate & local_region

    # 排除鸭子本身
    table_mask = table_mask & (duck_mask == 0)

    if prefer_bottom:
        # 偏向鸭子下半部分附近的桌面，但不只取正下方
        obj_h = y1 - y0 + 1
        bottom_start = int(y0 + 0.35 * obj_h)

        bottom_region = np.zeros((H, W), dtype=bool)
        by0 = max(0, bottom_start - int(bbox_pad) // 3)
        by1 = min(H - 1, y1 + int(bbox_pad))
        bottom_region[by0:by1 + 1, ex0:ex1 + 1] = True

        table_mask = table_mask & bottom_region

    table_mask = clean_binary_mask(
        table_mask.astype(np.uint8) * 255,
        min_area=min_component_area,
    )

    return table_mask


def maybe_limit_table_ratio(table_mask, duck_mask, max_table_ratio):
    """
    防止桌面 patch 太大。
    若 table 像素超过 duck 像素的 max_table_ratio 倍，则只保留离鸭子较近的桌面像素。
    """
    duck_pixels = int(duck_mask.sum())
    table_pixels = int(table_mask.sum())

    if duck_pixels <= 0 or table_pixels <= 0:
        return table_mask

    max_pixels = int(max_table_ratio * duck_pixels)
    if table_pixels <= max_pixels:
        return table_mask

    # 计算每个 table 像素到 duck mask 的距离，保留最近的 max_pixels 个
    inv_duck = (duck_mask == 0).astype(np.uint8)
    dist = cv2.distanceTransform(inv_duck, cv2.DIST_L2, 5)

    ys, xs = np.where(table_mask > 0)
    vals = dist[ys, xs]

    order = np.argsort(vals)
    keep_idx = order[:max_pixels]

    limited = np.zeros_like(table_mask, dtype=np.uint8)
    limited[ys[keep_idx], xs[keep_idx]] = 1

    limited = clean_binary_mask(limited * 255, min_area=20)

    print(
        f"[Limit] table patch 太大，已限制: "
        f"{table_pixels} -> {int(limited.sum())}, "
        f"max_table_ratio={max_table_ratio}"
    )

    return limited


def compute_local_table_patch_mask(
    depth,
    duck_mask,
    plane_normal,
    plane_d,
    fx,
    fy,
    cx,
    cy,
    bbox_pad=60,
    plane_thresh=0.012,
    prefer_bottom=True,
    auto_relax=True,
    min_table_pixels=80,
    max_table_ratio=1.2,
):
    """
    自动生成局部桌面 patch。

    如果第一次桌面点太少，会自动放宽：
      - 增大 bbox_pad
      - 增大 plane_thresh
      - 最后取消 bottom 偏置
    """
    attempts = []

    attempts.append({
        "bbox_pad": bbox_pad,
        "plane_thresh": plane_thresh,
        "prefer_bottom": prefer_bottom,
    })

    if auto_relax:
        attempts.append({
            "bbox_pad": int(bbox_pad * 1.4),
            "plane_thresh": plane_thresh * 1.5,
            "prefer_bottom": prefer_bottom,
        })
        attempts.append({
            "bbox_pad": int(bbox_pad * 1.8),
            "plane_thresh": plane_thresh * 2.0,
            "prefer_bottom": False,
        })

    best_mask = None
    best_pixels = -1

    for i, cfg in enumerate(attempts):
        mask = compute_local_table_patch_mask_once(
            depth=depth,
            duck_mask=duck_mask,
            plane_normal=plane_normal,
            plane_d=plane_d,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            bbox_pad=cfg["bbox_pad"],
            plane_thresh=cfg["plane_thresh"],
            prefer_bottom=cfg["prefer_bottom"],
        )

        pixels = int(mask.sum())

        print(
            f"[Attempt {i+1}] "
            f"bbox_pad={cfg['bbox_pad']}, "
            f"plane_thresh={cfg['plane_thresh']:.5f}, "
            f"prefer_bottom={cfg['prefer_bottom']}, "
            f"table_pixels={pixels}"
        )

        if pixels > best_pixels:
            best_pixels = pixels
            best_mask = mask

        if pixels >= min_table_pixels:
            best_mask = mask
            break

    if best_mask is None:
        best_mask = np.zeros_like(duck_mask, dtype=np.uint8)

    best_mask = maybe_limit_table_ratio(
        best_mask,
        duck_mask,
        max_table_ratio=max_table_ratio,
    )

    return best_mask


# ============================================================
# 可视化输出
# ============================================================

def overlay_debug(rgb, duck_mask, table_mask):
    """
    OpenCV 是 BGR。
    红色 = 鸭子
    绿色 = 桌面 patch
    """
    out = rgb.copy()

    duck_idx = duck_mask > 0
    table_idx = table_mask > 0

    red = np.array([0, 0, 255], dtype=np.float32)
    green = np.array([0, 255, 0], dtype=np.float32)

    alpha = 0.55

    out_f = out.astype(np.float32)

    out_f[duck_idx] = out_f[duck_idx] * (1.0 - alpha) + red * alpha
    out_f[table_idx] = out_f[table_idx] * (1.0 - alpha) + green * alpha

    return np.clip(out_f, 0, 255).astype(np.uint8)


def save_bbox_debug(rgb, duck_mask, out_path):
    vis = rgb.copy()
    x0, y0, x1, y1 = bbox_from_mask(duck_mask)
    cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 0, 255), 2)
    cv2.imwrite(out_path, vis)


def copy_joint_mask_if_needed(src, dst):
    if dst is None or dst.strip() == "":
        return

    dst_dir = os.path.dirname(os.path.abspath(dst))
    if dst_dir:
        ensure_dir(dst_dir)

    img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"无法读取 joint mask 用于复制: {src}")

    cv2.imwrite(dst, img)
    print(f"[Copy] joint mask 已复制到: {os.path.abspath(dst)}")


def run_sam3d_cmd_if_needed(cmd_template, rgb_path, depth_path, mask_path, out_dir):
    """
    可选：生成 joint mask 后自动运行 SAM3D。

    支持占位符：
      {rgb}
      {depth}
      {mask}
      {out_dir}

    例子：
      --sam3d-cmd "python pipeline3.py --rgb {rgb} --mask {mask} --out-dir ./output_3d_duck_table_joint"
    """
    if cmd_template is None or cmd_template.strip() == "":
        return

    cmd = cmd_template
    cmd = cmd.replace("{rgb}", rgb_path)
    cmd = cmd.replace("{depth}", depth_path)
    cmd = cmd.replace("{mask}", mask_path)
    cmd = cmd.replace("{out_dir}", out_dir)

    print("\n" + "=" * 80)
    print("[SAM3D] 准备运行命令：")
    print(cmd)
    print("=" * 80)

    subprocess.run(cmd, shell=True, check=True)

    print("\n[SAM3D] 命令执行完成。")


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--rgb", required=True, help="RGB 图像路径")
    parser.add_argument("--depth", required=True, help="Depth 图像路径")
    parser.add_argument("--duck-mask", required=True, help="纯鸭子 mask，例如 pure_duck_mask.png")
    parser.add_argument("--table-plane", required=True, help="table_plane.json")

    parser.add_argument("--fx", type=float, required=True)
    parser.add_argument("--fy", type=float, required=True)
    parser.add_argument("--cx", type=float, required=True)
    parser.add_argument("--cy", type=float, required=True)

    parser.add_argument("--depth-scale", type=float, default=1000.0)

    parser.add_argument("--bbox-pad", type=int, default=60)
    parser.add_argument("--plane-thresh", type=float, default=0.012)
    parser.add_argument("--min-table-pixels", type=int, default=80)
    parser.add_argument("--max-table-ratio", type=float, default=1.2)

    parser.add_argument("--no-bottom-prefer", action="store_true")
    parser.add_argument("--no-auto-relax", action="store_true")

    parser.add_argument("--out-dir", default="./output_duck_table_joint_mask")

    parser.add_argument(
        "--copy-joint-to",
        default="",
        help="可选：把 joint mask 额外复制到某个路径，比如 ./sam3d_joint_mask.png"
    )

    parser.add_argument(
        "--sam3d-cmd",
        default="",
        help=(
            "可选：生成 joint mask 后自动运行 SAM3D。"
            "支持占位符 {rgb} {depth} {mask} {out_dir}"
        )
    )

    args = parser.parse_args()

    ensure_dir(args.out_dir)

    print("=" * 80)
    print("[Start] 生成鸭子 + 局部桌面联合 mask")
    print("=" * 80)

    print(f"[RGB]         {args.rgb}")
    print(f"[Depth]       {args.depth}")
    print(f"[Duck mask]   {args.duck_mask}")
    print(f"[Table plane] {args.table_plane}")
    print(f"[Intrinsics]  fx={args.fx}, fy={args.fy}, cx={args.cx}, cy={args.cy}")

    rgb = load_rgb(args.rgb)
    depth = load_depth(args.depth, depth_scale=args.depth_scale)
    duck_mask = load_mask(args.duck_mask)

    H, W = depth.shape[:2]
    if duck_mask.shape[:2] != (H, W):
        raise RuntimeError(
            f"duck mask 尺寸 {duck_mask.shape[:2]} 和 depth 尺寸 {(H, W)} 不一致。"
            "请确认 pure_duck_mask.png 与当前 RGB/depth 是同一帧。"
        )

    if rgb.shape[:2] != (H, W):
        print(
            f"[WARN] RGB 尺寸 {rgb.shape[:2]} 和 depth 尺寸 {(H, W)} 不一致。"
            "debug overlay 可能不准确。"
        )

    plane_normal, plane_d = load_table_plane(args.table_plane)

    # 打印 duck bbox
    x0, y0, x1, y1 = bbox_from_mask(duck_mask)
    duck_pixels = int(duck_mask.sum())

    print("\n[Duck mask]")
    print(f"  bbox          = x[{x0},{x1}], y[{y0},{y1}]")
    print(f"  duck_pixels   = {duck_pixels}")

    table_patch_mask = compute_local_table_patch_mask(
        depth=depth,
        duck_mask=duck_mask,
        plane_normal=plane_normal,
        plane_d=plane_d,
        fx=args.fx,
        fy=args.fy,
        cx=args.cx,
        cy=args.cy,
        bbox_pad=args.bbox_pad,
        plane_thresh=args.plane_thresh,
        prefer_bottom=(not args.no_bottom_prefer),
        auto_relax=(not args.no_auto_relax),
        min_table_pixels=args.min_table_pixels,
        max_table_ratio=args.max_table_ratio,
    )

    joint_mask = ((duck_mask > 0) | (table_patch_mask > 0)).astype(np.uint8)

    table_pixels = int(table_patch_mask.sum())
    joint_pixels = int(joint_mask.sum())

    print("\n[Mask stats]")
    print(f"  duck_pixels        = {duck_pixels}")
    print(f"  table_patch_pixels = {table_pixels}")
    print(f"  joint_pixels       = {joint_pixels}")

    if duck_pixels > 0:
        print(f"  table / duck ratio = {table_pixels / duck_pixels:.4f}")

    if table_pixels < args.min_table_pixels:
        print(
            "\n[WARN] 桌面 patch 像素较少。可以尝试：\n"
            "  1. 增大 --bbox-pad，例如 90\n"
            "  2. 增大 --plane-thresh，例如 0.018\n"
            "  3. 加 --no-bottom-prefer\n"
        )

    # 保存结果
    out_duck = os.path.join(args.out_dir, "pure_duck_mask_copy.png")
    out_table = os.path.join(args.out_dir, "local_table_patch_mask.png")
    out_joint = os.path.join(args.out_dir, "duck_plus_local_table_mask.png")
    out_overlay = os.path.join(args.out_dir, "debug_overlay.png")
    out_bbox = os.path.join(args.out_dir, "debug_duck_bbox.png")
    out_info = os.path.join(args.out_dir, "mask_generation_info.json")

    cv2.imwrite(out_duck, duck_mask * 255)
    cv2.imwrite(out_table, table_patch_mask * 255)
    cv2.imwrite(out_joint, joint_mask * 255)

    if rgb.shape[:2] == depth.shape[:2]:
        overlay = overlay_debug(rgb, duck_mask, table_patch_mask)
        cv2.imwrite(out_overlay, overlay)
        save_bbox_debug(rgb, duck_mask, out_bbox)
    else:
        print("[WARN] 跳过 overlay，因为 RGB/depth 尺寸不一致。")

    info = {
        "rgb": args.rgb,
        "depth": args.depth,
        "duck_mask": args.duck_mask,
        "table_plane": args.table_plane,
        "intrinsics": {
            "fx": args.fx,
            "fy": args.fy,
            "cx": args.cx,
            "cy": args.cy,
        },
        "depth_scale": args.depth_scale,
        "bbox_pad": args.bbox_pad,
        "plane_thresh": args.plane_thresh,
        "bottom_prefer": not args.no_bottom_prefer,
        "auto_relax": not args.no_auto_relax,
        "duck_bbox": [x0, y0, x1, y1],
        "duck_pixels": duck_pixels,
        "table_patch_pixels": table_pixels,
        "joint_pixels": joint_pixels,
        "table_over_duck_ratio": float(table_pixels / max(duck_pixels, 1)),
        "plane_normal": plane_normal.tolist(),
        "plane_d": float(plane_d),
        "outputs": {
            "duck_mask_copy": out_duck,
            "local_table_patch_mask": out_table,
            "joint_mask": out_joint,
            "debug_overlay": out_overlay,
            "debug_duck_bbox": out_bbox,
        },
    }

    with open(out_info, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    copy_joint_mask_if_needed(out_joint, args.copy_joint_to)

    print("\n[Save]")
    print(f"  纯鸭子 mask 备份       : {os.path.abspath(out_duck)}")
    print(f"  局部桌面 patch mask    : {os.path.abspath(out_table)}")
    print(f"  鸭子+桌面联合 mask     : {os.path.abspath(out_joint)}")
    print(f"  debug overlay          : {os.path.abspath(out_overlay)}")
    print(f"  debug bbox             : {os.path.abspath(out_bbox)}")
    print(f"  info json              : {os.path.abspath(out_info)}")

    print("\n颜色说明 debug_overlay.png：")
    print("  红色 = 原始鸭子 mask")
    print("  绿色 = 局部桌面 patch")
    print("\n后面喂给 SAM3D 的 mask 是：")
    print(f"  {os.path.abspath(out_joint)}")

    # 可选：自动运行 SAM3D
    run_sam3d_cmd_if_needed(
        args.sam3d_cmd,
        rgb_path=args.rgb,
        depth_path=args.depth,
        mask_path=os.path.abspath(out_joint),
        out_dir=args.out_dir,
    )

    print("\n✅ 完成。")
    print("=" * 80)


if __name__ == "__main__":
    main()